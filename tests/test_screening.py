"""
Sync orchestration tests.

``services/screening.py`` is the highest-risk code in the project — it is the
whole import-parse-score pipeline, it touches four external APIs, and it was the
least covered. These tests drive it through a fake Workspace client so the full
path runs with no network, no Google credentials and no model server.

Several cases here are regressions for bugs in the previous implementation,
notably the one where an applicant without a readable resume link was written to
the tracking sheet but silently never appeared in the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.contrib.auth.models import User

from hiring_app.models import Campaign, Candidate
from hiring_app.services.google_workspace import (
    Q_EMAIL,
    Q_GITHUB,
    Q_NAME,
    Q_RESUME,
    WorkspaceClient,
    WorkspaceError,
)
from hiring_app.services.screening import (
    ensure_job_spec,
    rescore_campaign,
    resume_dir,
    sync_campaign,
)

pytestmark = pytest.mark.django_db

QID = {Q_EMAIL: "q-email", Q_NAME: "q-name", Q_RESUME: "q-resume", Q_GITHUB: "q-github"}
DRIVE = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view"


def response(response_id: str, *, email="", name="", resume="", github="") -> dict:
    """Build a Google Forms response payload in the API's actual shape."""
    answers = {}
    for qid, value in (
        (QID[Q_EMAIL], email), (QID[Q_NAME], name),
        (QID[Q_RESUME], resume), (QID[Q_GITHUB], github),
    ):
        if value:
            answers[qid] = {"textAnswers": {"answers": [{"value": value}]}}
    return {
        "responseId": response_id,
        "createTime": "2026-08-01T10:00:00.000Z",
        "answers": answers,
    }


class FakeWorkspaceClient:
    """Stands in for WorkspaceClient without touching Google.

    Reuses the real ``answer`` and ``extract_file_id`` implementations so the
    parsing logic under test is genuinely exercised, not re-implemented.
    """

    answer = staticmethod(WorkspaceClient.answer)
    extract_file_id = staticmethod(WorkspaceClient.extract_file_id)

    def __init__(self, responses=None, *, download_error=None, sheet_error=None):
        self.responses = responses or []
        self.download_error = download_error
        self.sheet_error = sheet_error
        self.appended: list[list] = []
        self.downloads: list[str] = []

    def list_responses(self, form_id):
        return self.responses

    def download_pdf(self, file_id, destination):
        self.downloads.append(file_id)
        if self.download_error:
            raise WorkspaceError(self.download_error)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-1.4 fake")
        return destination

    def append_rows(self, sheet_id, tab, rows):
        if self.sheet_error:
            raise WorkspaceError(self.sheet_error)
        self.appended.extend(rows)


@pytest.fixture(autouse=True)
def _fake_pdf(monkeypatch, tmp_path, settings):
    """Bypass pypdf — these tests cover orchestration, not PDF decoding."""
    settings.MEDIA_ROOT = tmp_path / "media"

    from matching.parsing.pdf import PDFExtraction

    def fake_extract(path, max_pages=15):
        return PDFExtraction(
            text="Backend engineer with 5 years of Python and Django experience. " * 12,
            page_count=1, pages_read=1, ok=True, reason="ok",
        )

    monkeypatch.setattr("hiring_app.services.screening.extract_pdf_text", fake_extract)


@pytest.fixture
def campaign(db):
    owner = User.objects.create_user(username="rec", password="test-pass-1234")
    return Campaign.objects.create(
        owner=owner, role_title="Backend Engineer", status=Campaign.Status.ACTIVE,
        jd_text="We need a Python engineer with Django.",
        form_id="form-1", sheet_id="sheet-1", question_ids=QID,
        job_spec={"role_title": "Backend Engineer", "must_have_skills": ["Python", "Django"],
                  "min_years": 3.0, "is_technical": True},
    )


class TestSync:
    def test_imports_and_scores_new_responses(self, campaign):
        client = FakeWorkspaceClient([
            response("r1", email="a@example.com", name="Ada", resume=DRIVE),
            response("r2", email="b@example.com", name="Bob", resume=DRIVE),
        ])
        outcome = sync_campaign(campaign, client)

        assert outcome.imported == 2
        assert outcome.failed == 0
        assert Candidate.objects.filter(campaign=campaign).count() == 2

        candidate = Candidate.objects.get(response_id="r1")
        assert candidate.email == "a@example.com"
        assert candidate.is_scored
        assert candidate.scored_at is not None
        assert candidate.submitted_at is not None

    def test_is_idempotent(self, campaign):
        """A sync interrupted halfway must be safe to re-run."""
        payload = [response("r1", email="a@example.com", resume=DRIVE)]
        sync_campaign(campaign, FakeWorkspaceClient(payload))
        second = sync_campaign(campaign, FakeWorkspaceClient(payload))

        assert second.imported == 0
        assert second.skipped == 1
        assert Candidate.objects.filter(campaign=campaign).count() == 1

    def test_candidate_without_resume_link_is_still_recorded(self, campaign):
        """Regression: these applicants used to vanish from the dashboard.

        The old code only appended a candidate inside the `if file_id:` branch,
        so anyone whose Drive link was missing or unparseable was written to the
        sheet and then never seen again.
        """
        client = FakeWorkspaceClient([response("r1", email="a@example.com", name="Ada")])
        outcome = sync_campaign(campaign, client)

        assert outcome.imported == 1
        candidate = Candidate.objects.get(response_id="r1")
        assert candidate.email == "a@example.com"
        assert any("No resume link" in w for w in candidate.warnings)

    def test_unparseable_drive_link_is_reported_not_swallowed(self, campaign):
        client = FakeWorkspaceClient([
            response("r1", email="a@example.com", resume="see my attachment")
        ])
        sync_campaign(campaign, client)

        candidate = Candidate.objects.get(response_id="r1")
        assert any("Drive file ID" in w for w in candidate.warnings)
        assert client.downloads == []

    def test_download_failure_keeps_the_candidate_with_a_warning(self, campaign):
        """A failed download must never look like a genuinely weak candidate."""
        client = FakeWorkspaceClient(
            [response("r1", email="a@example.com", resume=DRIVE)],
            download_error="Anyone-with-the-link sharing is not enabled.",
        )
        outcome = sync_campaign(campaign, client)

        assert outcome.imported == 1
        candidate = Candidate.objects.get(response_id="r1")
        assert any("sharing" in w for w in candidate.warnings)
        # Low confidence, not a confidently low score.
        assert candidate.confidence < 0.6

    def test_one_bad_response_does_not_abort_the_sync(self, campaign, monkeypatch):
        from hiring_app.services import screening

        original = screening._screen_one
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("corrupt response")
            return original(**kwargs)

        monkeypatch.setattr(screening, "_screen_one", flaky)

        outcome = sync_campaign(campaign, FakeWorkspaceClient([
            response("r1", email="a@example.com", resume=DRIVE),
            response("r2", email="b@example.com", resume=DRIVE),
        ]))

        assert outcome.failed == 1
        assert outcome.imported == 1
        assert outcome.errors and "corrupt response" in outcome.errors[0]

    def test_sheet_failure_does_not_lose_candidates(self, campaign):
        """The Sheet is a synced export, not the system of record."""
        client = FakeWorkspaceClient(
            [response("r1", email="a@example.com", resume=DRIVE)],
            sheet_error="quota exceeded",
        )
        outcome = sync_campaign(campaign, client)

        assert Candidate.objects.filter(campaign=campaign).count() == 1
        assert any("Sheet update failed" in e for e in outcome.errors)

    def test_writes_a_sheet_row_per_candidate(self, campaign):
        client = FakeWorkspaceClient([
            response("r1", email="a@example.com", name="Ada", resume=DRIVE),
        ])
        sync_campaign(campaign, client)
        assert len(client.appended) == 1
        assert "a@example.com" in client.appended[0]

    def test_github_handle_is_extracted_from_a_full_url(self, campaign):
        client = FakeWorkspaceClient([
            response("r1", email="a@example.com", resume=DRIVE,
                     github="https://github.com/octocat"),
        ])
        sync_campaign(campaign, client)
        assert Candidate.objects.get(response_id="r1").github_username == "octocat"

    def test_last_synced_timestamp_is_updated(self, campaign):
        assert campaign.last_synced_at is None
        sync_campaign(campaign, FakeWorkspaceClient([]))
        campaign.refresh_from_db()
        assert campaign.last_synced_at is not None

    def test_campaign_without_a_form_is_rejected(self, campaign):
        campaign.form_id = ""
        campaign.save()
        with pytest.raises(WorkspaceError, match="no application form"):
            sync_campaign(campaign, FakeWorkspaceClient([]))

    def test_progress_callback_is_driven(self, campaign):
        seen = []
        sync_campaign(
            campaign,
            FakeWorkspaceClient([
                response("r1", email="a@example.com", resume=DRIVE),
                response("r2", email="b@example.com", resume=DRIVE),
            ]),
            progress=lambda done, total, msg: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]


class TestRescore:
    def test_rescores_from_stored_text_without_google(self, campaign):
        sync_campaign(campaign, FakeWorkspaceClient([
            response("r1", email="a@example.com", resume=DRIVE),
        ]))
        candidate = Candidate.objects.get(response_id="r1")
        first_scored_at = candidate.scored_at

        # No client is passed at all — re-scoring must not need Google.
        outcome = rescore_campaign(campaign)

        assert outcome.scored == 1
        candidate.refresh_from_db()
        assert candidate.scored_at > first_scored_at

    def test_handles_an_empty_campaign(self, campaign):
        assert rescore_campaign(campaign).scored == 0


class TestJobSpecCaching:
    def test_existing_spec_is_reused_without_a_model_call(self, campaign):
        from matching.config import get_config
        from matching.pipeline import ScreeningPipeline

        pipeline = ScreeningPipeline(get_config())
        before = len(pipeline.provider.calls)
        spec = ensure_job_spec(campaign, pipeline)

        assert spec.must_have_skills == ["Python", "Django"]
        assert len(pipeline.provider.calls) == before  # nothing was generated

    def test_spec_is_distilled_and_persisted_when_missing(self, campaign):
        from matching.config import get_config
        from matching.pipeline import ScreeningPipeline

        campaign.job_spec = {}
        campaign.save()

        ensure_job_spec(campaign, ScreeningPipeline(get_config()))
        campaign.refresh_from_db()
        assert campaign.job_spec  # written back for the next sync


class TestResumeStorage:
    def test_paths_are_namespaced_per_owner_and_campaign(self, campaign):
        path = resume_dir(campaign)
        assert f"user_{campaign.owner_id}" in str(path)
        assert f"campaign_{campaign.id}" in str(path)
        assert path.exists()
