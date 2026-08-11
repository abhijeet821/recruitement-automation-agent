"""
Web-layer tests, concentrated on the defects fixed during the rewrite.

Each test here corresponds to a real bug in the previous implementation, so the
suite doubles as a regression record.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from hiring_app.models import Campaign, Candidate, GoogleOAuthToken

pytestmark = pytest.mark.django_db


@pytest.fixture
def user(db):
    return User.objects.create_user(username="recruiter", password="test-pass-1234")


@pytest.fixture
def other_user(db):
    return User.objects.create_user(username="rival", password="test-pass-1234")


@pytest.fixture
def campaign(user):
    return Campaign.objects.create(
        owner=user, role_title="Backend Engineer", status=Campaign.Status.ACTIVE,
        jd_text="We need a Python engineer.", form_id="form-1", form_url="https://x/f",
        sheet_id="sheet-1",
    )


@pytest.fixture
def client_in(client, user):
    client.force_login(user)
    return client


class TestTokenEncryption:
    def test_token_is_not_stored_in_plaintext(self, user):
        secret = '{"refresh_token": "super-secret-value"}'
        record = GoogleOAuthToken(user=user)
        record.token_json = secret
        record.save()

        record.refresh_from_db()
        assert "super-secret-value" not in record.encrypted_token
        assert record.token_json == secret

    def test_ciphertext_differs_between_saves(self, user, other_user):
        """Fernet includes a random IV, so identical plaintext must not collide."""
        a = GoogleOAuthToken(user=user)
        a.token_json = "same"
        b = GoogleOAuthToken(user=other_user)
        b.token_json = "same"
        assert a.encrypted_token != b.encrypted_token


class TestAccessControl:
    def test_campaign_of_another_user_is_not_visible(self, client, other_user, campaign):
        client.force_login(other_user)
        response = client.get(reverse("campaign_detail", args=[campaign.pk]))
        assert response.status_code == 404

    def test_anonymous_is_redirected_to_login(self, client, campaign):
        response = client.get(reverse("campaign_detail", args=[campaign.pk]))
        assert response.status_code == 302
        assert "/login/" in response["Location"]


class TestMutationsRequirePost:
    """Previously /sync-responses/ was a GET link — prefetchable and CSRF-able."""

    @pytest.mark.parametrize("name", [
        "sync_campaign", "rescore_campaign", "launch_campaign",
        "generate_jd", "save_jd", "send_invites", "send_outcomes", "campaign_delete",
    ])
    def test_get_is_rejected(self, client_in, campaign, name):
        response = client_in.get(reverse(name, args=[campaign.pk]))
        assert response.status_code == 405

    def test_google_disconnect_requires_post(self, client_in):
        assert client_in.get(reverse("google_disconnect")).status_code == 405


class TestCandidateModel:
    def test_duplicate_response_id_is_rejected(self, campaign):
        """Idempotent sync: re-importing a response must not duplicate it."""
        from django.db import IntegrityError

        Candidate.objects.create(campaign=campaign, response_id="r1", email="a@x.com")
        with pytest.raises(IntegrityError):
            Candidate.objects.create(campaign=campaign, response_id="r1", email="b@x.com")

    def test_same_response_id_allowed_in_different_campaigns(self, user, campaign):
        second = Campaign.objects.create(owner=user, role_title="Other")
        Candidate.objects.create(campaign=campaign, response_id="r1")
        Candidate.objects.create(campaign=second, response_id="r1")
        assert Candidate.objects.filter(response_id="r1").count() == 2

    def test_apply_score_populates_indexed_columns(self, campaign):
        from matching.schemas import CandidateScore

        candidate = Candidate.objects.create(campaign=campaign, response_id="r1")
        candidate.apply_score(CandidateScore(
            overall=82.5, recommendation="STRONG_YES", confidence=0.9
        ))
        candidate.save()
        candidate.refresh_from_db()

        assert candidate.overall_score == 82.5
        assert candidate.recommendation == "STRONG_YES"
        assert candidate.status == Candidate.Status.SCORED
        assert candidate.scored_at is not None

    def test_needs_review_tracks_confidence_not_score(self, campaign):
        """Low confidence means 'we could not tell', not 'they are weak'."""
        candidate = Candidate(campaign=campaign, overall_score=90.0, confidence=0.3)
        assert candidate.needs_review
        candidate.confidence = 0.8
        assert not candidate.needs_review

    def test_ordering_is_by_score_descending(self, campaign):
        Candidate.objects.create(campaign=campaign, response_id="a", overall_score=40)
        Candidate.objects.create(campaign=campaign, response_id="b", overall_score=90)
        Candidate.objects.create(campaign=campaign, response_id="c", overall_score=65)
        assert [c.overall_score for c in campaign.candidates.all()] == [90, 65, 40]


class TestRecruiterRating:
    def test_rating_is_saved(self, client_in, campaign):
        candidate = Candidate.objects.create(campaign=campaign, response_id="r1")
        response = client_in.post(
            reverse("rate_candidate", args=[campaign.pk, candidate.pk]),
            {"rating": "4", "note": "Strong systems background"},
        )
        assert response.status_code == 302
        candidate.refresh_from_db()
        assert candidate.recruiter_rating == 4
        assert "systems" in candidate.recruiter_note

    def test_out_of_range_rating_is_rejected(self, client_in, campaign):
        candidate = Candidate.objects.create(campaign=campaign, response_id="r1")
        client_in.post(
            reverse("rate_candidate", args=[campaign.pk, candidate.pk]), {"rating": "9"}
        )
        candidate.refresh_from_db()
        assert candidate.recruiter_rating is None


class TestICS:
    def test_naive_datetime_is_refused(self):
        """The old code silently used the server timezone, shifting every invite."""
        from hiring_app.services.google_workspace import WorkspaceClient

        client = WorkspaceClient.__new__(WorkspaceClient)
        with pytest.raises(ValueError, match="timezone-aware"):
            client.send_invite(
                "a@x.com", "s", "b",
                organiser_name="O", organiser_email="o@x.com", role="R",
                start=datetime(2026, 9, 1, 10, 0),
            )

    def test_ics_is_built_in_utc_with_escaping(self):
        from hiring_app.services.google_workspace import _build_ics

        ics = _build_ics(
            organiser_name="Acme, Inc.", organiser_email="o@x.com",
            attendee_email="c@x.com", role="Engineer, Backend",
            start=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            duration_minutes=45,
        )
        assert "DTSTART:20260901T100000Z" in ics
        assert "DTEND:20260901T104500Z" in ics
        # Unescaped commas corrupt an iCalendar property.
        assert "Engineer\\, Backend" in ics
        assert "Acme\\, Inc." in ics


class TestDriveFileId:
    @pytest.mark.parametrize("url,expected", [
        ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view", True),
        ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz012345", True),
        ("https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/edit", True),
        ("1AbCdEfGhIjKlMnOpQrStUvWxYz012345", True),
        ("not a link at all", False),
        ("", False),
    ])
    def test_extraction(self, url, expected):
        from hiring_app.services.google_workspace import WorkspaceClient

        assert bool(WorkspaceClient.extract_file_id(url)) is expected


class TestJobProgress:
    def test_percent_handles_zero_total(self, user, campaign):
        from hiring_app.models import BackgroundJob

        job = BackgroundJob.objects.create(
            owner=user, campaign=campaign, kind=BackgroundJob.Kind.SYNC
        )
        assert job.percent == 0

    def test_status_endpoint_reports_progress(self, client_in, user, campaign):
        from hiring_app.models import BackgroundJob

        BackgroundJob.objects.create(
            owner=user, campaign=campaign, kind=BackgroundJob.Kind.SYNC,
            status=BackgroundJob.Status.RUNNING, total=10, processed=4,
        )
        response = client_in.get(reverse("job_status", args=[campaign.pk]))
        assert response.status_code == 200
        assert response.json()["percent"] == 40
        assert response.json()["finished"] is False


class TestPages:
    def test_dashboard_renders(self, client_in):
        assert client_in.get(reverse("dashboard")).status_code == 200

    def test_campaign_page_renders(self, client_in, campaign):
        response = client_in.get(reverse("campaign_detail", args=[campaign.pk]))
        assert response.status_code == 200
        assert campaign.role_title in response.content.decode()

    def test_candidate_page_renders_score_breakdown(self, client_in, campaign):
        from matching.schemas import CandidateScore, DimensionScore

        candidate = Candidate.objects.create(campaign=campaign, response_id="r1",
                                             email="a@x.com", full_name="A Person")
        candidate.apply_score(CandidateScore(
            overall=71.0, recommendation="YES", confidence=0.8,
            summary="Solid backend profile.",
            dimensions=[DimensionScore(name="Required skills", score=0.8, weight=0.3,
                                       rationale="4/5 skills evidenced")],
        ))
        candidate.save()

        response = client_in.get(
            reverse("candidate_detail", args=[campaign.pk, candidate.pk])
        )
        body = response.content.decode()
        assert response.status_code == 200
        assert "Required skills" in body
        assert "4/5 skills evidenced" in body

    def test_new_campaign_creates_and_redirects(self, client_in):
        response = client_in.post(reverse("campaign_new"), {
            "role_title": "Data Engineer", "experience": "3 years",
        })
        assert response.status_code == 302
        assert Campaign.objects.filter(role_title="Data Engineer").exists()
