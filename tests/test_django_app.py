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


class TestPagination:
    """A 500-applicant campaign used to render 1,000 table rows in one response."""

    def test_page_is_capped(self, client_in, campaign):
        from hiring_app.views import CANDIDATES_PER_PAGE

        Candidate.objects.bulk_create([
            Candidate(campaign=campaign, response_id=f"r{i}", overall_score=i)
            for i in range(CANDIDATES_PER_PAGE + 15)
        ])
        response = client_in.get(reverse("campaign_detail", args=[campaign.pk]))

        assert response.status_code == 200
        assert len(response.context["candidates"]) == CANDIDATES_PER_PAGE
        assert response.context["is_paginated"] is True

    def test_counts_cover_the_whole_campaign_not_the_page(self, client_in, campaign):
        """"3 strong candidates" must not quietly mean "3 on this page"."""
        from hiring_app.views import CANDIDATES_PER_PAGE

        Candidate.objects.bulk_create([
            Candidate(campaign=campaign, response_id=f"r{i}", overall_score=90, confidence=0.9)
            for i in range(CANDIDATES_PER_PAGE + 10)
        ])
        response = client_in.get(reverse("campaign_detail", args=[campaign.pk]))

        assert response.context["total_candidates"] == CANDIDATES_PER_PAGE + 10
        assert response.context["strong_count"] == CANDIDATES_PER_PAGE + 10

    def test_highest_scores_appear_on_the_first_page(self, client_in, campaign):
        from hiring_app.views import CANDIDATES_PER_PAGE

        Candidate.objects.bulk_create([
            Candidate(campaign=campaign, response_id=f"r{i}", overall_score=i)
            for i in range(CANDIDATES_PER_PAGE + 5)
        ])
        page = client_in.get(reverse("campaign_detail", args=[campaign.pk])).context["candidates"]
        assert page[0].overall_score == CANDIDATES_PER_PAGE + 4

    def test_second_page_is_reachable(self, client_in, campaign):
        from hiring_app.views import CANDIDATES_PER_PAGE

        Candidate.objects.bulk_create([
            Candidate(campaign=campaign, response_id=f"r{i}", overall_score=i)
            for i in range(CANDIDATES_PER_PAGE + 5)
        ])
        response = client_in.get(
            reverse("campaign_detail", args=[campaign.pk]), {"page": 2}
        )
        assert len(response.context["candidates"]) == 5

    def test_out_of_range_page_falls_back_rather_than_500ing(self, client_in, campaign):
        Candidate.objects.create(campaign=campaign, response_id="r1")
        assert client_in.get(
            reverse("campaign_detail", args=[campaign.pk]), {"page": 999}
        ).status_code == 200

    def test_garbage_page_parameter_is_survivable(self, client_in, campaign):
        assert client_in.get(
            reverse("campaign_detail", args=[campaign.pk]), {"page": "'; DROP TABLE--"}
        ).status_code == 200

    def test_small_campaign_is_not_paginated(self, client_in, campaign):
        Candidate.objects.create(campaign=campaign, response_id="r1")
        response = client_in.get(reverse("campaign_detail", args=[campaign.pk]))
        assert response.context["is_paginated"] is False


class TestRateLimiting:
    """Only login/register were limited; the model-backed endpoints were open."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        """Reset the rate-limit counter and keep job execution out of tests.

        ``start_job`` would otherwise hand work to a real worker thread, which
        touches the database outside the test transaction and makes these tests
        non-deterministic. The job *row* is still created, which is what the
        assertions check.
        """
        from django.core.cache import cache

        cache.clear()
        monkeypatch.setattr("hiring_app.jobs.enqueue", lambda job: None)
        yield
        cache.clear()

    def test_repeated_jd_drafting_is_throttled(self, client_in, campaign, settings):
        from hiring_app.views import RATE_JD

        allowed = int(RATE_JD.split("/")[0])
        url = reverse("generate_jd", args=[campaign.pk])

        # Exhaust the allowance, then one more.
        for _ in range(allowed):
            client_in.post(url, {"role_title": "Engineer"})
        response = client_in.post(url, {"role_title": "Engineer"}, follow=True)

        messages = [str(m) for m in response.context["messages"]]
        assert any("too many" in m.lower() for m in messages)

    def test_throttling_explains_itself_instead_of_403ing(self, client_in, campaign):
        """block=False so the user gets a readable message, not a bare 403."""
        from hiring_app.views import RATE_RESCORE

        Candidate.objects.create(campaign=campaign, response_id="r1")
        allowed = int(RATE_RESCORE.split("/")[0])
        url = reverse("rescore_campaign", args=[campaign.pk])

        for _ in range(allowed + 1):
            response = client_in.post(url)
        assert response.status_code == 302  # redirect with a message, never 403

    def test_limits_are_per_user_not_per_ip(self, client, campaign, user, other_user):
        """An office behind one NAT address must not share a quota.

        Asserted on behaviour (was work queued?) rather than on flash messages —
        unread messages accumulate in storage, so an earlier user's warnings
        would otherwise surface in a later user's response.
        """
        from django.test import Client

        from hiring_app.models import BackgroundJob
        from hiring_app.views import RATE_RESCORE

        Candidate.objects.create(campaign=campaign, response_id="r1")
        allowed = int(RATE_RESCORE.split("/")[0])

        client.force_login(user)
        for _ in range(allowed + 2):
            client.post(reverse("rescore_campaign", args=[campaign.pk]))

        first_user_jobs = BackgroundJob.objects.filter(campaign=campaign).count()
        assert first_user_jobs <= allowed  # throttling actually bit

        other_campaign = Campaign.objects.create(owner=other_user, role_title="Other")
        Candidate.objects.create(campaign=other_campaign, response_id="x1")

        fresh = Client()  # separate cookie jar, no inherited message state
        fresh.force_login(other_user)
        fresh.post(reverse("rescore_campaign", args=[other_campaign.pk]))

        # The second user was unaffected by the first user's exhausted quota.
        assert BackgroundJob.objects.filter(campaign=other_campaign).exists()


class TestInterviewGuide:
    @pytest.fixture(autouse=True)
    def _canned_questions(self):
        """Give the shared fake provider a usable guide to return.

        Its default is a bare schema skeleton with no questions, which the view
        correctly refuses to save — so without this the tests would only ever
        exercise the failure path.
        """
        from matching.llm import get_provider

        get_provider().canned = {
            "Prepare interview questions": {
                "focus": "Confirm production Django depth.",
                "opening": "Thanks for making time.",
                "questions": [{
                    "question": "How did you cut p99 latency from 800ms to 120ms?",
                    "category": "project_deep_dive",
                    "grounded_in": "Cut p99 latency from 800ms to 120ms",
                    "why": "Tests whether they diagnosed it or inherited the fix.",
                    "listen_for": ["A specific bottleneck", "How they measured it"],
                    "follow_up": "What did you rule out first?",
                }],
            }
        }
        yield
        get_provider().canned = {}

    @pytest.fixture
    def candidate(self, campaign):
        from matching.schemas import CandidateScore

        c = Candidate.objects.create(
            campaign=campaign, response_id="r1", email="a@x.com", full_name="Ada",
            resume_text="Senior backend engineer. Built a billing service in Django. " * 20,
            profile={
                "full_name": "Ada",
                "summary": "Backend engineer, 6 years.",
                "total_years_experience": 6,
                "experience": [{
                    "title": "Backend Engineer", "company": "Acme", "start_year": 2020,
                    "description": "Cut p99 latency from 800ms to 120ms.",
                }],
                "skills": [{"name": "Python", "evidence": "professional"}],
                "raw_text": "Backend engineer with six years of Django. " * 20,
            },
        )
        c.apply_score(CandidateScore(overall=78, recommendation="YES", confidence=0.8,
                                     gaps=["No evidence of Kubernetes"]))
        c.save()
        return c

    def test_generates_and_persists_a_guide(self, client_in, campaign, candidate):
        response = client_in.post(
            reverse("generate_interview_guide", args=[campaign.pk, candidate.pk]),
            {"duration_minutes": "60"},
        )
        assert response.status_code == 302
        candidate.refresh_from_db()
        assert candidate.interview_duration == 60
        assert candidate.interview_guide  # cached, not regenerated on each view

    def test_duration_is_clamped_not_rejected(self, client_in, campaign, candidate):
        """A typo should not lose the request; nobody runs a 6-hour interview."""
        client_in.post(
            reverse("generate_interview_guide", args=[campaign.pk, candidate.pk]),
            {"duration_minutes": "9999"},
        )
        candidate.refresh_from_db()
        assert candidate.interview_duration == 180

    def test_garbage_duration_falls_back_to_a_default(self, client_in, campaign, candidate):
        client_in.post(
            reverse("generate_interview_guide", args=[campaign.pk, candidate.pk]),
            {"duration_minutes": "twenty"},
        )
        candidate.refresh_from_db()
        assert candidate.interview_duration == 45

    def test_requires_post(self, client_in, campaign, candidate):
        assert client_in.get(
            reverse("generate_interview_guide", args=[campaign.pk, candidate.pk])
        ).status_code == 405

    def test_another_users_candidate_is_not_reachable(self, client, other_user, campaign, candidate):
        client.force_login(other_user)
        assert client.post(
            reverse("generate_interview_guide", args=[campaign.pk, candidate.pk])
        ).status_code == 404

    def test_candidate_page_offers_the_duration_picker(self, client_in, campaign, candidate):
        body = client_in.get(
            reverse("candidate_detail", args=[campaign.pk, candidate.pk])
        ).content.decode()
        assert "Interview guide" in body
        assert 'name="duration_minutes"' in body
        assert "45 minutes" in body

    def test_stored_guide_is_rendered(self, client_in, campaign, candidate):
        candidate.interview_guide = {
            "duration_minutes": 45, "focus": "Check Django depth.", "opening": "Hello Ada.",
            "questions": [{
                "question": "How did you cut p99 latency?",
                "category": "project_deep_dive",
                "grounded_in": "Cut p99 latency from 800ms to 120ms",
                "why": "Tests whether they diagnosed it.",
                "listen_for": ["A specific bottleneck"],
                "follow_up": "What did you rule out?",
                "minutes": 8,
            }],
        }
        candidate.save()

        body = client_in.get(
            reverse("candidate_detail", args=[campaign.pk, candidate.pk])
        ).content.decode()
        assert "How did you cut p99 latency?" in body
        assert "Cut p99 latency from 800ms to 120ms" in body   # the grounding is shown
        assert "A specific bottleneck" in body
        assert "Project deep-dive" in body


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
