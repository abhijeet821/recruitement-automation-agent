"""
Tests for the demo-seeding command.

Runs against the deterministic fake provider, so it verifies the wiring —
campaign creation, idempotency, label seeding — without a model server.
"""

from __future__ import annotations

from io import StringIO

import pytest
from django.contrib.auth.models import User
from django.core.management import call_command

from hiring_app.models import Campaign, Candidate

pytestmark = pytest.mark.django_db


def seed(**kwargs) -> str:
    out = StringIO()
    call_command("seed_demo", stdout=out, stderr=StringIO(), **kwargs)
    return out.getvalue()


class TestSeeding:
    def test_creates_a_campaign_with_scored_candidates(self):
        seed(fast=True)

        campaign = Campaign.objects.get()
        assert campaign.role_title == "Backend Engineer (Python)"
        assert campaign.status == Campaign.Status.ACTIVE
        assert campaign.candidates.count() == 14
        assert all(c.is_scored for c in campaign.candidates.all())

    def test_seeds_recruiter_ratings_from_the_golden_labels(self):
        """This is what makes `evaluate_scorer --campaign N` work immediately."""
        seed(fast=True)

        rated = Candidate.objects.exclude(recruiter_rating__isnull=True)
        assert rated.count() == 14
        assert set(rated.values_list("recruiter_rating", flat=True)) <= set(range(6))

    def test_labels_are_marked_as_synthetic(self):
        """A seeded label must never be mistaken for a real recruiter judgement."""
        seed(fast=True)
        note = Candidate.objects.exclude(recruiter_note="").first().recruiter_note
        assert "synthetic" in note.lower()

    def test_campaign_records_its_provenance(self):
        seed(fast=True)
        campaign = Campaign.objects.get()
        assert campaign.question_ids.get("_source") == "seeded-demo"

    def test_form_id_stays_empty_so_it_is_not_treated_as_launched(self):
        """The campaign was never provisioned against Google; the UI keys on this."""
        seed(fast=True)
        assert Campaign.objects.get().form_id == ""

    def test_jd_is_graded(self):
        seed(fast=True)
        assert "score" in Campaign.objects.get().jd_quality


class TestIdempotency:
    def test_rerunning_does_not_duplicate_candidates(self):
        seed(fast=True)
        output = seed(fast=True)

        assert Candidate.objects.count() == 14
        assert Campaign.objects.count() == 1
        assert "already loaded" in output

    def test_reset_rebuilds_from_scratch(self):
        seed(fast=True)
        original_id = Campaign.objects.get().pk

        seed(fast=True, reset=True)

        assert Campaign.objects.count() == 1
        assert Campaign.objects.get().pk != original_id
        assert Candidate.objects.count() == 14


class TestAccountHandling:
    def test_uses_an_existing_account_rather_than_creating_one(self):
        user = User.objects.create_user(username="abhijeet", password="test-pass-1234")
        seed(fast=True)

        assert Campaign.objects.get().owner == user
        assert not User.objects.filter(username="demo").exists()

    def test_creates_an_account_and_prints_the_password(self):
        output = seed(fast=True)
        assert User.objects.filter(username="demo").exists()
        assert "password:" in output

    def test_named_user_is_honoured(self):
        User.objects.create_user(username="first", password="test-pass-1234")
        target = User.objects.create_user(username="second", password="test-pass-1234")
        seed(fast=True, user="second")
        assert Campaign.objects.get().owner == target

    def test_unknown_user_is_rejected(self):
        from django.core.management.base import CommandError

        with pytest.raises(CommandError, match="No user named"):
            seed(fast=True, user="nobody")
