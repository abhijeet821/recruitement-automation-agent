"""Scoring behaviour — the properties that must hold regardless of the model."""

from __future__ import annotations

import pytest

from matching.features.builder import FeatureBuilder
from matching.schemas import JobSpec, ResumeProfile, SkillMention
from matching.scoring.base import band
from matching.scoring.baselines import (
    LEGACY_KEYWORDS,
    JDKeywordBaseline,
    KeywordBaseline,
    RandomBaseline,
)
from matching.scoring.ensemble import DIMENSION_SPEC, EnsembleScorer


class TestDimensionWeights:
    def test_weights_sum_to_one(self):
        total = sum(spec["weight"] for spec in DIMENSION_SPEC.values())
        assert abs(total - 1.0) < 1e-9


class TestBands:
    def test_thresholds_are_monotone(self):
        assert band(90) == "STRONG_YES"
        assert band(75) == "STRONG_YES"
        assert band(60) == "YES"
        assert band(45) == "MAYBE"
        assert band(44.9) == "NO"
        assert band(0) == "NO"


class TestLegacyBaseline:
    def test_reproduces_the_original_keyword_count(self, job_spec):
        resume = ResumeProfile(raw_text="I know python, django and docker.")
        score = KeywordBaseline().score(resume, job_spec)
        assert score.overall == pytest.approx(100.0 * 3 / len(LEGACY_KEYWORDS), abs=0.01)

    def test_is_role_independent(self, strong_resume):
        """The defining flaw: the same resume scores identically for any role."""
        scorer = KeywordBaseline()
        engineering = JobSpec(role_title="Backend Engineer", must_have_skills=["Python"])
        marketing = JobSpec(role_title="Marketing Manager", must_have_skills=["SEO"])
        assert scorer.score(strong_resume, engineering).overall == \
               scorer.score(strong_resume, marketing).overall

    def test_declares_its_own_limitation(self, strong_resume, job_spec):
        score = KeywordBaseline().score(strong_resume, job_spec)
        assert any("Role-independent" in flag for flag in score.flags)


class TestJDKeywordBaseline:
    def test_is_role_aware(self, strong_resume):
        scorer = JDKeywordBaseline()
        matching_role = JobSpec(role_title="X", must_have_skills=["Python", "Django"])
        other_role = JobSpec(role_title="Y", must_have_skills=["Photoshop", "Illustrator"])
        assert scorer.score(strong_resume, matching_role).overall > \
               scorer.score(strong_resume, other_role).overall

    def test_handles_a_job_with_no_requirements(self, strong_resume):
        score = JDKeywordBaseline().score(strong_resume, JobSpec(role_title="X"))
        assert score.overall == 50.0
        assert score.confidence < 0.2


class TestRandomBaseline:
    def test_is_deterministic(self, strong_resume, job_spec):
        scorer = RandomBaseline()
        assert scorer.score(strong_resume, job_spec).overall == \
               scorer.score(strong_resume, job_spec).overall

    def test_reports_zero_confidence(self, strong_resume, job_spec):
        assert RandomBaseline().score(strong_resume, job_spec).confidence == 0.0


class TestEnsemble:
    def test_missing_github_drops_the_dimension_rather_than_zeroing_it(
        self, provider, strong_resume, job_spec, github_profile
    ):
        """Absent evidence must not be scored as negative evidence."""
        scorer = EnsembleScorer(provider=provider)

        without = scorer.score(strong_resume, job_spec, github=None)
        with_gh = scorer.score(strong_resume, job_spec, github=github_profile)

        assert not any(d.name == "Code evidence" for d in without.dimensions)
        assert any(d.name == "Code evidence" for d in with_gh.dimensions)

        # Weights are renormalised in both cases, so each score is out of 100.
        for result in (without, with_gh):
            assert abs(sum(d.weight for d in result.dimensions) - 1.0) < 1e-6

    def test_code_evidence_dropped_for_non_technical_roles(
        self, provider, strong_resume, github_profile
    ):
        non_technical = JobSpec(
            role_title="Sales Manager", must_have_skills=["negotiation"], is_technical=False
        )
        score = EnsembleScorer(provider=provider).score(
            strong_resume, non_technical, github=github_profile
        )
        assert not any(d.name == "Code evidence" for d in score.dimensions)

    def test_score_is_bounded(self, provider, strong_resume, weak_resume, job_spec):
        scorer = EnsembleScorer(provider=provider)
        for resume in (strong_resume, weak_resume):
            score = scorer.score(resume, job_spec)
            assert 0.0 <= score.overall <= 100.0
            assert 0.0 <= score.confidence <= 1.0

    def test_confidence_drops_when_extraction_failed(self, provider, job_spec):
        scorer = EnsembleScorer(provider=provider)
        broken = ResumeProfile(raw_text="", extraction_failed=True)
        score = scorer.score(broken, job_spec)
        assert score.confidence < 0.4
        assert any("could not be parsed" in f for f in score.flags)

    def test_low_confidence_is_flagged_for_review(self, provider, job_spec):
        scorer = EnsembleScorer(provider=provider)
        thin = ResumeProfile(raw_text="Hi.", extraction_failed=True)
        score = scorer.score(thin, job_spec)
        assert any("Low confidence" in f for f in score.flags)

    def test_critical_gap_penalty_applies_when_most_requirements_missing(
        self, provider, job_spec
    ):
        """A senior profile missing the core stack must not score as a hire."""
        scorer = EnsembleScorer(provider=provider)
        wrong_stack = ResumeProfile(
            full_name="R", total_years_experience=9.0,
            skills=[SkillMention(name="COBOL", years=9, evidence="professional")],
            raw_text="Nine years of COBOL on mainframes." * 40,
        )
        score = scorer.score(wrong_stack, job_spec)
        assert score.overall < 60
        assert any("required skills are met" in f for f in score.flags)

    def test_no_penalty_when_requirements_are_met(self, provider, strong_resume, job_spec):
        score = EnsembleScorer(provider=provider).score(strong_resume, job_spec)
        assert not any("required skills are met" in f for f in score.flags)

    def test_score_round_trips_through_json(self, provider, strong_resume, job_spec):
        """Scores are persisted as JSON, so serialisation must be lossless."""
        from matching.schemas import CandidateScore

        original = EnsembleScorer(provider=provider).score(strong_resume, job_spec)
        restored = CandidateScore.from_dict(original.to_dict())
        assert restored.overall == original.overall
        assert restored.recommendation == original.recommendation
        assert len(restored.dimensions) == len(original.dimensions)
        assert restored.features.values == original.features.values


class TestFeatures:
    def test_experience_fit_penalises_shortfall_but_not_harshly(self, provider):
        builder = FeatureBuilder(provider)
        job = JobSpec(role_title="X", min_years=4.0, must_have_skills=["Python"])

        near = ResumeProfile(total_years_experience=3.0,
                             skills=[SkillMention(name="Python")], raw_text="x" * 900)
        far = ResumeProfile(total_years_experience=0.5,
                            skills=[SkillMention(name="Python")], raw_text="x" * 900)

        near_fit = builder.build(near, job)[0].get("experience_fit")
        far_fit = builder.build(far, job)[0].get("experience_fit")

        assert near_fit > far_fit
        assert near_fit > 0.7   # one year short is a near miss
        assert far_fit < 0.4

    def test_education_not_penalised_when_no_requirement_stated(self, provider):
        builder = FeatureBuilder(provider)
        job = JobSpec(role_title="X", must_have_skills=["Python"], education_requirement="")
        resume = ResumeProfile(skills=[SkillMention(name="Python")], raw_text="x" * 900)
        assert builder.build(resume, job)[0].get("education_fit") == 1.0

    def test_all_features_are_in_the_unit_interval(
        self, provider, strong_resume, job_spec, github_profile
    ):
        features, _, _ = FeatureBuilder(provider).build(strong_resume, job_spec, github_profile)
        assert features.values
        for name, value in features.values.items():
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

    def test_every_feature_carries_provenance(
        self, provider, strong_resume, job_spec, github_profile
    ):
        """Each number must come with an explanation the recruiter can read."""
        features, _, _ = FeatureBuilder(provider).build(strong_resume, job_spec, github_profile)
        for name in features.values:
            assert features.provenance.get(name), f"{name} has no provenance"
