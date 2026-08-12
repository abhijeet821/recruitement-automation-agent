"""
Interview guide generation.

Two things are worth pinning down here. The budget maths is pure arithmetic and
must never produce a plan that overruns the booked slot. And the grounding rule
— every question cites something specific from the candidate's background — is
the difference between a useful guide and generic filler, so an ungrounded
question must be dropped rather than displayed.
"""

from __future__ import annotations

import pytest

from matching.generation.interview import (
    CATEGORY_MINUTES,
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    InterviewGuide,
    generate_interview_guide,
    plan_budget,
)
from matching.schemas import CandidateScore


class TestBudget:
    @pytest.mark.parametrize("duration", [15, 30, 45, 60, 75, 90, 120])
    def test_plan_never_overruns_the_slot(self, duration):
        """The point of the feature: the plan must fit the time actually booked."""
        budget = plan_budget(duration, has_github=True, has_gaps=True)
        assert budget.estimated_minutes <= budget.usable_minutes
        assert budget.usable_minutes < duration  # intro + their questions reserved

    def test_longer_interviews_get_more_questions(self):
        short = plan_budget(30)
        long = plan_budget(90)
        assert long.total > short.total

    def test_question_count_is_sane_at_both_extremes(self):
        assert plan_budget(15).total >= MIN_QUESTIONS - 1  # very short slots are tight
        assert plan_budget(180).total <= MAX_QUESTIONS

    def test_time_is_reserved_for_the_candidates_own_questions(self):
        budget = plan_budget(60)
        assert budget.wrap_up_minutes > 0
        assert budget.warmup_minutes > 0

    def test_short_slots_trim_the_ceremony(self):
        assert plan_budget(15).warmup_minutes < plan_budget(60).warmup_minutes

    def test_core_categories_always_present(self):
        budget = plan_budget(30)
        assert budget.counts.get("skill_verification", 0) >= 1
        assert budget.counts.get("project_deep_dive", 0) >= 1

    def test_github_dropped_when_there_is_no_public_code(self):
        assert "github_code" not in plan_budget(60, has_github=False).counts
        assert "github_code" in plan_budget(60, has_github=True).counts

    def test_gap_probes_only_when_something_is_unverified(self):
        assert "gap_probe" not in plan_budget(60, has_gaps=False).counts
        assert "gap_probe" in plan_budget(60, has_gaps=True).counts

    def test_dropped_category_time_is_redistributed(self):
        """A candidate with no public code gets more of everything else, not a
        shorter interview — mirroring how the scorer handles missing evidence."""
        with_gh = plan_budget(60, has_github=True, has_gaps=True)
        without_gh = plan_budget(60, has_github=False, has_gaps=True)
        non_github = sum(n for c, n in with_gh.counts.items() if c != "github_code")
        assert sum(without_gh.counts.values()) >= non_github

    def test_non_technical_roles_drop_code_questions(self):
        budget = plan_budget(60, has_github=True, is_technical=False)
        assert "github_code" not in budget.counts
        assert "system_design" not in budget.counts

    def test_system_design_reserved_for_senior_and_long_slots(self):
        assert "system_design" not in plan_budget(60, seniority="junior").counts
        assert "system_design" not in plan_budget(30, seniority="senior").counts

    def test_absurd_durations_are_handled(self):
        assert plan_budget(0).total >= 1
        assert plan_budget(600).total <= MAX_QUESTIONS

    def test_estimated_minutes_matches_the_category_costs(self):
        budget = plan_budget(60, has_github=True, has_gaps=True)
        expected = sum(CATEGORY_MINUTES[c] * n for c, n in budget.counts.items())
        assert budget.estimated_minutes == expected


class TestGeneration:
    def _payload(self, questions):
        return {"focus": "Confirm production Django depth.", "opening": "Hello.",
                "questions": questions}

    def test_produces_a_usable_guide(self, provider, strong_resume, job_spec):
        provider.canned = {"Prepare interview questions": self._payload([
            {
                "question": "You cut p99 checkout latency from 840ms to 190ms — what was slow?",
                "category": "project_deep_dive",
                "grounded_in": "cut p99 checkout latency from 840ms to 190ms",
                "why": "Tests whether they diagnosed the bottleneck or inherited the fix.",
                "listen_for": ["A specific bottleneck", "How they measured it"],
                "follow_up": "What did you rule out first?",
            },
        ])}
        guide = generate_interview_guide(
            strong_resume, job_spec, provider, duration_minutes=45
        )

        assert guide.ok
        assert guide.questions[0].category == "project_deep_dive"
        assert guide.questions[0].minutes == CATEGORY_MINUTES["project_deep_dive"]
        assert guide.focus
        assert guide.budget.duration_minutes == 45

    def test_ungrounded_questions_are_discarded(self, provider, strong_resume, job_spec):
        """A generic question that looks personalised is worse than no question."""
        provider.canned = {"Prepare interview questions": self._payload([
            {"question": "What is a REST API?", "category": "skill_verification",
             "grounded_in": "", "why": "generic", "listen_for": []},
            {"question": "Why advisory locks for the ledger?", "category": "project_deep_dive",
             "grounded_in": "idempotent ledger writes on PostgreSQL with advisory locks",
             "why": "Probes a real design decision.", "listen_for": ["Concurrency reasoning"]},
        ])}
        guide = generate_interview_guide(strong_resume, job_spec, provider)

        assert len(guide.questions) == 1
        assert "REST API?" not in guide.questions[0].question
        assert any("discarded" in w for w in guide.warnings)

    def test_all_ungrounded_is_reported_as_an_error(self, provider, strong_resume, job_spec):
        provider.canned = {"Prepare interview questions": self._payload([
            {"question": "What is SQL?", "category": "skill_verification",
             "grounded_in": "", "why": "", "listen_for": []},
        ])}
        guide = generate_interview_guide(strong_resume, job_spec, provider)
        assert not guide.ok
        assert guide.error

    def test_unknown_category_falls_back_safely(self, provider, strong_resume, job_spec):
        provider.canned = {"Prepare interview questions": self._payload([
            {"question": "Tell me about the ledger.", "category": "made_up_category",
             "grounded_in": "idempotent ledger writes", "why": "x", "listen_for": []},
        ])}
        guide = generate_interview_guide(strong_resume, job_spec, provider)
        assert guide.questions[0].category == "skill_verification"

    def test_thin_resume_is_refused_rather_than_guessed_at(self, provider, job_spec):
        from matching.schemas import ResumeProfile

        guide = generate_interview_guide(ResumeProfile(), job_spec, provider)
        assert not guide.ok
        assert "resume detail" in guide.error

    def test_model_failure_is_reported_not_raised(self, provider, strong_resume, job_spec):
        from matching.llm.base import LLMError

        def explode(*args, **kwargs):
            raise LLMError("model unavailable")

        provider.generate_json = explode
        guide = generate_interview_guide(strong_resume, job_spec, provider)
        assert not guide.ok
        assert "model unavailable" in guide.error

    def test_score_gaps_are_offered_to_the_model(self, provider, strong_resume, job_spec):
        """Unverified requirements are the most useful thing to ask about."""
        score = CandidateScore(gaps=["No evidence of Kubernetes"])
        generate_interview_guide(strong_resume, job_spec, provider, score=score)

        prompt = next(p for kind, p in provider.calls if kind == "generate_json")
        assert "Kubernetes" in prompt
        assert "fair chance to demonstrate" in prompt

    def test_prompt_forbids_personal_questions(self, provider, strong_resume, job_spec):
        generate_interview_guide(strong_resume, job_spec, provider)
        prompt = next(p for kind, p in provider.calls if kind == "generate_json")
        for banned in ("age", "family", "nationality", "religion", "marital status"):
            assert banned in prompt.lower()

    def test_questions_are_ordered_to_flow(self, provider, strong_resume, job_spec):
        """Concrete and warm first, abstract last."""
        provider.canned = {"Prepare interview questions": self._payload([
            {"question": "Behavioural one", "category": "behavioural",
             "grounded_in": "mentored 4 engineers", "why": "x", "listen_for": []},
            {"question": "Project one", "category": "project_deep_dive",
             "grounded_in": "billing service migration", "why": "x", "listen_for": []},
        ])}
        guide = generate_interview_guide(strong_resume, job_spec, provider)
        assert guide.questions[0].category == "project_deep_dive"
        assert guide.questions[-1].category == "behavioural"


class TestSerialisation:
    def test_guide_round_trips_through_json(self, provider, strong_resume, job_spec):
        """Guides are persisted in a JSONField, so this must be lossless."""
        provider.canned = {"Prepare interview questions": {
            "focus": "Depth check.", "opening": "Hi.",
            "questions": [{
                "question": "Why advisory locks?", "category": "project_deep_dive",
                "grounded_in": "advisory locks", "why": "design decision",
                "listen_for": ["concurrency"], "follow_up": "What else?",
            }],
        }}
        original = generate_interview_guide(
            strong_resume, job_spec, provider, duration_minutes=60
        )
        restored = InterviewGuide.from_dict(original.to_dict())

        assert restored.duration_minutes == 60
        assert len(restored.questions) == len(original.questions)
        assert restored.questions[0].grounded_in == "advisory locks"
        assert restored.questions[0].listen_for == ["concurrency"]
        assert restored.budget.counts == original.budget.counts
        assert restored.estimated_minutes == original.estimated_minutes

    def test_empty_dict_is_survivable(self):
        assert InterviewGuide.from_dict({}).questions == []
        assert InterviewGuide.from_dict(None).questions == []
