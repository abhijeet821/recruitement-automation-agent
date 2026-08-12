"""
Ablation and rubric-consistency analysis.

Both are about knowing whether the scoring configuration is defensible: does
each dimension earn its weight, and is the one non-deterministic component
stable enough to carry the weight it has.
"""

from __future__ import annotations

import pytest

from matching.evaluation.ablation import run_ablation
from matching.evaluation.consistency import measure_rubric_consistency
from matching.schemas import CandidateScore, DimensionScore


def score(overall: float, dims: dict[str, tuple[float, float]]) -> CandidateScore:
    """Build a CandidateScore from {name: (score, weight)}."""
    return CandidateScore(
        overall=overall,
        dimensions=[
            DimensionScore(name=n, score=s, weight=w) for n, (s, w) in dims.items()
        ],
    )


def linear(n: int, *, decisive: str = "A") -> tuple[list[CandidateScore], list[float]]:
    """Candidates whose ranking is driven entirely by one dimension.

    The other dimension is constant, so ablating the decisive one must destroy
    the ranking and ablating the constant one must change nothing.
    """
    scores, labels = [], []
    for i in range(n):
        strength = 1.0 - i / n
        dims = {"A": (0.5, 0.5), "B": (0.5, 0.5)}
        dims[decisive] = (strength, 0.5)
        # overall matches the weighted sum, no penalty applied
        overall = 100.0 * sum(s * w for s, w in dims.values())
        scores.append(score(overall, dims))
        labels.append(float(n - i))
    return scores, labels


class TestAblation:
    def test_removing_the_decisive_dimension_destroys_the_ranking(self):
        """With the only varying dimension gone every score is identical, so
        Spearman is undefined. That is maximal damage, not missing data, and the
        report must say so rather than showing a bare nan."""
        scores, labels = linear(10, decisive="A")
        report = run_ablation(scores, labels)

        dropped_a = next(r for r in report.rows if r.dropped == "A")
        assert dropped_a.collapsed
        assert dropped_a.delta_spearman < -0.5
        assert dropped_a.verdict.startswith("carries signal")

    def test_a_collapse_sorts_above_a_merely_bad_dimension(self):
        scores, labels = linear(10, decisive="A")
        report = run_ablation(scores, labels)
        assert report.rows[0].dropped == "A"

    def test_removing_a_constant_dimension_changes_nothing(self):
        scores, labels = linear(10, decisive="A")
        report = run_ablation(scores, labels)

        dropped_b = next(r for r in report.rows if r.dropped == "B")
        assert abs(dropped_b.delta_spearman) < 1e-9
        assert dropped_b.verdict == "no measurable effect"

    def test_rows_are_ordered_most_damaging_first(self):
        scores, labels = linear(10)
        report = run_ablation(scores, labels)
        deltas = [r.delta_spearman for r in report.rows]
        assert deltas == sorted(deltas)

    def test_baseline_is_recorded(self):
        scores, labels = linear(10)
        report = run_ablation(scores, labels)
        assert report.baseline_spearman == pytest.approx(1.0)
        assert report.n == 10

    def test_gap_penalty_is_held_constant(self):
        """Ablation must measure one thing. The penalty is a multiplier on the
        whole score, so it has to survive the recomputation unchanged."""
        from matching.evaluation.ablation import _recompute

        dims = {"A": (0.8, 0.5), "B": (0.4, 0.5)}
        base = 100.0 * sum(s * w for s, w in dims.values())   # 60.0
        penalised = score(base * 0.5, dims)                    # a 0.5x penalty

        # Dropping B leaves A at full weight: 100*0.8 = 80, still halved -> 40.
        assert _recompute(penalised, "B") == pytest.approx(40.0)

    def test_dropping_the_only_dimension_is_safe(self):
        from matching.evaluation.ablation import _recompute

        assert _recompute(score(50.0, {"A": (0.5, 1.0)}), "A") == 0.0

    def test_too_few_candidates_is_refused(self):
        scores, labels = linear(3)
        report = run_ablation(scores, labels)
        assert report.rows == []
        assert any("Too few" in n for n in report.notes)

    def test_widespread_inertness_is_explained_not_left_ambiguous(self):
        """The easiest result to misread: many dimensions showing no effect
        means correlated dimensions, not dead weight."""
        n = 10
        scores, labels = [], []
        for i in range(n):
            strength = 1.0 - i / n
            # Every dimension moves together — the ranking is over-determined.
            dims = {k: (strength, 0.25) for k in ("A", "B", "C", "D")}
            scores.append(score(100.0 * strength, dims))
            labels.append(float(n - i))

        report = run_ablation(scores, labels)
        assert all(abs(r.delta_spearman) < 1e-9 for r in report.rows)
        assert any("over-determined" in note for note in report.notes)

    def test_small_sample_is_flagged(self):
        scores, labels = linear(10)
        report = run_ablation(scores, labels)
        assert any("wide uncertainty" in n for n in report.notes)


class TestRubricConsistency:
    def test_identical_runs_report_zero_variance(self, provider, strong_resume, job_spec):
        """The fake provider is deterministic, so this pins the maths."""
        provider.canned = {"Assess this candidate": {
            "technical_depth": {"score": 8, "rationale": "", "evidence": ["x"]},
            "relevant_experience": {"score": 9, "rationale": "", "evidence": ["x"]},
            "impact_evidence": {"score": 7, "rationale": "", "evidence": ["x"]},
            "communication": {"score": 8, "rationale": "", "evidence": ["x"]},
            "summary": "s", "strengths": [], "gaps": [],
        }}
        report = measure_rubric_consistency(strong_resume, job_spec, provider, runs=4)

        assert report.overall.stdev == 0.0
        assert report.overall.spread == 0.0
        assert report.verdict.startswith("stable")
        assert report.score_swing() == 0.0

    def test_swing_is_expressed_in_final_score_points(self, provider, strong_resume, job_spec):
        """Variance only matters in the units the recruiter sees."""
        from matching.evaluation.consistency import ConsistencyReport, DimensionSpread

        report = ConsistencyReport(overall=DimensionSpread("overall", [0.4, 0.8]))
        # spread 0.4 x weight 0.17 x 100 = 6.8 points
        assert report.score_swing() == pytest.approx(6.8, abs=0.01)
        assert report.verdict.startswith("UNSTABLE")

    def test_large_swing_is_called_out(self):
        from matching.evaluation.consistency import ConsistencyReport, DimensionSpread

        report = ConsistencyReport(runs=3, overall=DimensionSpread("overall", [0.3, 0.9, 0.6]))
        assert report.score_swing() > 5

    def test_failed_runs_are_counted_not_hidden(self, provider, strong_resume, job_spec):
        from matching.llm.base import LLMError

        def explode(*args, **kwargs):
            raise LLMError("model down")

        provider.generate_json = explode
        report = measure_rubric_consistency(strong_resume, job_spec, provider, runs=3)

        assert report.failures == 3
        assert any("failed outright" in n for n in report.notes)
        assert any("Not enough successful runs" in n for n in report.notes)

    def test_progress_callback_is_driven(self, provider, strong_resume, job_spec):
        seen = []
        measure_rubric_consistency(
            strong_resume, job_spec, provider, runs=3,
            on_run=lambda i, t, v: seen.append(i),
        )
        assert seen == [1, 2, 3]
