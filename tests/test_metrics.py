"""Ranking metrics — the numbers every claim about the scorer rests on."""

from __future__ import annotations

import math

import pytest

from matching.evaluation.metrics import (
    bootstrap_ci,
    evaluate,
    kendall_tau,
    ndcg_at_k,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    spearman,
)


def test_perfect_ranking_scores_one():
    predicted = [90, 80, 70, 60, 50]
    actual = [5, 4, 3, 2, 1]
    rho, _ = spearman(predicted, actual)
    assert rho == pytest.approx(1.0)
    assert kendall_tau(predicted, actual) == pytest.approx(1.0)


def test_inverted_ranking_scores_minus_one():
    rho, _ = spearman([10, 20, 30, 40], [4, 3, 2, 1])
    assert rho == pytest.approx(-1.0)


def test_constant_predictions_are_undefined_not_zero():
    """A scorer giving everyone the same score is a failure, and must show as nan."""
    rho, _ = spearman([50, 50, 50, 50], [1, 2, 3, 4])
    assert math.isnan(rho)


def test_ndcg_rewards_putting_the_best_first():
    actual = [5, 4, 3, 2, 1, 0]
    good = ndcg_at_k([100, 90, 80, 70, 60, 50], actual, 5)
    bad = ndcg_at_k([50, 60, 70, 80, 90, 100], actual, 5)
    assert good == 1.0
    assert bad < good


def test_ndcg_is_nan_when_nothing_is_relevant():
    assert math.isnan(ndcg_at_k([10, 20, 30], [0, 0, 0], 3))


def test_precision_at_k_counts_only_the_top_k():
    predicted = [100, 90, 80, 20, 10]
    actual = [5, 4, 1, 0, 0]
    assert precision_at_k(predicted, actual, k=2, threshold=3.0) == 1.0
    assert precision_at_k(predicted, actual, k=3, threshold=3.0) == 2 / 3


def test_recall_at_k_catches_a_missed_strong_candidate():
    # The strongest candidate (label 5) is ranked last by the scorer.
    predicted = [10, 90, 80, 70]
    actual = [5, 4, 3, 1]
    assert recall_at_k(predicted, actual, k=2, threshold=4.0) == 0.5


def test_recall_is_nan_with_no_relevant_candidates():
    assert math.isnan(recall_at_k([1, 2, 3], [0, 0, 0], k=2, threshold=3.0))


def test_evaluate_flags_small_samples():
    result = evaluate("test", [90, 80, 70, 60], [4, 3, 2, 1])
    assert result.n == 4
    assert any("indicative" in note for note in result.notes)


def test_evaluate_rescales_labels_for_absolute_error():
    """Labels are 0-5, scores are 0-100; MAE must compare like with like."""
    result = evaluate("test", [100.0, 80.0, 60.0], [5.0, 4.0, 3.0], label_scale=5.0)
    assert result.mae == 0.0


def test_evaluate_refuses_to_rank_two_candidates():
    result = evaluate("test", [90, 10], [5, 1])
    assert math.isnan(result.spearman)
    assert any("meaningless" in note for note in result.notes)


class TestBootstrap:
    """The machinery that stops a point estimate being mistaken for a result."""

    def _rho(self, p, a):
        return spearman(p, a)[0]

    def test_interval_brackets_the_point_estimate(self):
        predicted = [90, 85, 80, 70, 65, 60, 50, 40, 30, 20]
        actual = [5, 5, 4, 4, 3, 3, 2, 2, 1, 0]
        low, high = bootstrap_ci(predicted, actual, self._rho, resamples=300)
        point, _ = spearman(predicted, actual)
        assert low <= point <= high

    def test_interval_is_reproducible(self):
        predicted = [90, 85, 80, 70, 65, 60, 50, 40, 30, 20]
        actual = [5, 5, 4, 4, 3, 3, 2, 2, 1, 0]
        assert bootstrap_ci(predicted, actual, self._rho, resamples=300) == \
               bootstrap_ci(predicted, actual, self._rho, resamples=300)

    def test_too_few_candidates_gives_nan(self):
        low, high = bootstrap_ci([3, 2, 1], [3, 2, 1], self._rho)
        assert math.isnan(low) and math.isnan(high)

    def test_noisier_scorer_has_a_wider_interval(self):
        actual = [5, 5, 4, 4, 3, 3, 2, 2, 1, 0]
        clean = [95, 90, 82, 78, 65, 62, 45, 40, 25, 10]
        noisy = [95, 20, 82, 30, 65, 90, 45, 70, 25, 55]
        clean_low, clean_high = bootstrap_ci(clean, actual, self._rho, resamples=400)
        noisy_low, noisy_high = bootstrap_ci(noisy, actual, self._rho, resamples=400)
        assert (noisy_high - noisy_low) > (clean_high - clean_low)

    def test_evaluate_attaches_intervals(self):
        result = evaluate(
            "s", [90, 85, 80, 70, 65, 60, 50, 40, 30, 20],
            [5, 5, 4, 4, 3, 3, 2, 2, 1, 0], resamples=300,
        )
        assert "spearman" in result.intervals
        assert result.interval_str("spearman").startswith("[")

    def test_intervals_can_be_disabled_for_speed(self):
        result = evaluate("s", [90, 80, 70, 60], [4, 3, 2, 1], with_intervals=False)
        assert result.intervals == {}


class TestPairedBootstrap:
    def _rho(self, p, a):
        return spearman(p, a)[0]

    def test_identical_scorers_show_no_difference(self):
        predicted = [90, 80, 70, 60, 50, 40, 30, 20]
        actual = [5, 4, 4, 3, 3, 2, 1, 0]
        stats = paired_bootstrap_delta(predicted, predicted, actual, self._rho, resamples=300)
        assert stats["delta"] == pytest.approx(0.0)
        assert not stats["significant"]

    def test_detects_a_large_consistent_difference(self):
        """A scorer that ranks perfectly vs one that ranks backwards."""
        actual = [5, 5, 4, 4, 3, 3, 2, 2, 1, 0, 0, 1]
        good = [100, 95, 85, 80, 70, 65, 50, 45, 30, 10, 12, 28]
        bad = list(reversed(good))
        stats = paired_bootstrap_delta(good, bad, actual, self._rho, resamples=400)
        assert stats["delta"] > 0
        assert stats["significant"]
        assert stats["low"] > 0
        assert stats["win_rate"] > 0.95

    def test_small_difference_is_reported_as_not_significant(self):
        """The case that matters: a real but underpowered gap must not be
        claimed as significant."""
        actual = [5, 4, 3, 2, 1, 0, 5, 4]
        a = [95, 80, 70, 60, 40, 10, 90, 78]
        b = [93, 82, 68, 62, 38, 12, 88, 80]
        stats = paired_bootstrap_delta(a, b, actual, self._rho, resamples=400)
        assert not stats["significant"]

    def test_mismatched_lengths_are_rejected(self):
        stats = paired_bootstrap_delta([1, 2, 3], [1, 2], [1, 2, 3], self._rho)
        assert math.isnan(stats["delta"])
