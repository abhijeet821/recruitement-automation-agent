"""Ranking metrics — the numbers every claim about the scorer rests on."""

from __future__ import annotations

import math

import pytest

from matching.evaluation.metrics import (
    evaluate,
    kendall_tau,
    ndcg_at_k,
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
