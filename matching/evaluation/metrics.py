"""
Ranking metrics for scorer evaluation.

The right question for a screening tool is not "is the score accurate" but
"does the ranking put the right people at the top", because a recruiter reads
the first page of results and stops. That makes rank-correlation and top-k
metrics the ones that matter, and absolute error largely beside the point.

Metrics implemented, and what each one is for:

    spearman        monotone agreement with the human ranking. The headline
                    number: does the scorer order candidates like a recruiter?
    kendall_tau     the same idea, more robust to small samples and ties
    ndcg_at_k       rewards putting the *strongest* candidates highest, with a
                    logarithmic discount — the closest match to how a ranked
                    shortlist is actually consumed
    precision_at_k  of the top k surfaced, what fraction were genuinely good?
    recall_at_k     of all the good candidates, how many made the top k? This is
                    the one that catches a scorer quietly dropping strong people
    mae / rmse      calibration, once scores are on a comparable scale
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class MetricResult:
    scorer: str
    n: int
    spearman: float = float("nan")
    spearman_p: float = float("nan")
    kendall_tau: float = float("nan")
    ndcg_at_5: float = float("nan")
    ndcg_at_10: float = float("nan")
    precision_at_5: float = float("nan")
    recall_at_5: float = float("nan")
    precision_at_10: float = float("nan")
    mae: float = float("nan")
    rmse: float = float("nan")
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "n": self.n,
            "spearman": _round(self.spearman),
            "spearman_p": _round(self.spearman_p, 4),
            "kendall_tau": _round(self.kendall_tau),
            "ndcg@5": _round(self.ndcg_at_5),
            "ndcg@10": _round(self.ndcg_at_10),
            "precision@5": _round(self.precision_at_5),
            "recall@5": _round(self.recall_at_5),
            "precision@10": _round(self.precision_at_10),
            "mae": _round(self.mae, 2),
            "rmse": _round(self.rmse, 2),
            "seconds": round(self.seconds, 1),
            "notes": self.notes,
        }


def _round(value: float, places: int = 3) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), places)


def spearman(predicted: list[float], actual: list[float]) -> tuple[float, float]:
    """Rank correlation and its p-value.

    Undefined when either side is constant (every candidate scored identically),
    which scipy signals with nan — passed through rather than papered over,
    because a constant scorer is a real and important failure to notice.
    """
    if len(predicted) < 3:
        return float("nan"), float("nan")
    if len(set(predicted)) == 1 or len(set(actual)) == 1:
        return float("nan"), float("nan")
    result = stats.spearmanr(predicted, actual)
    return float(result.statistic), float(result.pvalue)


def kendall_tau(predicted: list[float], actual: list[float]) -> float:
    if len(predicted) < 3 or len(set(predicted)) == 1 or len(set(actual)) == 1:
        return float("nan")
    return float(stats.kendalltau(predicted, actual).statistic)


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain with the standard 2^rel - 1 gain function."""
    total = 0.0
    for index, relevance in enumerate(relevances[:k]):
        total += (2**relevance - 1) / math.log2(index + 2)
    return total


def ndcg_at_k(predicted: list[float], actual: list[float], k: int = 10) -> float:
    """Normalised DCG: ranking quality against the best achievable ordering."""
    if not predicted or len(predicted) != len(actual):
        return float("nan")

    order = np.argsort(predicted)[::-1]
    ranked = [actual[i] for i in order]
    ideal = sorted(actual, reverse=True)

    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        # No relevant candidate at all: the ranking cannot be wrong.
        return float("nan")
    return dcg_at_k(ranked, k) / ideal_dcg


def precision_at_k(
    predicted: list[float], actual: list[float], k: int = 5, threshold: float = 3.0
) -> float:
    """Fraction of the top k whose true label is at or above ``threshold``."""
    if not predicted:
        return float("nan")
    k = min(k, len(predicted))
    order = np.argsort(predicted)[::-1][:k]
    return sum(1 for i in order if actual[i] >= threshold) / k


def recall_at_k(
    predicted: list[float], actual: list[float], k: int = 5, threshold: float = 3.0
) -> float:
    """Fraction of all genuinely-good candidates that reached the top k."""
    relevant = sum(1 for a in actual if a >= threshold)
    if relevant == 0:
        return float("nan")
    k = min(k, len(predicted))
    order = np.argsort(predicted)[::-1][:k]
    return sum(1 for i in order if actual[i] >= threshold) / relevant


def mae(predicted: list[float], actual: list[float]) -> float:
    if not predicted:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def rmse(predicted: list[float], actual: list[float]) -> float:
    if not predicted:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))


def evaluate(
    scorer_name: str,
    predicted: list[float],
    actual: list[float],
    *,
    label_scale: float = 5.0,
    relevance_threshold: float = 3.0,
    seconds: float = 0.0,
) -> MetricResult:
    """Compute the full metric set for one scorer's predictions."""
    result = MetricResult(scorer=scorer_name, n=len(predicted), seconds=seconds)

    if len(predicted) < 3:
        result.notes.append("Fewer than 3 candidates — rank metrics are meaningless.")
        return result

    result.spearman, result.spearman_p = spearman(predicted, actual)
    result.kendall_tau = kendall_tau(predicted, actual)
    result.ndcg_at_5 = ndcg_at_k(predicted, actual, 5)
    result.ndcg_at_10 = ndcg_at_k(predicted, actual, 10)
    result.precision_at_5 = precision_at_k(predicted, actual, 5, relevance_threshold)
    result.recall_at_5 = recall_at_k(predicted, actual, 5, relevance_threshold)
    result.precision_at_10 = precision_at_k(predicted, actual, 10, relevance_threshold)

    # Absolute error needs both sides on one scale: labels 0..label_scale are
    # projected onto the scorer's 0..100 range.
    scaled = [a / label_scale * 100.0 for a in actual]
    result.mae = mae(predicted, scaled)
    result.rmse = rmse(predicted, scaled)

    if math.isnan(result.spearman):
        result.notes.append("Spearman undefined — scores or labels are constant.")
    if len(predicted) < 20:
        result.notes.append(
            f"Only {len(predicted)} candidates; treat differences between scorers as indicative."
        )
    return result
