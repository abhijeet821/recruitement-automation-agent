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
    # 95% bootstrap intervals, keyed by metric name -> (low, high).
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def interval_str(self, metric: str) -> str:
        bounds = self.intervals.get(metric)
        if not bounds or any(math.isnan(b) for b in bounds):
            return ""
        return f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"

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
            "ci95": {
                k: [_round(v[0]), _round(v[1])] for k, v in self.intervals.items()
            },
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


def bootstrap_ci(
    predicted: list[float],
    actual: list[float],
    statistic,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a paired ranking statistic.

    Why this exists: on a small labelled set, a point estimate like
    "Spearman 0.964 vs 0.942" invites a conclusion the data cannot support. The
    two scorers are evaluated on the *same* candidates, so their errors are
    correlated and the apparent gap is often well inside sampling noise.
    Resampling candidates with replacement and recomputing the statistic shows
    how much of the number is signal.

    A fixed seed makes the interval reproducible, so re-running the harness does
    not silently move the reported bounds.
    """
    n = len(predicted)
    if n < 5:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    samples: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        # A resample can be degenerate (every label identical), which makes rank
        # correlation undefined. Skip those rather than poisoning the interval
        # with nan or silently coercing them to zero.
        try:
            value = statistic(predicted_arr[idx].tolist(), actual_arr[idx].tolist())
        except Exception:  # noqa: BLE001
            continue
        if value is not None and not math.isnan(value):
            samples.append(float(value))

    if len(samples) < resamples * 0.5:
        return float("nan"), float("nan")

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(samples, 100 * alpha)),
        float(np.percentile(samples, 100 * (1 - alpha))),
    )


def paired_bootstrap_delta(
    predicted_a: list[float],
    predicted_b: list[float],
    actual: list[float],
    statistic,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict:
    """Is scorer A genuinely better than scorer B?

    This is the comparison that matters, and it is *not* answered by looking at
    two separate confidence intervals. Both scorers are measured on the same
    candidates, so their errors move together; two heavily overlapping intervals
    can still hide a difference that is consistent on every resample.

    The correct test resamples the candidates once per iteration and recomputes
    **both** statistics on that same resample, then takes the difference. If the
    resulting interval excludes zero, the ordering held across resamples.
    ``win_rate`` reports how often A beat B directly.
    """
    n = len(actual)
    out = {
        "delta": float("nan"), "low": float("nan"), "high": float("nan"),
        "win_rate": float("nan"), "significant": False, "n": n,
    }
    if n < 5 or len(predicted_a) != n or len(predicted_b) != n:
        return out

    rng = np.random.default_rng(seed)
    a_arr = np.asarray(predicted_a, dtype=float)
    b_arr = np.asarray(predicted_b, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    try:
        base_a = statistic(predicted_a, actual)
        base_b = statistic(predicted_b, actual)
        out["delta"] = float(base_a - base_b)
    except Exception:  # noqa: BLE001
        return out

    deltas: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        labels = actual_arr[idx].tolist()
        try:
            value_a = statistic(a_arr[idx].tolist(), labels)
            value_b = statistic(b_arr[idx].tolist(), labels)
        except Exception:  # noqa: BLE001
            continue
        if value_a is None or value_b is None:
            continue
        if math.isnan(value_a) or math.isnan(value_b):
            continue
        deltas.append(float(value_a - value_b))

    if len(deltas) < resamples * 0.5:
        return out

    alpha = (1.0 - confidence) / 2.0
    out["low"] = float(np.percentile(deltas, 100 * alpha))
    out["high"] = float(np.percentile(deltas, 100 * (1 - alpha)))
    out["win_rate"] = float(np.mean([d > 0 for d in deltas]))
    # Significant only when the whole interval sits on one side of zero.
    out["significant"] = bool(out["low"] > 0 or out["high"] < 0)
    return out


def evaluate(
    scorer_name: str,
    predicted: list[float],
    actual: list[float],
    *,
    label_scale: float = 5.0,
    relevance_threshold: float = 3.0,
    seconds: float = 0.0,
    with_intervals: bool = True,
    resamples: int = 2000,
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

    if with_intervals:
        result.intervals["spearman"] = bootstrap_ci(
            predicted, actual, lambda p, a: spearman(p, a)[0], resamples=resamples
        )
        result.intervals["kendall_tau"] = bootstrap_ci(
            predicted, actual, kendall_tau, resamples=resamples
        )
        result.intervals["ndcg@5"] = bootstrap_ci(
            predicted, actual, lambda p, a: ndcg_at_k(p, a, 5), resamples=resamples
        )

    if math.isnan(result.spearman):
        result.notes.append("Spearman undefined — scores or labels are constant.")
    if len(predicted) < 20:
        result.notes.append(
            f"Only {len(predicted)} candidates; treat differences between scorers as indicative."
        )

    width = result.intervals.get("spearman")
    if width and not any(math.isnan(b) for b in width) and (width[1] - width[0]) > 0.3:
        # Name the scorer: these notes are pooled across the comparison table,
        # so an unattributed warning is not actionable.
        result.notes.append(
            f"{scorer_name}: Spearman 95% CI spans {width[1] - width[0]:.2f} — too "
            f"wide to rank scorers on this metric alone."
        )
    return result
