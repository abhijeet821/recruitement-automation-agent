"""
Ablation: does each scoring dimension earn its weight?

The eight dimension weights in ``scoring/ensemble.py`` are reasoned but, until
now, unvalidated. Reasoning is not evidence. This measures each one by removing
it, redistributing its weight over the rest, and re-measuring rank agreement
with the human labels.

Reading the output:

* **A large drop** when a dimension is removed means it is carrying real signal.
* **No change** means it is decorative — the other dimensions already encode
  whatever it contributes, and its weight could go to them.
* **An improvement** when removed means it is actively harmful, and that is the
  finding worth having, because nothing else in the system would reveal it.

## Why this is cheap

Ablating naively would re-score every candidate for every dimension — on a local
model that is hours. It is unnecessary: a ``CandidateScore`` already carries its
per-dimension scores and weights, so an ablated variant is pure arithmetic over
results already computed. One scoring pass, then N recomputations.

The critical-gap penalty is held constant across variants. It is a multiplier on
the whole score rather than a dimension, so ablating a dimension should not
silently change it — otherwise the measured delta would confound two effects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from matching.evaluation.metrics import ndcg_at_k, spearman
from matching.schemas import CandidateScore

logger = logging.getLogger(__name__)


@dataclass
class AblationRow:
    dropped: str
    spearman: float
    ndcg_at_5: float
    delta_spearman: float
    delta_ndcg: float
    mean_weight: float
    present_for: int          # candidates where the dimension had evidence
    # True when removing this dimension left every candidate with the same
    # score. Spearman is undefined on constant input, so the delta is nan — but
    # that is not "unknown", it is the strongest possible result: this dimension
    # was producing the entire ordering on its own.
    collapsed: bool = False

    @property
    def verdict(self) -> str:
        """Plain reading of the effect, thresholded loosely.

        The bands are deliberately coarse: on a small labelled set these deltas
        carry wide uncertainty, and pretending otherwise would be the same
        mistake the headline metrics used to make.
        """
        if self.collapsed:
            return "carries signal — ranking collapses"
        if math.isnan(self.delta_spearman):
            return "undefined"
        if self.delta_spearman <= -0.05:
            return "carries signal"
        if self.delta_spearman >= 0.05:
            return "HARMFUL — better without it"
        if abs(self.delta_spearman) < 0.01:
            return "no measurable effect"
        return "marginal"

    def to_dict(self) -> dict:
        return {
            "dropped": self.dropped,
            "spearman": round(self.spearman, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "delta_spearman": round(self.delta_spearman, 4),
            "delta_ndcg@5": round(self.delta_ndcg, 4),
            "mean_weight": round(self.mean_weight, 4),
            "present_for": self.present_for,
            "verdict": self.verdict,
        }


@dataclass
class AblationReport:
    n: int = 0
    baseline_spearman: float = float("nan")
    baseline_ndcg: float = float("nan")
    rows: list[AblationRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "baseline_spearman": round(self.baseline_spearman, 4),
            "baseline_ndcg@5": round(self.baseline_ndcg, 4),
            "rows": [r.to_dict() for r in self.rows],
            "notes": self.notes,
        }

    def as_table(self) -> str:
        headers = ["dimension dropped", "weight", "spearman", "Δ", "ndcg@5", "Δ", "verdict"]
        widths = [22, 7, 9, 8, 8, 8, 26]

        def row(cells):
            return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, widths, strict=True))

        lines = [row(headers), row(["-" * w for w in widths])]
        lines.append(row([
            "(none — full model)", "—", f"{self.baseline_spearman:.3f}", "—",
            f"{self.baseline_ndcg:.3f}", "—", "baseline",
        ]))
        for r in self.rows:
            lines.append(row([
                r.dropped, f"{r.mean_weight:.2f}",
                "collapse" if r.collapsed else f"{r.spearman:.3f}",
                f"{r.delta_spearman:+.3f}", f"{r.ndcg_at_5:.3f}",
                f"{r.delta_ndcg:+.3f}", r.verdict,
            ]))
        return "\n".join(lines)


def _recompute(score: CandidateScore, drop: str | None) -> float:
    """Recompute a score with one dimension removed and weights renormalised.

    Mirrors the ensemble exactly: drop the dimension, share its weight over the
    remaining ones proportionally, and reapply the same critical-gap penalty the
    full model applied — recovered from the original score so it stays fixed.
    """
    dimensions = [d for d in score.dimensions if drop is None or d.name != drop]
    if not dimensions:
        return 0.0

    total_weight = sum(d.weight for d in dimensions)
    if total_weight <= 0:
        return 0.0

    base = 100.0 * sum(d.score * (d.weight / total_weight) for d in dimensions)

    # The gap penalty is a multiplier on the whole score, not a dimension.
    # Recovering and reapplying it keeps the ablation measuring one thing.
    original_base = 100.0 * sum(d.score * d.weight for d in score.dimensions)
    penalty = (score.overall / original_base) if original_base > 0 else 1.0
    return base * penalty


def run_ablation(scores: list[CandidateScore], labels: list[float]) -> AblationReport:
    """Measure each dimension's contribution to rank agreement with the labels."""
    report = AblationReport(n=len(scores))

    if len(scores) < 5:
        report.notes.append("Too few candidates for ablation to mean anything.")
        return report

    baseline = [s.overall for s in scores]
    report.baseline_spearman, _ = spearman(baseline, labels)
    report.baseline_ndcg = ndcg_at_k(baseline, labels, 5)

    dimension_names: list[str] = []
    for score in scores:
        for dimension in score.dimensions:
            if dimension.name not in dimension_names:
                dimension_names.append(dimension.name)

    for name in dimension_names:
        ablated = [_recompute(s, name) for s in scores]
        collapsed = len(set(round(a, 9) for a in ablated)) == 1
        rho, _ = spearman(ablated, labels)
        ndcg = ndcg_at_k(ablated, labels, 5)

        present = [d.weight for s in scores for d in s.dimensions if d.name == name]
        report.rows.append(AblationRow(
            dropped=name,
            spearman=rho,
            ndcg_at_5=ndcg,
            # A collapse is maximal damage, not missing data: without this
            # dimension there is no ordering left at all.
            delta_spearman=(
                -1.0 - report.baseline_spearman if collapsed
                else rho - report.baseline_spearman
            ),
            delta_ndcg=ndcg - report.baseline_ndcg,
            mean_weight=sum(present) / len(present) if present else 0.0,
            present_for=len(present),
            collapsed=collapsed,
        ))

    # Most damaging to remove first. nan sorts unpredictably, so push undefined
    # rows to the end rather than letting them land arbitrarily.
    report.rows.sort(
        key=lambda r: (math.isnan(r.delta_spearman), r.delta_spearman)
    )

    if len(scores) < 30:
        report.notes.append(
            f"n={len(scores)}. These deltas carry wide uncertainty; treat them as "
            f"directional, not as grounds for retuning weights."
        )
    harmful = [r.dropped for r in report.rows
               if not math.isnan(r.delta_spearman) and r.delta_spearman >= 0.05]
    if harmful:
        report.notes.append(
            "Removing these *improved* rank agreement, which is worth "
            f"investigating before trusting them: {', '.join(harmful)}."
        )
    inert = [r.dropped for r in report.rows
             if not math.isnan(r.delta_spearman) and abs(r.delta_spearman) < 0.01]
    if inert:
        report.notes.append(
            "No measurable effect on this set — either genuinely redundant, or "
            f"the set is too small to show it: {', '.join(inert)}."
        )

    # The important caveat, and the easiest result to misread. When most
    # dimensions can be removed without moving the ranking, the likeliest
    # explanation is not that they are worthless — it is that they are strongly
    # correlated (a candidate strong on required skills tends to be strong on
    # role alignment and skill quality too), so the ordering is over-determined
    # and any one signal is redundant *given the others*. Ablation cannot
    # discriminate under those conditions, and dropping a dimension on this
    # evidence would be a mistake.
    if report.rows and len(inert) > len(report.rows) / 2:
        report.notes.append(
            f"{len(inert)} of {len(report.rows)} dimensions can be removed with no "
            f"effect on the ranking. That points to correlated dimensions and an "
            f"over-determined ordering on this set, not to dead weight. Ablation "
            f"cannot separate them here — it needs candidates that are strong on "
            f"some dimensions and weak on others, which a 14-row synthetic set "
            f"with cleanly separated labels does not provide."
        )
    return report
