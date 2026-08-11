"""
Adverse-impact auditing.

Redaction reduces the direct identity cue; it does not prove the resulting
ranking is fair. This module measures the outcome instead of assuming it, using
the standard US EEOC framework:

    selection rate       = selected / total, within a group
    impact ratio         = group selection rate / highest group's rate
    four-fifths rule     = an impact ratio below 0.80 is the conventional
                           threshold for adverse impact warranting review

Two honest caveats, both worth stating out loud in any discussion of this code:

1. The four-fifths rule is a screening heuristic, not a legal verdict, and it is
   unstable on small samples. A campaign with 12 applicants cannot support a
   meaningful conclusion, so the report carries an explicit
   ``sufficient_sample`` flag rather than pretending otherwise.
2. Auditing requires group data that this system does not and should not
   collect from resumes. Inferring ethnicity or gender from a name is itself a
   discriminatory act with poor accuracy. Group labels must come from a
   voluntary, separately-stored self-identification survey — which is why this
   module takes labels as an argument and never derives them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FOUR_FIFTHS = 0.80
MIN_GROUP_SIZE = 30


@dataclass
class GroupOutcome:
    group: str
    total: int
    selected: int

    @property
    def selection_rate(self) -> float:
        return self.selected / self.total if self.total else 0.0


@dataclass
class AuditReport:
    groups: list[GroupOutcome] = field(default_factory=list)
    impact_ratios: dict[str, float] = field(default_factory=dict)
    reference_group: str = ""
    flagged_groups: list[str] = field(default_factory=list)
    sufficient_sample: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def passes_four_fifths(self) -> bool:
        return not self.flagged_groups

    def to_dict(self) -> dict:
        return {
            "groups": [
                {
                    "group": g.group,
                    "total": g.total,
                    "selected": g.selected,
                    "selection_rate": round(g.selection_rate, 4),
                }
                for g in self.groups
            ],
            "impact_ratios": {k: round(v, 4) for k, v in self.impact_ratios.items()},
            "reference_group": self.reference_group,
            "flagged_groups": self.flagged_groups,
            "sufficient_sample": self.sufficient_sample,
            "passes_four_fifths": self.passes_four_fifths,
            "notes": self.notes,
        }


def adverse_impact(
    outcomes: list[tuple[str, bool]],
    *,
    min_group_size: int = MIN_GROUP_SIZE,
) -> AuditReport:
    """Compute selection rates and impact ratios.

    ``outcomes`` is a list of ``(group_label, was_selected)`` pairs drawn from
    voluntary self-identification, never inferred.
    """
    report = AuditReport()
    if not outcomes:
        report.notes.append("No outcome data supplied.")
        return report

    tallies: dict[str, list[int]] = {}
    for group, selected in outcomes:
        label = (group or "undisclosed").strip() or "undisclosed"
        bucket = tallies.setdefault(label, [0, 0])
        bucket[0] += 1
        bucket[1] += int(bool(selected))

    # "undisclosed" is reported for transparency but excluded from the ratio
    # comparison, since it is not a protected group.
    report.groups = sorted(
        (GroupOutcome(group=g, total=t, selected=s) for g, (t, s) in tallies.items()),
        key=lambda g: g.group,
    )
    comparable = [g for g in report.groups if g.group != "undisclosed" and g.total > 0]

    if len(comparable) < 2:
        report.notes.append("Fewer than two groups with data — no comparison possible.")
        return report

    reference = max(comparable, key=lambda g: g.selection_rate)
    report.reference_group = reference.group

    if reference.selection_rate == 0:
        report.notes.append("No candidate in any group was selected.")
        return report

    report.sufficient_sample = all(g.total >= min_group_size for g in comparable)
    if not report.sufficient_sample:
        smallest = min(g.total for g in comparable)
        report.notes.append(
            f"Smallest group has {smallest} candidates (recommended minimum "
            f"{min_group_size}). Treat these ratios as indicative only."
        )

    for group in comparable:
        ratio = group.selection_rate / reference.selection_rate
        report.impact_ratios[group.group] = ratio
        if ratio < FOUR_FIFTHS:
            report.flagged_groups.append(group.group)

    if report.flagged_groups:
        report.notes.append(
            f"Groups below the four-fifths threshold: {', '.join(report.flagged_groups)}. "
            "Review the scoring weights and the requirement list for proxies."
        )

    return report


def score_gap(
    scores: list[tuple[str, float]],
) -> dict[str, dict[str, float]]:
    """Mean/median score per group — a continuous complement to selection rate.

    A ranking can pass the four-fifths rule at one cut-off and fail at another;
    comparing the underlying score distributions catches disparities that a
    single threshold hides.
    """
    buckets: dict[str, list[float]] = {}
    for group, score in scores:
        buckets.setdefault((group or "undisclosed").strip() or "undisclosed", []).append(score)

    out: dict[str, dict[str, float]] = {}
    for group, values in buckets.items():
        ordered = sorted(values)
        count = len(ordered)
        median = (
            ordered[count // 2]
            if count % 2
            else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
        )
        out[group] = {
            "n": float(count),
            "mean": sum(ordered) / count,
            "median": median,
            "min": ordered[0],
            "max": ordered[-1],
        }
    return out
