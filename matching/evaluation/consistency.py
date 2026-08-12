"""
How stable is the LLM rubric?

The rubric carries 0.17 of the final score, and it is the one component that can
return a different answer to the identical question. That is the standard and
correct objection to LLM-as-judge, and until it is measured, the weight is an
act of faith.

This runs the rubric repeatedly over the *same* resume and reports the spread.
What the numbers mean:

* **Score standard deviation** — on the 0-1 dimension scale. Below ~0.05 the
  judge is effectively deterministic; above ~0.15 it is picking a number, and no
  single call should be trusted.
* **Range** — worst case seen. A recruiter comparing two candidates one point
  apart needs to know whether one point is inside the noise.
* **Verdict flip rate** — how often repeated runs would change the
  recommendation band. This is the number that actually matters, because it is
  the decision the score feeds, not the score itself.

If the spread is large, the honest responses are to lower the rubric weight, to
average several calls, or to drop temperature — not to quietly keep using one
sample.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from matching.llm import LLMProvider
from matching.schemas import GitHubProfile, JobSpec, ResumeProfile
from matching.scoring import rubric as rubric_module

logger = logging.getLogger(__name__)


@dataclass
class DimensionSpread:
    name: str
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def spread(self) -> float:
        return (max(self.values) - min(self.values)) if self.values else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": round(self.mean, 3),
            "stdev": round(self.stdev, 3),
            "min": round(min(self.values), 3) if self.values else None,
            "max": round(max(self.values), 3) if self.values else None,
            "range": round(self.spread, 3),
            "runs": len(self.values),
        }


@dataclass
class ConsistencyReport:
    candidate: str = ""
    runs: int = 0
    failures: int = 0
    dimensions: list[DimensionSpread] = field(default_factory=list)
    overall: DimensionSpread = field(default_factory=lambda: DimensionSpread("overall"))
    notes: list[str] = field(default_factory=list)

    @property
    def worst_stdev(self) -> float:
        return max((d.stdev for d in self.dimensions), default=0.0)

    @property
    def verdict(self) -> str:
        if self.overall.stdev < 0.05:
            return "stable — effectively deterministic"
        if self.overall.stdev < 0.10:
            return "acceptable — small run-to-run variation"
        if self.overall.stdev < 0.15:
            return "noisy — consider averaging several runs"
        return "UNSTABLE — a single call should not carry weight"

    def score_swing(self) -> float:
        """Worst-case swing in final points from rubric variance alone.

        The rubric contributes ``weight x 100`` points, so a spread of ``r`` on
        the 0-1 scale moves the final score by ``r x weight x 100``.
        """
        from matching.scoring.ensemble import DIMENSION_SPEC

        weight = DIMENSION_SPEC["Recruiter rubric"]["weight"]
        return self.overall.spread * weight * 100

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "runs": self.runs,
            "failures": self.failures,
            "overall": self.overall.to_dict(),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "verdict": self.verdict,
            "worst_case_score_swing": round(self.score_swing(), 2),
            "notes": self.notes,
        }

    def as_table(self) -> str:
        headers = ["dimension", "mean", "stdev", "min", "max", "range"]
        widths = [24, 7, 7, 7, 7, 7]

        def row(cells):
            return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, widths, strict=True))

        lines = [row(headers), row(["-" * w for w in widths])]
        for d in [*self.dimensions, self.overall]:
            lines.append(row([
                d.name, f"{d.mean:.3f}", f"{d.stdev:.3f}",
                f"{min(d.values):.3f}" if d.values else "—",
                f"{max(d.values):.3f}" if d.values else "—",
                f"{d.spread:.3f}",
            ]))
        return "\n".join(lines)


def measure_rubric_consistency(
    resume: ResumeProfile,
    job: JobSpec,
    provider: LLMProvider,
    *,
    runs: int = 5,
    github: GitHubProfile | None = None,
    temperature: float = 0.1,
    on_run=None,
) -> ConsistencyReport:
    """Run the rubric ``runs`` times on one resume and report the spread."""
    report = ConsistencyReport(candidate=resume.full_name or resume.email, runs=runs)

    collected: dict[str, list[float]] = {k: [] for k in rubric_module.RubricResult.DIMENSIONS}

    for index in range(runs):
        result = rubric_module.assess(resume, job, provider, github, temperature=temperature)
        if not result.ok:
            report.failures += 1
            logger.warning("Rubric run %d failed: %s", index + 1, result.error)
            if on_run:
                on_run(index + 1, runs, None)
            continue
        for key in rubric_module.RubricResult.DIMENSIONS:
            collected[key].append(result.scores[key])
        report.overall.values.append(result.overall)
        if on_run:
            on_run(index + 1, runs, result.overall)

    report.dimensions = [
        DimensionSpread(name=rubric_module.RubricResult.LABELS[k], values=v)
        for k, v in collected.items()
    ]

    if report.failures:
        report.notes.append(f"{report.failures}/{runs} run(s) failed outright.")
    if len(report.overall.values) < 2:
        report.notes.append("Not enough successful runs to measure variance.")
        return report

    swing = report.score_swing()
    report.notes.append(
        f"Worst case, rubric variance alone moves the final score by "
        f"{swing:.1f} points out of 100."
    )
    if swing >= 5:
        report.notes.append(
            "That is large enough to flip a recommendation band near a "
            "threshold. Average several calls, or lower the rubric weight."
        )

    unstable = [d.name for d in report.dimensions if d.stdev >= 0.15]
    if unstable:
        report.notes.append(
            f"Least reproducible dimension(s): {', '.join(unstable)}."
        )
    return report
