"""
The evaluation harness: run every scorer over a labelled set and compare.

The expensive work — LLM resume extraction and GitHub enrichment — is done once
per candidate and shared across scorers. Only scoring is repeated. That is what
makes it practical to re-run the whole comparison after a weight change, which
in turn is what makes tuning an experiment rather than a guess.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from matching.config import MatchingConfig, get_config
from matching.enrichment.github import analyse_github
from matching.evaluation.dataset import EvaluationSet, LabelledCandidate
from matching.evaluation.metrics import (
    MetricResult,
    evaluate,
    ndcg_at_k,
    paired_bootstrap_delta,
    spearman,
)
from matching.llm import get_provider
from matching.parsing.resume import parse_resume
from matching.schemas import CandidateScore, GitHubProfile, ResumeProfile
from matching.scoring.base import Scorer
from matching.scoring.baselines import JDKeywordBaseline, KeywordBaseline, RandomBaseline
from matching.scoring.ensemble import EnsembleScorer

logger = logging.getLogger(__name__)


@dataclass
class PreparedCandidate:
    """A candidate parsed once, ready to be scored by any number of scorers."""

    source: LabelledCandidate
    resume: ResumeProfile
    github: GitHubProfile | None = None


@dataclass
class ComparisonReport:
    dataset: str
    n: int
    results: list[MetricResult] = field(default_factory=list)
    per_candidate: dict[str, dict[str, float]] = field(default_factory=dict)
    # Paired bootstrap of (best scorer - baseline), keyed by metric name.
    head_to_head: dict[str, dict] = field(default_factory=dict)
    compared: tuple[str, str] | None = None

    def best(self, metric: str = "spearman") -> MetricResult | None:
        ranked = [r for r in self.results if not _is_nan(getattr(r, metric, float("nan")))]
        if not ranked:
            return None
        return max(ranked, key=lambda r: getattr(r, metric))

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "n": self.n,
            "results": [r.to_dict() for r in self.results],
            "head_to_head": self.head_to_head,
            "compared": list(self.compared) if self.compared else None,
            "per_candidate": self.per_candidate,
        }

    def as_table(self) -> str:
        """Fixed-width comparison table for the CLI."""
        headers = ["scorer", "n", "spearman", "95% CI", "ndcg@5", "95% CI", "P@5", "MAE", "secs"]
        widths = [20, 4, 8, 14, 7, 14, 5, 6, 6]

        def row(cells: list[str]) -> str:
            return "  ".join(c.ljust(w)[:w] for c, w in zip(cells, widths, strict=True))

        lines = [row(headers), row(["-" * w for w in widths])]
        for result in self.results:
            data = result.to_dict()
            lines.append(row([
                data["scorer"],
                str(data["n"]),
                _fmt(data["spearman"]),
                result.interval_str("spearman") or "—",
                _fmt(data["ndcg@5"]),
                result.interval_str("ndcg@5") or "—",
                _fmt(data["precision@5"], 2),
                _fmt(data["mae"], 1),
                f"{data['seconds']:.1f}",
            ]))
        return "\n".join(lines)

    def significance_summary(self) -> str:
        """Plain-English reading of the paired comparison."""
        if not self.head_to_head or not self.compared:
            return ""
        winner, baseline = self.compared
        lines = [f"Paired bootstrap: {winner} vs {baseline} (same candidates, 2000 resamples)"]
        for metric, stats in self.head_to_head.items():
            if _is_nan(stats.get("delta")):
                continue
            verdict = (
                "significant — the interval excludes zero"
                if stats["significant"]
                else "NOT significant — the interval contains zero"
            )
            lines.append(
                f"  {metric:<10} delta {stats['delta']:+.3f}  "
                f"95% CI [{stats['low']:+.3f}, {stats['high']:+.3f}]  "
                f"won {stats['win_rate']:.0%} of resamples  — {verdict}"
            )
        return "\n".join(lines)


def default_scorers(config: MatchingConfig | None = None) -> list[Scorer]:
    """The standard comparison ladder, weakest first."""
    return [
        RandomBaseline(),
        KeywordBaseline(),
        JDKeywordBaseline(),
        EnsembleScorer(config=config or get_config()),
    ]


def prepare(
    dataset: EvaluationSet,
    config: MatchingConfig | None = None,
    *,
    with_github: bool = True,
    progress: bool = True,
) -> list[PreparedCandidate]:
    """Parse (and optionally enrich) every candidate exactly once."""
    config = config or get_config()
    provider = get_provider(config)
    prepared: list[PreparedCandidate] = []

    for index, candidate in enumerate(dataset.candidates, start=1):
        if progress:
            print(f"  [{index}/{len(dataset)}] parsing {candidate.id}…", flush=True)

        resume = parse_resume(
            candidate.resume_text, provider, fallback_email=candidate.email
        )

        github = None
        username = candidate.github_username or resume.github_username
        if with_github and username and dataset.job.is_technical:
            github = analyse_github(username, config)

        prepared.append(PreparedCandidate(source=candidate, resume=resume, github=github))

    return prepared


def compare(
    dataset: EvaluationSet,
    scorers: list[Scorer] | None = None,
    config: MatchingConfig | None = None,
    *,
    with_github: bool = True,
    progress: bool = True,
) -> ComparisonReport:
    """Score the labelled set with each scorer and compute the metric table."""
    config = config or get_config()
    scorers = scorers or default_scorers(config)

    if progress:
        print(f"Preparing {len(dataset)} candidates…", flush=True)
    prepared = prepare(dataset, config, with_github=with_github, progress=progress)

    labels = [p.source.label for p in prepared]
    report = ComparisonReport(dataset=dataset.name, n=len(prepared))
    report.per_candidate = {
        p.source.id: {"label": p.source.label} for p in prepared
    }

    for scorer in scorers:
        if progress:
            print(f"Scoring with {scorer.name}…", flush=True)
        started = time.time()
        predictions: list[float] = []

        for item in prepared:
            try:
                result: CandidateScore = scorer.score(item.resume, dataset.job, item.github)
                predictions.append(result.overall)
            except Exception as exc:  # noqa: BLE001 - one failure must not void the run
                logger.error("%s failed on %s: %s", scorer.name, item.source.id, exc)
                predictions.append(0.0)
            report.per_candidate[item.source.id][scorer.name] = predictions[-1]

        elapsed = time.time() - started
        report.results.append(
            evaluate(scorer.name, predictions, labels, seconds=elapsed)
        )

    # Present strongest first so the table reads as a ladder.
    report.results.sort(
        key=lambda r: (-r.spearman if not _is_nan(r.spearman) else 1.0),
    )

    _add_head_to_head(report, labels)
    return report


def _add_head_to_head(report: ComparisonReport, labels: list[float]) -> None:
    """Paired-bootstrap the best scorer against the legacy keyword baseline.

    Two separate confidence intervals cannot answer "is A better than B" when
    both are measured on the same candidates — see ``paired_bootstrap_delta``.
    """
    best = report.best("spearman")
    baseline_name = "keyword_legacy"
    if best is None or best.scorer == baseline_name:
        return
    if baseline_name not in {r.scorer for r in report.results}:
        return

    ids = list(report.per_candidate)
    try:
        best_preds = [report.per_candidate[i][best.scorer] for i in ids]
        base_preds = [report.per_candidate[i][baseline_name] for i in ids]
        ordered_labels = [report.per_candidate[i]["label"] for i in ids]
    except KeyError:
        return

    _ = labels  # ordering comes from per_candidate to stay aligned with predictions

    report.compared = (best.scorer, baseline_name)
    report.head_to_head = {
        "spearman": paired_bootstrap_delta(
            best_preds, base_preds, ordered_labels, lambda p, a: spearman(p, a)[0]
        ),
        "ndcg@5": paired_bootstrap_delta(
            best_preds, base_preds, ordered_labels, lambda p, a: ndcg_at_k(p, a, 5)
        ),
    }


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _fmt(value, places: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"
