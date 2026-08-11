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
from matching.evaluation.metrics import MetricResult, evaluate
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
            "per_candidate": self.per_candidate,
        }

    def as_table(self) -> str:
        """Fixed-width comparison table for the CLI."""
        headers = ["scorer", "n", "spearman", "kendall", "ndcg@5", "P@5", "R@5", "MAE", "secs"]
        widths = [22, 4, 9, 8, 7, 6, 6, 7, 6]

        def row(cells: list[str]) -> str:
            return "  ".join(c.ljust(w)[:w] for c, w in zip(cells, widths, strict=True))

        lines = [row(headers), row(["-" * w for w in widths])]
        for result in self.results:
            data = result.to_dict()
            lines.append(row([
                data["scorer"],
                str(data["n"]),
                _fmt(data["spearman"]),
                _fmt(data["kendall_tau"]),
                _fmt(data["ndcg@5"]),
                _fmt(data["precision@5"]),
                _fmt(data["recall@5"]),
                _fmt(data["mae"], 1),
                f"{data['seconds']:.1f}",
            ]))
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
    return report


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _fmt(value, places: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"
