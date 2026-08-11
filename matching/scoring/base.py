"""The scorer interface shared by the production scorer and every baseline."""

from __future__ import annotations

import abc

from matching.schemas import CandidateScore, GitHubProfile, JobSpec, ResumeProfile


class Scorer(abc.ABC):
    """Anything that turns (resume, job, github) into a ``CandidateScore``.

    A single interface is what lets the evaluation harness treat the keyword
    baseline and the full ensemble as interchangeable and measure the difference
    between them on the same labelled data.
    """

    name: str = "base"
    version: str = "0"

    @abc.abstractmethod
    def score(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> CandidateScore:
        ...

    def score_many(
        self,
        items: list[tuple[ResumeProfile, JobSpec, GitHubProfile | None]],
    ) -> list[CandidateScore]:
        return [self.score(r, j, g) for r, j, g in items]


def band(overall: float) -> str:
    """Map a 0-100 score onto a recommendation band.

    Absolute thresholds rather than a per-campaign percentile, because a
    recruiter needs "is this person good enough" to mean the same thing in a
    5-applicant campaign and a 500-applicant one. Ranking within a campaign is
    still available by sorting on the raw score.
    """
    from matching.schemas import Recommendation

    if overall >= 75:
        return Recommendation.STRONG_YES.value
    if overall >= 60:
        return Recommendation.YES.value
    if overall >= 45:
        return Recommendation.MAYBE.value
    return Recommendation.NO.value
