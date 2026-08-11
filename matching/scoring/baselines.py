"""
Baselines the production scorer has to beat.

The hardcoded keyword counter that used to *be* the scoring engine is preserved
here verbatim as ``KeywordBaseline``. It was not deleted, because a claim like
"semantic matching is better" is worthless without a number attached, and the
number requires the old method to still be runnable.

Three baselines form a ladder of increasing sophistication:

    RandomBaseline      seeded noise — the floor any real method must clear
    KeywordBaseline     the original: 9 hardcoded, role-independent keywords
    JDKeywordBaseline   keyword overlap against the *actual* JD requirements —
                        a genuinely competitive baseline, and the honest one to
                        compare against, since beating pure noise proves little

``manage.py evaluate_scorer`` runs all of them against the labelled set and
prints the comparison.
"""

from __future__ import annotations

import hashlib

from matching.features.skills import normalise_skill
from matching.schemas import (
    CandidateScore,
    DimensionScore,
    GitHubProfile,
    JobSpec,
    ResumeProfile,
)
from matching.scoring.base import Scorer, band

# The exact list from the original implementation. Frozen deliberately: this is
# a historical artefact used for comparison, not a list anyone should extend.
LEGACY_KEYWORDS = [
    "python", "django", "api", "sql", "rest", "docker", "java", "node", "aws",
]


class KeywordBaseline(Scorer):
    """The original scorer: count hardcoded keywords in the resume text.

    Reproduced faithfully, including its defining flaw — the keyword list is
    fixed, so the score is completely independent of the role being hired for.
    A marketing candidate is scored on whether they mention Docker.
    """

    name = "keyword_legacy"
    version = "1.0"

    def score(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> CandidateScore:
        text = (resume.raw_text or "").lower()
        hits = [k for k in LEGACY_KEYWORDS if k in text]
        raw = len(hits)
        overall = 100.0 * raw / len(LEGACY_KEYWORDS)

        return CandidateScore(
            overall=round(overall, 2),
            recommendation=band(overall),
            dimensions=[
                DimensionScore(
                    name="Keyword hits",
                    score=raw / len(LEGACY_KEYWORDS),
                    weight=1.0,
                    rationale=f"{raw}/{len(LEGACY_KEYWORDS)} hardcoded keywords present",
                    evidence=hits,
                )
            ],
            confidence=0.2,
            summary=f"Legacy keyword score: {raw}/{len(LEGACY_KEYWORDS)} terms matched.",
            flags=["Role-independent scoring — the keyword list ignores the job description"],
            scorer=self.name,
            scorer_version=self.version,
        )


class JDKeywordBaseline(Scorer):
    """Substring overlap against the JD's own required skills.

    Role-aware, unlike the legacy scorer, but still literal: it cannot connect
    "PyTorch" to "deep learning frameworks", and it will happily match "Go"
    inside "Django". This is the baseline that makes the semantic scorer's
    improvement meaningful.
    """

    name = "keyword_jd"
    version = "1.0"

    def score(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> CandidateScore:
        required = job.must_have_skills or job.nice_to_have_skills
        if not required:
            return CandidateScore(
                overall=50.0, recommendation=band(50.0), confidence=0.1,
                summary="No requirements extracted from the JD.",
                scorer=self.name, scorer_version=self.version,
            )

        haystack = (resume.raw_text or "").lower()
        haystack += " " + " ".join(s.name.lower() for s in resume.skills)

        hits = [s for s in required if normalise_skill(s) in haystack or s.lower() in haystack]
        coverage = len(hits) / len(required)
        overall = 100.0 * coverage

        return CandidateScore(
            overall=round(overall, 2),
            recommendation=band(overall),
            dimensions=[
                DimensionScore(
                    name="JD keyword overlap",
                    score=coverage,
                    weight=1.0,
                    rationale=f"{len(hits)}/{len(required)} required skills found as substrings",
                    evidence=hits[:10],
                )
            ],
            confidence=0.35,
            summary=f"{len(hits)} of {len(required)} required skills matched literally.",
            scorer=self.name,
            scorer_version=self.version,
        )


class RandomBaseline(Scorer):
    """Deterministic pseudo-random scores — the floor.

    Seeded from the candidate's identity so a run is reproducible. Any scorer
    that cannot beat this is not measuring anything.
    """

    name = "random"
    version = "1.0"

    def score(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> CandidateScore:
        seed = hashlib.sha256(
            f"{resume.email}{resume.full_name}{job.role_title}".encode()
        ).digest()
        overall = 100.0 * (int.from_bytes(seed[:4], "big") / 0xFFFFFFFF)
        return CandidateScore(
            overall=round(overall, 2),
            recommendation=band(overall),
            confidence=0.0,
            summary="Random baseline.",
            scorer=self.name,
            scorer_version=self.version,
        )
