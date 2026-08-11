"""
The production scorer: features + rubric, combined into an explainable score.

Design notes worth defending:

**Why a weighted linear model and not something learned?**
Because there is no training data yet. A hiring model needs labelled outcomes —
who was interviewed, who performed — and a new deployment has none. A
transparent linear combination of documented features is auditable on day one,
degrades predictably, and can be explained to a candidate who asks why they were
rejected. Once enough outcomes accumulate, ``FEATURE_ORDER`` is already the
input layer for a learned ranker; the weights here become the prior it replaces.

**Why redistribute weight instead of imputing zeros?**
Missing evidence is not negative evidence. A candidate with no GitHub is not a
worse engineer than one with a mediocre GitHub — we simply know less. Scoring a
missing dimension as zero conflates "absent" with "bad" and systematically
penalises anyone whose work is proprietary. Instead the dimension is dropped and
its weight is spread across the dimensions that *do* have evidence, so every
candidate is scored out of 100 on what is actually known about them, with
``confidence`` reporting how much that is.
"""

from __future__ import annotations

import logging

from matching.config import MatchingConfig, get_config
from matching.fairness.redaction import redact_profile
from matching.features.builder import FeatureBuilder
from matching.features.skills import SkillMatchResult
from matching.llm import LLMProvider, get_provider
from matching.schemas import (
    CandidateScore,
    DimensionScore,
    FeatureVector,
    GitHubProfile,
    JobSpec,
    ResumeProfile,
)
from matching.scoring import rubric as rubric_module
from matching.scoring.base import Scorer, band

logger = logging.getLogger(__name__)

SCORER_VERSION = "2.0.0"

# Each dimension is a weighted blend of features. Weights sum to 1.0 across all
# dimensions; any dimension whose evidence is missing is dropped and its weight
# is redistributed proportionally over the rest.
DIMENSION_SPEC: dict[str, dict] = {
    "Required skills": {
        "weight": 0.28,
        "features": {"must_have_coverage": 0.6, "must_have_hard_coverage": 0.4},
    },
    "Recruiter rubric": {
        "weight": 0.17,
        "features": {},  # supplied by the LLM pass, not the feature vector
    },
    "Experience level": {
        "weight": 0.14,
        "features": {"experience_fit": 0.8, "career_progression": 0.2},
    },
    "Role alignment": {
        "weight": 0.12,
        "features": {"semantic_similarity": 0.6, "title_relevance": 0.4},
    },
    "Skill quality": {
        "weight": 0.11,
        "features": {"skill_depth": 0.6, "skill_recency": 0.4},
    },
    "Code evidence": {
        "weight": 0.09,
        "features": {
            "github_substance": 0.4,
            "github_language_match": 0.35,
            "github_recency": 0.25,
        },
    },
    "Preferred skills": {
        "weight": 0.05,
        "features": {"nice_to_have_coverage": 1.0},
    },
    "Education": {
        "weight": 0.04,
        "features": {"education_fit": 1.0},
    },
}


class EnsembleScorer(Scorer):
    """Feature-weighted scoring with an optional LLM rubric dimension."""

    name = "ensemble"
    version = SCORER_VERSION

    def __init__(
        self,
        provider: LLMProvider | None = None,
        config: MatchingConfig | None = None,
    ):
        self.config = config or get_config()
        self.provider = provider or get_provider(self.config)
        self.builder = FeatureBuilder(self.provider)

    def score(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> CandidateScore:
        features, must, nice = self.builder.build(resume, job, github)

        rubric_result = None
        if self.config.rubric_enabled:
            judged = (
                redact_profile(resume) if self.config.blind_screening else resume
            )
            rubric_result = rubric_module.assess(judged, job, self.provider, github)
            if not rubric_result.ok:
                logger.warning("Rubric unavailable, scoring on features only: %s",
                               rubric_result.error)

        dimensions = self._build_dimensions(features, rubric_result, job, github)
        base = 100.0 * sum(d.contribution for d in dimensions)

        penalty, penalty_reason = self._critical_gap_penalty(features, must, job)
        overall = base * penalty

        confidence = self._confidence(resume, features, rubric_result, github, job)
        flags = self._flags(resume, features, rubric_result, github, job, confidence)
        if penalty_reason:
            flags.insert(0, penalty_reason)

        return CandidateScore(
            overall=round(overall, 2),
            recommendation=band(overall),
            dimensions=dimensions,
            confidence=round(confidence, 3),
            summary=self._summary(rubric_result, must, job, overall),
            strengths=self._strengths(rubric_result, must, features),
            gaps=self._gaps(rubric_result, must),
            flags=flags,
            scorer=self.name,
            scorer_version=self.version,
            features=features,
        )

    # ── dimensions ───────────────────────────────────────────

    def _build_dimensions(
        self,
        features: FeatureVector,
        rubric_result,
        job: JobSpec,
        github: GitHubProfile | None,
    ) -> list[DimensionScore]:
        available: list[tuple[str, float, float, str, list[str]]] = []

        for label, spec in DIMENSION_SPEC.items():
            if label == "Recruiter rubric":
                if rubric_result is None or not rubric_result.ok:
                    continue
                available.append((
                    label,
                    rubric_result.overall,
                    spec["weight"],
                    rubric_result.summary
                    or "Structured recruiter-style assessment of the resume.",
                    [
                        f"{rubric_module.RubricResult.LABELS[k]}: "
                        f"{rubric_result.scores[k] * 10:.0f}/10"
                        for k in rubric_module.RubricResult.DIMENSIONS
                    ],
                ))
                continue

            if label == "Code evidence":
                # Dropped for non-technical roles and for candidates with no
                # public profile — see the module docstring.
                if not job.is_technical or not github or not github.found:
                    continue

            present = {
                name: w for name, w in spec["features"].items() if name in features.values
            }
            if not present:
                continue

            total = sum(present.values())
            value = sum(features.get(n) * w for n, w in present.items()) / total
            evidence = [features.provenance[n] for n in present if n in features.provenance]
            available.append((label, value, spec["weight"], _lead(evidence), evidence))

        if not available:
            return []

        # Redistribute the weight of dropped dimensions proportionally.
        # Weights are stored at full precision — rounding each one here would
        # make them sum to slightly more or less than 1.0 and skew the total.
        # Rounding is a presentation concern and belongs in the template.
        total_weight = sum(w for _, _, w, _, _ in available)
        return [
            DimensionScore(
                name=label,
                score=round(value, 4),
                weight=weight / total_weight,
                rationale=rationale,
                evidence=evidence,
            )
            for label, value, weight, rationale, evidence in available
        ]

    # ── critical gaps ────────────────────────────────────────

    def _critical_gap_penalty(
        self, features: FeatureVector, must: SkillMatchResult, job: JobSpec
    ) -> tuple[float, str]:
        """Multiplicative penalty for missing most of the hard requirements.

        A purely additive model treats requirements as interchangeable: enough
        seniority, education and adjacent skills can outvote the absence of the
        core competence. That produced a concrete failure on the evaluation set
        — a senior Java engineer scored 67/100 ("interview") for a Python role
        despite no Python, no Django and no testing evidence, carried by tenure
        and transferable infrastructure skills.

        Real screening does not work that way. Hard requirements are
        conjunctive: missing two thirds of them is disqualifying no matter how
        strong the rest of the profile is. The penalty ramps from 1.0 at half
        coverage down to 0.45 at none, so it degrades smoothly rather than
        acting as a cliff-edge filter that would drop genuine near-misses.
        """
        if not job.must_have_skills:
            return 1.0, ""

        hard = features.get("must_have_hard_coverage")
        if hard >= 0.5:
            return 1.0, ""

        penalty = 0.45 + 1.10 * hard
        missing = [m.required for m in must.missing][:4]
        reason = (
            f"Only {len(must.matched)}/{len(must.matches)} required skills are met — "
            f"score reduced by {(1 - penalty) * 100:.0f}%"
            + (f". Missing: {', '.join(missing)}" if missing else "")
        )
        return penalty, reason

    # ── confidence ───────────────────────────────────────────

    def _confidence(
        self,
        resume: ResumeProfile,
        features: FeatureVector,
        rubric_result,
        github: GitHubProfile | None,
        job: JobSpec,
    ) -> float:
        """How much evidence backs this score, in [0, 1].

        Separating confidence from the score is what lets the UI distinguish
        "we assessed this person and they are weak" from "we could not read
        their resume". The old system collapsed both into a zero.
        """
        signals: list[tuple[float, float]] = []  # (value, weight)

        signals.append((0.0 if resume.extraction_failed else 1.0, 0.30))

        length = len(resume.raw_text or "")
        signals.append((min(1.0, length / 2500.0), 0.20))

        signals.append((min(1.0, len(resume.skills) / 8.0), 0.15))

        dated = sum(1 for e in resume.experience if e.start_year)
        signals.append((min(1.0, dated / 2.0), 0.15))

        signals.append((1.0 if (rubric_result and rubric_result.ok) else 0.0, 0.10))

        if job.is_technical:
            signals.append((1.0 if (github and github.found) else 0.4, 0.10))
        else:
            signals.append((1.0, 0.10))

        return sum(v * w for v, w in signals) / sum(w for _, w in signals)

    # ── narrative ────────────────────────────────────────────

    def _flags(
        self,
        resume: ResumeProfile,
        features: FeatureVector,
        rubric_result,
        github: GitHubProfile | None,
        job: JobSpec,
        confidence: float,
    ) -> list[str]:
        flags: list[str] = []

        if resume.extraction_failed:
            flags.append("Resume could not be parsed — score is based on limited evidence")
        if len(resume.raw_text or "") < 600:
            flags.append("Very little resume text extracted (possibly a scanned PDF)")
        if confidence < 0.5:
            flags.append("Low confidence — review manually before deciding")
        if features.get("must_have_hard_coverage") < 0.34 and job.must_have_skills:
            flags.append("Under a third of the required skills are evidenced")
        if features.get("skill_recency", 1.0) < 0.35 and "skill_recency" in features.values:
            flags.append("Relevant skills appear to be several years out of date")
        if job.is_technical and github and github.username and not github.found:
            flags.append(f"GitHub profile '{github.username}' could not be read: {github.error}")
        if rubric_result is not None and not rubric_result.ok:
            flags.append("Rubric assessment unavailable — scored on structured features only")
        if rubric_result is not None and rubric_result.ok:
            flags.extend(rubric_result.concerns[:2])

        return flags

    def _summary(self, rubric_result, must: SkillMatchResult, job: JobSpec, overall: float) -> str:
        if rubric_result is not None and rubric_result.ok and rubric_result.summary:
            return rubric_result.summary
        hits = len(must.matched)
        total = len(must.matches)
        return (
            f"Scored {overall:.0f}/100 for {job.role_title or 'this role'} — "
            f"{hits} of {total} required skills evidenced."
        )

    def _strengths(self, rubric_result, must: SkillMatchResult, features: FeatureVector) -> list[str]:
        if rubric_result is not None and rubric_result.ok and rubric_result.strengths:
            return rubric_result.strengths
        strong = [m.describe() for m in must.matched if m.similarity >= 0.8][:4]
        if not strong:
            best = sorted(features.values.items(), key=lambda kv: kv[1], reverse=True)[:3]
            strong = [features.provenance.get(k, k) for k, _ in best]
        return strong

    def _gaps(self, rubric_result, must: SkillMatchResult) -> list[str]:
        missing = [m.required for m in must.missing][:5]
        if rubric_result is not None and rubric_result.ok and rubric_result.gaps:
            merged = list(dict.fromkeys(rubric_result.gaps + missing))
            return merged[:5]
        return [f"No evidence of {name}" for name in missing]


def _lead(evidence: list[str]) -> str:
    return evidence[0] if evidence else ""
