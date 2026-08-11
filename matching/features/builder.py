"""
Structured records -> a named feature vector.

Every feature is a documented function of the parsed resume, the job spec and
the GitHub profile, scaled to [0, 1]. Keeping feature extraction separate from
scoring is what makes the system inspectable and improvable: the same vector
feeds the hand-weighted scorer used today and would feed a learned model
trained on hiring outcomes tomorrow, with no change to anything upstream.

Each feature also records a plain-English provenance string, which is what the
UI shows the recruiter instead of a bare number.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date

import numpy as np

from matching.config import MatchingConfig, get_config
from matching.features.skills import (
    SkillMatcher,
    SkillMatchResult,
)
from matching.features.skills import (
    calibrate_document_similarity as calibrate,
)
from matching.llm import LLMProvider
from matching.schemas import FeatureVector, GitHubProfile, JobSpec, ResumeProfile

logger = logging.getLogger(__name__)

# The canonical feature order — the interface any trained model would consume.
FEATURE_ORDER = [
    "must_have_coverage",
    "must_have_hard_coverage",
    "nice_to_have_coverage",
    "skill_depth",
    "skill_recency",
    "experience_fit",
    "semantic_similarity",
    "title_relevance",
    "education_fit",
    "career_progression",
    "github_recency",
    "github_substance",
    "github_language_match",
]

# Evidence quality: a skill used professionally is worth more than one listed.
_EVIDENCE_WEIGHT = {"professional": 1.0, "project": 0.7, "self-reported": 0.35}

_SENIORITY_RANK = [
    (r"\bintern|trainee\b", 0),
    (r"\bjunior|jr\.?|associate|graduate|entry\b", 1),
    (r"\bsenior|sr\.?\b", 3),
    (r"\blead|principal|staff|head|director|vp\b", 4),
]

_DEGREE_RANK = {
    "phd": 4, "doctorate": 4, "d.phil": 4,
    "master": 3, "msc": 3, "m.s": 3, "mtech": 3, "mba": 3, "m.tech": 3,
    "bachelor": 2, "bsc": 2, "b.s": 2, "btech": 2, "b.tech": 2, "be": 2, "b.e": 2,
    "diploma": 1, "associate": 1,
}


class FeatureBuilder:
    """Builds a ``FeatureVector`` from the three structured inputs."""

    def __init__(self, provider: LLMProvider, config: MatchingConfig | None = None):
        self.provider = provider
        self.config = config or get_config()
        self.matcher = SkillMatcher(provider, self.config)

    def build(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        github: GitHubProfile | None = None,
    ) -> tuple[FeatureVector, SkillMatchResult, SkillMatchResult]:
        features = FeatureVector()

        # Must-haves and nice-to-haves are matched in one pass so the retrieval
        # and verification stages are shared rather than paid for twice.
        combined = self.matcher.match(
            list(job.must_have_skills) + list(job.nice_to_have_skills), resume.skills
        )
        split = len(job.must_have_skills)
        must = SkillMatchResult(
            matches=combined.matches[:split], verified=combined.verified, note=combined.note
        )
        nice = SkillMatchResult(
            matches=combined.matches[split:], verified=combined.verified, note=combined.note
        )

        self._skill_features(features, must, nice, job)
        self._experience_features(features, resume, job)
        self._semantic_features(features, resume, job)
        self._education_features(features, resume, job)
        self._github_features(features, github, job)

        return features, must, nice

    # ── skills ───────────────────────────────────────────────

    def _skill_features(
        self,
        features: FeatureVector,
        must: SkillMatchResult,
        nice: SkillMatchResult,
        job: JobSpec,
    ) -> None:
        if job.must_have_skills:
            hits = len(must.matched)
            features.set(
                "must_have_coverage", must.coverage(),
                f"{hits}/{len(must.matches)} required skills evidenced"
                + (f" — missing: {', '.join(m.required for m in must.missing[:4])}"
                   if must.missing else ""),
            )
            features.set(
                "must_have_hard_coverage", must.hard_coverage(),
                f"{hits} of {len(must.matches)} required skills clearly present",
            )
        else:
            # No extractable requirements: stay neutral rather than award a
            # free perfect score for satisfying an empty list.
            features.set("must_have_coverage", 0.5, "No specific requirements listed in the JD")
            features.set("must_have_hard_coverage", 0.5, "No specific requirements listed in the JD")

        if job.nice_to_have_skills:
            features.set(
                "nice_to_have_coverage", nice.coverage(),
                f"{len(nice.matched)}/{len(nice.matches)} preferred skills evidenced",
            )
        else:
            features.set("nice_to_have_coverage", 0.5, "No preferred skills listed")

        matched = must.matched + nice.matched
        if matched:
            depth = float(np.mean([
                _EVIDENCE_WEIGHT.get(m.evidence, 0.35) for m in matched
            ]))
            professional = sum(1 for m in matched if m.evidence == "professional")
            features.set(
                "skill_depth", depth,
                f"{professional}/{len(matched)} matched skills backed by professional experience",
            )

            this_year = date.today().year
            years = [m.recency_year for m in matched if m.recency_year]
            if years:
                # Linear decay to zero over 6 years since last use.
                staleness = [max(0.0, 1.0 - (this_year - y) / 6.0) for y in years]
                newest = max(years)
                features.set(
                    "skill_recency", float(np.mean(staleness)),
                    f"Relevant skills last used {this_year - newest} year(s) ago"
                    if this_year > newest else "Relevant skills in current use",
                )
            else:
                features.set("skill_recency", 0.5, "Resume gives no dates for skill usage")
        else:
            features.set("skill_depth", 0.0, "No required or preferred skills matched")
            features.set("skill_recency", 0.0, "No matched skills to date")

    # ── experience ───────────────────────────────────────────

    def _experience_features(
        self, features: FeatureVector, resume: ResumeProfile, job: JobSpec
    ) -> None:
        years = resume.inferred_years_experience()
        required = job.min_years or 0.0

        if required <= 0:
            score = 1.0
            why = f"{years:g} years experience; the role states no minimum"
        elif years >= required:
            score = 1.0
            why = f"{years:g} years experience meets the {required:g}-year requirement"
            ceiling = job.max_years
            if ceiling and years > ceiling * 1.6:
                # Substantially over the band is a mild retention/level-fit
                # concern, not a disqualification — hence a floor of 0.8.
                score = 0.8
                why = (
                    f"{years:g} years is well above the {required:g}-{ceiling:g} band "
                    f"— possible level mismatch"
                )
        else:
            # Concave shortfall curve: 1 year short of 4 is a near miss, 4 years
            # short of 4 is not. The 0.7 exponent keeps near-misses competitive.
            score = (years / required) ** 0.7
            why = f"{years:g} years experience against a {required:g}-year requirement"

        features.set("experience_fit", score, why)

        ranks = [
            (e.start_year or 0, _title_rank(e.title))
            for e in resume.experience if e.title
        ]
        ranks = [r for r in ranks if r[0]]
        if len(ranks) >= 2:
            ranks.sort()
            delta = ranks[-1][1] - ranks[0][1]
            # Map a -2..+3 seniority delta onto [0, 1].
            score = float(np.clip((delta + 2) / 5.0, 0.0, 1.0))
            features.set(
                "career_progression", score,
                "Rising seniority across roles" if delta > 0
                else "Level unchanged across roles" if delta == 0
                else "Seniority decreased across roles",
            )
        else:
            features.set("career_progression", 0.5, "Too few dated roles to assess progression")

    # ── semantics ────────────────────────────────────────────

    def _semantic_features(
        self, features: FeatureVector, resume: ResumeProfile, job: JobSpec
    ) -> None:
        job_text = _job_text(job)
        resume_text = _resume_text(resume)

        if not job_text or not resume_text:
            features.set("semantic_similarity", 0.0, "Not enough text to compare")
            features.set("title_relevance", 0.0, "Not enough text to compare")
            return

        titles = [e.title for e in resume.experience if e.title][:5]

        try:
            texts = [job_text, resume_text, job.role_title or job_text] + titles
            vectors = self.provider.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Document embedding failed: %s", exc)
            features.set("semantic_similarity", 0.0, f"Embedding unavailable: {exc}")
            features.set("title_relevance", 0.0, "Embedding unavailable")
            return

        overall = calibrate(float(vectors[0] @ vectors[1]))
        features.set(
            "semantic_similarity", overall,
            f"Overall resume/JD content overlap: {overall:.0%}",
        )

        if titles:
            title_vectors = vectors[3:]
            best = float(np.max(title_vectors @ vectors[2]))
            calibrated = calibrate(best)
            closest = titles[int(np.argmax(title_vectors @ vectors[2]))]
            features.set(
                "title_relevance", calibrated,
                f"Closest previous title: \"{closest}\"",
            )
        else:
            features.set("title_relevance", 0.3, "No job titles found in the resume")

    # ── education ────────────────────────────────────────────

    def _education_features(
        self, features: FeatureVector, resume: ResumeProfile, job: JobSpec
    ) -> None:
        requirement = (job.education_requirement or "").lower()
        if not requirement.strip():
            # No stated requirement: neutral-positive. Penalising an unstated
            # requirement would silently filter on credentials the employer
            # never asked for.
            features.set("education_fit", 1.0, "No education requirement stated")
            return

        needed = _degree_rank(requirement)
        held = max((_degree_rank(e.degree) for e in resume.education), default=0)

        if needed == 0:
            features.set("education_fit", 1.0, "No specific degree level required")
        elif held >= needed:
            best = max(resume.education, key=lambda e: _degree_rank(e.degree), default=None)
            features.set(
                "education_fit", 1.0,
                f"Holds {best.degree}" if best else "Education requirement met",
            )
        elif held > 0:
            features.set(
                "education_fit", 0.6,
                f"Education below the stated requirement ({job.education_requirement})",
            )
        else:
            features.set(
                "education_fit", 0.3,
                f"No degree found; the role asks for {job.education_requirement}",
            )

    # ── github ───────────────────────────────────────────────

    def _github_features(
        self, features: FeatureVector, github: GitHubProfile | None, job: JobSpec
    ) -> None:
        if not github or not github.found:
            # Deliberately NOT set to 0. Absent features are dropped by the
            # scorer and their weight redistributed — see scoring/ensemble.py.
            return

        if github.days_since_last_push is None:
            features.set("github_recency", 0.0, "No repository activity found")
        else:
            days = github.days_since_last_push
            # Half-life of ~6 months: active now = 1.0, a year idle ≈ 0.25.
            score = math.exp(-days / 260.0)
            features.set(
                "github_recency", score,
                f"Last public commit {days} day(s) ago; "
                f"{github.repos_pushed_last_year} repo(s) active in the past year",
            )

        # Logarithmic compression: 0 -> 10 stars matters, 900 -> 1000 does not.
        star_score = math.log1p(github.total_stars) / math.log1p(200)
        repo_score = math.log1p(github.original_repos) / math.log1p(20)
        substance = float(np.clip(
            0.40 * min(star_score, 1.0)
            + 0.30 * min(repo_score, 1.0)
            + 0.15 * github.original_repo_ratio
            + 0.15 * github.documented_repo_ratio,
            0.0, 1.0,
        ))
        features.set(
            "github_substance", substance,
            f"{github.original_repos} original repo(s), {github.total_stars} star(s), "
            f"{github.documented_repo_ratio:.0%} documented",
        )

        shares = github.language_share()
        if not shares or not job.all_skills():
            features.set("github_language_match", 0.0, "No language overlap computable")
            return

        wanted = {s.lower() for s in job.all_skills()}
        overlap = sum(
            share for language, share in shares.items()
            if language in wanted or any(language in w or w in language for w in wanted)
        )
        top = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)[:3]
        features.set(
            "github_language_match", float(np.clip(overlap, 0.0, 1.0)),
            "Public code is "
            + ", ".join(f"{lang} {share:.0%}" for lang, share in top)
            + f" — {overlap:.0%} in the stack this role asks for",
        )


# ── helpers ──────────────────────────────────────────────────

def _job_text(job: JobSpec) -> str:
    parts = [
        job.role_title,
        f"Seniority: {job.seniority}" if job.seniority else "",
        "Required: " + ", ".join(job.must_have_skills) if job.must_have_skills else "",
        "Preferred: " + ", ".join(job.nice_to_have_skills) if job.nice_to_have_skills else "",
        " ".join(job.responsibilities),
    ]
    text = " ".join(p for p in parts if p).strip()
    # Fall back to the raw JD when the spec came back nearly empty.
    return text if len(text) > 40 else (job.raw_jd or text)[:4000]


def _resume_text(resume: ResumeProfile) -> str:
    parts = [
        resume.summary,
        " ".join(f"{e.title} at {e.company}. {e.description}" for e in resume.experience),
        "Skills: " + ", ".join(s.name for s in resume.skills),
    ]
    text = " ".join(p for p in parts if p).strip()
    return text if len(text) > 60 else (resume.raw_text or text)[:4000]


def _title_rank(title: str) -> int:
    for pattern, rank in _SENIORITY_RANK:
        if re.search(pattern, (title or "").lower()):
            return rank
    return 2  # unmarked titles read as mid-level


def _degree_rank(text: str) -> int:
    lowered = (text or "").lower()
    for key, rank in sorted(_DEGREE_RANK.items(), key=lambda kv: -kv[1]):
        if key in lowered:
            return rank
    return 0
