"""
The data contracts every layer of the engine passes around.

These are plain dataclasses rather than Django models on purpose: the scoring
pipeline must be runnable offline (evaluation harness, notebooks, CI) with no
database. The web layer persists them as JSON in a ``JSONField``, so every type
here round-trips through ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

# ─────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────

def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class _Serialisable:
    """Mixin giving every schema a symmetric dict representation."""

    def to_dict(self) -> dict:
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict | None):
        """Tolerant constructor — unknown keys are dropped, missing keys default.

        Tolerance matters because these records are produced by an LLM and
        persisted across schema revisions; a new field must not break the
        deserialisation of rows written by an older version.
        """
        data = data or {}
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


# ─────────────────────────────────────────────────────────────
# Resume side
# ─────────────────────────────────────────────────────────────

@dataclass
class SkillMention(_Serialisable):
    """A skill as *evidenced* in a resume, not merely as a matched substring.

    ``years`` and ``last_used_year`` are what let the engine distinguish
    "used Python daily for four years, most recently this year" from
    "Python" listed once in a skills salad.
    """

    name: str
    years: float | None = None
    last_used_year: int | None = None
    # self-reported | project | professional — professional evidence outranks
    # a bare self-report on a skills list.
    evidence: str = "self-reported"
    context: str = ""

    def normalised(self) -> str:
        return self.name.strip().lower()


@dataclass
class WorkExperience(_Serialisable):
    title: str = ""
    company: str = ""
    start_year: int | None = None
    end_year: int | None = None   # None + is_current -> ongoing
    is_current: bool = False
    description: str = ""
    technologies: list[str] = field(default_factory=list)

    def duration_years(self, today_year: int | None = None) -> float:
        if self.start_year is None:
            return 0.0
        end = self.end_year or (today_year or date.today().year)
        return max(0.0, float(end - self.start_year))


@dataclass
class Education(_Serialisable):
    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    graduation_year: int | None = None


@dataclass
class ResumeProfile(_Serialisable):
    """Structured view of one resume, extracted by the LLM under a JSON schema."""

    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    github_username: str = ""
    linkedin_url: str = ""
    portfolio_url: str = ""
    summary: str = ""
    total_years_experience: float | None = None
    skills: list[SkillMention] = field(default_factory=list)
    experience: list[WorkExperience] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    # Kept for the keyword baseline, semantic embedding and audit trail.
    raw_text: str = ""
    # True when extraction failed and this is a degraded, text-only record.
    extraction_failed: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> ResumeProfile:
        data = dict(data or {})
        skills = [SkillMention.from_dict(s) for s in data.pop("skills", []) or []]
        experience = [WorkExperience.from_dict(e) for e in data.pop("experience", []) or []]
        education = [Education.from_dict(e) for e in data.pop("education", []) or []]
        names = {f.name for f in dataclasses.fields(cls)}
        obj = cls(**{k: v for k, v in data.items() if k in names})
        obj.skills, obj.experience, obj.education = skills, experience, education
        return obj

    def skill_names(self) -> list[str]:
        return [s.normalised() for s in self.skills if s.name.strip()]

    def inferred_years_experience(self) -> float:
        """Prefer the model's own estimate, else derive from the work history.

        Deriving takes the span from earliest start to latest end rather than
        summing roles, because summing double-counts overlapping/concurrent
        positions and inflates juniors who list internships alongside study.
        """
        if self.total_years_experience is not None:
            return max(0.0, float(self.total_years_experience))
        starts = [e.start_year for e in self.experience if e.start_year]
        if not starts:
            return 0.0
        this_year = date.today().year
        ends = [e.end_year or this_year for e in self.experience]
        return max(0.0, float(max(ends) - min(starts)))


# ─────────────────────────────────────────────────────────────
# Job side
# ─────────────────────────────────────────────────────────────

@dataclass
class JobSpec(_Serialisable):
    """Machine-readable requirements distilled from a free-text JD.

    Turning the JD into this shape is what makes scoring role-aware: every
    downstream comparison is against *these* requirements, so a marketing role
    is never scored on engineering vocabulary.
    """

    role_title: str = ""
    seniority: str = ""            # intern | junior | mid | senior | lead | principal
    min_years: float = 0.0
    max_years: float | None = None
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    domain: str = ""
    education_requirement: str = ""
    location: str = ""
    employment_type: str = ""
    raw_jd: str = ""
    # Does this role's quality depend on shipped code? Drives GitHub weighting.
    is_technical: bool = True

    def all_skills(self) -> list[str]:
        seen, out = set(), []
        for s in list(self.must_have_skills) + list(self.nice_to_have_skills):
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


# ─────────────────────────────────────────────────────────────
# GitHub side
# ─────────────────────────────────────────────────────────────

@dataclass
class RepoSummary(_Serialisable):
    name: str = ""
    description: str = ""
    language: str = ""
    stars: int = 0
    forks: int = 0
    size_kb: int = 0
    is_fork: bool = False
    is_archived: bool = False
    topics: list[str] = field(default_factory=list)
    pushed_at: str = ""
    created_at: str = ""
    has_description: bool = False


@dataclass
class GitHubProfile(_Serialisable):
    """Evidence of real, sustained engineering work — or its absence."""

    username: str = ""
    found: bool = False
    error: str = ""

    name: str = ""
    bio: str = ""
    company: str = ""
    blog: str = ""
    public_repos: int = 0
    followers: int = 0
    following: int = 0
    account_created_at: str = ""
    account_age_days: int = 0

    total_stars: int = 0
    total_forks: int = 0
    # Bytes per language across non-fork repos -> the candidate's real stack.
    language_bytes: dict[str, int] = field(default_factory=dict)
    top_repos: list[RepoSummary] = field(default_factory=list)

    original_repos: int = 0
    forked_repos: int = 0
    days_since_last_push: int | None = None
    repos_pushed_last_year: int = 0
    documented_repo_ratio: float = 0.0

    @classmethod
    def from_dict(cls, data: dict | None) -> GitHubProfile:
        data = dict(data or {})
        repos = [RepoSummary.from_dict(r) for r in data.pop("top_repos", []) or []]
        names = {f.name for f in dataclasses.fields(cls)}
        obj = cls(**{k: v for k, v in data.items() if k in names})
        obj.top_repos = repos
        return obj

    def language_share(self) -> dict[str, float]:
        total = sum(self.language_bytes.values())
        if total <= 0:
            return {}
        return {k.lower(): v / total for k, v in self.language_bytes.items()}

    @property
    def original_repo_ratio(self) -> float:
        total = self.original_repos + self.forked_repos
        return self.original_repos / total if total else 0.0


# ─────────────────────────────────────────────────────────────
# Features & scores
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureVector(_Serialisable):
    """Named, unit-interval features plus the provenance behind each one.

    Every feature is scaled to [0, 1] so weights are directly comparable and a
    score breakdown is readable without mental arithmetic. ``provenance`` holds
    the human-readable "why" that the UI surfaces next to each number.
    """

    values: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def set(self, name: str, value: float, why: str = "") -> None:
        self.values[name] = float(max(0.0, min(1.0, value)))
        if why:
            self.provenance[name] = why

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)

    def as_row(self, order: list[str]) -> list[float]:
        """Fixed-order vector — the interface to any future trained model."""
        return [self.get(n) for n in order]


class Recommendation(str, Enum):
    STRONG_YES = "STRONG_YES"
    YES = "YES"
    MAYBE = "MAYBE"
    NO = "NO"

    @property
    def label(self) -> str:
        return {
            "STRONG_YES": "Strong fit — fast-track",
            "YES": "Good fit — interview",
            "MAYBE": "Borderline — recruiter review",
            "NO": "Weak fit",
        }[self.value]


@dataclass
class DimensionScore(_Serialisable):
    """One interpretable axis of the final score."""

    name: str
    score: float                 # 0..1
    weight: float                # 0..1, weights across dimensions sum to 1
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass
class CandidateScore(_Serialisable):
    """The engine's output: a number, its decomposition, and its caveats."""

    overall: float = 0.0                      # 0..100
    recommendation: str = Recommendation.NO.value
    dimensions: list[DimensionScore] = field(default_factory=list)
    # Low when evidence is thin (no resume text, no GitHub, extraction failed) —
    # surfaced so a recruiter can tell "genuinely weak" from "we don't know".
    confidence: float = 0.0
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    scorer: str = ""
    scorer_version: str = ""
    features: FeatureVector = field(default_factory=FeatureVector)

    @classmethod
    def from_dict(cls, data: dict | None) -> CandidateScore:
        data = dict(data or {})
        dims = [DimensionScore.from_dict(d) for d in data.pop("dimensions", []) or []]
        feats = FeatureVector.from_dict(data.pop("features", {}) or {})
        names = {f.name for f in dataclasses.fields(cls)}
        obj = cls(**{k: v for k, v in data.items() if k in names})
        obj.dimensions, obj.features = dims, feats
        return obj

    @property
    def recommendation_enum(self) -> Recommendation:
        try:
            return Recommendation(self.recommendation)
        except ValueError:
            return Recommendation.NO
