"""
Semantic skill matching: embedding retrieval followed by LLM verification.

## Why not just cosine similarity

The obvious design is "embed both sides, threshold the cosine". I built that
first and measured it on labelled pairs (reproduce with
``manage.py calibrate_similarity``). bge-m3 raw cosine:

    TRUE   python           <-> python programming     0.843
    TRUE   postgresql       <-> postgres               0.931
    TRUE   rest api design  <-> rest                   0.607
    TRUE   docker           <-> containerisation       0.546
    FALSE  python           <-> java                   0.680   <-- higher than
    FALSE  postgresql       <-> oracle sql             0.694        two TRUE pairs
    FALSE  unit testing     <-> manual testing         0.642
    FALSE  django           <-> spring boot            0.409

The distributions overlap, and not by a little: ``python <-> java`` (0.680)
outranks the genuine match ``docker <-> containerisation`` (0.546). No single
threshold separates them, so a similarity cut-off cannot work — this is not a
tuning problem, it is what the model measures.

The reason is that embeddings encode **topical relatedness**, not
**substitutability**. Python and Java are maximally related — both are
general-purpose programming languages, they co-occur constantly in training
text — which is exactly the signal the embedding is designed to capture. But
"has Java" is not evidence of "knows Python", and that distinction is the entire
question a screening tool has to answer.

## The architecture that does work

Standard retrieve-then-rerank, motivated here by measurement:

    1. Exact match      normalised string equality -> 1.0, no model call
    2. Retrieve         embeddings pick the top-3 plausible resume skills per
                        requirement, using a deliberately *low* floor. This
                        stage is tuned for recall; precision is not its job.
    3. Verify           one batched LLM call judges the shortlist:
                        EXACT / STRONG / PARTIAL / NONE. The model does
                        understand that Java is not Python and that PyTorch is
                        a deep-learning framework.

Stage 2 turns "compare against every skill" into "compare against three", which
is what makes stage 3 affordable: one model call per candidate regardless of how
many skills are involved. Verdicts are cached by (requirement, skill) pair, and
those repeat massively across candidates for the same role.

If the model is unreachable the matcher degrades to embeddings alone with a
strict threshold — measurably worse, but still functional, and flagged as such.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from matching.config import MatchingConfig, get_config
from matching.llm import LLMError, LLMProvider
from matching.llm.cache import JSONCache
from matching.schemas import SkillMention

logger = logging.getLogger(__name__)

# Retrieval floor — deliberately permissive. Anything above this is merely
# *worth asking about*, not a match.
RETRIEVAL_FLOOR = 0.42
RETRIEVAL_TOP_K = 3

# Verdict -> credit. STRONG is not 1.0 because a transferable skill really is
# weaker evidence than the requirement itself.
VERDICT_SCORES = {"EXACT": 1.0, "STRONG": 0.85, "PARTIAL": 0.5, "NONE": 0.0}

# Credit at or above this counts a requirement as *satisfied*. Set above the
# PARTIAL credit of 0.5 on purpose: "Flask when we asked for Django" earns
# partial credit toward weighted coverage, but it does not mean the requirement
# is met, and hard coverage must reflect that.
MATCH_THRESHOLD = 0.6

# Fallback-only threshold, used when verification is unavailable. Set high
# because, per the measurements above, raw cosine cannot be trusted below it.
FALLBACK_SIMILARITY_FLOOR = 0.72

# Band used to rescale *document-level* similarity (resume vs JD), where topical
# relatedness genuinely is the quantity of interest.
DOC_SIMILARITY_FLOOR = 0.40
DOC_SIMILARITY_CEILING = 0.78

# Pure string normalisation — two spellings of one skill, not a scoring list.
_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "golang": "go", "c sharp": "c#", "csharp": "c#", "cpp": "c++",
    "postgres": "postgresql", "psql": "postgresql", "postgre sql": "postgresql",
    "k8s": "kubernetes", "gcp": "google cloud platform",
    "aws": "amazon web services", "amazon web services": "aws",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing",
    "restful": "rest", "rest apis": "rest api", "restful apis": "rest api",
    "reactjs": "react", "react.js": "react",
    "nodejs": "node.js", "node": "node.js",
    "vuejs": "vue", "vue.js": "vue", "nextjs": "next.js",
    "tf": "tensorflow", "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "drf": "django rest framework", "ci cd": "ci/cd",
    "unit tests": "unit testing", "pytest": "pytest",
}

_PUNCT = re.compile(r"[^a-z0-9+#./\s-]")
_SPACE = re.compile(r"\s+")

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "best_match": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["EXACT", "STRONG", "PARTIAL", "NONE"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["requirement", "best_match", "verdict"],
            },
        }
    },
    "required": ["verdicts"],
}

VERIFY_SYSTEM = (
    "You decide whether a candidate's skill satisfies a job requirement. You are "
    "strict: related is not the same as equivalent. A different programming "
    "language, a different database engine, or a different framework does not "
    "satisfy a requirement for a specific one."
)

VERIFY_PROMPT = """For each requirement below, decide whether any of the candidate's
listed skills satisfies it.

Verdicts:
- EXACT   the same skill, possibly spelled differently
          (PostgreSQL / Postgres; REST API design / REST APIs)
- STRONG  a different name for genuinely the same competence, or a specific
          instance of the requested general category
          (PyTorch satisfies "deep learning framework";
           Django REST Framework satisfies "REST API design")
- PARTIAL adjacent and transferable, but not the thing asked for
          (Flask for a Django requirement; MySQL for a PostgreSQL requirement)
- NONE    a different technology, or no relevant skill offered
          (Java for a Python requirement; Spring Boot for a Django requirement;
           manual testing for a unit testing requirement)

Judge each requirement independently. Pick the single best candidate skill.

{blocks}
"""


def normalise_skill(name: str) -> str:
    """Lowercase, strip punctuation noise, resolve spelling aliases."""
    text = _PUNCT.sub(" ", (name or "").lower().strip())
    text = _SPACE.sub(" ", text).strip(" -.")
    return _ALIASES.get(text, text)


def calibrate_document_similarity(raw: float) -> float:
    """Rescale a whole-document cosine onto [0, 1].

    Applied only to resume-vs-JD and title comparisons, where topical
    relatedness is the intended signal — never to individual skill decisions.
    """
    span = DOC_SIMILARITY_CEILING - DOC_SIMILARITY_FLOOR
    return float(np.clip((raw - DOC_SIMILARITY_FLOOR) / span, 0.0, 1.0))


# Backwards-compatible alias used by the feature builder.
calibrate = calibrate_document_similarity


@dataclass
class SkillMatch:
    """One JD requirement and the resume skill judged to satisfy it."""

    required: str
    matched: str = ""
    similarity: float = 0.0          # credit awarded, 0..1
    exact: bool = False
    verdict: str = "NONE"
    reason: str = ""
    evidence: str = ""               # professional | project | self-reported
    years: float | None = None
    recency_year: int | None = None
    verified: bool = False           # False when the LLM stage was unavailable

    @property
    def is_match(self) -> bool:
        return self.similarity >= MATCH_THRESHOLD

    def describe(self) -> str:
        if not self.is_match:
            return f"{self.required}: not evidenced"
        how = "exact match" if self.exact else f"via {self.matched}"
        bits = [how]
        if self.years:
            bits.append(f"{self.years:g}y")
        if self.evidence == "professional":
            bits.append("professional")
        elif self.evidence == "project":
            bits.append("project work")
        elif self.evidence == "self-reported":
            bits.append("listed only")
        return f"{self.required} ({', '.join(bits)})"


@dataclass
class SkillMatchResult:
    matches: list[SkillMatch] = field(default_factory=list)
    verified: bool = True
    note: str = ""

    @property
    def matched(self) -> list[SkillMatch]:
        return [m for m in self.matches if m.is_match]

    @property
    def missing(self) -> list[SkillMatch]:
        return [m for m in self.matches if not m.is_match]

    def coverage(self) -> float:
        """Credit-weighted coverage, so a PARTIAL counts for less than an EXACT."""
        if not self.matches:
            return 0.0
        return float(np.mean([m.similarity for m in self.matches]))

    def hard_coverage(self) -> float:
        """Plain fraction of requirements satisfied at or above the threshold."""
        if not self.matches:
            return 0.0
        return len(self.matched) / len(self.matches)


class SkillMatcher:
    """Retrieve-then-verify skill matcher."""

    def __init__(self, provider: LLMProvider, config: MatchingConfig | None = None):
        self.provider = provider
        self.config = config or get_config()
        self._cache = JSONCache(
            self.config.cache_dir, "skill_verdicts", self.config.cache_enabled
        )

    def match(
        self, required: list[str], resume_skills: list[SkillMention]
    ) -> SkillMatchResult:
        result = SkillMatchResult()
        if not required:
            return result

        have = [s for s in resume_skills if s.name and s.name.strip()]
        if not have:
            result.matches = [SkillMatch(required=r) for r in required]
            return result

        req_norm = [normalise_skill(r) for r in required]
        have_norm = [normalise_skill(s.name) for s in have]
        exact_index = {name: i for i, name in enumerate(have_norm)}

        matches: list[SkillMatch | None] = [None] * len(required)
        pending: list[int] = []

        # ── stage 1: exact ───────────────────────────────────
        for i, normalised in enumerate(req_norm):
            if normalised in exact_index:
                source = have[exact_index[normalised]]
                matches[i] = SkillMatch(
                    required=required[i], matched=source.name, similarity=1.0,
                    exact=True, verdict="EXACT", reason="identical after normalisation",
                    evidence=source.evidence, years=source.years,
                    recency_year=source.last_used_year, verified=True,
                )
            else:
                pending.append(i)

        if not pending:
            result.matches = [m for m in matches if m]
            return result

        # ── stage 2: retrieve ────────────────────────────────
        shortlists = self._retrieve(req_norm, have_norm, pending)

        # ── stage 3: verify ──────────────────────────────────
        verdicts, verified = self._verify(
            {
                required[i]: [have[j].name for j, _ in shortlists[i]]
                for i in pending
                if shortlists[i]
            }
        )
        result.verified = verified
        if not verified:
            result.note = (
                "Skill verification unavailable — fell back to strict embedding "
                "similarity, which is measurably less accurate."
            )

        name_to_skill = {s.name: s for s in have}
        for i in pending:
            shortlist = shortlists[i]
            if not shortlist:
                matches[i] = SkillMatch(required=required[i], verified=verified)
                continue

            payload = verdicts.get(required[i])
            if payload is None:
                # No verdict (model failed, or nothing shortlisted): fall back to
                # the strict cosine threshold.
                best_j, best_raw = shortlist[0]
                source = have[best_j]
                credit = 1.0 if best_raw >= FALLBACK_SIMILARITY_FLOOR else 0.0
                matches[i] = SkillMatch(
                    required=required[i],
                    matched=source.name if credit else "",
                    similarity=credit, verdict="EXACT" if credit else "NONE",
                    reason=f"unverified, cosine {best_raw:.2f}",
                    evidence=source.evidence if credit else "",
                    years=source.years if credit else None,
                    recency_year=source.last_used_year if credit else None,
                    verified=False,
                )
                continue

            verdict, best_match, reason = payload
            credit = VERDICT_SCORES.get(verdict, 0.0)
            source = name_to_skill.get(best_match)
            if source is None:
                # The model named a skill it was not offered — treat as no match
                # rather than trusting a fabricated pairing.
                source = have[shortlist[0][0]]
                if verdict != "NONE":
                    logger.debug(
                        "Verifier returned unlisted skill %r for %r", best_match, required[i]
                    )
                    credit = 0.0
                    verdict = "NONE"

            matched_name = source.name if credit > 0 else ""
            matches[i] = SkillMatch(
                required=required[i], matched=matched_name, similarity=credit,
                exact=False, verdict=verdict, reason=reason,
                evidence=source.evidence if credit > 0 else "",
                years=source.years if credit > 0 else None,
                recency_year=source.last_used_year if credit > 0 else None,
                verified=True,
            )

        result.matches = [m for m in matches if m]
        self._cache.flush()
        return result

    # ── stages ───────────────────────────────────────────────

    def _retrieve(
        self, req_norm: list[str], have_norm: list[str], pending: list[int]
    ) -> dict[int, list[tuple[int, float]]]:
        """Shortlist plausible resume skills per unresolved requirement."""
        shortlists: dict[int, list[tuple[int, float]]] = {i: [] for i in pending}
        try:
            vectors = self.provider.embed(req_norm + have_norm)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skill embedding failed: %s", exc)
            return shortlists

        req_vectors = vectors[: len(req_norm)]
        have_vectors = vectors[len(req_norm) :]
        similarity = req_vectors @ have_vectors.T  # rows are unit-normalised

        for i in pending:
            row = similarity[i]
            order = np.argsort(row)[::-1][:RETRIEVAL_TOP_K]
            shortlists[i] = [
                (int(j), float(row[j])) for j in order if row[j] >= RETRIEVAL_FLOOR
            ]
        return shortlists

    def _verify(
        self, shortlists: dict[str, list[str]]
    ) -> tuple[dict[str, tuple[str, str, str]], bool]:
        """Judge every shortlist in one batched call. Returns (verdicts, verified)."""
        if not shortlists:
            return {}, True

        verdicts: dict[str, tuple[str, str, str]] = {}
        uncached: dict[str, list[str]] = {}

        for requirement, candidates in shortlists.items():
            key = JSONCache.key("v2", requirement.lower(), "|".join(sorted(c.lower() for c in candidates)))
            hit = self._cache.get(key)
            if isinstance(hit, list) and len(hit) == 3:
                verdicts[requirement] = (hit[0], hit[1], hit[2])
            else:
                uncached[requirement] = candidates

        if not uncached:
            return verdicts, True

        blocks = "\n".join(
            f"Requirement: {requirement}\nCandidate skills: {', '.join(candidates)}"
            for requirement, candidates in uncached.items()
        )

        try:
            payload = self.provider.generate_json(
                VERIFY_PROMPT.format(blocks=blocks),
                schema=VERIFY_SCHEMA,
                system=VERIFY_SYSTEM,
                temperature=0.0,
            )
        except LLMError as exc:
            logger.error("Skill verification failed: %s", exc)
            return verdicts, False

        by_requirement = {r.lower(): r for r in uncached}
        for row in payload.get("verdicts") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("requirement") or "").strip().lower()
            requirement = by_requirement.get(name)
            if requirement is None:
                continue
            verdict = str(row.get("verdict") or "NONE").strip().upper()
            if verdict not in VERDICT_SCORES:
                verdict = "NONE"
            entry = (
                verdict,
                str(row.get("best_match") or "").strip(),
                str(row.get("reason") or "").strip()[:200],
            )
            verdicts[requirement] = entry
            key = JSONCache.key(
                "v2", requirement.lower(),
                "|".join(sorted(c.lower() for c in uncached[requirement])),
            )
            self._cache.put(key, list(entry))

        return verdicts, True


def match_skills(
    required: list[str],
    resume_skills: list[SkillMention],
    provider: LLMProvider,
    config: MatchingConfig | None = None,
) -> SkillMatchResult:
    """Convenience wrapper around :class:`SkillMatcher`."""
    return SkillMatcher(provider, config).match(required, resume_skills)
