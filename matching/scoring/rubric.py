"""
LLM-as-judge: a structured rubric assessment of fit.

The feature vector captures what is countable. It cannot read a bullet point and
recognise that "reduced p99 latency from 800ms to 120ms by rewriting the query
planner" is stronger evidence of engineering ability than "worked on backend
performance". That judgement is what this pass adds.

Three design choices keep it honest:

1. **Redacted input.** The judge sees the resume only after
   ``fairness.redaction`` has removed name, contact details and gendered terms.
   The model cannot condition on an identity it was never shown.
2. **Constrained output.** A JSON schema with fixed dimensions and a 0-10 range.
   Free-form judgement invites drift; a rubric makes assessments comparable
   between candidates.
3. **Mandatory evidence.** Every dimension must cite a quote from the resume.
   This is the cheapest available check against confabulation — a claim with no
   supporting quote is visible as such to the recruiter reviewing it.

The rubric is one weighted input to the final score, never the whole of it. A
single model call is too unstable to hand a hiring decision to.
"""

from __future__ import annotations

import logging

from matching.llm import LLMError, LLMProvider
from matching.schemas import GitHubProfile, JobSpec, ResumeProfile

logger = logging.getLogger(__name__)

MAX_RESUME_CHARS = 12_000

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "technical_depth": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "rationale": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "rationale", "evidence"],
        },
        "relevant_experience": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "rationale": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "rationale", "evidence"],
        },
        "impact_evidence": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "rationale": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "rationale", "evidence"],
        },
        "communication": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "rationale": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score", "rationale", "evidence"],
        },
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "technical_depth", "relevant_experience", "impact_evidence",
        "communication", "summary", "strengths", "gaps",
    ],
}

SYSTEM_PROMPT = (
    "You are an experienced technical recruiter assessing a candidate against a "
    "specific role. You judge only on evidence present in the document. You never "
    "speculate about a candidate's background, identity or personal circumstances. "
    "Identifying details have been redacted; the redaction is not a defect in the "
    "resume and must not affect your assessment."
)

PROMPT_TEMPLATE = """Assess this candidate against the role.

ROLE
Title: {role_title}
Seniority: {seniority}
Experience required: {min_years} years
Must have: {must_have}
Nice to have: {nice_to_have}
Responsibilities: {responsibilities}

{github_block}
CANDIDATE RESUME (identifying details redacted)
---
{resume_text}
---

Score each dimension from 0 to 10:

- technical_depth — Does the work described show real command of the required
  skills, or only exposure to them? Specific systems, scale and trade-offs score
  high; lists of technologies score low.
- relevant_experience — How closely does the actual work match this role's
  responsibilities? Judge the work, not the job title.
- impact_evidence — Are outcomes stated with specifics (metrics, scale, results),
  or only duties? "Reduced latency 40%" outranks "responsible for performance".
- communication — Is the resume clear, concrete and well organised? Judge the
  writing, not the formatting or the length.

Rules:
- Every dimension needs 1-3 short verbatim quotes from the resume in `evidence`.
  If you cannot find a supporting quote, the score must be 3 or lower.
- `gaps` must name required skills you could not find evidence for.
- `concerns` is only for things visible in the document that are relevant to
  performing this job. Never speculate about age, gender, nationality, health,
  family, or where a person studied.
"""


def _github_block(github: GitHubProfile | None, job: JobSpec) -> str:
    if not job.is_technical or not github or not github.found:
        return ""
    languages = ", ".join(
        f"{lang} {share:.0%}"
        for lang, share in sorted(
            github.language_share().items(), key=lambda kv: kv[1], reverse=True
        )[:4]
    )
    repos = "; ".join(
        f"{r.name} ({r.language or 'n/a'}, {r.stars}★)" for r in github.top_repos[:4]
    )
    return (
        "PUBLIC CODE (GitHub)\n"
        f"Original repositories: {github.original_repos}, total stars: {github.total_stars}\n"
        f"Languages by volume: {languages or 'unknown'}\n"
        f"Notable repositories: {repos or 'none'}\n"
        f"Last public commit: {github.days_since_last_push} days ago\n"
        "Treat this as supporting evidence for technical_depth only. "
        "A thin public profile is not evidence of weakness — many strong "
        "engineers work exclusively on private code.\n\n"
    )


class RubricResult:
    """Parsed rubric output, normalised to [0, 1] per dimension."""

    DIMENSIONS = ("technical_depth", "relevant_experience", "impact_evidence", "communication")
    LABELS = {
        "technical_depth": "Technical depth",
        "relevant_experience": "Relevant experience",
        "impact_evidence": "Evidence of impact",
        "communication": "Communication",
    }

    def __init__(self, payload: dict | None = None, error: str = ""):
        self.error = error
        payload = payload or {}
        self.scores: dict[str, float] = {}
        self.rationales: dict[str, str] = {}
        self.evidence: dict[str, list[str]] = {}

        for key in self.DIMENSIONS:
            node = payload.get(key) or {}
            if not isinstance(node, dict):
                node = {}
            raw = node.get("score", 0)
            value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.0
            self.scores[key] = max(0.0, min(1.0, value / 10.0))
            self.rationales[key] = str(node.get("rationale") or "").strip()
            self.evidence[key] = [
                str(e).strip() for e in (node.get("evidence") or []) if str(e).strip()
            ][:3]

        self.summary = str(payload.get("summary") or "").strip()
        self.strengths = [str(s).strip() for s in (payload.get("strengths") or []) if str(s).strip()][:5]
        self.gaps = [str(g).strip() for g in (payload.get("gaps") or []) if str(g).strip()][:5]
        self.concerns = [str(c).strip() for c in (payload.get("concerns") or []) if str(c).strip()][:5]

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def overall(self) -> float:
        """Weighted mean of the four dimensions, in [0, 1].

        Technical depth and relevant experience dominate; communication is real
        signal but a smaller share, since resume prose is often ghost-written or
        template-driven and is a weak proxy for on-the-job communication.
        """
        weights = {
            "technical_depth": 0.35,
            "relevant_experience": 0.35,
            "impact_evidence": 0.20,
            "communication": 0.10,
        }
        return sum(self.scores[k] * w for k, w in weights.items())


def assess(
    resume: ResumeProfile,
    job: JobSpec,
    provider: LLMProvider,
    github: GitHubProfile | None = None,
    *,
    temperature: float = 0.1,
) -> RubricResult:
    """Run the rubric. Never raises — failure returns a result with ``error`` set."""
    text = (resume.raw_text or resume.summary or "").strip()
    if not text:
        return RubricResult(error="no resume text to assess")

    prompt = PROMPT_TEMPLATE.format(
        role_title=job.role_title or "Unspecified",
        seniority=job.seniority or "unspecified",
        min_years=f"{job.min_years:g}",
        must_have=", ".join(job.must_have_skills) or "not specified",
        nice_to_have=", ".join(job.nice_to_have_skills) or "not specified",
        responsibilities="; ".join(job.responsibilities) or "not specified",
        github_block=_github_block(github, job),
        resume_text=text[:MAX_RESUME_CHARS],
    )

    try:
        payload = provider.generate_json(
            prompt, schema=RUBRIC_SCHEMA, system=SYSTEM_PROMPT, temperature=temperature
        )
    except LLMError as exc:
        logger.error("Rubric assessment failed: %s", exc)
        return RubricResult(error=str(exc))

    return RubricResult(payload)
