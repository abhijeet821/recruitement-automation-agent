"""
Job description text -> ``JobSpec``.

This is the other half of what makes scoring role-aware. The old system scored
every resume against one hardcoded list of nine engineering keywords, so a
Marketing Manager posting ranked candidates on whether they had used Docker.
Here the requirements come *from the posting*: must-haves, nice-to-haves,
seniority, years, and whether the role is technical at all — which is what
decides how much GitHub evidence should count.
"""

from __future__ import annotations

import logging
import re

from matching.llm import LLMError, LLMProvider
from matching.schemas import JobSpec

logger = logging.getLogger(__name__)

MAX_JD_CHARS = 12_000

JOBSPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "role_title": {"type": "string"},
        "seniority": {
            "type": "string",
            "enum": ["intern", "junior", "mid", "senior", "lead", "principal"],
        },
        "min_years": {"type": "number"},
        "max_years": {"type": "number"},
        "must_have_skills": {"type": "array", "items": {"type": "string"}},
        "nice_to_have_skills": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "domain": {"type": "string"},
        "education_requirement": {"type": "string"},
        "location": {"type": "string"},
        "employment_type": {"type": "string"},
        "is_technical": {"type": "boolean"},
    },
    "required": ["role_title", "must_have_skills", "min_years", "is_technical"],
}

SYSTEM_PROMPT = (
    "You convert job descriptions into structured hiring requirements. "
    "You extract only stated requirements and never invent them."
)

PROMPT_TEMPLATE = """Convert the job description below into structured requirements.

Rules:
- `must_have_skills`: concrete, checkable requirements (languages, frameworks,
  tools, methods). Split compound entries: "React/Redux" becomes two skills.
  Exclude soft skills and generic phrases like "team player" or "fast learner".
- `nice_to_have_skills`: anything described as preferred, bonus, a plus, or desirable.
- `min_years` / `max_years`: from the stated experience range. If only a single
  number is given, set `min_years` to it and omit `max_years`. If unstated, use 0.
- `is_technical`: true only when the job's core output is software, data,
  infrastructure or analysis — that is, when a public code portfolio would be
  meaningful evidence of ability. Set false for sales, marketing, HR, design,
  operations and similar roles.
- `responsibilities`: up to 8 short phrases.

JOB DESCRIPTION:
---
{jd_text}
---
"""

# Fallback only: used when the LLM is unreachable, so a campaign can still be
# created and scored (weakly) rather than blocking the recruiter entirely.
_TITLE_SENIORITY = [
    (r"\bprincipal\b|\bstaff\b|\bdistinguished\b", "principal", 8.0),
    (r"\blead\b|\bhead of\b|\bmanager\b", "lead", 6.0),
    (r"\bsenior\b|\bsr\.?\b|\bsde\s*3\b|\bl[45]\b", "senior", 5.0),
    (r"\bjunior\b|\bjr\.?\b|\bassociate\b|\bgraduate\b|\bentry\b", "junior", 0.0),
    (r"\bintern\b|\btrainee\b", "intern", 0.0),
]
_NON_TECHNICAL = re.compile(
    r"\b(sales|marketing|recruit|human resources|hr\b|account manager|customer success|"
    r"content writer|copywriter|social media|business development|operations manager)\b",
    re.IGNORECASE,
)
_YEARS = re.compile(r"(\d+)\s*(?:\+|plus)?\s*(?:-|to)?\s*(\d+)?\s*years?", re.IGNORECASE)


def parse_jobspec(
    jd_text: str,
    provider: LLMProvider,
    *,
    role_title: str = "",
    experience_hint: str = "",
) -> JobSpec:
    """Distil a JD into a ``JobSpec``. Never raises; degrades to heuristics."""
    jd_text = (jd_text or "").strip()
    if not jd_text:
        return _heuristic_spec("", role_title, experience_hint)

    prompt = PROMPT_TEMPLATE.format(jd_text=jd_text[:MAX_JD_CHARS])

    try:
        data = provider.generate_json(
            prompt, schema=JOBSPEC_SCHEMA, system=SYSTEM_PROMPT, temperature=0.0
        )
    except LLMError as exc:
        logger.error("JobSpec extraction failed, falling back to heuristics: %s", exc)
        return _heuristic_spec(jd_text, role_title, experience_hint)

    spec = JobSpec(
        role_title=_str(data.get("role_title")) or role_title,
        seniority=_str(data.get("seniority")),
        min_years=_num(data.get("min_years"), 0.0),
        max_years=_opt_num(data.get("max_years")),
        must_have_skills=_skills(data.get("must_have_skills")),
        nice_to_have_skills=_skills(data.get("nice_to_have_skills")),
        responsibilities=[_str(r) for r in _list(data.get("responsibilities")) if _str(r)][:8],
        domain=_str(data.get("domain")),
        education_requirement=_str(data.get("education_requirement")),
        location=_str(data.get("location")),
        employment_type=_str(data.get("employment_type")),
        is_technical=bool(data.get("is_technical", True)),
        raw_jd=jd_text,
    )

    # The recruiter typed the experience requirement into the form; trust that
    # over the model's reading of the prose it generated.
    hinted = _years_from_text(experience_hint)
    if hinted is not None:
        spec.min_years = hinted

    if not spec.seniority:
        spec.seniority = _seniority_from_years(spec.min_years)

    # A JD with no extractable must-haves would make skill coverage vacuous
    # (0/0 = perfect). Promote nice-to-haves rather than score against nothing.
    if not spec.must_have_skills and spec.nice_to_have_skills:
        spec.must_have_skills = spec.nice_to_have_skills[:5]

    return spec


# ── Heuristic fallback ───────────────────────────────────────

def _heuristic_spec(jd_text: str, role_title: str, experience_hint: str) -> JobSpec:
    blob = f"{role_title} {jd_text}"
    seniority, implied_years = "mid", 2.0
    for pattern, level, years in _TITLE_SENIORITY:
        if re.search(pattern, blob, re.IGNORECASE):
            seniority, implied_years = level, years
            break

    min_years = _years_from_text(experience_hint)
    if min_years is None:
        min_years = _years_from_text(jd_text)
    if min_years is None:
        min_years = implied_years

    return JobSpec(
        role_title=role_title or "Unspecified role",
        seniority=seniority,
        min_years=min_years,
        is_technical=not bool(_NON_TECHNICAL.search(blob)),
        raw_jd=jd_text,
    )


def _years_from_text(text: str) -> float | None:
    match = _YEARS.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _seniority_from_years(years: float) -> str:
    if years >= 8:
        return "principal"
    if years >= 6:
        return "lead"
    if years >= 4:
        return "senior"
    if years >= 2:
        return "mid"
    return "junior"


def _skills(value) -> list[str]:
    seen, out = set(), []
    for item in _list(value):
        name = _str(item)
        if not name:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out[:25]


def _str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _num(value, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _opt_num(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _list(value) -> list:
    return value if isinstance(value, list) else []
