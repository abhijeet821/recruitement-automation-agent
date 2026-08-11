"""
Resume text -> ``ResumeProfile``.

This is the step that replaces substring matching. Instead of asking "does the
string 'python' appear", we ask the model to build a record: which skills, with
how many years, last used when, evidenced professionally or merely listed. Every
downstream feature is computed from that record, which is why the engine can
tell a four-year Python engineer from someone who put Python on a skills line.

The extraction is schema-constrained, and the fields with a rigid grammar
(email, GitHub handle) are overridden by regex afterwards — the model is used
only where language understanding is genuinely needed.
"""

from __future__ import annotations

import logging
from datetime import date

from matching.llm import LLMError, LLMProvider
from matching.parsing.contacts import extract_contacts
from matching.schemas import Education, ResumeProfile, SkillMention, WorkExperience

logger = logging.getLogger(__name__)

# Resumes beyond this are truncated to fit the model context. 18k chars is
# roughly 5-6k tokens, comfortably inside the 8k window configured for Ollama,
# and longer than 99% of real CVs.
MAX_RESUME_CHARS = 18_000

RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "location": {"type": "string"},
        "github_username": {"type": "string"},
        "linkedin_url": {"type": "string"},
        "portfolio_url": {"type": "string"},
        "summary": {"type": "string"},
        "total_years_experience": {"type": "number"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "years": {"type": "number"},
                    "last_used_year": {"type": "integer"},
                    "evidence": {
                        "type": "string",
                        "enum": ["professional", "project", "self-reported"],
                    },
                    "context": {"type": "string"},
                },
                "required": ["name", "evidence"],
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                    "is_current": {"type": "boolean"},
                    "description": {"type": "string"},
                    "technologies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "company"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field_of_study": {"type": "string"},
                    "institution": {"type": "string"},
                    "graduation_year": {"type": "integer"},
                },
                "required": ["degree"],
            },
        },
        "certifications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["full_name", "skills", "experience"],
}

SYSTEM_PROMPT = (
    "You are a precise resume parser. You extract only what the document states. "
    "You never invent skills, employers, dates or credentials. When a value is "
    "absent, you leave the field empty rather than guessing."
)

PROMPT_TEMPLATE = """Extract structured data from the resume below.

Rules:
- Copy only what is present. Do not infer skills from job titles.
- For each skill set `evidence`:
    "professional"  = used in a paid role listed in the work history
    "project"       = used in a personal, academic or open-source project
    "self-reported" = only named in a skills list, with no supporting context
- `years` for a skill: how long the resume shows it in use. Omit if unclear.
- `last_used_year`: the most recent year the resume shows it in use. Current year is {current_year}.
- `total_years_experience`: total professional experience, excluding internships
  and study. Count overlapping roles once.
- Use 4-digit years. For an ongoing role set `is_current` true and omit `end_year`.

RESUME:
---
{resume_text}
---
"""


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_RESUME_CHARS:
        return text, False
    # Keep the head (identity, summary, recent roles) and the tail (education,
    # skills lists) — the middle of a long CV is the least decision-relevant.
    head = text[: int(MAX_RESUME_CHARS * 0.7)]
    tail = text[-int(MAX_RESUME_CHARS * 0.3) :]
    return f"{head}\n\n[... middle of resume omitted ...]\n\n{tail}", True


def parse_resume(
    text: str,
    provider: LLMProvider,
    *,
    fallback_email: str = "",
) -> ResumeProfile:
    """Build a ``ResumeProfile`` from resume text.

    Never raises. On LLM failure it returns a degraded profile flagged with
    ``extraction_failed=True`` — the pipeline then reports low confidence rather
    than a misleadingly low score.
    """
    text = (text or "").strip()
    contacts = extract_contacts(text)

    if not text:
        return ResumeProfile(
            email=fallback_email or contacts["email"],
            raw_text="",
            extraction_failed=True,
        )

    prompt_text, truncated = _truncate(text)
    prompt = PROMPT_TEMPLATE.format(
        resume_text=prompt_text, current_year=date.today().year
    )

    try:
        data = provider.generate_json(
            prompt, schema=RESUME_SCHEMA, system=SYSTEM_PROMPT, temperature=0.0
        )
    except LLMError as exc:
        logger.error("Resume extraction failed: %s", exc)
        profile = ResumeProfile(raw_text=text, extraction_failed=True)
        profile.email = fallback_email or contacts["email"]
        profile.github_username = contacts["github_username"]
        profile.linkedin_url = contacts["linkedin_url"]
        return profile

    profile = _profile_from_payload(data, text)

    # Regex wins for rigid-grammar fields — see parsing/contacts.py.
    profile.email = contacts["email"] or profile.email or fallback_email
    profile.phone = contacts["phone"] or profile.phone
    profile.github_username = (
        contacts["github_username"] or _clean_github(profile.github_username)
    )
    profile.linkedin_url = contacts["linkedin_url"] or profile.linkedin_url

    if truncated:
        logger.info("Resume truncated to %d chars for extraction", MAX_RESUME_CHARS)

    return profile


def _profile_from_payload(data: dict, raw_text: str) -> ResumeProfile:
    """Convert the model payload into a profile, dropping malformed entries.

    A single bad list item must not lose the whole extraction, so each nested
    record is converted defensively and skipped on failure.
    """
    profile = ResumeProfile(
        full_name=_str(data.get("full_name")),
        email=_str(data.get("email")),
        phone=_str(data.get("phone")),
        location=_str(data.get("location")),
        github_username=_str(data.get("github_username")),
        linkedin_url=_str(data.get("linkedin_url")),
        portfolio_url=_str(data.get("portfolio_url")),
        summary=_str(data.get("summary")),
        total_years_experience=_num(data.get("total_years_experience")),
        certifications=[_str(c) for c in _list(data.get("certifications")) if _str(c)],
        raw_text=raw_text,
    )

    for item in _list(data.get("skills")):
        if not isinstance(item, dict):
            # Some models emit a bare list of strings despite the schema.
            if isinstance(item, str) and item.strip():
                profile.skills.append(SkillMention(name=item.strip()))
            continue
        name = _str(item.get("name"))
        if not name:
            continue
        profile.skills.append(
            SkillMention(
                name=name,
                years=_num(item.get("years")),
                last_used_year=_int(item.get("last_used_year")),
                evidence=_str(item.get("evidence")) or "self-reported",
                context=_str(item.get("context")),
            )
        )

    for item in _list(data.get("experience")):
        if not isinstance(item, dict):
            continue
        profile.experience.append(
            WorkExperience(
                title=_str(item.get("title")),
                company=_str(item.get("company")),
                start_year=_int(item.get("start_year")),
                end_year=_int(item.get("end_year")),
                is_current=bool(item.get("is_current")),
                description=_str(item.get("description")),
                technologies=[_str(t) for t in _list(item.get("technologies")) if _str(t)],
            )
        )

    for item in _list(data.get("education")):
        if not isinstance(item, dict):
            continue
        profile.education.append(
            Education(
                degree=_str(item.get("degree")),
                field_of_study=_str(item.get("field_of_study")),
                institution=_str(item.get("institution")),
                graduation_year=_int(item.get("graduation_year")),
            )
        )

    return profile


def _clean_github(value: str) -> str:
    value = (value or "").strip().strip("/@")
    if "github.com/" in value:
        value = value.split("github.com/", 1)[1]
    return value.split("/")[0].strip().lower()


def _str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _num(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _list(value) -> list:
    return value if isinstance(value, list) else []
