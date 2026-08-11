"""
Blind screening: strip identity signals before the LLM judges a resume.

Language models carry the biases of their training data. Given a name, a
university, or a gendered pronoun, a model's assessment of identical
qualifications can shift. The mitigation used here is the one that decades of
orchestra audition research supports: remove the identity signal from the
evidence before the judgement is made.

What is removed:
    name, email, phone, street address, personal URLs, photos-by-reference,
    gendered titles and pronouns, marital/nationality/religion markers,
    and (optionally) institution names.

What is deliberately kept:
    dates, years of experience, skills, employers, job titles, achievements —
    everything that is actually job-relevant. Redaction that destroys signal is
    not fairness, it is just noise.

Note the honest limit: this reduces the *direct* cue, not every proxy. A resume
can still leak identity through a fraternity, a language, a location. The
adverse-impact audit in ``fairness/audit.py`` exists precisely because
redaction alone is not a proof of fairness.
"""

from __future__ import annotations

import re

from matching.schemas import ResumeProfile

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}")
_URL_PERSONAL = re.compile(
    r"https?://(?:www\.)?(?:linkedin\.com|facebook\.com|twitter\.com|x\.com|instagram\.com)/\S+",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\s]{3,30}\b(?:street|st\.|road|rd\.|avenue|ave\.|lane|ln\.|"
    r"boulevard|blvd\.|drive|dr\.|apartment|apt\.|flat|sector|nagar|colony)\b[^\n,]*",
    re.IGNORECASE,
)
_POSTCODE = re.compile(r"\b\d{5,6}(?:-\d{4})?\b")

_GENDERED = re.compile(
    r"\b(he|him|his|she|her|hers|mr\.?|mrs\.?|ms\.?|miss|sir|madam|male|female|"
    r"husband|wife|father|mother|son|daughter)\b",
    re.IGNORECASE,
)
_PROTECTED = re.compile(
    r"^.*\b(date of birth|d\.?o\.?b\.?|age\s*[:\-]|marital status|nationality|"
    r"religion|caste|gender|sex\s*[:\-]|citizenship|visa status|passport)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_PHOTO = re.compile(r"^.*\b(photograph|photo attached|passport size|headshot)\b.*$",
                    re.IGNORECASE | re.MULTILINE)

_INSTITUTION = re.compile(
    r"\b(?:[A-Z][A-Za-z&.\-]*\s+){0,4}"
    r"(?:University|Institute of Technology|College|Polytechnic|Universidad|Universität)"
    r"(?:\s+of\s+[A-Z][A-Za-z]*)?",
)

REDACTED = "[REDACTED]"


def redact_text(
    text: str,
    *,
    name: str = "",
    email: str = "",
    phone: str = "",
    redact_institutions: bool = False,
) -> str:
    """Return ``text`` with identity signals replaced by ``[REDACTED]``."""
    if not text:
        return ""

    out = text

    # Known values first — they are exact and cannot be over-matched.
    for value in (email, phone):
        if value and len(value.strip()) > 3:
            out = out.replace(value.strip(), REDACTED)

    if name and len(name.strip()) > 2:
        out = _redact_name(out, name)

    out = _EMAIL.sub(REDACTED, out)
    out = _URL_PERSONAL.sub(REDACTED, out)
    out = _ADDRESS.sub(REDACTED, out)
    out = _PROTECTED.sub(REDACTED, out)
    out = _PHOTO.sub(REDACTED, out)
    out = _PHONE.sub(_redact_if_phone, out)
    out = _POSTCODE.sub(REDACTED, out)
    out = _GENDERED.sub(REDACTED, out)

    if redact_institutions:
        out = _INSTITUTION.sub("[INSTITUTION]", out)

    return re.sub(r"(?:\[REDACTED\]\s*){2,}", f"{REDACTED} ", out).strip()


def _redact_name(text: str, name: str) -> str:
    """Redact the full name and each of its parts as standalone words."""
    parts = [p for p in re.split(r"\s+", name.strip()) if len(p) > 2]
    if not parts:
        return text
    # Longest first, so "John Smith" is replaced before "John".
    for token in sorted([name.strip(), *parts], key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(token)}\b", REDACTED, text, flags=re.IGNORECASE)
    return text


def _redact_if_phone(match: re.Match) -> str:
    """Only redact digit runs long enough to be a phone number.

    Guards against destroying "2019 - 2023" or "improved throughput by 40%",
    which carry real signal.
    """
    digits = re.sub(r"\D", "", match.group(0))
    return REDACTED if 7 <= len(digits) <= 15 else match.group(0)


def redact_profile(profile: ResumeProfile, *, redact_institutions: bool = False) -> ResumeProfile:
    """Return a copy of ``profile`` safe to send to an LLM judge.

    Skills, experience, education level and dates survive intact; only the
    identity fields are cleared.
    """
    clone = ResumeProfile.from_dict(profile.to_dict())

    clone.raw_text = redact_text(
        profile.raw_text,
        name=profile.full_name,
        email=profile.email,
        phone=profile.phone,
        redact_institutions=redact_institutions,
    )
    clone.summary = redact_text(
        profile.summary, name=profile.full_name, email=profile.email, phone=profile.phone
    )
    for role, source in zip(clone.experience, profile.experience, strict=True):
        role.description = redact_text(source.description, name=profile.full_name)

    clone.full_name = REDACTED
    clone.email = ""
    clone.phone = ""
    clone.location = ""
    clone.linkedin_url = ""
    clone.portfolio_url = ""

    if redact_institutions:
        for entry in clone.education:
            entry.institution = "[INSTITUTION]"

    return clone
