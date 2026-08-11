"""
Deterministic extraction of contact handles from resume text.

These fields are *not* delegated to the LLM. A regex for an email address or a
github.com URL is exact, free, and instant, whereas a model will occasionally
hallucinate a plausible-looking handle — and a hallucinated GitHub username
means the enrichment step scores the wrong person's code, which is the worst
possible failure in a hiring tool.

Rule of thumb the codebase follows: regex for anything with a rigid grammar,
LLM only for genuine language understanding.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?:(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]?\d{3,4}[\s.\-]?\d{3,4})")

_GITHUB = re.compile(
    r"(?:github\.com/|github:\s*|git\s*hub:\s*)@?([A-Za-z0-9](?:[A-Za-z0-9\-]{0,37}[A-Za-z0-9])?)",
    re.IGNORECASE,
)
_LINKEDIN = re.compile(
    r"(?:linkedin\.com/(?:in|pub)/)([A-Za-z0-9\-_%]{3,100})", re.IGNORECASE
)

# github.com paths that are site features, not user profiles.
_GITHUB_RESERVED = {
    "about", "blog", "explore", "features", "pricing", "topics", "trending",
    "marketplace", "sponsors", "collections", "events", "login", "join",
    "settings", "notifications", "issues", "pulls", "search", "orgs", "apps",
    "readme", "gist", "raw", "www",
}


def find_emails(text: str) -> list[str]:
    seen, out = set(), []
    for match in _EMAIL.findall(text or ""):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            out.append(match)
    return out


def find_phone(text: str) -> str:
    for candidate in _PHONE.findall(text or ""):
        digits = re.sub(r"\D", "", candidate)
        # 7 digits filters out years, zip codes and "2019 - 2023" ranges.
        if 7 <= len(digits) <= 15:
            return candidate.strip()
    return ""


def find_github_username(text: str) -> str:
    """Return the first plausible GitHub username, or ``""``.

    Trailing path segments are stripped, so ``github.com/octocat/my-repo``
    yields ``octocat``.
    """
    for match in _GITHUB.finditer(text or ""):
        username = match.group(1).strip().strip("/").lower()
        if username and username not in _GITHUB_RESERVED and not username.endswith(".git"):
            return username
    return ""


def find_linkedin_url(text: str) -> str:
    match = _LINKEDIN.search(text or "")
    if not match:
        return ""
    return f"https://www.linkedin.com/in/{match.group(1).strip('/')}"


def extract_contacts(text: str) -> dict[str, str]:
    emails = find_emails(text)
    return {
        "email": emails[0] if emails else "",
        "phone": find_phone(text),
        "github_username": find_github_username(text),
        "linkedin_url": find_linkedin_url(text),
    }
