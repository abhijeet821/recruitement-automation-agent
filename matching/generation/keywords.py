"""
The recruiter keyword taxonomy used to draft and grade job descriptions.

Three separate concerns live here, and they are separate on purpose:

1. **Structure** — the sections recruiters expect and candidates look for. A JD
   missing a compensation range or a work model gets fewer and worse-targeted
   applications, regardless of how well the rest is written.

2. **Attraction language** — the terms that make a posting appealing without
   making it vague. Used as prompt guidance, and measured afterwards.

3. **Coded language** — this is the evidence-backed part. Gaucher, Friesen and
   Kay (2011) showed that job adverts using masculine-coded wording
   ("aggressive", "dominant", "ninja") measurably reduce the number of women who
   apply, without changing the role itself. Age-coded phrasing ("young",
   "digital native", "recent graduate") and ableist idioms have similar
   filtering effects. These lists let the system *measure* that property of a
   generated JD instead of asserting it is fine.

The lists are intentionally conservative. Flagging a word is advisory — the
recruiter is shown the finding and decides — not an automatic rewrite.
"""

from __future__ import annotations

# ── Structural sections a strong JD contains ─────────────────
# (canonical name -> phrases that indicate the section is present)
REQUIRED_SECTIONS: dict[str, list[str]] = {
    "About the role": ["about the role", "the role", "role overview", "about this position", "the opportunity"],
    "Responsibilities": ["responsibilities", "what you'll do", "what you will do", "your impact", "day to day"],
    "Requirements": ["requirements", "must have", "must-have", "what we're looking for", "qualifications", "you have"],
    "Nice to have": ["nice to have", "nice-to-have", "bonus", "preferred", "a plus", "desirable"],
    "What we offer": ["what we offer", "benefits", "perks", "we offer", "compensation", "salary"],
    "Work model": ["remote", "hybrid", "on-site", "onsite", "in office", "location"],
    "How to apply": ["how to apply", "apply", "application", "next steps", "hiring process"],
    "Equal opportunity": [
        "equal opportunity", "regardless of", "diverse", "inclusive",
        "we welcome", "all backgrounds", "encourage applications",
    ],
}

# ── Attraction terms: appealing and concrete ─────────────────
ATTRACTION_TERMS: list[str] = [
    "ownership", "impact", "growth", "mentorship", "learning budget",
    "flexible hours", "remote", "career progression", "autonomy",
    "collaborative", "professional development", "well-defined",
]

# ── Coded language (advisory flags) ──────────────────────────
# Gaucher, Friesen & Kay (2011), "Evidence That Gendered Wording in Job
# Advertisements Exists and Sustains Gender Inequality" — representative subset.
MASCULINE_CODED: list[str] = [
    "aggressive", "ambitious", "assertive", "autonomous", "battle", "boast",
    "challenging", "competitive", "confident", "decisive", "determined",
    "dominant", "driven", "fearless", "force", "independent", "individual",
    "intellectual", "lead", "objective", "outspoken", "principle", "relentless",
    "self-reliant", "superior", "ninja", "rockstar", "rock star", "guru",
    "wizard", "hacker", "hero", "warrior", "crush", "dominate", "killer",
]

FEMININE_CODED: list[str] = [
    "affectionate", "collaborate", "collaborative", "committed", "compassion",
    "connect", "considerate", "cooperative", "depend", "empathy", "loyal",
    "nurture", "pleasant", "polite", "responsible", "supportive", "sympathetic",
    "together", "trust", "understanding", "warm", "interpersonal",
]

AGE_CODED: list[str] = [
    "young", "youthful", "energetic", "digital native", "recent graduate",
    "fresh graduate", "new grad", "student mentality", "mature",
    "high energy", "fast-paced young",
]

ABLEIST_TERMS: list[str] = [
    "walk", "stand for long", "able-bodied", "hear clearly", "see clearly",
    "physically fit", "crazy", "insane", "lame", "blind to", "tone deaf",
]

EXCLUSIONARY_TERMS: list[str] = [
    "native speaker", "native english", "no career gaps", "must be local",
    "cultural fit", "work hard play hard", "family-like", "wear many hats",
    "unlimited hours", "willing to sacrifice",
]

# Why each flagged category matters, shown to the recruiter alongside the flag.
FLAG_EXPLANATIONS: dict[str, str] = {
    "masculine_coded": (
        "Masculine-coded wording is associated with fewer applications from "
        "women (Gaucher et al., 2011). Consider neutral alternatives."
    ),
    "age_coded": (
        "Age-coded wording discourages older applicants and is legally risky in "
        "many jurisdictions. Describe the work, not the person's age."
    ),
    "ableist": (
        "Physical or idiomatic ableist language excludes disabled applicants. "
        "State only genuine, essential requirements of the job."
    ),
    "exclusionary": (
        "This phrasing narrows the applicant pool on grounds unrelated to "
        "ability, or signals poor working conditions."
    ),
}

NEUTRAL_ALTERNATIVES: dict[str, str] = {
    "ninja": "specialist", "rockstar": "high-performing engineer",
    "rock star": "high-performing engineer", "guru": "expert",
    "wizard": "expert", "hero": "key contributor", "warrior": "advocate",
    "aggressive": "proactive", "dominant": "leading", "fearless": "willing to take initiative",
    "crush": "exceed", "dominate": "lead", "killer": "excellent",
    "young": "early-career", "energetic": "motivated",
    "digital native": "comfortable with modern tooling",
    "cultural fit": "alignment with how we work",
    "work hard play hard": "a balanced, sustainable pace",
    "native speaker": "fluent",
    "crazy": "demanding", "insane": "substantial",
}


def role_keywords(role_title: str, must_have: list[str], nice_to_have: list[str]) -> list[str]:
    """The concrete terms a JD should contain so it is findable and specific.

    These are the searchable keywords — job boards and ATS keyword search both
    match on them, and candidates scan for them to decide whether to read on.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for value in [role_title, *must_have, *nice_to_have]:
        cleaned = (value or "").strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            terms.append(cleaned)
    return terms
