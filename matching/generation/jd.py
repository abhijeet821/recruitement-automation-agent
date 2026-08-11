"""
Job description drafting and grading.

The old version sent a one-line prompt to the model and printed whatever came
back. Two things are added here:

* **Keyword-directed drafting.** The prompt carries the recruiter taxonomy from
  ``keywords.py`` — the sections the JD must contain, the concrete skill terms
  that make it searchable, and an explicit instruction to avoid coded language.

* **A measured quality report.** After generation the draft is graded on section
  coverage, keyword coverage, inclusive-language flags and readability. This
  turns "the AI wrote a JD" into "the JD scores 82/100 and is missing a
  compensation range", which is a thing a recruiter can act on.

The grader is pure string analysis — no model call — so it is instant,
deterministic, and equally applicable to a JD the recruiter typed by hand.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from matching.generation.keywords import (
    ABLEIST_TERMS,
    AGE_CODED,
    ATTRACTION_TERMS,
    EXCLUSIONARY_TERMS,
    FEMININE_CODED,
    FLAG_EXPLANATIONS,
    MASCULINE_CODED,
    NEUTRAL_ALTERNATIVES,
    REQUIRED_SECTIONS,
    role_keywords,
)
from matching.llm import LLMError, LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an experienced technical recruiter who writes job descriptions that "
    "attract strong, diverse applicant pools. You write concretely and never use "
    "clichés, hype, or coded language."
)

PROMPT_TEMPLATE = """Write a job description for this role.

Role: {role_title}
Experience required: {experience}
{skills_block}{extras_block}
Include every one of these sections, using these exact headings:
{sections}

Requirements for the writing:
- 350-500 words. Use bullet points inside sections.
- Name specific technologies, systems and scale. Avoid vague phrases like
  "fast-paced environment" or "wear many hats".
- Under Responsibilities, describe outcomes the person owns, not a task list.
- Under Requirements, list only genuinely required things. Anything optional
  belongs under "Nice to have" — long required lists deter strong applicants who
  do not match every line.
- Use "you" to address the candidate. Never use "he", "she", "he/she", or
  any gendered term.
- Never use: ninja, rockstar, guru, wizard, hero, aggressive, dominant,
  young, energetic, digital native, work hard play hard, cultural fit.
- Include a brief, genuine equal-opportunity statement.
- Write plain prose. No preamble, no closing commentary, no markdown fences.

Output only the job description.
"""


@dataclass
class JDQualityReport:
    """A gradeable assessment of a job description."""

    score: float = 0.0                       # 0..100
    word_count: int = 0
    sections_present: list[str] = field(default_factory=list)
    sections_missing: list[str] = field(default_factory=list)
    keywords_present: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    attraction_terms: list[str] = field(default_factory=list)
    masculine_coded: list[str] = field(default_factory=list)
    feminine_coded: list[str] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    avg_sentence_words: float = 0.0

    @property
    def gender_balance(self) -> str:
        """Which way the wording leans, in plain language.

        Reported as a direction rather than a number because the underlying
        research is about relative balance, not a precise quantity.
        """
        masculine, feminine = len(self.masculine_coded), len(self.feminine_coded)
        if masculine == feminine:
            return "neutral"
        if masculine > feminine + 2:
            return "strongly masculine-coded"
        if masculine > feminine:
            return "slightly masculine-coded"
        if feminine > masculine + 2:
            return "strongly feminine-coded"
        return "slightly feminine-coded"

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "word_count": self.word_count,
            "sections_present": self.sections_present,
            "sections_missing": self.sections_missing,
            "keywords_present": self.keywords_present,
            "keywords_missing": self.keywords_missing,
            "attraction_terms": self.attraction_terms,
            "masculine_coded": self.masculine_coded,
            "feminine_coded": self.feminine_coded,
            "gender_balance": self.gender_balance,
            "flags": self.flags,
            "suggestions": self.suggestions,
            "avg_sentence_words": round(self.avg_sentence_words, 1),
        }


def generate_jd(
    role_title: str,
    experience: str,
    provider: LLMProvider,
    *,
    must_have: list[str] | None = None,
    nice_to_have: list[str] | None = None,
    company: str = "",
    location: str = "",
    work_model: str = "",
    salary_range: str = "",
) -> str:
    """Draft a JD. On model failure, returns a usable structured template.

    The fallback is a real skeleton rather than an error string, so a recruiter
    is never blocked from launching a campaign by an unavailable model.
    """
    must_have = must_have or []
    nice_to_have = nice_to_have or []

    skills_block = ""
    if must_have:
        skills_block += f"Required skills to feature: {', '.join(must_have)}\n"
    if nice_to_have:
        skills_block += f"Preferred skills to feature: {', '.join(nice_to_have)}\n"

    extras = []
    if company:
        extras.append(f"Company: {company}")
    if location:
        extras.append(f"Location: {location}")
    if work_model:
        extras.append(f"Work model: {work_model}")
    if salary_range:
        extras.append(f"Compensation range to state explicitly: {salary_range}")
    extras_block = ("\n".join(extras) + "\n") if extras else ""

    prompt = PROMPT_TEMPLATE.format(
        role_title=role_title or "the role",
        experience=experience or "not specified",
        skills_block=skills_block,
        extras_block=extras_block,
        sections="\n".join(f"- {name}" for name in REQUIRED_SECTIONS),
    )

    try:
        text = provider.generate(prompt, system=SYSTEM_PROMPT, temperature=0.6)
    except LLMError as exc:
        logger.error("JD generation failed: %s", exc)
        return _fallback_jd(role_title, experience, must_have, nice_to_have, location, salary_range)

    text = _strip_wrapper(text)
    if len(text.split()) < 80:
        logger.warning("JD generation returned only %d words; using template", len(text.split()))
        return _fallback_jd(role_title, experience, must_have, nice_to_have, location, salary_range)
    return text


def analyse_jd(
    jd_text: str,
    *,
    role_title: str = "",
    must_have: list[str] | None = None,
    nice_to_have: list[str] | None = None,
) -> JDQualityReport:
    """Grade a job description. Pure text analysis — no model call."""
    report = JDQualityReport()
    text = (jd_text or "").strip()
    if not text:
        report.suggestions.append("The job description is empty.")
        return report

    lowered = text.lower()
    words = re.findall(r"\b[\w'-]+\b", text)
    report.word_count = len(words)

    # ── sections ─────────────────────────────────────────────
    for section, markers in REQUIRED_SECTIONS.items():
        if any(marker in lowered for marker in markers):
            report.sections_present.append(section)
        else:
            report.sections_missing.append(section)

    # ── searchable keywords ──────────────────────────────────
    wanted = role_keywords(role_title, must_have or [], nice_to_have or [])
    for term in wanted:
        if _mentions(lowered, term):
            report.keywords_present.append(term)
        else:
            report.keywords_missing.append(term)

    # ── attraction & coded language ──────────────────────────
    report.attraction_terms = [t for t in ATTRACTION_TERMS if t in lowered]
    report.masculine_coded = _find_terms(lowered, MASCULINE_CODED)
    report.feminine_coded = _find_terms(lowered, FEMININE_CODED)

    _flag(report, "masculine_coded", report.masculine_coded)
    _flag(report, "age_coded", _find_terms(lowered, AGE_CODED))
    _flag(report, "ableist", _find_terms(lowered, ABLEIST_TERMS))
    _flag(report, "exclusionary", _find_terms(lowered, EXCLUSIONARY_TERMS))

    # ── readability ──────────────────────────────────────────
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    report.avg_sentence_words = (len(words) / len(sentences)) if sentences else 0.0

    report.score = _score(report, wanted)
    report.suggestions = _suggestions(report)
    return report


# ── internals ────────────────────────────────────────────────

# Words carrying no search value, ignored when matching a multi-word skill.
_FILLER = {"and", "or", "the", "a", "an", "of", "in", "with", "for", "design",
           "development", "experience", "knowledge", "skills"}


def _mentions(lowered: str, term: str) -> bool:
    """Is this skill mentioned, allowing for reasonable rewording?

    An exact-substring test produces false alarms that erode trust in the whole
    report: a JD saying "REST APIs" would be reported as missing "REST API
    design", and a title like "Backend Engineer (Python)" never matches
    verbatim. A term counts as present when its meaningful tokens all appear.
    """
    term = (term or "").strip().lower()
    if not term:
        return True
    if term in lowered:
        return True

    # Drop parentheticals: "Backend Engineer (Python)" -> "backend engineer".
    stripped = re.sub(r"\([^)]*\)", " ", term).strip()
    if stripped and stripped != term and stripped in lowered:
        return True

    tokens = [t for t in re.findall(r"[a-z0-9+#.]+", stripped or term)
              if len(t) > 1 and t not in _FILLER]
    if not tokens:
        return True
    # Singular/plural tolerance: "api" should match "apis".
    return all(
        re.search(rf"\b{re.escape(token)}s?\b", lowered) for token in tokens
    )


def _find_terms(lowered: str, terms: list[str]) -> list[str]:
    """Whole-word matching, so "lead" does not fire inside "leadership"."""
    found = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found.append(term)
    return found


def _flag(report: JDQualityReport, category: str, terms: list[str]) -> None:
    if not terms:
        return
    report.flags.append({
        "category": category,
        "terms": terms,
        "explanation": FLAG_EXPLANATIONS.get(category, ""),
        "alternatives": {
            t: NEUTRAL_ALTERNATIVES[t] for t in terms if t in NEUTRAL_ALTERNATIVES
        },
    })


def _score(report: JDQualityReport, wanted: list[str]) -> float:
    """Composite 0-100 quality score.

    Weighted toward the two things that most affect applicant volume and
    quality: whether the posting is complete, and whether it is written in
    language that does not filter people out before they apply.
    """
    section_score = len(report.sections_present) / len(REQUIRED_SECTIONS)
    keyword_score = (len(report.keywords_present) / len(wanted)) if wanted else 1.0

    # Length: 350-600 words is the sweet spot; penalise both extremes.
    if 350 <= report.word_count <= 600:
        length_score = 1.0
    elif report.word_count < 350:
        length_score = max(0.0, report.word_count / 350)
    else:
        length_score = max(0.4, 1.0 - (report.word_count - 600) / 900)

    # Each flagged term costs 8 points of the inclusivity component.
    flagged = sum(len(f["terms"]) for f in report.flags)
    inclusivity = max(0.0, 1.0 - 0.08 * flagged)

    readability = 1.0 if report.avg_sentence_words <= 25 else max(0.4, 25 / report.avg_sentence_words)

    return 100.0 * (
        0.30 * section_score
        + 0.25 * keyword_score
        + 0.25 * inclusivity
        + 0.10 * length_score
        + 0.10 * readability
    )


def _suggestions(report: JDQualityReport) -> list[str]:
    out: list[str] = []
    if report.sections_missing:
        out.append("Add missing sections: " + ", ".join(report.sections_missing))
    if report.keywords_missing:
        out.append(
            "These required skills are not mentioned anywhere, which hurts search "
            "visibility: " + ", ".join(report.keywords_missing[:6])
        )
    for flag in report.flags:
        swaps = ", ".join(f'"{k}" → "{v}"' for k, v in flag["alternatives"].items())
        out.append(
            f"{flag['category'].replace('_', ' ').title()}: "
            f"{', '.join(flag['terms'])}. {flag['explanation']}"
            + (f" Suggested: {swaps}" if swaps else "")
        )
    if report.word_count < 250:
        out.append(f"Only {report.word_count} words — too thin to convey the role.")
    elif report.word_count > 800:
        out.append(f"{report.word_count} words — long postings lose readers; aim for 350-500.")
    if report.avg_sentence_words > 25:
        out.append(
            f"Sentences average {report.avg_sentence_words:.0f} words. Shorter sentences read better."
        )
    if not report.attraction_terms:
        out.append(
            "No terms conveying ownership, growth or impact — these materially "
            "improve response rates."
        )
    return out


def _strip_wrapper(text: str) -> str:
    """Remove markdown fences and conversational preamble."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:\w+)?\s*(.*?)```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    lines = text.splitlines()
    if lines and re.match(
        r"^(sure|here'?s|certainly|of course|below is|i'?ve|absolutely)\b",
        lines[0].strip(), re.IGNORECASE,
    ):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _fallback_jd(
    role_title: str,
    experience: str,
    must_have: list[str],
    nice_to_have: list[str],
    location: str,
    salary_range: str,
) -> str:
    """A structured template used when the model is unavailable."""
    def bullets(items: list[str], empty: str) -> str:
        return "\n".join(f"- {i}" for i in items) if items else f"- {empty}"

    return f"""{role_title or 'Role title'}

About the role
We are hiring {'a ' + role_title if role_title else 'for this role'}{' in ' + location if location else ''}.
[Describe the team, the product, and the problem this person will own.]

Responsibilities
- [Outcome this person owns, not a task list]
- [System or area of ownership]
- [Collaboration and review expectations]

Requirements
- {experience or 'X'} years of relevant experience
{bullets(must_have, '[Required skill]')}

Nice to have
{bullets(nice_to_have, '[Preferred skill]')}

What we offer
- Compensation: {salary_range or '[state the range explicitly]'}
- [Benefits, learning budget, leave policy]

Work model
- {location or '[Location]'} — [remote / hybrid / on-site]

How to apply
- Submit your application through the form linked in this posting.

Equal opportunity
We welcome applications from people of all backgrounds and are committed to an
inclusive hiring process. If you need an adjustment at any stage, tell us.

[This draft was produced from a template because the AI service was
unavailable. Please complete the bracketed sections before publishing.]
"""
