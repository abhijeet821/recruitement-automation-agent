"""
Personalised interview guides.

A screening score tells a recruiter *who* to interview. It does not help them
run the interview. This module closes that gap: given the parsed resume, the job
requirements, the GitHub profile and the score breakdown, it produces questions
grounded in **this specific candidate's** evidence, sized to fit the slot the
company actually books.

## Grounding is the whole point

The failure mode for generated interview questions is genericity — "What is a
REST API?" could be asked of anybody, tells you nothing, and wastes a slot the
company is paying for. Every question here must therefore cite the exact detail
that prompted it, and the prompt forbids questions that could be asked without
having read the CV. A question with no citation is dropped at parse time rather
than shown, because an ungrounded question is worse than no question: it looks
personalised and is not.

The score's own **gaps** are fed in deliberately. If the scorer could not find
evidence of a required skill, that is precisely what the interview should
resolve — and it turns an automated judgement into something the candidate gets
to answer for themselves, which is the fair way round.

## Sizing to the slot

Question count is derived from the booked duration, not guessed. Time is
budgeted per category (a project deep-dive genuinely takes longer than a skill
check), warm-up and the candidate's own questions are reserved, and categories
with no supporting evidence are dropped with their time redistributed — the same
principle the scorer uses for missing dimensions.

## What is deliberately excluded

No questions about age, family, nationality, health, religion, marital status,
visa status, or the personal reasons behind a career break. These are unlawful
to screen on in most jurisdictions and irrelevant to whether someone can do the
job. Employment gaps may be asked about only in terms of *what the person worked
on*, never why they were away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from matching.llm import LLMError, LLMProvider
from matching.schemas import CandidateScore, GitHubProfile, JobSpec, ResumeProfile

logger = logging.getLogger(__name__)

# Reserved either side of the questions themselves.
WARMUP_MINUTES = 5
WRAP_UP_MINUTES = 5          # the candidate's own questions — never squeeze this
MIN_QUESTIONS = 3
MAX_QUESTIONS = 24

# How long a good answer to each kind of question actually takes, including the
# follow-ups that make it worth asking.
CATEGORY_MINUTES: dict[str, int] = {
    "skill_verification": 4,
    "project_deep_dive": 8,
    "github_code": 6,
    "gap_probe": 5,
    "experience_depth": 6,
    "system_design": 12,
    "behavioural": 5,
}

# Share of the usable time each category gets, before dropping unavailable ones.
BASE_WEIGHTS: dict[str, float] = {
    "skill_verification": 0.22,
    "project_deep_dive": 0.24,
    "github_code": 0.10,
    "gap_probe": 0.14,
    "experience_depth": 0.16,
    "system_design": 0.06,
    "behavioural": 0.08,
}

CATEGORY_LABELS: dict[str, str] = {
    "skill_verification": "Skill verification",
    "project_deep_dive": "Project deep-dive",
    "github_code": "Public code",
    "gap_probe": "Unverified requirement",
    "experience_depth": "Experience depth",
    "system_design": "System design",
    "behavioural": "Working style",
}

COMMON_DURATIONS = [15, 30, 45, 60, 75, 90, 120]

# What to keep when the slot is too short for everything, most valuable first.
# Trimming by cost instead would strip the project deep-dive — the single most
# informative question type — from every short interview, which is exactly wrong.
# Skill verification leads because it is cheap and tests the requirement directly.
KEEP_PRIORITY = [
    "skill_verification",
    "project_deep_dive",
    "gap_probe",
    "experience_depth",
    "github_code",
    "behavioural",
    "system_design",
]


# ─────────────────────────────────────────────────────────────
# Budget
# ─────────────────────────────────────────────────────────────

@dataclass
class QuestionBudget:
    """How many questions of each kind fit in the booked slot."""

    duration_minutes: int
    warmup_minutes: int
    wrap_up_minutes: int
    usable_minutes: int
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def estimated_minutes(self) -> int:
        """Time the questions themselves are expected to consume."""
        return sum(CATEGORY_MINUTES[c] * n for c, n in self.counts.items())

    @property
    def fits(self) -> bool:
        return self.estimated_minutes <= self.usable_minutes

    def to_dict(self) -> dict:
        return {
            "duration_minutes": self.duration_minutes,
            "warmup_minutes": self.warmup_minutes,
            "wrap_up_minutes": self.wrap_up_minutes,
            "usable_minutes": self.usable_minutes,
            "counts": self.counts,
            "total": self.total,
            "estimated_minutes": self.estimated_minutes,
        }


def plan_budget(
    duration_minutes: int,
    *,
    has_github: bool = False,
    has_gaps: bool = False,
    is_technical: bool = True,
    seniority: str = "mid",
) -> QuestionBudget:
    """Turn a booked slot into a per-category question count.

    Categories with nothing to draw on are dropped and their time is shared out
    across the rest, so a candidate with no public code gets more project and
    experience questions rather than a short interview.
    """
    duration_minutes = max(10, int(duration_minutes))

    # Very short slots cannot afford the full ceremony.
    warmup = WARMUP_MINUTES if duration_minutes > 20 else 2
    wrap_up = WRAP_UP_MINUTES if duration_minutes > 20 else 3
    usable = max(6, duration_minutes - warmup - wrap_up)

    weights = dict(BASE_WEIGHTS)

    if not has_github:
        weights.pop("github_code", None)
    if not has_gaps:
        weights.pop("gap_probe", None)
    if not is_technical:
        # Without code to discuss, the time belongs in experience and working style.
        weights.pop("github_code", None)
        weights.pop("system_design", None)
        weights["experience_depth"] = weights.get("experience_depth", 0) + 0.12
        weights["behavioural"] = weights.get("behavioural", 0) + 0.08
    # System design only earns a slot for senior roles in a long enough session.
    if seniority in {"intern", "junior"} or usable < 40:
        weights.pop("system_design", None)

    total_weight = sum(weights.values()) or 1.0
    counts: dict[str, int] = {}

    for category, weight in weights.items():
        minutes = usable * (weight / total_weight)
        counts[category] = max(0, round(minutes / CATEGORY_MINUTES[category]))

    # Prefer to include the categories that carry an interview — but these are
    # preferences, not guarantees. A 15-minute screen cannot afford an 8-minute
    # project deep-dive on top of everything else, and quietly overrunning the
    # booked slot would defeat the point of budgeting at all.
    counts["skill_verification"] = max(1, counts.get("skill_verification", 0))
    counts["project_deep_dive"] = max(1, counts.get("project_deep_dive", 0))
    if has_gaps:
        counts["gap_probe"] = max(1, counts.get("gap_probe", 0))

    counts = {c: n for c, n in counts.items() if n > 0}
    budget = QuestionBudget(duration_minutes, warmup, wrap_up, usable, counts)

    # Drop the least valuable category first until the plan fits, keeping at
    # least one question.
    guard = 0
    while budget.estimated_minutes > usable and budget.total > 1 and guard < 60:
        guard += 1
        present = [c for c in KEEP_PRIORITY if budget.counts.get(c, 0) > 0]
        if not present:
            break
        budget.counts[present[-1]] -= 1
        budget.counts = {c: n for c, n in budget.counts.items() if n > 0}

    # Trimming and rounding are coarse, so a usable gap can remain. Fill it by
    # cycling through the eligible categories in priority order rather than
    # topping up a single one — otherwise a long slot fills with seven rapid
    # skill checks instead of the depth a long interview is booked for.
    eligible = [c for c in KEEP_PRIORITY if c in weights]
    guard = 0
    added = True
    while added and budget.total < MAX_QUESTIONS and guard < 60:
        guard += 1
        added = False
        for category in eligible:
            remaining = usable - budget.estimated_minutes
            if remaining >= CATEGORY_MINUTES[category] and budget.total < MAX_QUESTIONS:
                budget.counts[category] = budget.counts.get(category, 0) + 1
                added = True

    if budget.total > MAX_QUESTIONS:
        # Absurdly long slot: cap it and say so rather than generating 40 questions.
        overflow = budget.total - MAX_QUESTIONS
        for category in sorted(budget.counts, key=lambda c: -budget.counts[c]):
            take = min(overflow, budget.counts[category] - 1)
            budget.counts[category] -= take
            overflow -= take
            if overflow <= 0:
                break

    return budget


# ─────────────────────────────────────────────────────────────
# Guide
# ─────────────────────────────────────────────────────────────

@dataclass
class InterviewQuestion:
    question: str
    category: str = "skill_verification"
    # The exact CV/GitHub detail that prompted this question. Without it the
    # question is generic and gets dropped.
    grounded_in: str = ""
    why: str = ""
    listen_for: list[str] = field(default_factory=list)
    follow_up: str = ""
    minutes: int = 5

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category.replace("_", " ").title())

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "category_label": self.category_label,
            "grounded_in": self.grounded_in,
            "why": self.why,
            "listen_for": self.listen_for,
            "follow_up": self.follow_up,
            "minutes": self.minutes,
        }


@dataclass
class InterviewGuide:
    candidate_name: str = ""
    role_title: str = ""
    duration_minutes: int = 45
    budget: QuestionBudget | None = None
    questions: list[InterviewQuestion] = field(default_factory=list)
    opening: str = ""
    focus: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.questions)

    @property
    def estimated_minutes(self) -> int:
        return sum(q.minutes for q in self.questions)

    def by_category(self) -> dict[str, list[InterviewQuestion]]:
        grouped: dict[str, list[InterviewQuestion]] = {}
        for question in self.questions:
            grouped.setdefault(question.category, []).append(question)
        return grouped

    def to_dict(self) -> dict:
        return {
            "candidate_name": self.candidate_name,
            "role_title": self.role_title,
            "duration_minutes": self.duration_minutes,
            "budget": self.budget.to_dict() if self.budget else None,
            "questions": [q.to_dict() for q in self.questions],
            "opening": self.opening,
            "focus": self.focus,
            "warnings": self.warnings,
            "error": self.error,
            "estimated_minutes": self.estimated_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> InterviewGuide:
        data = dict(data or {})
        questions = [
            InterviewQuestion(
                question=q.get("question", ""),
                category=q.get("category", "skill_verification"),
                grounded_in=q.get("grounded_in", ""),
                why=q.get("why", ""),
                listen_for=list(q.get("listen_for") or []),
                follow_up=q.get("follow_up", ""),
                minutes=int(q.get("minutes") or 5),
            )
            for q in data.get("questions") or []
        ]
        budget_data = data.get("budget") or {}
        budget = (
            QuestionBudget(
                duration_minutes=budget_data.get("duration_minutes", 45),
                warmup_minutes=budget_data.get("warmup_minutes", WARMUP_MINUTES),
                wrap_up_minutes=budget_data.get("wrap_up_minutes", WRAP_UP_MINUTES),
                usable_minutes=budget_data.get("usable_minutes", 35),
                counts=budget_data.get("counts", {}),
            )
            if budget_data else None
        )
        return cls(
            candidate_name=data.get("candidate_name", ""),
            role_title=data.get("role_title", ""),
            duration_minutes=data.get("duration_minutes", 45),
            budget=budget,
            questions=questions,
            opening=data.get("opening", ""),
            focus=data.get("focus", ""),
            warnings=list(data.get("warnings") or []),
            error=data.get("error", ""),
        )


GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "opening": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORY_MINUTES)},
                    "grounded_in": {"type": "string"},
                    "why": {"type": "string"},
                    "listen_for": {"type": "array", "items": {"type": "string"}},
                    "follow_up": {"type": "string"},
                },
                "required": ["question", "category", "grounded_in", "why", "listen_for"],
            },
        },
    },
    "required": ["questions", "focus"],
}

SYSTEM_PROMPT = (
    "You are an experienced technical interviewer preparing to interview one "
    "specific candidate. You write questions that could only be asked of this "
    "person, because they refer to what this person actually did. You never ask "
    "textbook definition questions, and you never ask about a candidate's "
    "personal circumstances."
)

PROMPT_TEMPLATE = """Prepare interview questions for this candidate.

ROLE
{role_title} ({seniority}) — {min_years}+ years
Required: {must_have}
Preferred: {nice_to_have}

CANDIDATE RESUME
---
{resume}
---
{github_block}{gaps_block}
INTERVIEW SLOT
{duration} minutes total. After {warmup} min of introductions and reserving
{wrap_up} min for the candidate's own questions, you have {usable} minutes.

Produce exactly this many questions per category:
{plan}

CATEGORY MEANINGS
- skill_verification — confirm a required skill they claim, by asking about a
  decision or trade-off only someone who used it would know
- project_deep_dive — go deep on something specific they built: why that design,
  what broke, what they would change
- github_code — about their actual public repositories, named
- gap_probe — a required skill with no evidence in the resume. Ask openly so they
  can demonstrate it; never imply they lack it
- experience_depth — scale, ownership, and the hardest problem in a role they held
- system_design — a design problem in the domain they have actually worked in
- behavioural — collaboration, disagreement, mentoring, drawn from a real
  situation described in the resume

RULES
- Every question must include `grounded_in`: a short verbatim detail from the
  resume or GitHub that prompted it. If you cannot cite something specific, do
  not write the question.
- Never ask a question answerable from a textbook without having read this CV.
  "What is an index?" is banned. "You cut report generation from 45 minutes to 6
  using materialised views — what made the original query slow?" is right.
- `listen_for`: 2-3 concrete things a strong answer contains.
- `follow_up`: one probe to use if the first answer is shallow.
- Never ask about age, family, nationality, religion, health, marital status,
  visa status, or the personal reasons behind any career break. You may ask what
  someone worked on during a period, never why they were away.
- `focus`: one sentence on what this interview most needs to establish.
- `opening`: one warm, specific opening line referencing their background.
"""


def _resume_digest(resume: ResumeProfile, limit: int = 6000) -> str:
    parts: list[str] = []
    if resume.summary:
        parts.append(f"Summary: {resume.summary}")
    if resume.total_years_experience:
        parts.append(f"Total experience: {resume.total_years_experience:g} years")

    for role in resume.experience[:6]:
        span = ""
        if role.start_year:
            span = f" ({role.start_year}–{'present' if role.is_current else role.end_year or '?'})"
        parts.append(f"\n{role.title} at {role.company}{span}\n{role.description}")

    if resume.skills:
        parts.append(
            "\nSkills: "
            + ", ".join(
                f"{s.name}" + (f" ({s.years:g}y, {s.evidence})" if s.years else f" ({s.evidence})")
                for s in resume.skills[:30]
            )
        )
    if resume.education:
        parts.append(
            "\nEducation: "
            + "; ".join(f"{e.degree} {e.field_of_study}".strip() for e in resume.education)
        )

    text = "\n".join(parts).strip()
    # Fall back to raw text when extraction produced little structure.
    if len(text) < 200 and resume.raw_text:
        text = resume.raw_text[:limit]
    return text[:limit]


def _github_block(github: GitHubProfile | None) -> str:
    if not github or not github.found:
        return ""
    repos = "\n".join(
        f"  - {r.name} ({r.language or 'n/a'}, {r.stars}★): {r.description or 'no description'}"
        for r in github.top_repos[:5]
    )
    languages = ", ".join(
        f"{lang} {share:.0%}"
        for lang, share in sorted(github.language_share().items(), key=lambda kv: kv[1], reverse=True)[:4]
    )
    return (
        f"\nPUBLIC CODE (github.com/{github.username})\n"
        f"{github.original_repos} original repos, {github.total_stars} stars total\n"
        f"Languages: {languages or 'unknown'}\n"
        f"Notable repositories:\n{repos}\n"
    )


def _gaps_block(score: CandidateScore | None) -> str:
    if not score or not score.gaps:
        return ""
    listed = "\n".join(f"  - {g}" for g in score.gaps[:6])
    return (
        "\nREQUIREMENTS WITH NO EVIDENCE IN THE RESUME\n"
        "(the screening step could not verify these; the interview should give "
        "the candidate a fair chance to demonstrate them)\n"
        f"{listed}\n"
    )


def generate_interview_guide(
    resume: ResumeProfile,
    job: JobSpec,
    provider: LLMProvider,
    *,
    duration_minutes: int = 45,
    github: GitHubProfile | None = None,
    score: CandidateScore | None = None,
) -> InterviewGuide:
    """Build a duration-sized, evidence-grounded interview guide.

    Never raises — failures come back on ``error`` so a view can report them.
    """
    guide = InterviewGuide(
        candidate_name=resume.full_name,
        role_title=job.role_title,
        duration_minutes=duration_minutes,
    )

    digest = _resume_digest(resume)
    if len(digest) < 120:
        guide.error = "Not enough resume detail to write grounded questions."
        return guide

    has_gaps = bool(score and score.gaps)
    budget = plan_budget(
        duration_minutes,
        has_github=bool(github and github.found),
        has_gaps=has_gaps,
        is_technical=job.is_technical,
        seniority=job.seniority or "mid",
    )
    guide.budget = budget

    plan_lines = "\n".join(
        f"  {CATEGORY_LABELS.get(c, c)} ({c}): {n}" for c, n in budget.counts.items()
    )

    prompt = PROMPT_TEMPLATE.format(
        role_title=job.role_title or "the role",
        seniority=job.seniority or "unspecified",
        min_years=f"{job.min_years:g}",
        must_have=", ".join(job.must_have_skills) or "not specified",
        nice_to_have=", ".join(job.nice_to_have_skills) or "not specified",
        resume=digest,
        github_block=_github_block(github),
        gaps_block=_gaps_block(score),
        duration=budget.duration_minutes,
        warmup=budget.warmup_minutes,
        wrap_up=budget.wrap_up_minutes,
        usable=budget.usable_minutes,
        plan=plan_lines,
    )

    try:
        payload = provider.generate_json(
            prompt, schema=GUIDE_SCHEMA, system=SYSTEM_PROMPT, temperature=0.4
        )
    except LLMError as exc:
        logger.error("Interview guide generation failed: %s", exc)
        guide.error = str(exc)
        return guide

    guide.focus = str(payload.get("focus") or "").strip()
    guide.opening = str(payload.get("opening") or "").strip()

    dropped_ungrounded = 0
    for raw in payload.get("questions") or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("question") or "").strip()
        grounded = str(raw.get("grounded_in") or "").strip()
        if not text:
            continue
        if not grounded:
            # An ungrounded question is worse than none — it looks personalised
            # and is not. Drop it rather than present it.
            dropped_ungrounded += 1
            continue

        category = str(raw.get("category") or "skill_verification").strip()
        if category not in CATEGORY_MINUTES:
            category = "skill_verification"

        guide.questions.append(InterviewQuestion(
            question=text,
            category=category,
            grounded_in=grounded[:300],
            why=str(raw.get("why") or "").strip()[:300],
            listen_for=[str(x).strip() for x in (raw.get("listen_for") or []) if str(x).strip()][:4],
            follow_up=str(raw.get("follow_up") or "").strip()[:300],
            minutes=CATEGORY_MINUTES[category],
        ))

    if dropped_ungrounded:
        guide.warnings.append(
            f"{dropped_ungrounded} question(s) were discarded for not citing "
            f"anything specific in the candidate's background."
        )

    if not guide.questions:
        guide.error = "The model returned no usable, grounded questions. Try again."
        return guide

    # Order so the interview flows: warm and concrete first, abstract later.
    order = [
        "project_deep_dive", "experience_depth", "skill_verification",
        "github_code", "gap_probe", "system_design", "behavioural",
    ]
    guide.questions.sort(key=lambda q: order.index(q.category) if q.category in order else 99)

    over = guide.estimated_minutes - budget.usable_minutes
    if over > 0:
        guide.warnings.append(
            f"These questions are budgeted at {guide.estimated_minutes} minutes, "
            f"{over} more than the {budget.usable_minutes} available. Treat the "
            f"later ones as optional."
        )
    return guide
