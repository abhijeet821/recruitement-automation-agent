"""
The screening pipeline — one entry point for the whole engine.

    PDF ──► text ──► ResumeProfile ──┐
                                     ├──► FeatureVector ──► CandidateScore
    JD  ──► JobSpec ─────────────────┤
    GitHub handle ──► GitHubProfile ─┘

The web layer imports this and nothing deeper. That boundary is what keeps the
Django code free of scoring logic and the scoring code free of Django, so the
same pipeline runs identically from a view, a background job, the evaluation
harness, or a notebook.

Nothing here raises on bad input. Every stage degrades: an unreadable PDF, an
unreachable model, a missing GitHub account all produce a result carrying
warnings and reduced confidence, never an exception that loses the candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from matching.config import MatchingConfig, get_config
from matching.enrichment.github import analyse_github
from matching.generation.interview import InterviewGuide, generate_interview_guide
from matching.generation.jd import JDQualityReport, analyse_jd, generate_jd
from matching.llm import LLMProvider, get_provider
from matching.parsing.jobspec import parse_jobspec
from matching.parsing.pdf import PDFExtraction, extract_pdf_text
from matching.parsing.resume import parse_resume
from matching.schemas import CandidateScore, GitHubProfile, JobSpec, ResumeProfile
from matching.scoring.ensemble import EnsembleScorer

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Everything the pipeline learned about one candidate."""

    resume: ResumeProfile
    score: CandidateScore
    github: GitHubProfile | None = None
    pdf: PDFExtraction | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.resume.extraction_failed

    def to_dict(self) -> dict:
        return {
            "resume": self.resume.to_dict(),
            "score": self.score.to_dict(),
            "github": self.github.to_dict() if self.github else None,
            "warnings": self.warnings,
        }


class ScreeningPipeline:
    """Orchestrates parsing, enrichment and scoring."""

    def __init__(
        self,
        config: MatchingConfig | None = None,
        provider: LLMProvider | None = None,
    ):
        self.config = config or get_config()
        self.provider = provider or get_provider(self.config)
        self.scorer = EnsembleScorer(provider=self.provider, config=self.config)

    # ── job side ─────────────────────────────────────────────

    def draft_jd(
        self,
        role_title: str,
        experience: str,
        **kwargs,
    ) -> tuple[str, JDQualityReport]:
        """Draft a job description and grade it in one step."""
        text = generate_jd(role_title, experience, self.provider, **kwargs)
        report = analyse_jd(
            text,
            role_title=role_title,
            must_have=kwargs.get("must_have"),
            nice_to_have=kwargs.get("nice_to_have"),
        )
        return text, report

    def interview_guide(
        self,
        resume: ResumeProfile,
        job: JobSpec,
        *,
        duration_minutes: int = 45,
        github: GitHubProfile | None = None,
        score: CandidateScore | None = None,
    ) -> InterviewGuide:
        """Build an interview guide sized to the booked slot.

        Takes the score so the questions can target exactly the requirements the
        screening step could not verify — which is both the most useful thing to
        ask about and the fairest, since it lets the candidate answer for
        themselves rather than being filtered on an inference.
        """
        return generate_interview_guide(
            resume, job, self.provider,
            duration_minutes=duration_minutes, github=github, score=score,
        )

    def build_job_spec(
        self,
        jd_text: str,
        *,
        role_title: str = "",
        experience: str = "",
    ) -> JobSpec:
        """Distil a JD into structured requirements — do this once per campaign.

        The result is cached on the campaign rather than recomputed per
        candidate: it is identical for every applicant and costs a model call.
        """
        return parse_jobspec(
            jd_text, self.provider, role_title=role_title, experience_hint=experience
        )

    # ── candidate side ───────────────────────────────────────

    def screen_pdf(
        self,
        pdf_path: str | Path,
        job: JobSpec,
        *,
        email: str = "",
        github_username: str = "",
        with_github: bool = True,
    ) -> ScreeningResult:
        """Screen a candidate from a resume PDF on disk."""
        extraction = extract_pdf_text(pdf_path)
        warnings = list(extraction.warnings)

        if not extraction.ok:
            warnings.append(f"Resume could not be read ({extraction.reason})")

        result = self.screen_text(
            extraction.text,
            job,
            email=email,
            github_username=github_username,
            with_github=with_github,
        )
        result.pdf = extraction
        result.warnings = warnings + result.warnings

        if not extraction.ok:
            result.resume.extraction_failed = True
            if extraction.looks_scanned:
                result.score.flags.insert(
                    0, "Resume appears to be a scanned image — no text could be read"
                )
        return result

    def screen_text(
        self,
        resume_text: str,
        job: JobSpec,
        *,
        email: str = "",
        github_username: str = "",
        with_github: bool = True,
    ) -> ScreeningResult:
        """Screen a candidate from resume text."""
        warnings: list[str] = []

        resume = parse_resume(resume_text, self.provider, fallback_email=email)
        if resume.extraction_failed and resume_text.strip():
            warnings.append(
                "Structured extraction failed — scoring from raw text only, "
                "confidence is reduced."
            )

        github = None
        handle = (github_username or resume.github_username or "").strip()
        if with_github and handle and job.is_technical:
            github = analyse_github(handle, self.config)
            if not github.found:
                warnings.append(f"GitHub '{handle}' unavailable: {github.error}")

        try:
            score = self.scorer.score(resume, job, github)
        except Exception as exc:  # noqa: BLE001 - a scoring bug must not lose the candidate
            logger.exception("Scoring failed for %s", resume.email or email)
            warnings.append(f"Scoring failed: {exc}")
            score = CandidateScore(
                overall=0.0,
                confidence=0.0,
                summary="Scoring failed — this candidate needs manual review.",
                flags=[f"Scoring error: {exc}"],
                scorer=self.scorer.name,
                scorer_version=self.scorer.version,
            )

        return ScreeningResult(resume=resume, score=score, github=github, warnings=warnings)

    # ── diagnostics ──────────────────────────────────────────

    def health(self) -> dict:
        """Report on backend reachability, for the UI banner and `manage.py doctor`."""
        ok, detail = self.provider.health()
        return {
            "provider": self.config.provider,
            "model": getattr(self.config, f"{self.config.provider}_model", ""),
            "healthy": ok,
            "detail": detail,
            "blind_screening": self.config.blind_screening,
            "rubric_enabled": self.config.rubric_enabled,
            "github_token_set": bool(self.config.github_token),
        }
