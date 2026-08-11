"""
Campaign orchestration: the glue between Google Workspace and the matching engine.

Two operations live here, both designed to be interruptible and re-runnable:

``sync_campaign``   pull new Form responses, download resumes, screen them
``rescore_campaign`` re-screen already-imported candidates after a JD or
                     weighting change, without re-downloading anything

Both are idempotent. Responses are keyed by Google's ``responseId`` under a
unique constraint, so a sync that dies halfway through can simply be run again;
already-imported candidates are skipped and partially-processed ones are
completed. That matters because this work is slow — a local model spends roughly
half a minute per candidate — and any long job will eventually be interrupted.

Each candidate is committed in its own transaction as soon as it is screened,
rather than batching at the end. One malformed resume can then never cost the
work already done on the previous forty.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from hiring_app.models import Campaign, Candidate
from hiring_app.services.google_workspace import (
    Q_EMAIL,
    Q_GITHUB,
    Q_LINKEDIN,
    Q_NAME,
    Q_RESUME,
    RESPONSES_TAB,
    WorkspaceClient,
    WorkspaceError,
)
from matching.config import get_config
from matching.parsing.contacts import find_github_username
from matching.parsing.pdf import extract_pdf_text
from matching.pipeline import ScreeningPipeline
from matching.schemas import JobSpec

logger = logging.getLogger("hiring_app")

ProgressFn = Callable[[int, int, str], None]


@dataclass
class SyncOutcome:
    imported: int = 0
    scored: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.imported} new", f"{self.scored} scored"]
        if self.skipped:
            parts.append(f"{self.skipped} already imported")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


def resume_dir(campaign: Campaign) -> Path:
    """Per-campaign resume storage, namespaced by owner.

    The old layout put every user's resumes in one flat directory keyed by Drive
    file ID. Namespacing by owner and campaign keeps one recruiter's candidate
    documents out of another's, and makes deleting a campaign's personal data a
    single directory removal.
    """
    path = Path(settings.MEDIA_ROOT) / "resumes" / f"user_{campaign.owner_id}" / f"campaign_{campaign.id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_job_spec(campaign: Campaign, pipeline: ScreeningPipeline) -> JobSpec:
    """Return the campaign's JobSpec, distilling it once and caching it.

    This costs a model call, and it is identical for every applicant, so it is
    computed at most once per campaign rather than per candidate.
    """
    spec = campaign.get_job_spec()
    if spec.must_have_skills or not campaign.jd_text.strip():
        return spec

    spec = pipeline.build_job_spec(
        campaign.jd_text,
        role_title=campaign.role_title,
        experience=campaign.experience_required,
    )
    campaign.set_job_spec(spec)
    campaign.save(update_fields=["job_spec", "updated_at"])
    logger.info(
        "Distilled JobSpec for campaign %s: %d must-have skills",
        campaign.id, len(spec.must_have_skills),
    )
    return spec


def sync_campaign(
    campaign: Campaign,
    client: WorkspaceClient,
    *,
    progress: ProgressFn | None = None,
) -> SyncOutcome:
    """Import and screen every new response for a campaign."""
    outcome = SyncOutcome()
    if not campaign.form_id:
        raise WorkspaceError("This campaign has no application form yet.")

    responses = client.list_responses(campaign.form_id)
    known = set(
        Candidate.objects.filter(campaign=campaign)
        .values_list("response_id", flat=True)
    )
    pending = [r for r in responses if r.get("responseId") not in known]
    outcome.skipped = len(responses) - len(pending)

    if not pending:
        campaign.last_synced_at = timezone.now()
        campaign.save(update_fields=["last_synced_at", "updated_at"])
        return outcome

    pipeline = ScreeningPipeline(get_config())
    job_spec = ensure_job_spec(campaign, pipeline)
    question_ids = campaign.question_ids or {}
    storage = resume_dir(campaign)
    sheet_rows: list[list] = []

    for index, response in enumerate(pending, start=1):
        response_id = response.get("responseId", "")
        if progress:
            progress(index, len(pending), f"Screening applicant {index} of {len(pending)}")

        try:
            candidate = _screen_one(
                campaign=campaign,
                client=client,
                pipeline=pipeline,
                job_spec=job_spec,
                response=response,
                question_ids=question_ids,
                storage=storage,
            )
        except Exception as exc:  # noqa: BLE001 - one applicant must not end the sync
            logger.exception("Failed to screen response %s", response_id)
            outcome.failed += 1
            outcome.errors.append(f"{response_id}: {exc}")
            continue

        outcome.imported += 1
        if candidate.is_scored:
            outcome.scored += 1

        sheet_rows.append([
            candidate.response_id,
            candidate.submitted_at.isoformat() if candidate.submitted_at else "",
            candidate.full_name,
            candidate.email,
            round(candidate.overall_score, 1),
            f"{candidate.confidence_percent}%",
            candidate.recommendation,
            candidate.resume_drive_link,
            candidate.github_username,
            candidate.get_status_display(),
        ])

    # The Sheet is a synced export, not the system of record — a failure to
    # write it must not roll back candidates already stored in the database.
    if sheet_rows and campaign.sheet_id:
        try:
            client.append_rows(campaign.sheet_id, RESPONSES_TAB, sheet_rows)
        except WorkspaceError as exc:
            logger.warning("Could not update the tracking sheet: %s", exc)
            outcome.errors.append(f"Sheet update failed: {exc}")

    campaign.last_synced_at = timezone.now()
    campaign.save(update_fields=["last_synced_at", "updated_at"])
    return outcome


def _screen_one(
    *,
    campaign: Campaign,
    client: WorkspaceClient,
    pipeline: ScreeningPipeline,
    job_spec: JobSpec,
    response: dict,
    question_ids: dict,
    storage: Path,
) -> Candidate:
    """Import one response, fetch its resume, screen it, and persist the result."""
    response_id = response.get("responseId", "")
    answer = client.answer

    email = answer(response, question_ids.get(Q_EMAIL)) or response.get("respondentEmail", "")
    full_name = answer(response, question_ids.get(Q_NAME))
    drive_link = answer(response, question_ids.get(Q_RESUME))
    github_raw = answer(response, question_ids.get(Q_GITHUB))
    linkedin = answer(response, question_ids.get(Q_LINKEDIN))

    github_username = find_github_username(github_raw) or find_github_username(
        f"github.com/{github_raw.strip().strip('/@')}" if github_raw.strip() else ""
    )

    candidate = Candidate(
        campaign=campaign,
        response_id=response_id,
        email=email,
        full_name=full_name,
        github_username=github_username,
        resume_drive_link=drive_link,
        submitted_at=parse_datetime(response.get("createTime", "") or "") or timezone.now(),
    )

    warnings: list[str] = []
    resume_text = ""

    file_id = client.extract_file_id(drive_link)
    if not drive_link:
        warnings.append("No resume link was provided.")
    elif not file_id:
        warnings.append(f"Could not read a Google Drive file ID from '{drive_link[:80]}'.")
    else:
        candidate.resume_file_id = file_id
        destination = storage / f"{file_id}.pdf"
        try:
            if not destination.exists():
                client.download_pdf(file_id, destination)
            candidate.resume_path = str(destination)
            extraction = extract_pdf_text(destination)
            resume_text = extraction.text
            warnings.extend(extraction.warnings)
            if not extraction.ok:
                warnings.append(f"Resume could not be read ({extraction.reason}).")
        except WorkspaceError as exc:
            warnings.append(str(exc))

    result = pipeline.screen_text(
        resume_text,
        job_spec,
        email=email,
        github_username=github_username,
        with_github=bool(github_username),
    )

    candidate.resume_text = resume_text
    candidate.profile = result.resume.to_dict()
    candidate.github_profile = result.github.to_dict() if result.github else {}
    candidate.apply_score(result.score)
    candidate.warnings = warnings + result.warnings

    # Fill blanks from the parsed resume — candidates routinely mistype the
    # form fields, and the resume itself is usually more reliable.
    candidate.full_name = candidate.full_name or result.resume.full_name
    candidate.email = candidate.email or result.resume.email
    candidate.github_username = candidate.github_username or result.resume.github_username
    if linkedin and not result.resume.linkedin_url:
        candidate.profile["linkedin_url"] = linkedin

    with transaction.atomic():
        candidate.save()
    return candidate


def rescore_campaign(
    campaign: Campaign,
    *,
    progress: ProgressFn | None = None,
    only_unscored: bool = False,
) -> SyncOutcome:
    """Re-screen stored candidates — after a JD edit or a weighting change.

    Uses the resume text already in the database, so no Google call is needed
    and the operation works even with Google disconnected.
    """
    outcome = SyncOutcome()
    pipeline = ScreeningPipeline(get_config())
    job_spec = ensure_job_spec(campaign, pipeline)

    queryset = campaign.candidates.all()
    if only_unscored:
        queryset = queryset.filter(scored_at__isnull=True)
    candidates = list(queryset)

    for index, candidate in enumerate(candidates, start=1):
        if progress:
            progress(index, len(candidates), f"Re-scoring {index} of {len(candidates)}")
        try:
            result = pipeline.screen_text(
                candidate.resume_text,
                job_spec,
                email=candidate.email,
                github_username=candidate.github_username,
                with_github=bool(candidate.github_username),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Re-scoring failed for candidate %s", candidate.pk)
            outcome.failed += 1
            outcome.errors.append(f"{candidate.email or candidate.pk}: {exc}")
            continue

        candidate.profile = result.resume.to_dict()
        if result.github:
            candidate.github_profile = result.github.to_dict()
        candidate.apply_score(result.score)
        candidate.warnings = result.warnings
        candidate.save(update_fields=[
            "profile", "github_profile", "score_detail", "overall_score",
            "confidence", "recommendation", "status", "scored_at",
            "warnings", "updated_at",
        ])
        outcome.scored += 1

    return outcome
