"""
HTTP layer.

Deliberately thin: parse the request, call a service, report the outcome. No
scoring logic, no Google API calls, no state juggling.

Corrections to the previous version:

* Every state-changing endpoint is POST-only. ``/sync-responses/`` used to be a
  GET link, so a browser prefetch or a crawler could trigger an expensive,
  irreversible operation.
* Failures reach the user. Previously every view swallowed exceptions into the
  log and redirected as though nothing had happened, so a campaign that failed
  to launch looked identical to one that succeeded. Errors are now surfaced
  through ``django.contrib.messages``.
* Long work is enqueued rather than executed inline.
* Candidates are queried, not loaded from a JSON file and filtered in Python.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from django_ratelimit.decorators import ratelimit

from hiring_app import jobs
from hiring_app.models import BackgroundJob, Campaign, Candidate
from hiring_app.services import google_auth
from hiring_app.services.google_workspace import (
    CredentialsExpired,
    WorkspaceClient,
    WorkspaceError,
)
from hiring_app.services.linkedin import LinkedInError, post_job
from hiring_app.services.screening import ensure_job_spec
from matching.config import get_config
from matching.generation.jd import analyse_jd
from matching.pipeline import ScreeningPipeline

logger = logging.getLogger("hiring_app")

INTERVIEW_SLOT_MINUTES = 45


# ─────────────────────────────────────────────────────────────
# Public pages
# ─────────────────────────────────────────────────────────────

def landing(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "hiring_app/landing.html")


@ratelimit(key="ip", rate="10/m", method="POST", block=True)
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    error = None
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("dashboard")
        error = "Invalid username or password."
    return render(request, "hiring_app/login.html", {"error": error})


@ratelimit(key="ip", rate="5/m", method="POST", block=True)
def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    error = None
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password1 = request.POST.get("password1", "")
        password2 = request.POST.get("password2", "")

        if not (username and email and password1):
            error = "All fields are required."
        elif password1 != password2:
            error = "Passwords do not match."
        elif len(password1) < 8:
            error = "Password must be at least 8 characters."
        elif User.objects.filter(username__iexact=username).exists():
            error = "That username is taken."
        elif User.objects.filter(email__iexact=email).exists():
            error = "That email is already registered."
        else:
            user = User.objects.create_user(username=username, email=email, password=password1)
            login(request, user)
            return redirect("dashboard")

    return render(request, "hiring_app/register.html", {"error": error})


# ─────────────────────────────────────────────────────────────
# Google OAuth
# ─────────────────────────────────────────────────────────────

@login_required
def google_connect(request):
    try:
        flow = google_auth.build_flow(request)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",  # forces a refresh token on every consent
        )
    except google_auth.OAuthNotConfigured as exc:
        return render(request, "hiring_app/google_connect.html",
                      {"error": str(exc), "has_google": False})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not start the Google OAuth flow")
        return render(request, "hiring_app/google_connect.html",
                      {"error": f"Could not start Google sign-in: {exc}", "has_google": False})

    request.session["google_oauth_state"] = state
    return redirect(authorization_url)


@login_required
def google_oauth_callback(request):
    state = request.session.pop("google_oauth_state", None)
    if not state or state != request.GET.get("state"):
        # A mismatched state is a CSRF signal on the OAuth callback.
        return render(request, "hiring_app/google_connect.html", {
            "error": "The sign-in request could not be verified. Please try again.",
            "has_google": False,
        })

    try:
        flow = google_auth.build_flow(request, state=state)
        flow.fetch_token(authorization_response=request.build_absolute_uri())
        google_auth.save_credentials(request.user, flow.credentials)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google OAuth callback failed for %s", request.user.username)
        return render(request, "hiring_app/google_connect.html",
                      {"error": f"Google sign-in failed: {exc}", "has_google": False})

    messages.success(request, "Google account connected.")
    return redirect("dashboard")


@login_required
@require_POST
def google_disconnect(request):
    from hiring_app.models import GoogleOAuthToken

    GoogleOAuthToken.objects.filter(user=request.user).delete()
    messages.info(request, "Google account disconnected.")
    return redirect("dashboard")


# ─────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    campaigns = (
        Campaign.objects.filter(owner=request.user)
        .annotate(
            n_candidates=Count("candidates"),
            n_strong=Count("candidates", filter=Q(candidates__overall_score__gte=75)),
            avg_score=Avg("candidates__overall_score"),
        )
    )
    totals = Candidate.objects.filter(campaign__owner=request.user).aggregate(
        total=Count("id"),
        strong=Count("id", filter=Q(overall_score__gte=75)),
        interviewed=Count("id", filter=Q(status=Candidate.Status.INVITED)),
    )

    return render(request, "hiring_app/dashboard.html", {
        "campaigns": campaigns,
        "total_campaigns": len(campaigns),
        "active_campaigns": sum(1 for c in campaigns if c.status == Campaign.Status.ACTIVE),
        "total_candidates": totals["total"] or 0,
        "strong_candidates": totals["strong"] or 0,
        "interviewed": totals["interviewed"] or 0,
        "has_google": google_auth.has_google(request.user),
        "engine": _engine_status(),
    })


def _engine_status() -> dict:
    """Health of the scoring backend, shown as a banner when it is down."""
    try:
        return ScreeningPipeline(get_config()).health()
    except Exception as exc:  # noqa: BLE001
        return {"healthy": False, "detail": str(exc), "provider": get_config().provider}


# ─────────────────────────────────────────────────────────────
# Campaigns
# ─────────────────────────────────────────────────────────────

def _own_campaign(request, campaign_id: int) -> Campaign:
    """Fetch a campaign, scoped to the requesting user.

    Scoping the lookup itself means an ID from another account 404s rather than
    leaking whether it exists.
    """
    return get_object_or_404(Campaign, pk=campaign_id, owner=request.user)


@login_required
def campaign_new(request):
    if request.method != "POST":
        return render(request, "hiring_app/campaign_new.html")

    role_title = request.POST.get("role_title", "").strip()
    if not role_title:
        messages.error(request, "A role title is required.")
        return render(request, "hiring_app/campaign_new.html")

    campaign = Campaign.objects.create(
        owner=request.user,
        role_title=role_title,
        experience_required=request.POST.get("experience", "").strip(),
        location=request.POST.get("location", "").strip(),
        work_model=request.POST.get("work_model", "").strip(),
        salary_range=request.POST.get("salary_range", "").strip(),
    )
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
def campaign_detail(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    candidates = list(campaign.candidates.all())

    return render(request, "hiring_app/campaign.html", {
        "campaign": campaign,
        "candidates": candidates,
        "job_spec": campaign.get_job_spec(),
        "jd_quality": campaign.jd_quality or None,
        "other_campaigns": Campaign.objects.filter(owner=request.user).exclude(pk=campaign.pk)[:20],
        "has_google": google_auth.has_google(request.user),
        "active_job": jobs.active_job(campaign),
        "engine": _engine_status(),
        "strong_count": sum(1 for c in candidates if c.overall_score >= 75),
        "review_count": sum(1 for c in candidates if c.needs_review),
    })


@login_required
@require_POST
def campaign_delete(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    # Remove stored resumes along with the record — candidate PII should not
    # outlive the campaign it was collected for.
    import shutil

    from hiring_app.services.screening import resume_dir

    shutil.rmtree(resume_dir(campaign), ignore_errors=True)
    campaign.delete()
    messages.info(request, "Campaign and all associated candidate data deleted.")
    return redirect("dashboard")


@login_required
@require_POST
def generate_jd(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    campaign.role_title = request.POST.get("role_title", campaign.role_title).strip()
    campaign.experience_required = request.POST.get("experience", "").strip()
    campaign.location = request.POST.get("location", "").strip()
    campaign.work_model = request.POST.get("work_model", "").strip()
    campaign.salary_range = request.POST.get("salary_range", "").strip()

    must_have = _split(request.POST.get("must_have", ""))
    nice_to_have = _split(request.POST.get("nice_to_have", ""))

    pipeline = ScreeningPipeline(get_config())
    try:
        jd_text, report = pipeline.draft_jd(
            campaign.role_title,
            campaign.experience_required,
            must_have=must_have,
            nice_to_have=nice_to_have,
            location=campaign.location,
            work_model=campaign.work_model,
            salary_range=campaign.salary_range,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("JD generation failed")
        messages.error(request, f"Could not draft the job description: {exc}")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    campaign.jd_text = jd_text
    campaign.jd_quality = report.to_dict()
    campaign.save()

    if report.flags:
        messages.warning(
            request,
            f"The draft scored {report.score:.0f}/100. "
            f"{len(report.suggestions)} suggestion(s) below — review before launching.",
        )
    else:
        messages.success(request, f"Draft ready — quality score {report.score:.0f}/100.")
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
@require_POST
def save_jd(request, campaign_id: int):
    """Save a hand-edited JD and re-grade it (no model call needed)."""
    campaign = _own_campaign(request, campaign_id)
    campaign.jd_text = request.POST.get("jd_text", "")
    spec = campaign.get_job_spec()
    campaign.jd_quality = analyse_jd(
        campaign.jd_text,
        role_title=campaign.role_title,
        must_have=spec.must_have_skills,
        nice_to_have=spec.nice_to_have_skills,
    ).to_dict()
    # The JD changed, so the cached requirement spec is stale.
    campaign.job_spec = {}
    campaign.save()
    messages.success(request, "Job description saved.")
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
@require_POST
def launch_campaign(request, campaign_id: int):
    """Create the Google Form and Sheet, and optionally post to LinkedIn."""
    campaign = _own_campaign(request, campaign_id)

    if not campaign.jd_text.strip():
        messages.error(request, "Draft a job description before launching.")
        return redirect("campaign_detail", campaign_id=campaign.pk)
    if campaign.form_id:
        messages.info(request, "This campaign is already live.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    try:
        credentials = google_auth.load_credentials(request.user)
    except CredentialsExpired as exc:
        messages.error(request, str(exc))
        return redirect("google_connect")
    if credentials is None:
        messages.error(request, "Connect your Google account before launching a campaign.")
        return redirect("google_connect")

    try:
        client = WorkspaceClient(credentials)
        assets = client.create_campaign_assets(campaign.role_title, campaign.jd_text)
    except WorkspaceError as exc:
        messages.error(request, f"Could not launch the campaign: {exc}")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    campaign.form_id = assets["form_id"]
    campaign.form_url = assets["form_url"]
    campaign.sheet_id = assets["sheet_id"]
    campaign.sheet_url = assets["sheet_url"]
    campaign.question_ids = assets["question_ids"]
    campaign.status = Campaign.Status.ACTIVE
    campaign.save()

    # Distil the JD into requirements now, so the first sync does not pay for it.
    try:
        ensure_job_spec(campaign, ScreeningPipeline(get_config()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not pre-compute the job spec: %s", exc)

    token = request.POST.get("linkedin_token", "").strip()
    urn = request.POST.get("linkedin_urn", "").strip()
    if token and urn:
        try:
            campaign.linkedin_post_id = post_job(
                token, urn, campaign.role_title, campaign.jd_text, campaign.form_url
            )
            campaign.save(update_fields=["linkedin_post_id", "updated_at"])
            messages.success(request, "Posted to LinkedIn.")
        except LinkedInError as exc:
            messages.warning(request, f"Campaign is live, but the LinkedIn post failed: {exc}")

    messages.success(request, "Campaign is live. Share the application form link.")
    return redirect("campaign_detail", campaign_id=campaign.pk)


# ─────────────────────────────────────────────────────────────
# Background operations
# ─────────────────────────────────────────────────────────────

@login_required
@require_POST
def sync_campaign_view(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    if not campaign.form_id:
        messages.error(request, "This campaign has no application form yet.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    try:
        if google_auth.load_credentials(request.user) is None:
            messages.error(request, "Connect your Google account first.")
            return redirect("google_connect")
    except CredentialsExpired as exc:
        messages.error(request, str(exc))
        return redirect("google_connect")

    jobs.start_job(request.user, campaign, BackgroundJob.Kind.SYNC)
    messages.info(request, "Sync started — screening runs in the background.")
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
@require_POST
def rescore_campaign_view(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    if not campaign.candidates.exists():
        messages.info(request, "There are no candidates to re-score.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    jobs.start_job(request.user, campaign, BackgroundJob.Kind.SCORE)
    messages.info(request, "Re-scoring started.")
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
def job_status(request, campaign_id: int):
    """JSON progress for the page to poll while a job runs."""
    campaign = _own_campaign(request, campaign_id)
    job = (
        BackgroundJob.objects.filter(campaign=campaign).order_by("-created_at").first()
    )
    if job is None:
        return JsonResponse({"status": "idle"})
    return JsonResponse({
        "status": job.status,
        "kind": job.kind,
        "processed": job.processed,
        "total": job.total,
        "percent": job.percent,
        "message": job.message,
        "error": job.error,
        "finished": job.is_finished,
    })


# ─────────────────────────────────────────────────────────────
# Candidates
# ─────────────────────────────────────────────────────────────

@login_required
def candidate_detail(request, campaign_id: int, candidate_id: int):
    campaign = _own_campaign(request, campaign_id)
    candidate = get_object_or_404(Candidate, pk=candidate_id, campaign=campaign)
    score = candidate.get_score()
    return render(request, "hiring_app/candidate.html", {
        "campaign": campaign,
        "candidate": candidate,
        "score": score,
        "profile": candidate.get_profile(),
        "github": candidate.get_github(),
        "job_spec": campaign.get_job_spec(),
    })


@login_required
@require_POST
def rate_candidate(request, campaign_id: int, candidate_id: int):
    """Record the recruiter's own 0-5 rating.

    This is the label the scorer is evaluated against. Collecting it inside the
    normal review flow is what makes a real, non-synthetic evaluation set
    accumulate as a by-product of using the product.
    """
    campaign = _own_campaign(request, campaign_id)
    candidate = get_object_or_404(Candidate, pk=candidate_id, campaign=campaign)

    raw = request.POST.get("rating", "")
    try:
        rating = int(raw)
    except (TypeError, ValueError):
        messages.error(request, "Invalid rating.")
        return redirect("candidate_detail", campaign_id=campaign.pk, candidate_id=candidate.pk)

    if not 0 <= rating <= 5:
        messages.error(request, "Rating must be between 0 and 5.")
        return redirect("candidate_detail", campaign_id=campaign.pk, candidate_id=candidate.pk)

    candidate.recruiter_rating = rating
    candidate.recruiter_note = request.POST.get("note", "").strip()[:2000]
    candidate.save(update_fields=["recruiter_rating", "recruiter_note", "updated_at"])
    messages.success(request, "Rating saved — it will be used to evaluate the scorer.")
    return redirect("candidate_detail", campaign_id=campaign.pk, candidate_id=candidate.pk)


@login_required
@require_POST
def send_invites(request, campaign_id: int):
    campaign = _own_campaign(request, campaign_id)
    selected = request.POST.getlist("candidate_ids")
    when_raw = request.POST.get("interview_date", "")

    if not selected:
        messages.error(request, "Select at least one candidate.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    start = _parse_local_datetime(when_raw)
    if start is None:
        messages.error(request, "Enter a valid interview date and time.")
        return redirect("campaign_detail", campaign_id=campaign.pk)
    if start < timezone.now():
        messages.error(request, "The interview time is in the past.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    try:
        credentials = google_auth.load_credentials(request.user)
        if credentials is None:
            messages.error(request, "Connect your Google account first.")
            return redirect("google_connect")
        client = WorkspaceClient(credentials)
        sender = client.sender_address()
    except (CredentialsExpired, WorkspaceError) as exc:
        messages.error(request, str(exc))
        return redirect("campaign_detail", campaign_id=campaign.pk)

    candidates = list(campaign.candidates.filter(pk__in=selected))
    sent, failed = 0, []
    from datetime import timedelta

    for offset, candidate in enumerate(candidates):
        if not candidate.email:
            failed.append(f"{candidate.full_name or candidate.pk}: no email address")
            continue
        slot = start + timedelta(minutes=offset * INTERVIEW_SLOT_MINUTES)
        try:
            client.send_invite(
                candidate.email,
                f"Interview invitation — {campaign.role_title}",
                (
                    f"Hello{' ' + candidate.full_name if candidate.full_name else ''},\n\n"
                    f"Thank you for applying for the {campaign.role_title} role. "
                    f"We would like to invite you to an interview.\n\n"
                    f"The calendar invitation is attached. If the time does not suit "
                    f"you, reply to this email and we will rearrange.\n\n"
                    f"Best regards,\nThe hiring team"
                ),
                organiser_name=request.user.get_full_name() or request.user.username,
                organiser_email=sender,
                role=campaign.role_title,
                start=slot,
                duration_minutes=INTERVIEW_SLOT_MINUTES,
            )
        except WorkspaceError as exc:
            failed.append(f"{candidate.email}: {exc}")
            continue

        candidate.status = Candidate.Status.INVITED
        candidate.interview_at = slot
        candidate.save(update_fields=["status", "interview_at", "updated_at"])
        sent += 1

    if sent:
        messages.success(request, f"Sent {sent} interview invitation(s).")
    for problem in failed[:5]:
        messages.error(request, f"Invitation failed — {problem}")
    return redirect("campaign_detail", campaign_id=campaign.pk)


@login_required
@require_POST
def send_outcomes(request, campaign_id: int):
    """Send offer and rejection emails.

    Only candidates explicitly listed in ``decision_scope`` are contacted. The
    old implementation emailed *every* candidate in the campaign whenever this
    ran, so a second click mailed everyone twice, including people already
    rejected.
    """
    campaign = _own_campaign(request, campaign_id)
    scope_ids = request.POST.getlist("decision_scope")
    hired_ids = set(request.POST.getlist("hired_ids"))

    if not scope_ids:
        messages.error(request, "Select which candidates to notify.")
        return redirect("campaign_detail", campaign_id=campaign.pk)

    try:
        credentials = google_auth.load_credentials(request.user)
        if credentials is None:
            messages.error(request, "Connect your Google account first.")
            return redirect("google_connect")
        client = WorkspaceClient(credentials)
    except (CredentialsExpired, WorkspaceError) as exc:
        messages.error(request, str(exc))
        return redirect("campaign_detail", campaign_id=campaign.pk)

    candidates = list(campaign.candidates.filter(pk__in=scope_ids))
    rows, sent, failed = [], 0, []

    for candidate in candidates:
        if not candidate.email:
            failed.append(f"{candidate.full_name or candidate.pk}: no email address")
            continue

        hired = str(candidate.pk) in hired_ids
        if hired:
            subject = f"Your application for {campaign.role_title}"
            body = (
                f"Hello{' ' + candidate.full_name if candidate.full_name else ''},\n\n"
                f"We are delighted to move forward with your application for the "
                f"{campaign.role_title} role. Someone from our team will be in touch "
                f"shortly with the details.\n\nCongratulations, and welcome aboard.\n\n"
                f"Best regards,\nThe hiring team"
            )
            new_status = Candidate.Status.HIRED
        else:
            subject = f"Update on your application for {campaign.role_title}"
            body = (
                f"Hello{' ' + candidate.full_name if candidate.full_name else ''},\n\n"
                f"Thank you for taking the time to apply for the {campaign.role_title} "
                f"role and for sharing your experience with us. After careful "
                f"consideration we have decided to progress other candidates on this "
                f"occasion.\n\nWe genuinely appreciate your interest and wish you well "
                f"with your search.\n\nBest regards,\nThe hiring team"
            )
            new_status = Candidate.Status.REJECTED

        try:
            client.send_email(candidate.email, subject, body)
        except WorkspaceError as exc:
            failed.append(f"{candidate.email}: {exc}")
            continue

        candidate.status = new_status
        candidate.save(update_fields=["status", "updated_at"])
        rows.append([
            timezone.now().isoformat(),
            candidate.email,
            "OFFER" if hired else "REJECTED",
            round(candidate.overall_score, 1),
        ])
        sent += 1

    if rows and campaign.sheet_id:
        try:
            client.log_outcomes(campaign.sheet_id, rows)
        except WorkspaceError as exc:
            messages.warning(request, f"Emails sent, but the sheet log failed: {exc}")

    if sent:
        messages.success(request, f"Sent {sent} decision email(s).")
    for problem in failed[:5]:
        messages.error(request, f"Email failed — {problem}")

    if not campaign.candidates.filter(status__in=[
        Candidate.Status.NEW, Candidate.Status.SCORED, Candidate.Status.INVITED
    ]).exists():
        campaign.status = Campaign.Status.CLOSED
        campaign.save(update_fields=["status", "updated_at"])

    return redirect("campaign_detail", campaign_id=campaign.pk)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _split(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").replace("\n", ",").split(",") if part.strip()]


def _parse_local_datetime(raw: str):
    """Parse a ``datetime-local`` value into an aware datetime.

    ``<input type="datetime-local">`` submits a naive wall-clock value. It is
    interpreted in the configured ``TIME_ZONE`` and made timezone-aware here, at
    the boundary — the calendar layer refuses naive datetimes precisely so this
    conversion cannot be forgotten.
    """
    from datetime import datetime

    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(raw, fmt)
        except (TypeError, ValueError):
            continue
        return timezone.make_aware(naive, timezone.get_current_timezone())
    return None
