"""
Background job runner.

Screening is slow: downloading a PDF, extracting it, three model calls per
candidate. Forty applicants is comfortably twenty minutes of work against a
120-second gunicorn timeout. The previous design ran all of it inside the
request handler, which meant a large campaign could not be synced at all.

Here a view creates a ``BackgroundJob`` row, hands the work to a small thread
pool and returns immediately; the page polls the job for progress.

**This is deliberately a thread pool and not Celery.** Celery is the right
answer at scale, but it needs a broker (Redis) and a second process, which is
real operational weight for a system whose bottleneck is a single local model
that can only serve one request at a time anyway. A bounded pool with
database-backed progress gives the same user-visible behaviour with no extra
infrastructure. The trade-offs are accepted knowingly and are not hidden:

* Jobs live in one process. Two gunicorn workers each get their own pool, so
  ``MAX_WORKERS`` is per process, and a restart marks running jobs as failed
  (``reap_stale_jobs``) rather than leaving them stuck at "running" forever.
* There is no retry or scheduling.

``run_job`` is written so that swapping in Celery is a change to this file
alone: make ``enqueue`` a ``.delay()`` call and the rest of the app is unchanged.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from django.db import close_old_connections
from django.utils import timezone

from hiring_app.models import BackgroundJob, Campaign

logger = logging.getLogger("hiring_app")

# One local model serves one request at a time, so extra concurrency here buys
# nothing and would only multiply memory pressure.
MAX_WORKERS = 2

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=MAX_WORKERS, thread_name_prefix="hireai-job"
            )
        return _executor


def enqueue(job: BackgroundJob) -> None:
    """Schedule a job for background execution."""
    _pool().submit(_run, job.pk)
    logger.info("Enqueued %s job %s", job.kind, job.pk)


def _run(job_id: int) -> None:
    """Execute a job. Runs in a worker thread; must never raise."""
    # Each thread gets its own database connection; stale ones are closed at
    # both ends so a long job cannot exhaust the connection pool.
    close_old_connections()
    try:
        job = BackgroundJob.objects.select_related("campaign", "owner").get(pk=job_id)
    except BackgroundJob.DoesNotExist:
        logger.warning("Job %s vanished before it could run", job_id)
        return

    job.status = BackgroundJob.Status.RUNNING
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])

    def progress(done: int, total: int, message: str) -> None:
        BackgroundJob.objects.filter(pk=job.pk).update(
            processed=done, total=total, message=message[:500]
        )

    try:
        if job.kind == BackgroundJob.Kind.SYNC:
            message = _run_sync(job, progress)
        elif job.kind == BackgroundJob.Kind.SCORE:
            message = _run_rescore(job, progress)
        else:
            raise ValueError(f"Unknown job kind: {job.kind}")
    except Exception as exc:  # noqa: BLE001 - the failure belongs in the row, not the log alone
        logger.exception("Job %s failed", job.pk)
        BackgroundJob.objects.filter(pk=job.pk).update(
            status=BackgroundJob.Status.FAILED,
            error=str(exc)[:2000],
            message="Failed",
            finished_at=timezone.now(),
        )
    else:
        BackgroundJob.objects.filter(pk=job.pk).update(
            status=BackgroundJob.Status.DONE,
            message=message[:500],
            finished_at=timezone.now(),
        )
    finally:
        close_old_connections()


def _run_sync(job: BackgroundJob, progress) -> str:
    from hiring_app.services.google_auth import load_credentials
    from hiring_app.services.google_workspace import WorkspaceClient
    from hiring_app.services.screening import sync_campaign

    credentials = load_credentials(job.owner)
    if credentials is None:
        raise RuntimeError("No Google account is connected.")

    client = WorkspaceClient(credentials)
    outcome = sync_campaign(job.campaign, client, progress=progress)

    if outcome.errors:
        # Partial success is reported, not swallowed: the job succeeds but the
        # per-candidate failures stay visible in the UI.
        BackgroundJob.objects.filter(pk=job.pk).update(
            error="\n".join(outcome.errors[:20])[:2000]
        )
    return outcome.summary()


def _run_rescore(job: BackgroundJob, progress) -> str:
    from hiring_app.services.screening import rescore_campaign

    outcome = rescore_campaign(job.campaign, progress=progress)
    if outcome.errors:
        BackgroundJob.objects.filter(pk=job.pk).update(
            error="\n".join(outcome.errors[:20])[:2000]
        )
    return outcome.summary()


_reaped = False


def _reap_once() -> None:
    """Run the stale-job sweep at most once per process, on first use.

    Doing this lazily rather than in ``AppConfig.ready()`` avoids querying the
    database before the app registry is ready, which breaks ``migrate`` on a
    fresh database.
    """
    global _reaped
    if _reaped:
        return
    with _lock:
        if _reaped:
            return
        _reaped = True
    try:
        reap_stale_jobs()
    except Exception as exc:  # noqa: BLE001 - never block a real request over cleanup
        logger.debug("Stale job cleanup skipped: %s", exc)


def start_job(user, campaign: Campaign, kind: str) -> BackgroundJob:
    """Create and enqueue a job, reusing one that is already in flight.

    Guards against a recruiter clicking "Sync" repeatedly and queueing several
    expensive duplicate runs.
    """
    _reap_once()
    existing = BackgroundJob.objects.filter(
        campaign=campaign,
        kind=kind,
        status__in=[BackgroundJob.Status.PENDING, BackgroundJob.Status.RUNNING],
    ).first()
    if existing:
        return existing

    job = BackgroundJob.objects.create(owner=user, campaign=campaign, kind=kind)
    enqueue(job)
    return job


def active_job(campaign: Campaign) -> BackgroundJob | None:
    _reap_once()
    return BackgroundJob.objects.filter(
        campaign=campaign,
        status__in=[BackgroundJob.Status.PENDING, BackgroundJob.Status.RUNNING],
    ).first()


def reap_stale_jobs() -> int:
    """Mark jobs orphaned by a process restart as failed.

    Without this a job that was running when the server stopped would show
    "running" forever, and ``start_job`` would refuse to start a replacement.
    Called once at application startup.
    """
    stale = BackgroundJob.objects.filter(
        status__in=[BackgroundJob.Status.PENDING, BackgroundJob.Status.RUNNING]
    )
    count = stale.count()
    if count:
        stale.update(
            status=BackgroundJob.Status.FAILED,
            error="Interrupted by a server restart. Run it again.",
            message="Interrupted",
            finished_at=timezone.now(),
        )
        logger.info("Reaped %d stale background job(s)", count)
    return count
