"""
Relational state for the hiring app.

This replaces the previous design, where campaign state lived in JSON files
under ``BASE_DIR/campaigns/user_{id}/``. That layout had three defects that a
database fixes outright:

* On an ephemeral filesystem (Railway, Heroku, any container) every redeploy
  silently destroyed all candidate data.
* ``save_state()`` was a read-modify-write with no locking, so two gunicorn
  workers handling concurrent requests could lose each other's updates.
* Nothing could be queried. Sorting candidates by score, filtering by status or
  aggregating across campaigns all meant loading and scanning every file.

Structured artefacts produced by the matching engine (``ResumeProfile``,
``CandidateScore``, ``JobSpec``) are stored in ``JSONField`` columns rather than
being shredded into relational tables. They are read as whole documents, they
evolve with the engine's schema, and their dataclasses already round-trip
through dicts. The few values needed for sorting and filtering are denormalised
into indexed columns alongside.
"""

from __future__ import annotations

from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from hiring_app.crypto import decrypt, encrypt
from matching.schemas import CandidateScore, GitHubProfile, JobSpec, ResumeProfile


class GoogleOAuthToken(models.Model):
    """Per-user Google credentials, encrypted at rest."""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="google_token")
    encrypted_token = models.TextField(
        help_text="Fernet-encrypted Google OAuth2 credentials JSON"
    )
    scopes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_refreshed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "Google OAuth token"

    def __str__(self) -> str:
        return f"GoogleOAuthToken({self.user.username})"

    # The plaintext is exposed only through this property, so no call site can
    # accidentally read or write the raw column.
    @property
    def token_json(self) -> str:
        return decrypt(self.encrypted_token)

    @token_json.setter
    def token_json(self, value: str) -> None:
        self.encrypted_token = encrypt(value or "")


class Campaign(models.Model):
    """One hiring campaign: a role, its JD, and the Google artefacts behind it."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        CLOSED = "closed", "Closed"

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="campaigns")
    role_title = models.CharField(max_length=200)
    experience_required = models.CharField(max_length=100, blank=True, default="")
    location = models.CharField(max_length=200, blank=True, default="")
    work_model = models.CharField(max_length=50, blank=True, default="")
    salary_range = models.CharField(max_length=120, blank=True, default="")

    jd_text = models.TextField(blank=True, default="")
    # JDQualityReport.to_dict() — section/keyword coverage and inclusivity flags.
    jd_quality = models.JSONField(default=dict, blank=True)
    # JobSpec.to_dict() — distilled once per campaign, reused for every candidate.
    job_spec = models.JSONField(default=dict, blank=True)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    form_id = models.CharField(max_length=120, blank=True, default="")
    form_url = models.URLField(blank=True, default="")
    sheet_id = models.CharField(max_length=120, blank=True, default="")
    sheet_url = models.URLField(blank=True, default="")
    # Google Forms answers are keyed by opaque question IDs, so the mapping from
    # ID to meaning must be captured when the form is created — it cannot be
    # recovered from the question titles later.
    question_ids = models.JSONField(default=dict, blank=True)

    linkedin_post_id = models.CharField(max_length=200, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_synced_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "-created_at"]),
            models.Index(fields=["owner", "status"]),
        ]

    def __str__(self) -> str:
        return f"{self.role_title} ({self.status})"

    def get_job_spec(self) -> JobSpec:
        spec = JobSpec.from_dict(self.job_spec)
        if not spec.role_title:
            spec.role_title = self.role_title
        if not spec.raw_jd:
            spec.raw_jd = self.jd_text
        return spec

    def set_job_spec(self, spec: JobSpec) -> None:
        self.job_spec = spec.to_dict()

    @property
    def is_live(self) -> bool:
        return self.status == self.Status.ACTIVE and bool(self.form_url)

    @property
    def candidate_count(self) -> int:
        return self.candidates.count()


class Candidate(models.Model):
    """One applicant, their parsed artefacts, and their score."""

    class Status(models.TextChoices):
        NEW = "new", "New"
        SCORED = "scored", "Scored"
        INVITED = "invited", "Interview invited"
        HIRED = "hired", "Offer sent"
        REJECTED = "rejected", "Rejected"

    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name="candidates")
    response_id = models.CharField(max_length=200)

    email = models.EmailField(blank=True, default="")
    full_name = models.CharField(max_length=200, blank=True, default="")
    github_username = models.CharField(max_length=120, blank=True, default="")

    resume_drive_link = models.URLField(blank=True, default="")
    resume_file_id = models.CharField(max_length=200, blank=True, default="")
    resume_path = models.CharField(max_length=500, blank=True, default="")
    resume_text = models.TextField(blank=True, default="")

    # Structured artefacts from the matching engine.
    profile = models.JSONField(default=dict, blank=True)          # ResumeProfile
    github_profile = models.JSONField(default=dict, blank=True)   # GitHubProfile
    score_detail = models.JSONField(default=dict, blank=True)     # CandidateScore

    # Denormalised for ordering and filtering — the JSON blobs above are the
    # source of truth, these exist so the database can sort without loading them.
    overall_score = models.FloatField(default=0.0, db_index=True)
    confidence = models.FloatField(default=0.0)
    recommendation = models.CharField(max_length=20, blank=True, default="")

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.NEW)
    interview_at = models.DateTimeField(null=True, blank=True)

    # Closes the evaluation loop: a recruiter's own 0-5 rating is the label the
    # scorer is measured against. Exported by `manage.py export_labels` into an
    # evaluation set, which is how the weights get validated on real data
    # instead of the synthetic starter set.
    recruiter_rating = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(5)],
        help_text="Recruiter's own 0-5 relevance rating, used to evaluate the scorer",
    )
    recruiter_note = models.TextField(blank=True, default="")

    warnings = models.JSONField(default=list, blank=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    scored_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-overall_score", "-created_at"]
        constraints = [
            # Idempotent sync: re-importing a Google Forms response must never
            # create a duplicate candidate.
            models.UniqueConstraint(
                fields=["campaign", "response_id"], name="unique_response_per_campaign"
            )
        ]
        indexes = [
            models.Index(fields=["campaign", "-overall_score"]),
            models.Index(fields=["campaign", "status"]),
        ]

    def __str__(self) -> str:
        return f"{self.email or self.full_name or self.response_id} ({self.overall_score:.0f})"

    # ── typed accessors ──────────────────────────────────────

    def get_profile(self) -> ResumeProfile:
        return ResumeProfile.from_dict(self.profile)

    def get_github(self) -> GitHubProfile | None:
        return GitHubProfile.from_dict(self.github_profile) if self.github_profile else None

    def get_score(self) -> CandidateScore:
        return CandidateScore.from_dict(self.score_detail)

    def apply_score(self, score: CandidateScore) -> None:
        """Persist a score into both the JSON blob and the indexed columns."""
        self.score_detail = score.to_dict()
        self.overall_score = score.overall
        self.confidence = score.confidence
        self.recommendation = score.recommendation
        self.scored_at = timezone.now()
        if self.status == self.Status.NEW:
            self.status = self.Status.SCORED

    @property
    def is_scored(self) -> bool:
        return self.scored_at is not None

    @property
    def score_percent(self) -> int:
        return int(round(self.overall_score))

    @property
    def confidence_percent(self) -> int:
        return int(round(self.confidence * 100))

    @property
    def needs_review(self) -> bool:
        """Low confidence means "we could not tell", not "they are weak"."""
        return self.confidence < 0.5


class BackgroundJob(models.Model):
    """Progress record for work that runs outside the request cycle.

    Syncing a campaign downloads and parses every new resume and makes several
    model calls per candidate — minutes of work, against a 120-second gunicorn
    timeout. Running it in the request was the old design's most reliable way to
    produce a 502. Now a view enqueues a job and returns immediately; this row
    is what the UI polls for progress.
    """

    class Kind(models.TextChoices):
        SYNC = "sync", "Sync responses"
        SCORE = "score", "Re-score candidates"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        DONE = "done", "Completed"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="jobs")
    campaign = models.ForeignKey(
        Campaign, on_delete=models.CASCADE, related_name="jobs", null=True, blank=True
    )
    kind = models.CharField(max_length=20, choices=Kind.choices)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)

    total = models.IntegerField(default=0)
    processed = models.IntegerField(default=0)
    message = models.CharField(max_length=500, blank=True, default="")
    error = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["campaign", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.kind}:{self.status} ({self.processed}/{self.total})"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.DONE, self.Status.FAILED)

    @property
    def percent(self) -> int:
        if not self.total:
            return 0 if self.status == self.Status.PENDING else 100
        return int(round(100 * self.processed / self.total))
