"""
Load the labelled evaluation set as a fully scored campaign.

    python manage.py seed_demo
    python manage.py seed_demo --fast          # skip the LLM rubric (~3x quicker)
    python manage.py seed_demo --reset         # rebuild from scratch

Why this exists: the screening half of the product needs Google Workspace
credentials to get candidates in, which makes it awkward to show. This command
puts the same 14 labelled candidates into a campaign directly, so the candidate
table, score breakdowns, evidence panels and the recruiter-rating flow are all
usable with nothing but a local model running.

The recruiter ratings are seeded from the golden-set labels, which means
``evaluate_scorer --campaign <id>`` works immediately and demonstrates the
feedback loop end to end. Those labels are synthetic — the command says so, and
so does the campaign it creates.
"""

from __future__ import annotations

import time

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.utils.crypto import get_random_string

from hiring_app.models import Campaign, Candidate
from matching.config import get_config
from matching.evaluation.dataset import load_dataset
from matching.generation.jd import analyse_jd
from matching.pipeline import ScreeningPipeline

DEMO_MARKER = "seeded-demo"


class Command(BaseCommand):
    help = "Load the labelled golden set as a scored demo campaign."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            help="Username to own the campaign (default: first existing user, "
                 "or a new 'demo' account)",
        )
        parser.add_argument("--dataset", help="Path to a labelled JSON set")
        parser.add_argument(
            "--fast", action="store_true",
            help="Skip the LLM rubric pass — roughly 3x quicker, measurably less accurate",
        )
        parser.add_argument(
            "--with-github", action="store_true",
            help="Attempt GitHub enrichment (the sample resumes use placeholder "
                 "handles, so this will mostly 404 and burn rate limit)",
        )
        parser.add_argument(
            "--reset", action="store_true",
            help="Delete any existing demo campaign first",
        )

    def handle(self, *args, **options):
        try:
            dataset = load_dataset(options["dataset"])
        except (FileNotFoundError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        user = self._get_user(options["user"])
        config = get_config()
        if options["fast"]:
            config = config.with_overrides(rubric_enabled=False)

        pipeline = ScreeningPipeline(config)
        healthy, detail = pipeline.provider.health()
        if not healthy:
            raise CommandError(
                f"The scoring engine is not reachable ({detail}). "
                f"Run `python manage.py doctor` first."
            )

        campaign = self._get_campaign(user, dataset, reset=options["reset"])

        pending = [
            c for c in dataset.candidates
            if not Candidate.objects.filter(
                campaign=campaign, response_id=f"demo-{c.id}"
            ).exists()
        ]

        if not pending:
            self.stdout.write(self.style.WARNING(
                "Every candidate is already loaded. Use --reset to rebuild."
            ))
            self._report(campaign)
            return

        estimate = len(pending) * (12 if options["fast"] else 30)
        self.stdout.write(
            f"Screening {len(pending)} candidate(s) with "
            f"{config.provider}/{config.ollama_model}."
        )
        self.stdout.write(self.style.WARNING(
            f"This runs the real pipeline — expect roughly "
            f"{estimate // 60}m {estimate % 60}s."
        ))

        job_spec = campaign.get_job_spec()
        started = time.time()

        # Only draw an overwriting progress line on a real terminal; piped into
        # a file or a log, carriage returns just produce noise.
        interactive = bool(getattr(self.stdout, "isatty", lambda: False)())

        for index, source in enumerate(pending, start=1):
            label = f"[{index}/{len(pending)}] {source.id}"
            if interactive:
                self.stdout.write(f"  {label} screening…".ljust(60), ending="\r")
                self.stdout.flush()

            result = pipeline.screen_text(
                source.resume_text,
                job_spec,
                email=source.email,
                github_username=source.github_username,
                with_github=options["with_github"],
            )

            candidate = Candidate(
                campaign=campaign,
                response_id=f"demo-{source.id}",
                email=source.email or result.resume.email,
                full_name=result.resume.full_name,
                github_username=result.resume.github_username,
                resume_text=source.resume_text,
                profile=result.resume.to_dict(),
                github_profile=result.github.to_dict() if result.github else {},
                warnings=result.warnings,
                submitted_at=timezone.now(),
                # Seeded from the golden-set label so the feedback loop and
                # `evaluate_scorer --campaign` are demonstrable immediately.
                recruiter_rating=int(source.label),
                recruiter_note=f"Synthetic label from {dataset.name}. {source.notes}".strip(),
            )
            candidate.apply_score(result.score)
            candidate.save()

            self.stdout.write(
                f"  {label} {candidate.score_percent:>3}/100 "
                f"{candidate.recommendation:<11} (label {int(source.label)})"
            )

        elapsed = time.time() - started
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Loaded {len(pending)} candidate(s) in {elapsed:.0f}s "
            f"({elapsed / max(1, len(pending)):.1f}s each)."
        ))
        self._report(campaign)

    # ── helpers ──────────────────────────────────────────────

    def _get_user(self, username: str | None) -> User:
        if username:
            try:
                return User.objects.get(username=username)
            except User.DoesNotExist as exc:
                raise CommandError(f"No user named '{username}'.") from exc

        existing = User.objects.order_by("id").first()
        if existing:
            self.stdout.write(f"Using existing account: {existing.username}")
            return existing

        # No accounts yet — generate a real random password and print it, rather
        # than silently baking in a guessable default.
        password = get_random_string(14)
        user = User.objects.create_user(
            username="demo", email="demo@example.com", password=password
        )
        self.stdout.write(self.style.SUCCESS(
            f"Created account 'demo' with password: {password}"
        ))
        return user

    def _get_campaign(self, user: User, dataset, *, reset: bool) -> Campaign:
        job = dataset.job

        # Identify the seeded campaign by owner + role + "never launched"
        # (form_id empty). A JSONField `contains` lookup would be neater but is
        # unsupported on SQLite, which is the default development database.
        existing = Campaign.objects.filter(
            owner=user, role_title=job.role_title, form_id=""
        ).first()

        if existing and reset:
            existing.candidates.all().delete()
            existing.delete()
            existing = None

        if existing:
            return existing

        campaign = Campaign.objects.create(
            owner=user,
            role_title=job.role_title,
            experience_required=(
                f"{job.min_years:g}-{job.max_years:g} years"
                if job.max_years else f"{job.min_years:g}+ years"
            ),
            location=job.location,
            work_model=job.employment_type,
            jd_text=job.raw_jd,
            job_spec=job.to_dict(),
            status=Campaign.Status.ACTIVE,
            # Records provenance so a seeded campaign is never mistaken for one
            # with real applicants. form_id stays empty on purpose: the campaign
            # was never launched against Google, and the UI keys off that.
            question_ids={"_source": DEMO_MARKER, "_dataset": dataset.name},
            last_synced_at=timezone.now(),
        )

        campaign.jd_quality = analyse_jd(
            campaign.jd_text,
            role_title=job.role_title,
            must_have=job.must_have_skills,
            nice_to_have=job.nice_to_have_skills,
        ).to_dict()
        campaign.save(update_fields=["jd_quality"])
        return campaign

    def _report(self, campaign: Campaign) -> None:
        candidates = list(campaign.candidates.all())
        strong = sum(1 for c in candidates if c.overall_score >= 75)
        review = sum(1 for c in candidates if c.needs_review)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Demo campaign ready"))
        self.stdout.write(f"  Role:       {campaign.role_title}")
        self.stdout.write(f"  Owner:      {campaign.owner.username}")
        self.stdout.write(f"  Candidates: {len(candidates)} ({strong} strong, {review} need review)")
        self.stdout.write(f"  URL:        http://127.0.0.1:8000/campaign/{campaign.pk}/")
        self.stdout.write("")
        self.stdout.write(
            f"  Benchmark the scorer against these labels:\n"
            f"    python manage.py evaluate_scorer --campaign {campaign.pk}"
        )
