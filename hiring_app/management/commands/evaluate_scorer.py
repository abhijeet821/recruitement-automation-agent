"""
Benchmark the scorer against a labelled set.

    python manage.py evaluate_scorer
    python manage.py evaluate_scorer --dataset path/to/set.json --json results.json
    python manage.py evaluate_scorer --campaign 3      # use real recruiter labels

This is the command that turns "the AI ranks candidates" into a defensible
claim. It runs the production scorer and every baseline over the same labelled
data and prints the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from matching.config import get_config
from matching.evaluation import compare, load_dataset
from matching.evaluation.dataset import EvaluationSet, LabelledCandidate


class Command(BaseCommand):
    help = "Compare the scorer against baselines on a labelled evaluation set."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", help="Path to a labelled JSON set")
        parser.add_argument(
            "--campaign", type=int,
            help="Build the set from a campaign's recruiter ratings instead",
        )
        parser.add_argument("--json", help="Write the full report to this path")
        parser.add_argument(
            "--no-github", action="store_true",
            help="Skip GitHub enrichment (faster, avoids API rate limits)",
        )

    def handle(self, *args, **options):
        if options["campaign"]:
            dataset = _dataset_from_campaign(options["campaign"])
        else:
            try:
                dataset = load_dataset(options["dataset"])
            except (FileNotFoundError, ValueError) as exc:
                raise CommandError(str(exc)) from exc

        config = get_config()
        self.stdout.write(self.style.MIGRATE_HEADING("Evaluation set"))
        self.stdout.write(f"  {dataset.summary()}")
        self.stdout.write(f"  Role: {dataset.job.role_title}")
        self.stdout.write(f"  Provider: {config.provider} / {config.ollama_model}\n")

        report = compare(dataset, with_github=not options["no_github"], progress=True)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Results"))
        self.stdout.write(report.as_table())

        best = report.best("spearman")
        baseline = next((r for r in report.results if r.scorer == "keyword_legacy"), None)
        if best and baseline and baseline.spearman == baseline.spearman:
            delta = best.spearman - baseline.spearman
            self.stdout.write("")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Best: {best.scorer} (Spearman {best.spearman:.3f}), "
                    f"{delta:+.3f} against the legacy keyword scorer."
                )
            )

        notes = {note for r in report.results for note in r.notes}
        if notes:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Caveats"))
            for note in sorted(notes):
                self.stdout.write(f"  - {note}")

        if options["json"]:
            Path(options["json"]).write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            self.stdout.write(f"\nFull report written to {options['json']}")


def _dataset_from_campaign(campaign_id: int) -> EvaluationSet:
    """Build an evaluation set from a campaign's recruiter ratings.

    This is the point of collecting ratings in the UI: once a recruiter has
    reviewed real applicants, the scorer can be validated against their
    judgement instead of synthetic labels.
    """
    from hiring_app.models import Campaign

    try:
        campaign = Campaign.objects.get(pk=campaign_id)
    except Campaign.DoesNotExist as exc:
        raise CommandError(f"No campaign with id {campaign_id}") from exc

    rated = campaign.candidates.exclude(recruiter_rating__isnull=True)
    if rated.count() < 5:
        raise CommandError(
            f"Campaign {campaign_id} has only {rated.count()} rated candidates. "
            "Rate at least 5 (ideally 30+) before the metrics mean anything."
        )

    return EvaluationSet(
        name=f"campaign_{campaign_id}_{campaign.role_title}",
        job=campaign.get_job_spec(),
        candidates=[
            LabelledCandidate(
                id=str(c.pk),
                label=float(c.recruiter_rating),
                resume_text=c.resume_text,
                github_username=c.github_username,
                email=c.email,
            )
            for c in rated
        ],
    )
