"""
Measure what each scoring dimension actually contributes.

    python manage.py ablate_scorer
    python manage.py ablate_scorer --campaign 2 --json ablation.json

Drops each dimension in turn, redistributes its weight, and re-measures rank
agreement with the human labels. Answers the question the hand-tuned weights
otherwise leave open: is this dimension earning its place, or is it decoration?
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from matching.config import get_config
from matching.evaluation import load_dataset, prepare
from matching.evaluation.ablation import run_ablation
from matching.scoring.ensemble import EnsembleScorer


class Command(BaseCommand):
    help = "Ablate each scoring dimension and report its contribution."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", help="Path to a labelled JSON set")
        parser.add_argument(
            "--campaign", type=int,
            help="Use a campaign's recruiter ratings instead of a JSON set",
        )
        parser.add_argument("--json", help="Write the full report here")
        parser.add_argument("--no-github", action="store_true")

    def handle(self, *args, **options):
        # A campaign's candidates already carry persisted scores, and ablation is
        # arithmetic over those — so this path needs no model calls at all.
        if options["campaign"]:
            self._from_campaign(options)
            return

        try:
            dataset = load_dataset(options["dataset"])
        except (FileNotFoundError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        config = get_config()
        self.stdout.write(f"{dataset.summary()}")
        self.stdout.write(
            "Scoring once, then recomputing each ablation arithmetically "
            "(re-scoring per dimension would take hours).\n"
        )

        prepared = prepare(
            dataset, config, with_github=not options["no_github"], progress=True
        )
        scorer = EnsembleScorer(config=config)

        self.stdout.write("Scoring…")
        scores = []
        for index, item in enumerate(prepared, start=1):
            self.stdout.write(f"  [{index}/{len(prepared)}] {item.source.id}")
            scores.append(scorer.score(item.resume, dataset.job, item.github))

        report = run_ablation(scores, [p.source.label for p in prepared])
        self.stdout.write("")
        self._render(report)
        self._write(report, options)

    def _from_campaign(self, options) -> None:
        """Ablate from already-persisted scores — instant, no model calls."""
        from hiring_app.models import Campaign

        try:
            campaign = Campaign.objects.get(pk=options["campaign"])
        except Campaign.DoesNotExist as exc:
            raise CommandError(f"No campaign with id {options['campaign']}") from exc

        rated = list(
            campaign.candidates
            .exclude(recruiter_rating__isnull=True)
            .exclude(scored_at__isnull=True)
        )
        if len(rated) < 5:
            raise CommandError(
                f"Campaign {campaign.pk} has {len(rated)} rated and scored candidates; "
                f"at least 5 are needed."
            )

        self.stdout.write(
            f"Campaign {campaign.pk}: {campaign.role_title} — "
            f"{len(rated)} rated candidates (using stored scores, no re-scoring)\n"
        )

        report = run_ablation(
            [c.get_score() for c in rated],
            [float(c.recruiter_rating) for c in rated],
        )
        self._render(report)
        self._write(report, options)

    def _render(self, report) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("Dimension ablation"))
        self.stdout.write(report.as_table())
        self.stdout.write("")
        self.stdout.write(
            "  Δ is the change in Spearman when that dimension is removed.\n"
            "  Negative means removing it hurt, i.e. it was carrying signal."
        )
        if report.notes:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Caveats"))
            for note in report.notes:
                self.stdout.write(f"  - {note}")

    def _write(self, report, options) -> None:
        if options["json"]:
            Path(options["json"]).write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            self.stdout.write(f"\nWritten to {options['json']}")
