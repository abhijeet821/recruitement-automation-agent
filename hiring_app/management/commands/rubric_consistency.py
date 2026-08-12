"""
Measure how reproducible the LLM rubric is.

    python manage.py rubric_consistency --candidate 15 --runs 5
    python manage.py rubric_consistency --runs 7 --json consistency.json

Runs the rubric repeatedly on the same resume and reports the spread. The rubric
carries 0.17 of the final score and is the only component that can answer the
same question differently twice — this quantifies what that costs.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from matching.config import get_config
from matching.evaluation.consistency import measure_rubric_consistency
from matching.evaluation.dataset import load_dataset
from matching.llm import get_provider
from matching.parsing.resume import parse_resume


class Command(BaseCommand):
    help = "Measure run-to-run variance of the LLM rubric."

    def add_arguments(self, parser):
        parser.add_argument("--candidate", type=int, help="Score an existing candidate by id")
        parser.add_argument("--dataset", help="Otherwise, use a labelled set")
        parser.add_argument("--runs", type=int, default=5)
        parser.add_argument(
            "--temperature", type=float, default=0.1,
            help="Sampling temperature. The default matches production; raise it "
                 "to show how much of the stability is the configuration.",
        )
        parser.add_argument("--json", help="Write the report here")

    def handle(self, *args, **options):
        runs = max(2, options["runs"])
        config = get_config()
        provider = get_provider(config)

        resume, job, github = self._subject(options, provider, config)

        self.stdout.write(
            f"Running the rubric {runs}x on the same resume "
            f"({config.provider}/{config.ollama_model}, temperature {options['temperature']})."
        )
        self.stdout.write(self.style.WARNING(
            f"Expect roughly {runs * 25}s.\n"
        ))

        def progress(index, total, overall):
            got = f"overall {overall:.3f}" if overall is not None else "FAILED"
            self.stdout.write(f"  run {index}/{total}: {got}")

        report = measure_rubric_consistency(
            resume, job, provider, runs=runs, github=github,
            temperature=options['temperature'], on_run=progress
        )

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Rubric spread — {report.candidate}"))
        self.stdout.write(report.as_table())

        self.stdout.write("")
        style = self.style.SUCCESS if report.overall.stdev < 0.10 else self.style.WARNING
        self.stdout.write(style(f"Verdict: {report.verdict}"))
        for note in report.notes:
            self.stdout.write(f"  - {note}")

        if options.get("json"):
            Path(options["json"]).write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )

    def _subject(self, options, provider, config):
        """Resolve which resume to test, preferring a real stored candidate."""
        if options["candidate"]:
            from hiring_app.models import Candidate

            try:
                candidate = Candidate.objects.select_related("campaign").get(
                    pk=options["candidate"]
                )
            except Candidate.DoesNotExist as exc:
                raise CommandError(f"No candidate with id {options['candidate']}") from exc
            self.stdout.write(f"Subject: candidate {candidate.pk} — {candidate.email}")
            return (
                candidate.get_profile(),
                candidate.campaign.get_job_spec(),
                candidate.get_github(),
            )

        dataset = load_dataset(options["dataset"])
        source = dataset.candidates[0]
        self.stdout.write(f"Subject: {dataset.name} — {source.id}")
        resume = parse_resume(source.resume_text, provider, fallback_email=source.email)
        return resume, dataset.job, None
