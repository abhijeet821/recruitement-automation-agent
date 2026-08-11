"""
Run an adverse-impact audit over a campaign's outcomes.

    python manage.py audit_fairness --campaign 3 --groups groups.csv

Group labels are supplied from a file of ``candidate_email,group`` rows and must
come from voluntary self-identification. The system never infers demographics
from a resume: name-based ethnicity inference is both inaccurate and itself a
discriminatory act, and doing it to "check for bias" would introduce the exact
harm it claims to detect.

Without a labels file the command still reports the score distribution and the
selection rate overall, which is useful on its own for spotting a threshold that
is quietly cutting almost everyone.
"""

from __future__ import annotations

import csv
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from hiring_app.models import Campaign, Candidate
from matching.fairness import adverse_impact, score_gap


class Command(BaseCommand):
    help = "Audit a campaign's screening outcomes for adverse impact."

    def add_arguments(self, parser):
        parser.add_argument("--campaign", type=int, required=True)
        parser.add_argument(
            "--groups", help="CSV of 'email,group' rows from voluntary self-identification"
        )
        parser.add_argument(
            "--threshold", type=float, default=60.0,
            help="Score at or above which a candidate counts as selected (default 60)",
        )

    def handle(self, *args, **options):
        try:
            campaign = Campaign.objects.get(pk=options["campaign"])
        except Campaign.DoesNotExist as exc:
            raise CommandError(f"No campaign with id {options['campaign']}") from exc

        candidates = list(campaign.candidates.all())
        if not candidates:
            raise CommandError("This campaign has no candidates.")

        threshold = options["threshold"]
        self.stdout.write(self.style.MIGRATE_HEADING(f"Campaign: {campaign.role_title}"))
        self.stdout.write(f"  Candidates: {len(candidates)}")
        self.stdout.write(f"  Selection threshold: score >= {threshold}\n")

        # ── overall distribution ─────────────────────────────
        selected = [c for c in candidates if c.overall_score >= threshold]
        invited = [c for c in candidates if c.status == Candidate.Status.INVITED]
        self.stdout.write(self.style.MIGRATE_HEADING("Outcomes"))
        self.stdout.write(
            f"  Above threshold: {len(selected)}/{len(candidates)} "
            f"({100 * len(selected) / len(candidates):.0f}%)"
        )
        self.stdout.write(f"  Actually invited: {len(invited)}")

        low_confidence = [c for c in candidates if c.needs_review]
        if low_confidence:
            self.stdout.write(self.style.WARNING(
                f"  {len(low_confidence)} candidate(s) scored with low confidence — "
                f"their scores reflect thin evidence, not weak profiles."
            ))

        # ── group audit ──────────────────────────────────────
        if not options["groups"]:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                "No --groups file supplied, so no adverse-impact ratios were computed.\n"
                "Group data must come from voluntary self-identification; this tool "
                "will not infer it from names or resumes."
            ))
            return

        groups = _load_groups(Path(options["groups"]))
        by_email = {c.email.lower(): c for c in candidates if c.email}

        outcomes, scores, unmatched = [], [], 0
        for email, group in groups.items():
            candidate = by_email.get(email.lower())
            if candidate is None:
                unmatched += 1
                continue
            outcomes.append((group, candidate.overall_score >= threshold))
            scores.append((group, candidate.overall_score))

        if unmatched:
            self.stdout.write(self.style.WARNING(
                f"\n  {unmatched} row(s) in the groups file matched no candidate."
            ))
        if not outcomes:
            raise CommandError("No group rows matched any candidate in this campaign.")

        report = adverse_impact(outcomes)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Selection rates by group"))
        for group in report.groups:
            ratio = report.impact_ratios.get(group.group)
            ratio_text = f"impact ratio {ratio:.2f}" if ratio is not None else "reference/excluded"
            self.stdout.write(
                f"  {group.group:<20} {group.selected:>3}/{group.total:<3} "
                f"= {group.selection_rate:.1%}   {ratio_text}"
            )

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Score distribution by group"))
        for group, stats in sorted(score_gap(scores).items()):
            self.stdout.write(
                f"  {group:<20} n={stats['n']:>4.0f}  mean={stats['mean']:5.1f}  "
                f"median={stats['median']:5.1f}  range={stats['min']:.0f}-{stats['max']:.0f}"
            )

        self.stdout.write("")
        if report.passes_four_fifths:
            self.stdout.write(self.style.SUCCESS(
                "  No group falls below the four-fifths threshold."
            ))
        else:
            self.stdout.write(self.style.ERROR(
                f"  Adverse impact indicated for: {', '.join(report.flagged_groups)}"
            ))
        for note in report.notes:
            self.stdout.write(self.style.WARNING(f"  Note: {note}"))


def _load_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        raise CommandError(f"Groups file not found: {path}")
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                groups[row[0].strip()] = row[1].strip()
    if not groups:
        raise CommandError(f"No usable rows in {path} (expected 'email,group').")
    return groups
