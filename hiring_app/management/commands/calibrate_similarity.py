"""
Measure how well raw embedding similarity separates true from false skill pairs.

    python manage.py calibrate_similarity

This command produced the finding that drove the skill-matching architecture.
Running it against bge-m3 shows that the similarity distributions for genuinely
equivalent skills and for merely related ones **overlap**:

    python <-> java  (different)   scores higher than
    docker <-> containerisation  (equivalent)

Because no threshold separates those, skill matching cannot be a cosine cut-off.
It is instead embedding retrieval followed by LLM verification — see
``matching/features/skills.py``.

Re-run this after changing the embedding model. If a future model *does*
separate the classes cleanly, the verification stage becomes optional and the
pipeline gets materially faster.
"""

from __future__ import annotations

import numpy as np
from django.core.management.base import BaseCommand

from matching.config import get_config
from matching.features.skills import normalise_skill
from matching.llm import get_provider

# Hand-labelled pairs. EQUIVALENT means "having the second satisfies a
# requirement for the first"; DIFFERENT means it does not.
PAIRS: list[tuple[str, str, str]] = [
    ("EQUIVALENT", "python", "python programming"),
    ("EQUIVALENT", "postgresql", "postgres"),
    ("EQUIVALENT", "rest api design", "restful api development"),
    ("EQUIVALENT", "unit testing", "pytest"),
    ("EQUIVALENT", "docker", "containerisation"),
    ("EQUIVALENT", "kubernetes", "k8s"),
    ("EQUIVALENT", "deep learning", "pytorch"),
    ("EQUIVALENT", "ci/cd", "github actions"),
    ("EQUIVALENT", "django", "django rest framework"),
    ("EQUIVALENT", "javascript", "es6"),
    ("DIFFERENT", "python", "java"),
    ("DIFFERENT", "django", "spring boot"),
    ("DIFFERENT", "postgresql", "oracle sql"),
    ("DIFFERENT", "python", "node.js"),
    ("DIFFERENT", "docker", "jenkins"),
    ("DIFFERENT", "unit testing", "manual testing"),
    ("DIFFERENT", "react", "angular"),
    ("DIFFERENT", "pytorch", "tensorflow"),
    ("DIFFERENT", "python", "adobe illustrator"),
    ("DIFFERENT", "django", "enterprise sales"),
]


class Command(BaseCommand):
    help = "Measure embedding separation between equivalent and different skill pairs."

    def handle(self, *args, **options):
        config = get_config()
        provider = get_provider(config)

        texts: list[str] = []
        for _, left, right in PAIRS:
            texts.extend([normalise_skill(left), normalise_skill(right)])

        self.stdout.write(f"Embedding model: {config.ollama_embed_model}\n")
        vectors = provider.embed(texts)

        rows: list[tuple[str, str, str, float]] = []
        for index, (kind, left, right) in enumerate(PAIRS):
            similarity = float(vectors[2 * index] @ vectors[2 * index + 1])
            rows.append((kind, left, right, similarity))

        self.stdout.write(self.style.MIGRATE_HEADING("Pairwise similarity"))
        for kind, left, right, similarity in sorted(rows, key=lambda r: -r[3]):
            style = self.style.SUCCESS if kind == "EQUIVALENT" else self.style.WARNING
            self.stdout.write(style(f"  {similarity:.3f}  {kind:<11} {left} <-> {right}"))

        equivalent = np.array([s for k, _, _, s in rows if k == "EQUIVALENT"])
        different = np.array([s for k, _, _, s in rows if k == "DIFFERENT"])

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Distributions"))
        self.stdout.write(
            f"  EQUIVALENT  n={len(equivalent):2d}  "
            f"mean={equivalent.mean():.3f}  min={equivalent.min():.3f}  max={equivalent.max():.3f}"
        )
        self.stdout.write(
            f"  DIFFERENT   n={len(different):2d}  "
            f"mean={different.mean():.3f}  min={different.min():.3f}  max={different.max():.3f}"
        )

        overlap = float(different.max() - equivalent.min())
        self.stdout.write("")
        if overlap > 0:
            misranked = int(sum(1 for s in equivalent if s < different.max()))
            self.stdout.write(self.style.ERROR(
                f"  OVERLAP of {overlap:.3f}: the highest DIFFERENT pair "
                f"({different.max():.3f}) outscores {misranked} EQUIVALENT pair(s)."
            ))
            self.stdout.write(
                "  No single cosine threshold can separate these classes, which is "
                "why skill matching uses embedding retrieval plus LLM verification "
                "rather than a similarity cut-off."
            )
            best_f1, best_threshold = _best_threshold(equivalent, different)
            self.stdout.write(
                f"  For reference, the best possible single threshold is "
                f"{best_threshold:.3f} at F1={best_f1:.3f} — not good enough to "
                f"decide a hiring signal on."
            )
        else:
            self.stdout.write(self.style.SUCCESS(
                f"  CLEAN SEPARATION. A threshold at "
                f"{(equivalent.min() + different.max()) / 2:.3f} would classify "
                f"every pair correctly — the verification stage could be relaxed."
            ))


def _best_threshold(equivalent: np.ndarray, different: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over all candidate thresholds."""
    best = (0.0, 0.0)
    for threshold in np.arange(0.30, 0.95, 0.005):
        true_positive = int((equivalent >= threshold).sum())
        false_positive = int((different >= threshold).sum())
        false_negative = int((equivalent < threshold).sum())
        if true_positive == 0:
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best
