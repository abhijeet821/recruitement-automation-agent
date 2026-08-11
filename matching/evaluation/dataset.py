"""
The labelled evaluation set.

A scorer cannot be improved without something to measure it against. The format
is deliberately simple — one JSON file per role, holding the JD and a list of
candidates each carrying a human relevance label — so that labelling is a
half-hour of a recruiter's time rather than a data-engineering project.

Labels use a 0-5 relevance grade rather than a binary hire/no-hire, because
graded relevance is what NDCG needs and because real screening decisions are
graded in practice:

    5  outstanding — fast-track to final round
    4  strong — interview
    3  worth a conversation
    2  weak but not absurd
    1  clearly unsuitable
    0  irrelevant (wrong field entirely)

``golden_set.json`` in this directory ships with a small synthetic set so the
harness runs out of the box. Real labels from real applicants replace it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from matching.schemas import JobSpec

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "golden_set.json"


@dataclass
class LabelledCandidate:
    id: str
    label: float                      # 0-5 human relevance grade
    resume_text: str = ""
    resume_path: str = ""
    github_username: str = ""
    email: str = ""
    # Optional, voluntarily self-reported group label for the fairness audit.
    # Never inferred from the resume — see fairness/audit.py.
    group: str = ""
    notes: str = ""


@dataclass
class EvaluationSet:
    name: str
    job: JobSpec
    candidates: list[LabelledCandidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.candidates)

    def labels(self) -> list[float]:
        return [c.label for c in self.candidates]

    def summary(self) -> str:
        counts: dict[float, int] = {}
        for candidate in self.candidates:
            counts[candidate.label] = counts.get(candidate.label, 0) + 1
        spread = ", ".join(f"{int(k)}:{v}" for k, v in sorted(counts.items(), reverse=True))
        return f"{self.name}: {len(self)} candidates (label counts — {spread})"


def load_dataset(path: str | Path | None = None) -> EvaluationSet:
    """Load a labelled set from JSON, validating the parts that matter."""
    path = Path(path or DEFAULT_DATASET)
    if not path.exists():
        raise FileNotFoundError(
            f"No evaluation set at {path}. "
            f"Create one from the format in {DEFAULT_DATASET.name}."
        )

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    job = JobSpec.from_dict(payload.get("job") or {})
    if not job.role_title:
        raise ValueError(f"{path}: job.role_title is required")

    candidates: list[LabelledCandidate] = []
    for index, row in enumerate(payload.get("candidates") or []):
        if "label" not in row:
            logger.warning("%s: candidate %d has no label, skipping", path.name, index)
            continue
        resume_text = row.get("resume_text", "")
        resume_path = row.get("resume_path", "")
        if not resume_text and resume_path:
            # Resolve resume paths relative to the dataset file so a set stays
            # portable between machines.
            candidate_path = (path.parent / resume_path).resolve()
            if candidate_path.exists():
                resume_text = candidate_path.read_text(encoding="utf-8", errors="ignore")
            else:
                logger.warning("%s: resume_path %s not found", path.name, resume_path)

        candidates.append(
            LabelledCandidate(
                id=str(row.get("id") or f"candidate-{index}"),
                label=float(row["label"]),
                resume_text=resume_text,
                resume_path=resume_path,
                github_username=row.get("github_username", ""),
                email=row.get("email", ""),
                group=row.get("group", ""),
                notes=row.get("notes", ""),
            )
        )

    if not candidates:
        raise ValueError(f"{path}: no labelled candidates found")

    return EvaluationSet(
        name=payload.get("name") or path.stem, job=job, candidates=candidates
    )


def save_dataset(dataset: EvaluationSet, path: str | Path) -> None:
    payload = {
        "name": dataset.name,
        "job": dataset.job.to_dict(),
        "candidates": [
            {
                "id": c.id,
                "label": c.label,
                "resume_text": c.resume_text,
                "resume_path": c.resume_path,
                "github_username": c.github_username,
                "email": c.email,
                "group": c.group,
                "notes": c.notes,
            }
            for c in dataset.candidates
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
