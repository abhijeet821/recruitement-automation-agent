"""Offline scorer evaluation."""

from matching.evaluation.dataset import (
    DEFAULT_DATASET,
    EvaluationSet,
    LabelledCandidate,
    load_dataset,
    save_dataset,
)
from matching.evaluation.harness import (
    ComparisonReport,
    PreparedCandidate,
    compare,
    default_scorers,
    prepare,
)
from matching.evaluation.metrics import (
    MetricResult,
    evaluate,
    kendall_tau,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    spearman,
)

__all__ = [
    "DEFAULT_DATASET",
    "EvaluationSet",
    "LabelledCandidate",
    "load_dataset",
    "save_dataset",
    "ComparisonReport",
    "PreparedCandidate",
    "compare",
    "default_scorers",
    "prepare",
    "MetricResult",
    "evaluate",
    "kendall_tau",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "spearman",
]
