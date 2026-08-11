"""Bias mitigation and outcome auditing."""

from matching.fairness.audit import (
    FOUR_FIFTHS,
    AuditReport,
    GroupOutcome,
    adverse_impact,
    score_gap,
)
from matching.fairness.redaction import redact_profile, redact_text

__all__ = [
    "FOUR_FIFTHS",
    "AuditReport",
    "GroupOutcome",
    "adverse_impact",
    "score_gap",
    "redact_profile",
    "redact_text",
]
