"""Structured records -> named feature vectors."""

from matching.features.builder import FEATURE_ORDER, FeatureBuilder
from matching.features.skills import (
    MATCH_THRESHOLD,
    VERDICT_SCORES,
    SkillMatch,
    SkillMatcher,
    SkillMatchResult,
    calibrate_document_similarity,
    match_skills,
    normalise_skill,
)

__all__ = [
    "FEATURE_ORDER",
    "FeatureBuilder",
    "MATCH_THRESHOLD",
    "VERDICT_SCORES",
    "SkillMatch",
    "SkillMatcher",
    "SkillMatchResult",
    "calibrate_document_similarity",
    "match_skills",
    "normalise_skill",
]
