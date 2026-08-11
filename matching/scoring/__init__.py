"""Feature vectors and rubric assessments -> explainable scores."""

from matching.scoring.base import Scorer, band
from matching.scoring.baselines import (
    LEGACY_KEYWORDS,
    JDKeywordBaseline,
    KeywordBaseline,
    RandomBaseline,
)
from matching.scoring.ensemble import DIMENSION_SPEC, SCORER_VERSION, EnsembleScorer
from matching.scoring.rubric import RubricResult, assess

__all__ = [
    "Scorer",
    "band",
    "LEGACY_KEYWORDS",
    "JDKeywordBaseline",
    "KeywordBaseline",
    "RandomBaseline",
    "DIMENSION_SPEC",
    "SCORER_VERSION",
    "EnsembleScorer",
    "RubricResult",
    "assess",
]
