"""Job description drafting and grading."""

from matching.generation.jd import JDQualityReport, analyse_jd, generate_jd
from matching.generation.keywords import (
    ATTRACTION_TERMS,
    MASCULINE_CODED,
    REQUIRED_SECTIONS,
    role_keywords,
)

__all__ = [
    "JDQualityReport",
    "analyse_jd",
    "generate_jd",
    "ATTRACTION_TERMS",
    "MASCULINE_CODED",
    "REQUIRED_SECTIONS",
    "role_keywords",
]
