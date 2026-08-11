"""
matching — the candidate/job matching engine.

This package is deliberately free of any Django import. It is a plain Python
library that takes a job description and a candidate's artifacts (resume PDF,
GitHub handle) and produces an explainable fit score. That independence is what
makes it testable in isolation, benchmarkable offline, and reusable outside the
web app (e.g. from the evaluation harness or a notebook).

Layering, bottom-up:

    llm/          provider abstraction (Ollama for local dev, Gemini for hosted)
    parsing/      unstructured bytes -> structured records (PDF, resume, JD)
    enrichment/   external evidence (GitHub activity analysis)
    features/     structured records -> a named, documented feature vector
    scoring/      feature vector -> explainable score (+ a keyword baseline)
    fairness/     PII redaction for blind screening, adverse-impact auditing
    evaluation/   offline metrics and scorer comparison
    generation/   JD drafting with recruiter keyword coverage
"""

from matching.config import MatchingConfig, get_config, set_config
from matching.schemas import (
    CandidateScore,
    DimensionScore,
    Education,
    FeatureVector,
    GitHubProfile,
    JobSpec,
    Recommendation,
    RepoSummary,
    ResumeProfile,
    SkillMention,
    WorkExperience,
)

__all__ = [
    "MatchingConfig",
    "get_config",
    "set_config",
    "CandidateScore",
    "DimensionScore",
    "Education",
    "FeatureVector",
    "GitHubProfile",
    "JobSpec",
    "Recommendation",
    "RepoSummary",
    "ResumeProfile",
    "SkillMention",
    "WorkExperience",
]

__version__ = "2.0.0"
