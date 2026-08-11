"""
Shared test fixtures.

The whole suite runs against ``FakeProvider`` — a deterministic stub — so tests
need no Ollama server, no network, and no GPU, and give identical results every
run. The real provider is exercised separately by ``manage.py doctor`` and
``manage.py evaluate_scorer``, which are integration tools rather than unit
tests.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "my_hiring_project.settings")

from matching.config import MatchingConfig, set_config  # noqa: E402
from matching.llm.fake import FakeProvider  # noqa: E402
from matching.schemas import (  # noqa: E402
    Education,
    GitHubProfile,
    JobSpec,
    ResumeProfile,
    SkillMention,
    WorkExperience,
)


@pytest.fixture(autouse=True)
def _fake_engine(tmp_path):
    """Point the engine at the fake provider and an isolated cache directory."""
    from matching.llm.factory import reset_provider_cache

    set_config(MatchingConfig(
        provider="fake",
        cache_dir=tmp_path / "cache",
        cache_enabled=False,
        blind_screening=True,
        rubric_enabled=False,
    ))
    reset_provider_cache()
    yield
    reset_provider_cache()


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def job_spec():
    return JobSpec(
        role_title="Backend Engineer (Python)",
        seniority="mid",
        min_years=3.0,
        max_years=7.0,
        must_have_skills=["Python", "Django", "PostgreSQL", "Docker"],
        nice_to_have_skills=["AWS", "Redis"],
        responsibilities=["Build REST APIs", "Own service reliability"],
        is_technical=True,
        raw_jd="We are hiring a backend engineer with Python and Django experience.",
    )


@pytest.fixture
def strong_resume():
    return ResumeProfile(
        full_name="Priya Ramanathan",
        email="priya@example.com",
        phone="+91 98450 11223",
        location="Bengaluru",
        github_username="example-priya",
        summary="Backend engineer with 7 years building Python services.",
        total_years_experience=7.0,
        skills=[
            SkillMention(name="Python", years=7, last_used_year=2026, evidence="professional"),
            SkillMention(name="Django", years=6, last_used_year=2026, evidence="professional"),
            SkillMention(name="PostgreSQL", years=5, last_used_year=2026, evidence="professional"),
            SkillMention(name="Docker", years=4, last_used_year=2025, evidence="professional"),
            SkillMention(name="AWS", years=3, last_used_year=2025, evidence="project"),
        ],
        experience=[
            WorkExperience(
                title="Senior Backend Engineer", company="Razorline",
                start_year=2021, is_current=True,
                description="Led migration of the billing service. She reduced p99 latency.",
            ),
            WorkExperience(
                title="Backend Engineer", company="Finwave",
                start_year=2018, end_year=2021, description="Built REST APIs in Django.",
            ),
        ],
        education=[Education(degree="B.Tech", field_of_study="Computer Science",
                             institution="NIT", graduation_year=2018)],
        raw_text=(
            "Priya Ramanathan\npriya@example.com | +91 98450 11223 | Bengaluru\n"
            "Senior Backend Engineer with 7 years of Python and Django experience. "
            "She led the migration of a billing service and reduced p99 latency from "
            "840ms to 190ms. Built REST APIs, PostgreSQL data models and Docker "
            "deployments on AWS." * 3
        ),
    )


@pytest.fixture
def weak_resume():
    return ResumeProfile(
        full_name="Clara Benoit",
        email="clara@example.com",
        summary="Graphic designer and illustrator.",
        total_years_experience=6.0,
        skills=[
            SkillMention(name="Adobe Illustrator", evidence="professional"),
            SkillMention(name="Photoshop", evidence="professional"),
        ],
        experience=[
            WorkExperience(title="Senior Graphic Designer", company="Atelier Rouge",
                           start_year=2020, is_current=True)
        ],
        raw_text="Clara Benoit. Graphic designer. Adobe Illustrator, Photoshop, InDesign." * 5,
    )


@pytest.fixture
def github_profile():
    return GitHubProfile(
        username="example-priya",
        found=True,
        public_repos=24,
        followers=90,
        total_stars=140,
        language_bytes={"Python": 800_000, "JavaScript": 120_000},
        original_repos=18,
        forked_repos=6,
        days_since_last_push=5,
        repos_pushed_last_year=11,
        documented_repo_ratio=0.8,
        account_age_days=2400,
    )
