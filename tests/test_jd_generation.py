"""Job description grading — the measurable half of JD generation."""

from __future__ import annotations

from matching.generation.jd import analyse_jd, generate_jd

GOOD_JD = """
About the role
We are hiring a Backend Engineer to own our payments services. You will work on
systems handling four million requests a day.

Responsibilities
- Own the design of REST APIs that power checkout
- Model relational data on PostgreSQL and keep queries fast
- Write automated tests and review your teammates' code

Requirements
- 3-6 years building backend services
- Strong Python and Django
- PostgreSQL and Docker in production

Nice to have
- AWS, Kubernetes, Celery

What we offer
- Compensation: 18-28 LPA
- Learning budget, generous leave, real ownership and career progression

Work model
- Hybrid, based in Bengaluru

How to apply
- Submit the application form linked in this posting

Equal opportunity
We welcome applications from people of all backgrounds. Tell us if you need an
adjustment at any stage and we will make it.
"""


class TestSectionCoverage:
    def test_complete_jd_finds_every_section(self):
        report = analyse_jd(GOOD_JD, role_title="Backend Engineer")
        assert report.sections_missing == []
        assert report.score > 70

    def test_missing_sections_are_named(self):
        report = analyse_jd("We need a backend engineer. Apply now.")
        assert "What we offer" in report.sections_missing
        assert any("missing sections" in s.lower() for s in report.suggestions)


class TestKeywordCoverage:
    def test_tolerates_reasonable_rewording(self):
        """"REST APIs" in the text must satisfy a "REST API design" requirement."""
        report = analyse_jd(GOOD_JD, must_have=["REST API design", "Python", "PostgreSQL"])
        assert report.keywords_missing == []

    def test_ignores_parenthetical_in_role_title(self):
        report = analyse_jd(GOOD_JD, role_title="Backend Engineer (Python)")
        assert "Backend Engineer (Python)" not in report.keywords_missing

    def test_genuinely_absent_skill_is_reported(self):
        report = analyse_jd(GOOD_JD, must_have=["Rust", "Kafka"])
        assert "Rust" in report.keywords_missing
        assert "Kafka" in report.keywords_missing


class TestCodedLanguage:
    def test_flags_masculine_coded_wording(self):
        report = analyse_jd(GOOD_JD + "\nWe want a rockstar ninja who will dominate.")
        categories = [f["category"] for f in report.flags]
        assert "masculine_coded" in categories
        flagged = next(f for f in report.flags if f["category"] == "masculine_coded")
        assert "rockstar" in flagged["terms"]
        # An actionable alternative must be offered, not just a complaint.
        assert flagged["alternatives"]["rockstar"]

    def test_flags_age_coded_wording(self):
        report = analyse_jd(GOOD_JD + "\nLooking for a young, energetic digital native.")
        assert "age_coded" in [f["category"] for f in report.flags]

    def test_flags_exclusionary_wording(self):
        report = analyse_jd(GOOD_JD + "\nMust be a native speaker and a good cultural fit.")
        assert "exclusionary" in [f["category"] for f in report.flags]

    def test_clean_jd_has_no_flags(self):
        assert analyse_jd(GOOD_JD).flags == []

    def test_whole_word_matching_avoids_false_positives(self):
        """"lead" is coded; "leadership" and "leading" should not trip it."""
        report = analyse_jd("You will show leadership while leading the team.")
        terms = [t for f in report.flags for t in f["terms"]]
        assert "lead" not in terms

    def test_gender_balance_is_reported_directionally(self):
        assert analyse_jd(GOOD_JD + " aggressive competitive dominant fearless driven") \
            .gender_balance.endswith("masculine-coded")


class TestScoring:
    def test_empty_jd_scores_zero(self):
        report = analyse_jd("")
        assert report.score == 0.0

    def test_coded_language_reduces_the_score(self):
        clean = analyse_jd(GOOD_JD).score
        coded = analyse_jd(GOOD_JD + "\nWe want a rockstar ninja guru wizard hero.").score
        assert coded < clean

    def test_overlong_jd_is_penalised(self):
        """Isolate length: both variants share sentence structure, so only the
        word count differs and the readability component cannot confound it."""
        filler = "\nYou will own a service and improve it steadily. "
        ideal = GOOD_JD + filler * 40        # ~440 words, the target band
        overlong = GOOD_JD + filler * 160    # ~1400 words

        assert 350 <= analyse_jd(ideal).word_count <= 600
        assert analyse_jd(overlong).score < analyse_jd(ideal).score
        assert any("long postings" in s for s in analyse_jd(overlong).suggestions)

    def test_too_short_jd_is_penalised(self):
        thin = analyse_jd("About the role\nWe need an engineer. Apply now.")
        assert any("too thin" in s for s in thin.suggestions)


class TestGeneration:
    def test_falls_back_to_a_usable_template_when_the_model_fails(self, provider):
        from matching.llm.base import LLMError

        def explode(*args, **kwargs):
            raise LLMError("model unavailable")

        provider.generate = explode
        text = generate_jd("Backend Engineer", "3 years", provider,
                           must_have=["Python"], nice_to_have=["AWS"])

        # A structured skeleton the recruiter can finish, not an error string.
        assert "Responsibilities" in text
        assert "Equal opportunity" in text
        assert "Python" in text
        assert "AI service was" in text

    def test_strips_conversational_preamble(self, provider):
        provider.generate = lambda *a, **k: (
            "Sure! Here's the job description:\n\nAbout the role\n" + "Detail. " * 100
        )
        assert not generate_jd("X", "1 year", provider).startswith("Sure")
