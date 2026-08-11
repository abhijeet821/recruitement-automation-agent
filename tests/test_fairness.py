"""Blind-screening redaction and adverse-impact auditing."""

from __future__ import annotations

from matching.fairness import adverse_impact, redact_profile, redact_text, score_gap
from matching.fairness.redaction import REDACTED


class TestRedaction:
    def test_removes_email_and_phone(self):
        out = redact_text("Reach me at priya@example.com or +91 98450 11223")
        assert "priya@example.com" not in out
        assert "98450" not in out

    def test_removes_name_and_its_parts(self):
        out = redact_text("Priya Ramanathan led the team. Priya shipped it.", name="Priya Ramanathan")
        assert "Priya" not in out
        assert "Ramanathan" not in out

    def test_removes_gendered_pronouns(self):
        out = redact_text("She led the migration and his team supported her.")
        for word in ("She", "his", "her"):
            assert word not in out.split()

    def test_removes_protected_attribute_lines(self):
        out = redact_text("Name: X\nDate of Birth: 12 May 1994\nSkills: Python")
        assert "1994" not in out
        assert "Python" in out

    def test_keeps_job_relevant_signal(self):
        """Redaction that destroys signal is noise, not fairness."""
        text = (
            "Senior Backend Engineer at Razorline, 2021-2024. "
            "Reduced p99 latency from 840ms to 190ms. Python, Django, PostgreSQL."
        )
        out = redact_text(text, name="Priya Ramanathan")
        for keep in ("Backend Engineer", "2021", "840ms", "Python", "Django", "Razorline"):
            assert keep in out

    def test_does_not_redact_year_ranges_as_phone_numbers(self):
        out = redact_text("Worked there 2019 - 2023 and grew revenue 40%")
        assert "2019" in out and "2023" in out and "40%" in out

    def test_institutions_kept_by_default(self):
        text = "BSc from Trinity College Dublin"
        assert "Trinity College" in redact_text(text)

    def test_institutions_removed_when_requested(self):
        out = redact_text("BSc from Trinity College Dublin", redact_institutions=True)
        assert "Trinity College" not in out

    def test_redact_profile_clears_identity_but_keeps_skills(self, strong_resume):
        clean = redact_profile(strong_resume)
        assert clean.full_name == REDACTED
        assert clean.email == ""
        assert clean.location == ""
        # Everything job-relevant survives.
        assert len(clean.skills) == len(strong_resume.skills)
        assert clean.skills[0].name == "Python"
        assert len(clean.experience) == len(strong_resume.experience)
        assert clean.experience[0].title == "Senior Backend Engineer"

    def test_redact_profile_does_not_mutate_the_original(self, strong_resume):
        original_name = strong_resume.full_name
        redact_profile(strong_resume)
        assert strong_resume.full_name == original_name
        assert strong_resume.email == "priya@example.com"

    def test_redacts_pronoun_inside_experience_description(self, strong_resume):
        clean = redact_profile(strong_resume)
        assert "She" not in clean.experience[0].description


class TestAdverseImpact:
    def test_equal_rates_pass(self):
        outcomes = [("A", True)] * 10 + [("A", False)] * 10
        outcomes += [("B", True)] * 10 + [("B", False)] * 10
        report = adverse_impact(outcomes, min_group_size=5)
        assert report.passes_four_fifths
        assert report.impact_ratios["B"] == 1.0

    def test_detects_disparity_below_four_fifths(self):
        # Group A selected 80%, group B 20% -> ratio 0.25
        outcomes = [("A", True)] * 8 + [("A", False)] * 2
        outcomes += [("B", True)] * 2 + [("B", False)] * 8
        report = adverse_impact(outcomes, min_group_size=5)
        assert not report.passes_four_fifths
        assert "B" in report.flagged_groups
        assert report.reference_group == "A"

    def test_small_samples_are_flagged_as_unreliable(self):
        outcomes = [("A", True), ("A", False), ("B", True), ("B", False)]
        report = adverse_impact(outcomes, min_group_size=30)
        assert not report.sufficient_sample
        assert any("indicative" in note for note in report.notes)

    def test_undisclosed_group_excluded_from_ratios(self):
        outcomes = [("A", True)] * 5 + [("B", True)] * 5 + [("undisclosed", False)] * 5
        report = adverse_impact(outcomes, min_group_size=1)
        assert "undisclosed" not in report.impact_ratios
        # It is still reported for transparency.
        assert any(g.group == "undisclosed" for g in report.groups)

    def test_single_group_cannot_be_compared(self):
        report = adverse_impact([("A", True), ("A", False)])
        assert report.impact_ratios == {}
        assert any("two groups" in note for note in report.notes)

    def test_empty_input_is_safe(self):
        assert adverse_impact([]).groups == []


class TestScoreGap:
    def test_reports_distribution_per_group(self):
        stats = score_gap([("A", 80.0), ("A", 60.0), ("B", 40.0)])
        assert stats["A"]["mean"] == 70.0
        assert stats["A"]["median"] == 70.0
        assert stats["B"]["n"] == 1.0
        assert stats["A"]["max"] == 80.0
