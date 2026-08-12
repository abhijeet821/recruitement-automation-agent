"""
Google Workspace client tests.

The previous implementation wrapped nearly every Google call in
``except: pass``, so a permission error, an expired token and a successful call
were indistinguishable. These tests pin down the replacement: every failure mode
maps to a specific, actionable message, and the provisioning path captures the
question IDs without which responses can never be parsed.

The Google services are MagicMocks — the discovery API is a fluent chain, which
mocks cleanly, and none of this needs credentials or network.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from email import message_from_string
from unittest.mock import MagicMock

import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from hiring_app.services.google_workspace import (
    OUTCOMES_TAB,
    RESPONSES_TAB,
    CredentialsExpired,
    WorkspaceClient,
    WorkspaceError,
    _wrap,
)


def http_error(status: int, reason: str = "boom") -> HttpError:
    resp = MagicMock()
    resp.status = status
    resp.reason = reason
    error = HttpError(resp, b'{"error": {"message": "boom"}}')
    error.reason = reason
    return error


def bare_client() -> WorkspaceClient:
    """A client with mocked services, bypassing __init__'s discovery build."""
    client = WorkspaceClient.__new__(WorkspaceClient)
    client.credentials = MagicMock()
    client.forms = MagicMock()
    client.sheets = MagicMock()
    client.drive = MagicMock()
    client.gmail = MagicMock()
    return client


class TestErrorMapping:
    """Every Google failure must become something a recruiter can act on."""

    def test_expired_credentials_are_distinguishable(self):
        wrapped = _wrap("doing a thing", RefreshError("token revoked"))
        assert isinstance(wrapped, CredentialsExpired)
        assert "reconnect" in str(wrapped).lower()

    def test_403_explains_permissions(self):
        message = str(_wrap("creating a form", http_error(403)))
        assert "403" in message and "permission" in message.lower()

    def test_404_says_the_resource_is_gone(self):
        assert "404" in str(_wrap("reading a sheet", http_error(404)))

    def test_429_says_to_retry(self):
        message = str(_wrap("listing responses", http_error(429)))
        assert "rate limit" in message.lower()

    def test_unknown_errors_still_carry_context(self):
        assert "doing a thing" in str(_wrap("doing a thing", ValueError("odd")))

    def test_credentials_expired_is_a_workspace_error(self):
        """So a single `except WorkspaceError` in a view catches both."""
        assert issubclass(CredentialsExpired, WorkspaceError)


class TestClientConstruction:
    def test_missing_credentials_are_rejected_immediately(self):
        with pytest.raises(CredentialsExpired, match="No Google account"):
            WorkspaceClient(None)


class TestSenderAddress:
    def test_returns_the_authenticated_address(self):
        client = bare_client()
        client.gmail.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "recruiter@example.com"
        }
        assert client.sender_address() == "recruiter@example.com"

    def test_missing_address_raises_rather_than_returning_a_placeholder(self):
        """The old code fell back to 'recruiter@example.com', so invites went out
        with a fabricated organiser address."""
        client = bare_client()
        client.gmail.users.return_value.getProfile.return_value.execute.return_value = {}
        with pytest.raises(WorkspaceError, match="did not return an email"):
            client.sender_address()

    def test_api_failure_is_wrapped(self):
        client = bare_client()
        client.gmail.users.return_value.getProfile.return_value.execute.side_effect = \
            http_error(403)
        with pytest.raises(WorkspaceError, match="403"):
            client.sender_address()


class TestCampaignProvisioning:
    def _wire(self, client, question_titles):
        client.sheets.spreadsheets.return_value.create.return_value.execute.return_value = {
            "spreadsheetId": "sheet-1", "spreadsheetUrl": "https://sheets/1",
        }
        client.forms.forms.return_value.create.return_value.execute.return_value = {
            "formId": "form-1"
        }
        client.forms.forms.return_value.batchUpdate.return_value.execute.return_value = {}
        client.forms.forms.return_value.get.return_value.execute.return_value = {
            "responderUri": "https://forms/respond",
            "items": [
                {"title": t, "questionItem": {"question": {"questionId": f"q{i}"}}}
                for i, t in enumerate(question_titles)
            ],
        }
        # Sheet has no tabs yet, so _ensure_tab must create the Responses tab.
        client.sheets.spreadsheets.return_value.get.return_value.execute.return_value = {
            "sheets": []
        }

    def test_captures_question_ids(self):
        """Forms answers are keyed by opaque IDs — without this mapping the
        responses can never be parsed, so it must be captured at creation."""
        from hiring_app.services.google_workspace import (
            Q_EMAIL,
            Q_GITHUB,
            Q_LINKEDIN,
            Q_NAME,
            Q_RESUME,
        )

        client = bare_client()
        self._wire(client, [Q_NAME, Q_EMAIL, Q_RESUME, Q_GITHUB, Q_LINKEDIN])

        assets = client.create_campaign_assets("Backend Engineer", "Write Python.")

        assert assets["form_id"] == "form-1"
        assert assets["sheet_id"] == "sheet-1"
        assert assets["form_url"] == "https://forms/respond"
        assert Q_EMAIL in assets["question_ids"]
        assert Q_RESUME in assets["question_ids"]

    def test_fails_loudly_when_essential_questions_are_missing(self):
        """Better to fail at creation than to produce a campaign whose responses
        silently cannot be read."""
        from hiring_app.services.google_workspace import Q_NAME

        client = bare_client()
        self._wire(client, [Q_NAME])  # no Email, no Resume

        with pytest.raises(WorkspaceError, match="could not be identified"):
            client.create_campaign_assets("Backend Engineer", "Write Python.")

    def test_long_jd_is_truncated_on_a_word_boundary(self):
        from hiring_app.services.google_workspace import Q_EMAIL, Q_NAME, Q_RESUME

        client = bare_client()
        self._wire(client, [Q_NAME, Q_EMAIL, Q_RESUME])
        client.create_campaign_assets("Role", "word " * 2000)

        body = client.forms.forms.return_value.batchUpdate.call_args.kwargs["body"]
        description = body["requests"][0]["updateFormInfo"]["info"]["description"]
        assert len(description) <= 3910
        assert description.endswith("…")

    def test_sheet_creation_failure_is_wrapped(self):
        client = bare_client()
        client.sheets.spreadsheets.return_value.create.return_value.execute.side_effect = \
            http_error(403)
        with pytest.raises(WorkspaceError, match="403"):
            client.create_campaign_assets("Role", "JD")


class TestListResponses:
    def test_follows_pagination(self):
        """A campaign with more than one page of applicants must not silently
        lose everyone after the first page."""
        client = bare_client()
        pages = [
            {"responses": [{"responseId": "r1"}], "nextPageToken": "p2"},
            {"responses": [{"responseId": "r2"}]},
        ]
        client.forms.forms.return_value.responses.return_value.list.return_value.execute.side_effect = pages

        out = client.list_responses("form-1")
        assert [r["responseId"] for r in out] == ["r1", "r2"]

    def test_empty_form_returns_empty_list(self):
        client = bare_client()
        client.forms.forms.return_value.responses.return_value.list.return_value.execute.return_value = {}
        assert client.list_responses("form-1") == []

    def test_failure_is_wrapped(self):
        client = bare_client()
        client.forms.forms.return_value.responses.return_value.list.return_value.execute.side_effect = \
            http_error(429)
        with pytest.raises(WorkspaceError, match="rate limit"):
            client.list_responses("form-1")


class TestAnswerParsing:
    def test_reads_a_text_answer(self):
        payload = {"answers": {"q1": {"textAnswers": {"answers": [{"value": " Ada "}]}}}}
        assert WorkspaceClient.answer(payload, "q1") == "Ada"

    def test_missing_question_id_is_safe(self):
        assert WorkspaceClient.answer({"answers": {}}, None) == ""

    def test_unanswered_question_returns_empty(self):
        assert WorkspaceClient.answer({"answers": {}}, "q1") == ""


class TestTabHandling:
    def test_existing_tab_is_not_recreated(self):
        """The old code ran addSheet on every sync inside a bare except."""
        client = bare_client()
        client.sheets.spreadsheets.return_value.get.return_value.execute.return_value = {
            "sheets": [{"properties": {"title": RESPONSES_TAB}}]
        }
        client._ensure_tab("sheet-1", RESPONSES_TAB, ["A"])
        client.sheets.spreadsheets.return_value.batchUpdate.assert_not_called()

    def test_missing_tab_is_created_with_a_header(self):
        client = bare_client()
        client.sheets.spreadsheets.return_value.get.return_value.execute.return_value = {
            "sheets": []
        }
        client._ensure_tab("sheet-1", OUTCOMES_TAB, ["Timestamp", "Email"])
        client.sheets.spreadsheets.return_value.batchUpdate.assert_called_once()

    def test_append_rows_is_a_noop_for_empty_input(self):
        client = bare_client()
        client.append_rows("sheet-1", RESPONSES_TAB, [])
        client.sheets.spreadsheets.return_value.values.return_value.append.assert_not_called()


class TestMail:
    def _sent_message(self, client):
        """Decode the Gmail payload back into a parsed email message.

        Bodies are base64-encoded inside their MIME parts, so the message has to
        be parsed rather than string-matched.
        """
        body = client.gmail.users.return_value.messages.return_value.send.call_args.kwargs["body"]
        raw = base64.urlsafe_b64decode(body["raw"]).decode()
        return message_from_string(raw)

    @staticmethod
    def _part_text(message, subtype: str) -> str:
        for part in message.walk():
            if part.get_content_subtype() == subtype:
                return part.get_payload(decode=True).decode()
        return ""

    def test_plain_email_is_well_formed(self):
        client = bare_client()
        client.send_email("c@example.com", "Subject line", "Body text")
        message = self._sent_message(client)

        assert message["To"] == "c@example.com"
        assert message["Subject"] == "Subject line"
        assert "Body text" in self._part_text(message, "plain")

    def test_invite_attaches_a_calendar_part(self):
        client = bare_client()
        start = datetime.now(UTC) + timedelta(days=1)
        client.send_invite(
            "c@example.com", "Interview", "Details",
            organiser_name="Hiring Team", organiser_email="r@example.com",
            role="Backend Engineer", start=start,
        )
        message = self._sent_message(client)

        calendar = self._part_text(message, "calendar")
        assert "BEGIN:VCALENDAR" in calendar
        assert "METHOD:REQUEST" in calendar
        assert "ATTENDEE" in calendar and "c@example.com" in calendar
        assert start.strftime("%Y%m%dT%H%M%SZ") in calendar

    def test_send_failure_names_the_recipient(self):
        client = bare_client()
        client.gmail.users.return_value.messages.return_value.send.return_value.execute.side_effect = \
            http_error(403)
        with pytest.raises(WorkspaceError, match="c@example.com"):
            client.send_email("c@example.com", "s", "b")
