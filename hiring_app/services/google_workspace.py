"""
Google Workspace integration: Forms, Sheets, Drive, Gmail.

Kept deliberately close to the original design — creating a Form and a tracking
Sheet per campaign, and pulling responses from the Form — because that flow
works and recruiters already live in those tools. What changed is the error
handling and the correctness bugs.

Fixed here:

* **Silent failures.** The previous version ended most methods in
  ``except: pass``. A failed Drive download produced a zero-byte file, empty
  text and a score of zero, which is indistinguishable from a genuinely weak
  candidate. Every operation now either succeeds or raises
  ``WorkspaceError`` with a message the UI can show.
* **Naive datetimes in calendar invites.** ``strptime`` produced a naive
  datetime and ``.astimezone()`` then silently assumed the *server's* timezone,
  so invites were hours off in any container running UTC. Aware datetimes are
  now required at the boundary.
* **Repeated header writes.** "Create the Responses sheet, ignore the error if
  it exists" ran on every sync inside a bare except. Existence is now checked.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger("hiring_app")

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/gmail.send",
]

# Question titles are the contract between form creation and response parsing.
Q_NAME = "Full name"
Q_EMAIL = "Email"
Q_EXPERIENCE = "Years of experience"
Q_FIT = "Why are you a fit for this role?"
Q_RESUME = "Resume (Google Drive link to a PDF)"
Q_GITHUB = "GitHub profile URL"
Q_LINKEDIN = "LinkedIn URL"

RESPONSES_TAB = "Responses"
OUTCOMES_TAB = "Outcomes"


class WorkspaceError(RuntimeError):
    """A Google Workspace operation failed."""


class CredentialsExpired(WorkspaceError):
    """The user's Google authorisation is no longer valid; they must reconnect."""


def _wrap(operation: str, exc: Exception) -> WorkspaceError:
    """Turn a Google API exception into something a user can act on."""
    if isinstance(exc, RefreshError):
        return CredentialsExpired(
            "Your Google authorisation has expired or been revoked. "
            "Reconnect your Google account to continue."
        )
    if isinstance(exc, HttpError):
        status = exc.resp.status if exc.resp else "?"
        if status == 403:
            return WorkspaceError(
                f"Google denied {operation} (403). The account may lack the required "
                f"permission, or the API is not enabled in the Cloud project."
            )
        if status == 404:
            return WorkspaceError(f"{operation}: the Google resource no longer exists (404).")
        if status == 429:
            return WorkspaceError(f"{operation}: Google rate limit hit. Try again shortly.")
        return WorkspaceError(f"{operation} failed ({status}): {exc.reason}")
    return WorkspaceError(f"{operation} failed: {exc}")


class WorkspaceClient:
    """Thin, explicit wrapper over the Google APIs this app uses."""

    def __init__(self, credentials):
        if credentials is None:
            raise CredentialsExpired("No Google account is connected.")
        self.credentials = credentials
        try:
            self.forms = build("forms", "v1", credentials=credentials, cache_discovery=False)
            self.sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)
            self.drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
            self.gmail = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        except Exception as exc:  # noqa: BLE001
            raise _wrap("connecting to Google", exc) from exc

    # ── identity ─────────────────────────────────────────────

    def sender_address(self) -> str:
        try:
            profile = self.gmail.users().getProfile(userId="me").execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap("reading your Gmail profile", exc) from exc
        address = profile.get("emailAddress")
        if not address:
            raise WorkspaceError("Google did not return an email address for this account.")
        return address

    # ── campaign provisioning ────────────────────────────────

    def create_campaign_assets(self, role_title: str, jd_text: str) -> dict:
        """Create the tracking Sheet and the application Form for a campaign."""
        try:
            spreadsheet = self.sheets.spreadsheets().create(
                body={"properties": {"title": f"Applications — {role_title}"}}
            ).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap("creating the tracking spreadsheet", exc) from exc

        sheet_id = spreadsheet["spreadsheetId"]
        sheet_url = spreadsheet["spreadsheetUrl"]

        try:
            form = self.forms.forms().create(
                body={"info": {"title": f"Application — {role_title}"}}
            ).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap("creating the application form", exc) from exc

        form_id = form["formId"]

        # The Forms API caps the description; truncate on a word boundary so the
        # JD does not end mid-word.
        description = jd_text if len(jd_text) <= 3900 else jd_text[:3900].rsplit(" ", 1)[0] + "…"

        requests = [
            {"updateFormInfo": {"info": {"description": description},
                                "updateMask": "description"}},
            _text_question(Q_NAME, 0),
            _text_question(Q_EMAIL, 1),
            _choice_question(Q_EXPERIENCE, ["0-1", "2-3", "4-6", "7-10", "10+"], 2),
            _text_question(Q_FIT, 3, paragraph=True),
            _text_question(
                Q_RESUME, 4,
                description="Upload your PDF to Google Drive, set sharing to "
                            "'Anyone with the link', and paste the link here.",
            ),
            _text_question(
                Q_GITHUB, 5, required=False,
                description="Optional. Public repositories are used as supporting "
                            "evidence only — leaving this blank never counts against you.",
            ),
            _text_question(Q_LINKEDIN, 6, required=False),
        ]

        try:
            self.forms.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
            meta = self.forms.forms().get(formId=form_id).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap("adding questions to the form", exc) from exc

        question_ids = {
            title: qid
            for title in (Q_NAME, Q_EMAIL, Q_EXPERIENCE, Q_FIT, Q_RESUME, Q_GITHUB, Q_LINKEDIN)
            if (qid := _question_id(meta, title))
        }
        missing = {Q_EMAIL, Q_RESUME} - set(question_ids)
        if missing:
            # Without these IDs responses cannot be parsed at all, so fail loudly
            # now rather than producing an unusable campaign.
            raise WorkspaceError(
                f"The form was created but these questions could not be identified: "
                f"{', '.join(sorted(missing))}. Delete the campaign and retry."
            )

        self._ensure_tab(sheet_id, RESPONSES_TAB, [
            "Response ID", "Submitted", "Name", "Email", "Score", "Confidence",
            "Recommendation", "Resume link", "GitHub", "Status",
        ])

        return {
            "form_id": form_id,
            "form_url": meta.get("responderUri", ""),
            "sheet_id": sheet_id,
            "sheet_url": sheet_url,
            "question_ids": question_ids,
        }

    # ── responses ────────────────────────────────────────────

    def list_responses(self, form_id: str) -> list[dict]:
        """Fetch every response, following pagination."""
        out: list[dict] = []
        page_token = None
        try:
            while True:
                params = {"formId": form_id}
                if page_token:
                    params["pageToken"] = page_token
                payload = self.forms.forms().responses().list(**params).execute()
                out.extend(payload.get("responses", []))
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
        except Exception as exc:  # noqa: BLE001
            raise _wrap("reading form responses", exc) from exc
        return out

    @staticmethod
    def answer(response: dict, question_id: str | None) -> str:
        if not question_id:
            return ""
        answers = (
            response.get("answers", {})
            .get(question_id, {})
            .get("textAnswers", {})
            .get("answers", [])
        )
        return answers[0].get("value", "").strip() if answers else ""

    # ── drive ────────────────────────────────────────────────

    @staticmethod
    def extract_file_id(url: str) -> str:
        """Pull a Drive file ID out of any of its several link formats."""
        if not url:
            return ""
        patterns = [
            r"/file/d/([a-zA-Z0-9_-]{20,})",
            r"[?&]id=([a-zA-Z0-9_-]{20,})",
            r"/document/d/([a-zA-Z0-9_-]{20,})",
            r"/open\?id=([a-zA-Z0-9_-]{20,})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        # A bare ID pasted without a URL.
        stripped = url.strip()
        if re.fullmatch(r"[a-zA-Z0-9_-]{25,}", stripped):
            return stripped
        return ""

    def download_pdf(self, file_id: str, destination: str | Path) -> Path:
        """Download a Drive file. Raises on failure rather than leaving a stub."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temp path and rename, so a failure never leaves a
        # truncated file that a later run would treat as a valid cached resume.
        temp = destination.with_suffix(".part")

        try:
            request = self.drive.files().get_media(fileId=file_id)
            with io.FileIO(temp, "wb") as handle:
                downloader = MediaIoBaseDownload(handle, request, chunksize=1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except Exception as exc:  # noqa: BLE001
            temp.unlink(missing_ok=True)
            if isinstance(exc, HttpError) and exc.resp is not None and exc.resp.status == 404:
                raise WorkspaceError(
                    "The resume link is not accessible. The candidate may not have "
                    "set sharing to 'Anyone with the link'."
                ) from exc
            raise _wrap("downloading the resume from Drive", exc) from exc

        if temp.stat().st_size == 0:
            temp.unlink(missing_ok=True)
            raise WorkspaceError("The downloaded resume is empty.")

        temp.replace(destination)
        return destination

    # ── sheets ───────────────────────────────────────────────

    def _tab_exists(self, sheet_id: str, title: str) -> bool:
        try:
            meta = self.sheets.spreadsheets().get(spreadsheetId=sheet_id).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap("reading the spreadsheet", exc) from exc
        return any(s["properties"]["title"] == title for s in meta.get("sheets", []))

    def _ensure_tab(self, sheet_id: str, title: str, header: list[str]) -> None:
        """Create a tab with a header row, only if it does not already exist."""
        if self._tab_exists(sheet_id, title):
            return
        try:
            self.sheets.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": title}}}]},
            ).execute()
            self.sheets.spreadsheets().values().append(
                spreadsheetId=sheet_id, range=f"{title}!A1",
                valueInputOption="RAW", body={"values": [header]},
            ).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap(f"creating the '{title}' tab", exc) from exc

    def append_rows(self, sheet_id: str, tab: str, rows: list[list]) -> None:
        if not rows:
            return
        try:
            self.sheets.spreadsheets().values().append(
                spreadsheetId=sheet_id, range=f"{tab}!A1",
                valueInputOption="RAW", body={"values": rows},
            ).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap(f"writing to the '{tab}' tab", exc) from exc

    def log_outcomes(self, sheet_id: str, rows: list[list]) -> None:
        self._ensure_tab(sheet_id, OUTCOMES_TAB, ["Timestamp", "Email", "Decision", "Score"])
        self.append_rows(sheet_id, OUTCOMES_TAB, rows)

    # ── gmail ────────────────────────────────────────────────

    def send_email(self, to: str, subject: str, body: str) -> None:
        message = MIMEMultipart()
        message["To"] = to
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))
        self._send(message)

    def send_invite(
        self,
        to: str,
        subject: str,
        body: str,
        *,
        organiser_name: str,
        organiser_email: str,
        role: str,
        start: datetime,
        duration_minutes: int = 45,
    ) -> None:
        """Send an interview invitation with an iCalendar attachment.

        ``start`` must be timezone-aware. The original code accepted a naive
        datetime and converted it using the server's local timezone, so the same
        request produced different invite times depending on where it ran.
        """
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError(
                "send_invite requires a timezone-aware datetime; a naive one "
                "would be interpreted in the server's timezone and silently "
                "shift the meeting."
            )

        ics = _build_ics(
            organiser_name=organiser_name,
            organiser_email=organiser_email,
            attendee_email=to,
            role=role,
            start=start,
            duration_minutes=duration_minutes,
        )

        message = MIMEMultipart("mixed")
        message["To"] = to
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        part = MIMEText(ics, "calendar", "utf-8")
        part.add_header("Content-Class", "urn:content-classes:calendarmessage")
        part.add_header("Content-Type", 'text/calendar; method=REQUEST; name="invite.ics"')
        part.add_header("Content-Disposition", 'attachment; filename="invite.ics"')
        message.attach(part)

        self._send(message)

    def _send(self, message: MIMEMultipart) -> None:
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        try:
            self.gmail.users().messages().send(userId="me", body={"raw": raw}).execute()
        except Exception as exc:  # noqa: BLE001
            raise _wrap(f"sending mail to {message['To']}", exc) from exc


# ── form question helpers ────────────────────────────────────

def _text_question(
    title: str, index: int, *, paragraph: bool = False,
    required: bool = True, description: str = "",
) -> dict:
    item = {
        "title": title,
        "questionItem": {
            "question": {"required": required, "textQuestion": {"paragraph": paragraph}}
        },
    }
    if description:
        item["description"] = description
    return {"createItem": {"item": item, "location": {"index": index}}}


def _choice_question(title: str, options: list[str], index: int) -> dict:
    item = {
        "title": title,
        "questionItem": {
            "question": {
                "required": True,
                "choiceQuestion": {
                    "type": "RADIO", "options": [{"value": o} for o in options]
                },
            }
        },
    }
    return {"createItem": {"item": item, "location": {"index": index}}}


def _question_id(meta: dict, title: str) -> str:
    for item in meta.get("items", []):
        if item.get("title") == title:
            return item.get("questionItem", {}).get("question", {}).get("questionId", "")
    return ""


def _ics_escape(text: str) -> str:
    """Escape per RFC 5545 — unescaped commas silently corrupt the event."""
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _build_ics(
    *, organiser_name: str, organiser_email: str, attendee_email: str,
    role: str, start: datetime, duration_minutes: int,
) -> str:

    def stamp(moment: datetime) -> str:
        return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    end = start + timedelta(minutes=duration_minutes)
    return "\r\n".join([
        "BEGIN:VCALENDAR",
        "PRODID:-//HireAI//Recruitment Agent//EN",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uuid.uuid4().hex}@hireai",
        f"DTSTAMP:{stamp(datetime.now(UTC))}",
        f"DTSTART:{stamp(start)}",
        f"DTEND:{stamp(end)}",
        f"SUMMARY:Interview — {_ics_escape(role)}",
        f"DESCRIPTION:Interview for the {_ics_escape(role)} position.",
        f"ORGANIZER;CN={_ics_escape(organiser_name)}:mailto:{organiser_email}",
        f"ATTENDEE;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{attendee_email}",
        "STATUS:CONFIRMED",
        "SEQUENCE:0",
        "END:VEVENT",
        "END:VCALENDAR",
    ])
