"""Google OAuth flow and per-user credential loading."""

from __future__ import annotations

import json
import logging
import os

from django.conf import settings
from django.utils import timezone
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from hiring_app.crypto import DecryptionError
from hiring_app.models import GoogleOAuthToken
from hiring_app.services.google_workspace import SCOPES, CredentialsExpired

logger = logging.getLogger("hiring_app")


class OAuthNotConfigured(RuntimeError):
    """No client secrets are available in the environment."""


def build_flow(request, state: str | None = None) -> Flow:
    """Build an OAuth Flow from the env var (production) or file (development)."""
    redirect_uri = request.build_absolute_uri("/google/oauth2callback/")
    raw_secrets = getattr(settings, "GOOGLE_CLIENT_SECRETS", "")

    if raw_secrets:
        try:
            client_config = json.loads(raw_secrets)
        except json.JSONDecodeError as exc:
            raise OAuthNotConfigured(
                "GOOGLE_CLIENT_SECRETS is set but is not valid JSON."
            ) from exc
        flow = Flow.from_client_config(client_config, scopes=SCOPES, state=state)
    else:
        secrets_file = os.path.join(settings.BASE_DIR, "client_secrets.json")
        if not os.path.exists(secrets_file):
            raise OAuthNotConfigured(
                "Google OAuth is not configured. Either set the "
                "GOOGLE_CLIENT_SECRETS environment variable, or place "
                "client_secrets.json in the project root."
            )
        flow = Flow.from_client_secrets_file(secrets_file, scopes=SCOPES, state=state)

    flow.redirect_uri = redirect_uri
    return flow


def load_credentials(user) -> Credentials | None:
    """Load a user's Google credentials, refreshing them if expired.

    Returns ``None`` when the user has never connected an account. Raises
    ``CredentialsExpired`` when a connection exists but is no longer usable, so
    the UI can tell "not set up yet" apart from "needs reconnecting".
    """
    try:
        record = GoogleOAuthToken.objects.get(user=user)
    except GoogleOAuthToken.DoesNotExist:
        return None

    try:
        payload = json.loads(record.token_json)
    except DecryptionError as exc:
        logger.error("Token decryption failed for %s: %s", user.username, exc)
        raise CredentialsExpired(str(exc)) from exc
    except json.JSONDecodeError as exc:
        logger.error("Stored token for %s is corrupt", user.username)
        raise CredentialsExpired("Stored Google credentials are corrupt.") from exc

    credentials = Credentials.from_authorized_user_info(payload, SCOPES)

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(GoogleRequest())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token refresh failed for %s: %s", user.username, exc)
            raise CredentialsExpired(
                "Your Google authorisation could not be refreshed — it was most "
                "likely revoked. Please reconnect your Google account."
            ) from exc
        record.token_json = credentials.to_json()
        record.last_refreshed_at = timezone.now()
        record.save(update_fields=["encrypted_token", "last_refreshed_at", "updated_at"])
        logger.info("Refreshed Google token for %s", user.username)

    if not credentials.valid:
        raise CredentialsExpired(
            "Your Google authorisation is no longer valid. Please reconnect."
        )

    return credentials


def save_credentials(user, credentials: Credentials) -> GoogleOAuthToken:
    record, _ = GoogleOAuthToken.objects.get_or_create(user=user)
    record.token_json = credentials.to_json()
    record.scopes = " ".join(credentials.scopes or [])
    record.last_refreshed_at = timezone.now()
    record.save()
    return record


def has_google(user) -> bool:
    return GoogleOAuthToken.objects.filter(user=user).exists()
