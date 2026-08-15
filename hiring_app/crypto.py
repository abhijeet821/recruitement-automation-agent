"""
Encryption for OAuth credentials at rest.

The previous version stored Google refresh tokens as plaintext in a
``TextField``. A refresh token is a long-lived key to a recruiter's Gmail and
Drive, so a database dump, a stray backup or a read-only SQL injection was a
full account compromise for every connected user.

Tokens are now sealed with Fernet (AES-128-CBC + HMAC-SHA256). The key comes
from ``FIELD_ENCRYPTION_KEY`` when set, and is otherwise derived from
``SECRET_KEY`` via HKDF so that an existing deployment keeps working without
extra configuration — with the consequence, documented loudly in the README,
that rotating ``SECRET_KEY`` invalidates stored tokens and users must reconnect.
"""

from __future__ import annotations

import base64
import logging

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from django.conf import settings

logger = logging.getLogger("hiring_app")


class DecryptionError(RuntimeError):
    """Stored ciphertext could not be decrypted with the current key."""


def _fernet() -> Fernet:
    configured = getattr(settings, "FIELD_ENCRYPTION_KEY", "")
    if configured:
        return Fernet(configured.encode() if isinstance(configured, str) else configured)

    # Derive a stable 32-byte key from SECRET_KEY. HKDF with a fixed info label
    # keeps this key domain-separated from any other use of SECRET_KEY.
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"hireai.oauth.token.v1",
    ).derive(settings.SECRET_KEY.encode())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt(plaintext: str) -> str:
    if plaintext is None:
        return ""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise DecryptionError(
            "Stored credentials could not be decrypted. This usually means "
            "SECRET_KEY or FIELD_ENCRYPTION_KEY changed since they were saved; "
            "the affected user must reconnect their Google account."
        ) from exc
