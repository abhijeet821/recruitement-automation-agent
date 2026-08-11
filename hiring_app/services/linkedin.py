"""
LinkedIn job posting.

Kept as-is in capability — post the opening with a link to the application form
— but no longer failing silently. The previous version wrapped the whole call in
``except: pass`` and returned ``None``, so a rejected token, an expired session
and a successful post were indistinguishable to the recruiter.

Known limitation, unchanged and worth stating plainly: the access token and
author URN are entered by hand per campaign rather than obtained through a
LinkedIn OAuth flow. Proper OAuth requires a reviewed LinkedIn application; the
manual token keeps the feature demonstrable without that approval. The code path
is the same one an OAuth token would use.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("hiring_app")

API_URL = "https://api.linkedin.com/v2/ugcPosts"
MAX_COMMENTARY = 2800  # LinkedIn's limit is 3000; leave room for the wrapper text


class LinkedInError(RuntimeError):
    """Posting to LinkedIn failed."""


def post_job(
    access_token: str,
    author_urn: str,
    role_title: str,
    jd_text: str,
    form_url: str,
    *,
    timeout: float = 20.0,
) -> str:
    """Publish a job post and return its LinkedIn post ID.

    Raises ``LinkedInError`` with an actionable message on failure.
    """
    if not access_token or not author_urn:
        raise LinkedInError("A LinkedIn access token and author URN are both required.")
    if not author_urn.startswith(("urn:li:person:", "urn:li:organization:")):
        raise LinkedInError(
            "The author URN must look like 'urn:li:person:XXXX' or "
            "'urn:li:organization:XXXX'."
        )

    body = _compose(role_title, jd_text, form_url)
    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": body},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LinkedInError(f"Could not reach LinkedIn: {exc}") from exc

    if response.status_code in (200, 201):
        post_id = response.json().get("id", "")
        logger.info("Posted to LinkedIn: %s", post_id)
        return post_id

    if response.status_code == 401:
        raise LinkedInError("LinkedIn rejected the access token (401). It may have expired.")
    if response.status_code == 403:
        raise LinkedInError(
            "LinkedIn denied the post (403). The token is likely missing the "
            "'w_member_social' permission."
        )
    if response.status_code == 422:
        raise LinkedInError(
            "LinkedIn rejected the post content (422). Check that the author URN "
            "matches the token's owner."
        )
    raise LinkedInError(f"LinkedIn returned {response.status_code}: {response.text[:200]}")


def _compose(role_title: str, jd_text: str, form_url: str) -> str:
    """Build the post body: hook, link, then as much of the JD as fits.

    The apply link goes near the top on purpose — LinkedIn truncates long posts
    behind a "see more" fold, and a link below it is effectively invisible.
    """
    header = f"We're hiring: {role_title}\n\nApply here: {form_url}\n\n"
    remaining = MAX_COMMENTARY - len(header)
    excerpt = (jd_text or "").strip()
    if len(excerpt) > remaining:
        excerpt = excerpt[:remaining].rsplit(" ", 1)[0] + "…"
    return header + excerpt
