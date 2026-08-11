"""
GitHub profile analysis.

For an engineering role a public repository history is the strongest available
evidence that someone can actually build things — a resume asserts, code
demonstrates. This module turns a username into measurable signals:

    breadth   which languages, weighted by how much code is actually in them
    depth     stars/forks earned, repository size, documentation discipline
    recency   days since last push, repos touched in the last year
    substance ratio of original repositories to forks

Deliberate limits, because signal quality matters more than signal quantity:

* Forks are excluded from language and star aggregates. Forking Linux does not
  make someone a kernel developer.
* Star counts are compressed logarithmically — the difference between 0 and 10
  stars is meaningful, between 900 and 1000 is not.
* Absence is never punished. A candidate with no GitHub is marked
  ``found=False`` and the GitHub dimension is dropped with its weight
  redistributed, rather than scored zero. People with heavy job commitments,
  proprietary-only work, or caregiving responsibilities have thin public
  profiles for reasons unrelated to competence, and penalising that would bake
  a bias straight into the ranking.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import requests

from matching.config import MatchingConfig, get_config
from matching.schemas import GitHubProfile, RepoSummary

logger = logging.getLogger(__name__)


class GitHubClient:
    """Thin GitHub REST client with explicit rate-limit reporting."""

    def __init__(self, config: MatchingConfig | None = None):
        self.config = config or get_config()
        self.api = self.config.github_api.rstrip("/")
        self._session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hireai-matching-engine",
        }
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"
        self._session.headers.update(headers)

    def _get(self, path: str, params: dict | None = None):
        """Return parsed JSON, or raise ``GitHubError`` with an actionable message."""
        url = f"{self.api}{path}"
        try:
            response = self._session.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            raise GitHubError(f"network error contacting GitHub: {exc}") from exc

        if response.status_code == 404:
            raise GitHubNotFound(f"no such GitHub resource: {path}")

        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining == "0":
                reset = response.headers.get("X-RateLimit-Reset", "")
                hint = (
                    "Set GITHUB_TOKEN to raise the limit from 60 to 5000 requests/hour."
                    if not self.config.github_token
                    else "Wait for the window to reset."
                )
                raise GitHubRateLimited(f"GitHub rate limit exhausted (resets at {reset}). {hint}")
            raise GitHubError(f"GitHub refused the request: {response.text[:200]}")

        if response.status_code >= 400:
            raise GitHubError(f"GitHub returned {response.status_code}: {response.text[:200]}")

        return response.json()

    def user(self, username: str) -> dict:
        return self._get(f"/users/{username}")

    def repos(self, username: str, limit: int) -> list[dict]:
        """Fetch up to ``limit`` repositories, most recently pushed first."""
        out: list[dict] = []
        page = 1
        while len(out) < limit and page <= 4:
            batch = self._get(
                f"/users/{username}/repos",
                params={"per_page": 100, "page": page, "sort": "pushed", "type": "owner"},
            )
            if not isinstance(batch, list) or not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    def languages(self, username: str, repo: str) -> dict[str, int]:
        return self._get(f"/repos/{username}/{repo}/languages") or {}


class GitHubError(RuntimeError):
    """GitHub could not be queried."""


class GitHubNotFound(GitHubError):
    """The username does not exist."""


class GitHubRateLimited(GitHubError):
    """The API rate limit is exhausted."""


def analyse_github(
    username: str,
    config: MatchingConfig | None = None,
    client: GitHubClient | None = None,
) -> GitHubProfile:
    """Build a ``GitHubProfile``. Never raises — failures land in ``error``."""
    config = config or get_config()
    username = (username or "").strip().strip("@/").lower()

    if not username:
        return GitHubProfile(found=False, error="no username provided")

    client = client or GitHubClient(config)

    try:
        user = client.user(username)
    except GitHubNotFound:
        return GitHubProfile(username=username, found=False, error="user not found")
    except GitHubError as exc:
        logger.warning("GitHub lookup failed for %s: %s", username, exc)
        return GitHubProfile(username=username, found=False, error=str(exc))

    try:
        repos = client.repos(username, config.github_max_repos)
    except GitHubError as exc:
        logger.warning("GitHub repo listing failed for %s: %s", username, exc)
        repos = []

    profile = GitHubProfile(
        username=username,
        found=True,
        name=user.get("name") or "",
        bio=user.get("bio") or "",
        company=user.get("company") or "",
        blog=user.get("blog") or "",
        public_repos=user.get("public_repos") or 0,
        followers=user.get("followers") or 0,
        following=user.get("following") or 0,
        account_created_at=user.get("created_at") or "",
    )
    profile.account_age_days = _days_since(profile.account_created_at) or 0

    now = datetime.now(UTC)
    originals = [r for r in repos if not r.get("fork")]
    profile.original_repos = len(originals)
    profile.forked_repos = len(repos) - len(originals)

    documented = 0
    last_push_days: list[int] = []

    for repo in originals:
        profile.total_stars += repo.get("stargazers_count") or 0
        profile.total_forks += repo.get("forks_count") or 0

        # Repository size (KB) as a proxy for how much code sits in the primary
        # language. The exact per-language byte counts need one extra API call
        # per repository; with no token that budget (60/hr) is gone after a
        # single candidate, so the detailed call is reserved for the top repos
        # when a token is configured.
        language = (repo.get("language") or "").strip()
        if language:
            profile.language_bytes[language] = (
                profile.language_bytes.get(language, 0) + max(1, repo.get("size") or 0)
            )

        if (repo.get("description") or "").strip():
            documented += 1

        days = _days_since(repo.get("pushed_at"))
        if days is not None:
            last_push_days.append(days)
            if days <= 365:
                profile.repos_pushed_last_year += 1

    if config.github_token and originals:
        _enrich_language_bytes(client, username, originals[:10], profile)

    profile.documented_repo_ratio = documented / len(originals) if originals else 0.0
    profile.days_since_last_push = min(last_push_days) if last_push_days else None

    profile.top_repos = [
        RepoSummary(
            name=r.get("name") or "",
            description=(r.get("description") or "")[:300],
            language=r.get("language") or "",
            stars=r.get("stargazers_count") or 0,
            forks=r.get("forks_count") or 0,
            size_kb=r.get("size") or 0,
            is_fork=bool(r.get("fork")),
            is_archived=bool(r.get("archived")),
            topics=list(r.get("topics") or [])[:8],
            pushed_at=r.get("pushed_at") or "",
            created_at=r.get("created_at") or "",
            has_description=bool((r.get("description") or "").strip()),
        )
        for r in sorted(
            originals,
            key=lambda r: (r.get("stargazers_count") or 0, r.get("size") or 0),
            reverse=True,
        )[:8]
    ]

    logger.info(
        "GitHub %s: %d original repos, %d stars, last push %s days ago",
        username, profile.original_repos, profile.total_stars, profile.days_since_last_push,
    )
    _ = now  # retained for clarity of the recency computations above
    return profile


def _enrich_language_bytes(
    client: GitHubClient, username: str, repos: list[dict], profile: GitHubProfile
) -> None:
    """Replace size-proxy language weights with true byte counts for top repos."""
    detailed: dict[str, int] = {}
    for repo in repos:
        name = repo.get("name")
        if not name:
            continue
        try:
            for language, byte_count in client.languages(username, name).items():
                detailed[language] = detailed.get(language, 0) + int(byte_count)
        except GitHubRateLimited:
            logger.info("Rate limit hit during language enrichment; keeping size proxy")
            return
        except GitHubError:
            continue
    if detailed:
        profile.language_bytes = detailed


def _days_since(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0, (datetime.now(UTC) - moment).days)
