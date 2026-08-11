"""External integrations and campaign orchestration."""

from hiring_app.services.google_auth import (
    OAuthNotConfigured,
    build_flow,
    has_google,
    load_credentials,
    save_credentials,
)
from hiring_app.services.google_workspace import (
    SCOPES,
    CredentialsExpired,
    WorkspaceClient,
    WorkspaceError,
)
from hiring_app.services.linkedin import LinkedInError, post_job
from hiring_app.services.screening import (
    SyncOutcome,
    ensure_job_spec,
    rescore_campaign,
    sync_campaign,
)

__all__ = [
    "OAuthNotConfigured",
    "build_flow",
    "has_google",
    "load_credentials",
    "save_credentials",
    "SCOPES",
    "CredentialsExpired",
    "WorkspaceClient",
    "WorkspaceError",
    "LinkedInError",
    "post_job",
    "SyncOutcome",
    "ensure_job_spec",
    "rescore_campaign",
    "sync_campaign",
]
