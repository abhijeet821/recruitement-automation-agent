from django.contrib import admin

from hiring_app.models import BackgroundJob, Campaign, Candidate, GoogleOAuthToken


@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = ("role_title", "owner", "status", "candidate_count", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("role_title", "owner__username")
    readonly_fields = ("created_at", "updated_at", "last_synced_at")


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = (
        "email", "campaign", "overall_score", "recommendation",
        "confidence", "recruiter_rating", "status",
    )
    list_filter = ("status", "recommendation", "campaign")
    search_fields = ("email", "full_name", "github_username")
    readonly_fields = ("created_at", "updated_at", "scored_at")


@admin.register(BackgroundJob)
class BackgroundJobAdmin(admin.ModelAdmin):
    list_display = ("kind", "campaign", "status", "processed", "total", "created_at")
    list_filter = ("kind", "status")
    readonly_fields = ("created_at", "started_at", "finished_at")


@admin.register(GoogleOAuthToken)
class GoogleOAuthTokenAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "last_refreshed_at")
    # The token column is deliberately not listed or searchable — it holds
    # encrypted credentials and must not be casually exposed in the admin.
    exclude = ("encrypted_token",)
    readonly_fields = ("created_at", "updated_at", "last_refreshed_at", "scopes")
