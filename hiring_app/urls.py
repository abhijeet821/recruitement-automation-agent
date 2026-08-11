from django.contrib.auth.views import LogoutView
from django.urls import path

from hiring_app import views

urlpatterns = [
    # Public
    path("", views.landing, name="landing"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", LogoutView.as_view(), name="logout"),

    # Google OAuth
    path("google/connect/", views.google_connect, name="google_connect"),
    path("google/oauth2callback/", views.google_oauth_callback, name="google_oauth_callback"),
    path("google/disconnect/", views.google_disconnect, name="google_disconnect"),

    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),

    # Campaigns
    path("campaign/new/", views.campaign_new, name="campaign_new"),
    path("campaign/<int:campaign_id>/", views.campaign_detail, name="campaign_detail"),
    path("campaign/<int:campaign_id>/delete/", views.campaign_delete, name="campaign_delete"),
    path("campaign/<int:campaign_id>/jd/generate/", views.generate_jd, name="generate_jd"),
    path("campaign/<int:campaign_id>/jd/save/", views.save_jd, name="save_jd"),
    path("campaign/<int:campaign_id>/launch/", views.launch_campaign, name="launch_campaign"),

    # Background work
    path("campaign/<int:campaign_id>/sync/", views.sync_campaign_view, name="sync_campaign"),
    path("campaign/<int:campaign_id>/rescore/", views.rescore_campaign_view, name="rescore_campaign"),
    path("campaign/<int:campaign_id>/job-status/", views.job_status, name="job_status"),

    # Candidates
    path(
        "campaign/<int:campaign_id>/candidate/<int:candidate_id>/",
        views.candidate_detail, name="candidate_detail",
    ),
    path(
        "campaign/<int:campaign_id>/candidate/<int:candidate_id>/rate/",
        views.rate_candidate, name="rate_candidate",
    ),

    # Actions
    path("campaign/<int:campaign_id>/invites/", views.send_invites, name="send_invites"),
    path("campaign/<int:campaign_id>/outcomes/", views.send_outcomes, name="send_outcomes"),
]
