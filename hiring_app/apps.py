import logging

from django.apps import AppConfig

logger = logging.getLogger("hiring_app")


class HiringAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hiring_app"

    # Cleanup of jobs orphaned by a process restart deliberately does NOT run
    # here. Querying the database from ready() runs before the app registry is
    # fully populated and breaks `migrate` on a fresh database. It is instead
    # done lazily on the first job operation — see hiring_app.jobs.
