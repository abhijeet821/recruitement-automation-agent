"""
Diagnose the environment.

    python manage.py doctor

Checks every external dependency the app needs and reports what is wrong and
how to fix it. Written because the common failure — "why is every candidate
scoring zero" — is almost always a missing Ollama model or an unset key, and
that should take five seconds to discover rather than an afternoon.
"""

from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection

from matching.config import get_config
from matching.llm import LLMUnavailable, get_provider


class Command(BaseCommand):
    help = "Check that the scoring engine, database and integrations are usable."

    def handle(self, *args, **options):
        config = get_config()
        problems: list[str] = []

        self.stdout.write(self.style.MIGRATE_HEADING("Configuration"))
        self._line("DEBUG", str(settings.DEBUG), ok=not settings.DEBUG or True)
        self._line("Provider", config.provider)
        self._line("Blind screening", str(config.blind_screening))
        self._line("Rubric enabled", str(config.rubric_enabled))
        self._line("Cache dir", str(config.cache_dir))

        # ── database ─────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nDatabase"))
        try:
            connection.ensure_connection()
            self._ok(f"connected ({connection.vendor})")
        except Exception as exc:  # noqa: BLE001
            self._fail(f"cannot connect: {exc}")
            problems.append("Database unreachable.")

        # ── LLM provider ─────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nScoring engine"))
        if config.provider == "ollama":
            self._check_ollama(config, problems)
        elif config.provider == "gemini":
            self._check_gemini(config, problems)
        else:
            self._warn(f"provider '{config.provider}' — no checks defined")

        # ── integrations ─────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nIntegrations"))
        if config.github_token:
            self._ok("GITHUB_TOKEN set (5000 requests/hour)")
        else:
            self._warn(
                "GITHUB_TOKEN not set — GitHub is limited to 60 requests/hour, "
                "roughly two candidates before throttling"
            )

        if settings.GOOGLE_CLIENT_SECRETS:
            self._ok("Google OAuth configured from the environment")
        elif (settings.BASE_DIR / "client_secrets.json").exists():
            self._ok("Google OAuth configured from client_secrets.json")
        else:
            self._warn(
                "Google OAuth not configured — campaigns cannot be launched. "
                "Scoring still works."
            )

        # ── summary ──────────────────────────────────────────
        self.stdout.write("")
        if problems:
            self.stdout.write(self.style.ERROR(f"{len(problems)} blocking problem(s):"))
            for problem in problems:
                self.stdout.write(f"  - {problem}")
        else:
            self.stdout.write(self.style.SUCCESS("Everything the scorer needs is available."))

    # ── checks ───────────────────────────────────────────────

    def _check_ollama(self, config, problems: list[str]) -> None:
        self._line("Host", config.ollama_host)
        try:
            provider = get_provider(config)
            installed = provider.list_models()
        except LLMUnavailable as exc:
            self._fail(str(exc))
            problems.append("Ollama is not reachable. Start it with: ollama serve")
            return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"unexpected error: {exc}")
            problems.append(str(exc))
            return

        self._ok(f"reachable — {len(installed)} model(s) installed")

        for label, wanted in (("chat", config.ollama_model), ("embedding", config.ollama_embed_model)):
            # Ollama reports "name" and "name:latest" interchangeably.
            if any(m == wanted or m.split(":")[0] == wanted.split(":")[0] for m in installed):
                self._ok(f"{label} model '{wanted}' present")
            else:
                self._fail(f"{label} model '{wanted}' missing")
                problems.append(f"Run: ollama pull {wanted}")

        try:
            vectors = provider.embed(["doctor check"])
            self._ok(f"embeddings working ({vectors.shape[1]} dimensions)")
        except Exception as exc:  # noqa: BLE001
            self._fail(f"embedding call failed: {exc}")
            problems.append("Embeddings are not working; scoring will be degraded.")

    def _check_gemini(self, config, problems: list[str]) -> None:
        if not config.gemini_api_key:
            self._fail("GEMINI_API_KEY is not set")
            problems.append("Set GEMINI_API_KEY, or switch to LLM_PROVIDER=ollama.")
            return
        try:
            provider = get_provider(config)
            ok, detail = provider.health()
        except Exception as exc:  # noqa: BLE001
            self._fail(str(exc))
            problems.append(str(exc))
            return
        if ok:
            self._ok(f"Gemini reachable ({config.gemini_model})")
        else:
            self._fail(detail)
            problems.append(f"Gemini unreachable: {detail}")

    # ── output helpers ───────────────────────────────────────

    def _line(self, label: str, value: str, ok: bool = True) -> None:
        self.stdout.write(f"  {label:<18} {value}")

    def _ok(self, message: str) -> None:
        self.stdout.write(self.style.SUCCESS(f"  [ok]   {message}"))

    def _warn(self, message: str) -> None:
        self.stdout.write(self.style.WARNING(f"  [warn] {message}"))

    def _fail(self, message: str) -> None:
        self.stdout.write(self.style.ERROR(f"  [fail] {message}"))
