"""
Configuration module for re-skin.
Loads settings from environment variables using python-dotenv.

Relative paths (GOOGLE_SERVICE_ACCOUNT_FILE, DATABASE_URL) are resolved against
BASE_DIR so they work correctly regardless of the process working directory.

Inside Docker containers, WORKDIR is /app and ./data is mounted at /app/data,
so a relative path like ./secrets/gdrive-sa.json resolves to /app/secrets/gdrive-sa.json
and sqlite:///./data/app.db resolves to sqlite:////app/data/app.db.

Locally, BASE_DIR is the backend/ directory (two levels up from this file),
which keeps the same structure as before.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (no-op in containers where vars are injected; harmless).
load_dotenv()

# Single authoritative base directory.  Override via APP_BASE_DIR if needed.
# Default: inside container => /app  (the WORKDIR); locally => backend/
BASE_DIR: Path = Path(
    os.environ.get("APP_BASE_DIR", str(Path(__file__).parent.parent))
).resolve()


def _resolve_path(raw: str) -> str:
    """Return *raw* as an absolute path, resolved against BASE_DIR if relative."""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def _resolve_db_url(url: str) -> str:
    """Resolve the file path inside a sqlite:/// URL against BASE_DIR."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        rest = url[len(prefix):]
        if rest and rest != ":memory:":
            return f"{prefix}{_resolve_path(rest)}"
    return url


def _assistant_edit_timeout_sec() -> int:
    """Trusted local render limit; configured independently of its SQL lease."""
    try:
        value = int(os.getenv("ASSISTANT_EDIT_TIMEOUT_SEC", "600"))
    except ValueError:
        raise ValueError("ASSISTANT_EDIT_TIMEOUT_SEC must be an integer within [30,600]") from None
    if not 30 <= value <= 600:
        raise ValueError("ASSISTANT_EDIT_TIMEOUT_SEC must be an integer within [30,600]")
    return value


class Settings:
    """Application settings loaded from environment."""

    # Private operator beta: disabled and deny-all unless explicitly configured.
    ASSISTANT_ENABLED: bool = os.getenv("ASSISTANT_ENABLED", "").lower() == "true"
    ASSISTANT_AGENT_ENABLED: bool = os.getenv("ASSISTANT_AGENT_ENABLED", "").lower() == "true"
    ASSISTANT_PROJECT_IDS: str = os.getenv("ASSISTANT_PROJECT_IDS", "")
    ASSISTANT_ALL_PROJECTS: bool = os.getenv("ASSISTANT_ALL_PROJECTS", "").lower() == "true"
    # Knowledge base (AICP-293, docs/knowledge-feedback-design.md §10): all off
    # by default. KNOWLEDGE_PROJECT_IDS narrows the assistant project scope
    # further ("" = no projects, "*" = the whole assistant scope).
    KNOWLEDGE_ENABLED: bool = os.getenv("KNOWLEDGE_ENABLED", "").lower() == "true"
    KNOWLEDGE_PROJECT_IDS: str = os.getenv("KNOWLEDGE_PROJECT_IDS", "")
    KNOWLEDGE_CAPTURE_ENABLED: bool = os.getenv("KNOWLEDGE_CAPTURE_ENABLED", "").lower() == "true"
    # The knowledge tick's automatic historical import. Off by default: the
    # importer is not yet limited to KNOWLEDGE_PROJECT_IDS (it reads every
    # project), so it runs only by explicit choice (AICP-298).
    KNOWLEDGE_IMPORT_TICK: bool = os.getenv("KNOWLEDGE_IMPORT_TICK", "").lower() == "true"
    KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA: bool = os.getenv("KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA", "").lower() == "true"
    KNOWLEDGE_FEEDBACK_UI: bool = os.getenv("KNOWLEDGE_FEEDBACK_UI", "").lower() == "true"
    KNOWLEDGE_ADVISOR_MODE: str = os.getenv("KNOWLEDGE_ADVISOR_MODE", "off")
    ASSISTANT_BROKER_SOCKET: str = os.getenv(
        "ASSISTANT_BROKER_SOCKET", "/run/reskin-assistant/broker.sock"
    )
    ASSISTANT_ASR_SOCKET: str = os.getenv("ASSISTANT_ASR_SOCKET", "")
    # A 29.7s portrait fade plus its preview measured 154.1s with unchanged
    # quality at the supervisor's 2 CPU limit, exceeding the former 120s cap.
    # This is one bounded total edit deadline; the 120s SQL lease is still
    # renewed and cancellation stays live.
    ASSISTANT_EDIT_TIMEOUT_SEC: int = _assistant_edit_timeout_sec()

    # kie.ai (Seedance)
    KIE_API_KEY: str = os.getenv("KIE_API_KEY", "")

    # kie.ai image-to-image model behind the Run reference-preprocessing phase
    # and the per-segment reference regeneration (docs/gpt-image-2-preprocess.md).
    # Env-overridable on purpose: the three variants below share ONE request
    # schema (same `input_urls` / `aspect_ratio` / `resolution` fields, same
    # response), so switching between them — or rolling back — needs no code
    # change and no image rebuild. It DOES need `docker compose up -d
    # --scale worker=2` (a recreate): `env_file` is baked into a container at
    # creation, so `docker compose restart` keeps the OLD value — verified
    # 2026-09-12.
    #   gpt-image-2-5-flare-image-to-image     default; OpenAI's own default
    #                                          variant, positioned for volume
    #   gpt-image-2-5-sunburst-image-to-image  same price (6 credits at 1K),
    #                                          positioned as "most capable"
    #   gpt-image-2-image-to-image             the pre-2026-09 model; rollback
    KIE_IMAGE_MODEL: str = os.getenv(
        "KIE_IMAGE_MODEL", "gpt-image-2-5-flare-image-to-image"
    )
    # Output resolution for that model: 1K | 2K | 4K. 1K is what every run has
    # used to date and costs 6 credits; 2K costs 10 and is the only tier that
    # exceeds a 1080p source frame (1K comes back 941px wide on a 9:16 input).
    # NOTE: 2K/4K with aspect_ratio="auto" is a hard create-failure on
    # gpt-image-2 and only became legal with 2.5 — verified live 2026-09-12.
    KIE_IMAGE_RESOLUTION: str = os.getenv("KIE_IMAGE_RESOLUTION", "1K")

    # Google Drive
    GOOGLE_SERVICE_ACCOUNT_FILE: str = _resolve_path(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "./secrets/gdrive-sa.json")
    )
    GDRIVE_DEFAULT_FOLDER_ID: str = os.getenv("GDRIVE_DEFAULT_FOLDER_ID", "")

    # Google Drive folder that burned-in overlay results are auto-uploaded to.
    OVERLAY_GDRIVE_FOLDER_ID: str = os.getenv(
        "OVERLAY_GDRIVE_FOLDER_ID", ""
    )

    # Google Drive folder for database backups (used by scripts/backup_db.sh).
    GDRIVE_BACKUP_FOLDER_ID: str = os.getenv(
        "GDRIVE_BACKUP_FOLDER_ID", ""
    )

    # Google Sheets — "Final Creatives" export (reuses GOOGLE_SERVICE_ACCOUNT_FILE).
    # The target spreadsheet must be shared with the service account email.
    FINAL_CREATIVES_SPREADSHEET_ID: str = os.getenv(
        "FINAL_CREATIVES_SPREADSHEET_ID",
        "",
    )
    FINAL_CREATIVES_TAB: str = os.getenv("FINAL_CREATIVES_TAB", "Final Creatives")

    # Run automation — scheduled batch generation from the persona spreadsheet
    # (see app/automation.py). The spreadsheet must be shared with the
    # service-account email. Filters/quota are configured per Run in the UI.
    AUTOMATION_SPREADSHEET_ID: str = os.getenv(
        "AUTOMATION_SPREADSHEET_ID",
        "",
    )
    AUTOMATION_SHEET_TAB: str = os.getenv("AUTOMATION_SHEET_TAB", "Daily Queue")
    # Default webhook for batch-completion notifications; a per-automation
    # webhook_url (DB) overrides it. No code default (the URL carries a
    # secret path): unset -> webhooks disabled. https only; see
    # app/webhook_safety.py.
    AUTOMATION_WEBHOOK_URL: str = os.getenv("AUTOMATION_WEBHOOK_URL", "").strip()
    # Comma-separated hosts a webhook may target (default: the host of
    # AUTOMATION_WEBHOOK_URL).
    AUTOMATION_WEBHOOK_ALLOWED_HOSTS: str = os.getenv(
        "AUTOMATION_WEBHOOK_ALLOWED_HOSTS", ""
    )
    # Max copies launched per automation per tick (also the in-flight cap:
    # the next batch only starts after the previous one completes+notifies).
    AUTOMATION_MAX_PER_BATCH: int = int(os.getenv("AUTOMATION_MAX_PER_BATCH", "10"))

    # Daily kie.ai spend cap (AICP-155, see app/spend.py). Every NEW
    # credit-spending action (run create/copy/copy-batch/automation launch,
    # retry, rerun, range rerun, GPT Image regenerations, transcription,
    # localisation, captions) reserves its estimated cost against the current
    # UTC day and is refused with HTTP 429 once the day's total would exceed
    # this many USD. Work already admitted (an in-flight run's remaining
    # segments, poll-loop resubmits, recovery) is never refused — it is only
    # measured. Unset -> 250 (the owner's 2026-10-16 decision), so the cap is
    # ON without an .env change. "0" (or any value <= 0) DISABLES enforcement;
    # spend is still recorded and shown in the header.
    KIE_DAILY_BUDGET_USD: float = float(os.getenv("KIE_DAILY_BUDGET_USD", "250"))
    # USD per kie.ai credit. kie.ai's pricing page lists prices in both units
    # (e.g. 20 credits = $0.10), i.e. 200 credits per USD.
    KIE_USD_PER_CREDIT: float = float(os.getenv("KIE_USD_PER_CREDIT", "0.005"))
    # Flat estimate for one LLM call (Gemini transcription, GPT translation) —
    # kie.ai bills those by tokens, so this is a deliberately generous guess
    # (20 credits = $0.10). It is what admission RESERVES per call. The rebuild
    # storyboard / character-check calls then settle their draw to the
    # response's own ``credits_consumed`` (AICP-382: ~0.25 cr a shot call,
    # ~2 cr the global call, ~0.45 cr a character check), so this stays the
    # booked figure only for calls whose response carries no such field.
    KIE_LLM_CALL_EST_CREDITS: float = float(
        os.getenv("KIE_LLM_CALL_EST_CREDITS", "20")
    )
    # --- Fal, the RESERVE provider (AICP-312 Stage 1 / AICP-330) -------------
    # docs/fal-provider-design.md §9. kie.ai stays primary; nothing routes to
    # Fal by default. FAL_ENABLED gates ADMISSION only: with it off, tasks that
    # already exist keep being polled, fetched and recovered while the key is
    # present, so turning it off in an incident never orphans paid work.
    # Validated at fal-worker start-up by pipeline_provider.validate_settings().
    FAL_API_KEY: str = os.getenv("FAL_API_KEY", "").strip()
    FAL_ENABLED: bool = os.getenv("FAL_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    # Explicit per-project allowlist (comma-separated UUIDs); empty = none.
    FAL_PROJECT_IDS: str = os.getenv("FAL_PROJECT_IDS", "")
    # Per-input cap of the single-POST CDN upload (rule 4.3).
    FAL_MAX_INPUT_BYTES: int = int(os.getenv("FAL_MAX_INPUT_BYTES", str(90 * 1024 * 1024)))
    # RQ timeout of every claim-holding Fal job; the claim lease outlasts it.
    FAL_JOB_TIMEOUT: int = int(os.getenv("FAL_JOB_TIMEOUT", "1800"))
    FAL_CLAIM_LEASE_SEC: int = int(
        os.getenv("FAL_CLAIM_LEASE_SEC", str(int(os.getenv("FAL_JOB_TIMEOUT", "1800")) + 300))
    )
    # `submitted` is bounded from accepted_at; reaching it requests a cancel.
    FAL_POLL_DEADLINE_SEC: int = int(os.getenv("FAL_POLL_DEADLINE_SEC", "3600"))
    # An unconfirmed cancel moves to needs_reconciliation after this long.
    FAL_CANCEL_CONFIRM_SEC: int = int(os.getenv("FAL_CANCEL_CONFIRM_SEC", "21600"))
    FAL_POLL_INTERVAL_SEC: int = int(os.getenv("FAL_POLL_INTERVAL_SEC", "15"))
    # needs_reconciliation keeps being polled, slowly.
    FAL_SLOW_POLL_SEC: int = int(os.getenv("FAL_SLOW_POLL_SEC", "600"))
    # A `queued` row with no claim and no live job is re-dispatched after this.
    FAL_QUEUED_GRACE_SEC: int = int(os.getenv("FAL_QUEUED_GRACE_SEC", "120"))
    # Fetch: at most this many attempts, within this window counted from the
    # FIRST started attempt (AICP-330 B1).
    FAL_FETCH_MAX_ATTEMPTS: int = int(os.getenv("FAL_FETCH_MAX_ATTEMPTS", "5"))
    FAL_FETCH_WINDOW_SEC: int = int(os.getenv("FAL_FETCH_WINDOW_SEC", "3600"))
    # Retention Fal applies to OUTPUT media (submit header) and to uploaded
    # INPUTS (upload header). Both must outlast what can happen before we, or
    # Fal, read them; validate_settings() checks the bounds.
    FAL_OUTPUT_LIFECYCLE_SEC: int = int(os.getenv("FAL_OUTPUT_LIFECYCLE_SEC", "86400"))
    FAL_INPUT_LIFECYCLE_SEC: int = int(os.getenv("FAL_INPUT_LIFECYCLE_SEC", "28800"))
    # ffprobe check of a downloaded result: duration within this of output_sec.
    FAL_OUTPUT_DURATION_TOLERANCE_SEC: float = float(
        os.getenv("FAL_OUTPUT_DURATION_TOLERANCE_SEC", "1.5")
    )
    # Fal recovery tick of the fal-worker.
    FAL_TICK_INTERVAL_SEC: int = int(os.getenv("FAL_TICK_INTERVAL_SEC", "60"))

    # "Rebuild from reference" project type (docs/rebuild-from-reference-design.md
    # §7.6). Off by default: while false create_project refuses
    # project_type=rebuild, create_run refuses kling-4-0 and images-only runs, and
    # the type is hidden in the Create Project / Move-to-project forms, so neither
    # new enum label (project_type 'rebuild', run_model 'kling-4-0') can reach a
    # row. Existing rebuild rows stay readable either way. `.env`, needs `up -d`.
    REBUILD_ENABLED: bool = os.getenv("REBUILD_ENABLED", "false").strip().lower() in (
        "1", "true", "yes",
    )

    # "by Assistant" autopilot (AICP-363, docs/autopilot-design.md). Off by
    # default: with it false no assistant project can be created and the
    # autopilot ticker does nothing (staged rollout, design §9).
    AUTOPILOT_ENABLED: bool = os.getenv("AUTOPILOT_ENABLED", "").lower() == "true"
    # Deploy admission gate (AICP-350, app/ops_gate.py): after this many minutes
    # paused the dashboard banner turns red and shows the force-end command.
    # The gate itself never expires (fail closed).
    OPS_GATE_WARN_AFTER_MIN: int = int(os.getenv("OPS_GATE_WARN_AFTER_MIN", "45"))
    AUTOPILOT_TICK_INTERVAL_SEC: int = int(os.getenv("AUTOPILOT_TICK_INTERVAL_SEC", "30"))
    AUTOPILOT_DEFAULT_TOLERANCE: float = float(os.getenv("AUTOPILOT_DEFAULT_TOLERANCE", "0.10"))
    AUTOPILOT_STALL_SEC: int = int(os.getenv("AUTOPILOT_STALL_SEC", "10800"))
    AUTOPILOT_REJECT_GRACE_SEC: int = int(os.getenv("AUTOPILOT_REJECT_GRACE_SEC", "600"))
    # Subtitle-removal flow: a CONFIRMED located caption band (the review's
    # band.verdict "located") is saved without an eye check. "false" asks the
    # operator for every located band (docs/autopilot-subtitle-flow.md §1.1).
    AUTOPILOT_AUTO_CAPTION_BAND: bool = (
        os.getenv("AUTOPILOT_AUTO_CAPTION_BAND", "true").lower() != "false")
    # Chat notifications of the assistant's questions (AICP-374,
    # docs/autopilot-operator-ux.md §2): a Telegram bot called directly. Off
    # unless both the token and the chat id are set; also needs
    # PUBLIC_BASE_URL for the links. Independent of AUTOPILOT_ENABLED and the
    # deploy hold. The token is a secret: never logged, never in Linear.
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    AUTOPILOT_NOTIFY_INTERVAL_SEC: int = int(os.getenv("AUTOPILOT_NOTIFY_INTERVAL_SEC", "10"))

    # Reframe workflow (AICP-251, docs/reframe-design.md). The longest window
    # one generation may cover (R6; the model takes 300 s), how far the pinned
    # source audio runs past the window end, and the RQ timeouts of its local
    # jobs. A claim lease is always the job's RQ timeout + 300 s, so a live
    # worker never outlives its claim.
    REFRAME_MAX_WINDOW_SEC: float = float(os.getenv("REFRAME_MAX_WINDOW_SEC", "120"))
    REFRAME_AUDIO_RUNON_MAX_SEC: float = float(os.getenv("REFRAME_AUDIO_RUNON_MAX_SEC", "3"))
    REFRAME_PREP_JOB_TIMEOUT: int = int(os.getenv("REFRAME_PREP_JOB_TIMEOUT", "1800"))
    # AICP-358 segment caption fill: RQ timeout of prepare/composite, and the
    # claim lease (longer than the timeout, so a live job never loses its claim).
    SEGMENT_FILL_JOB_TIMEOUT: int = int(os.getenv("SEGMENT_FILL_JOB_TIMEOUT", "1800"))
    SEGMENT_FILL_LEASE_SEC: int = int(os.getenv("SEGMENT_FILL_LEASE_SEC", "2400"))
    # Batch VOID fills (2026-11-03, docs/segment-fill-batch-design.md):
    # members per batch; how long a member whose Fal task only the operator can
    # resolve may hold the batch once every other member settled (then it is
    # detached and the rest promote); the default per-run VOID allowance the
    # autopilot starts without asking (a project's void_auto_usd_per_run wins);
    # auto-accept of detector-clean composites whose mask touches no face.
    SEGMENT_FILL_BATCH_MAX: int = int(os.getenv("SEGMENT_FILL_BATCH_MAX", "12"))
    SEGMENT_FILL_BATCH_HOLD_SEC: int = int(os.getenv("SEGMENT_FILL_BATCH_HOLD_SEC", "1800"))
    AUTOPILOT_VOID_AUTO_USD: float = float(os.getenv("AUTOPILOT_VOID_AUTO_USD", "1.0"))
    AUTOPILOT_AUTO_ACCEPT_FILLS: bool = (
        os.getenv("AUTOPILOT_AUTO_ACCEPT_FILLS", "true").lower() != "false")
    # AICP-389: the localisation QA round's free brand check (trailing "-s")
    # and its fix asks. Off by default: enable only after the brand-asr
    # container's RSS/wall time were measured on a real final. Turning it off
    # stops NEW checks and refuses NEW fix admissions; a patch already
    # admitted still finishes (rollback: docs/brand-trailing-sibilant.md §7).
    AUTOPILOT_BRAND_FIX_ENABLED: bool = (
        os.getenv("AUTOPILOT_BRAND_FIX_ENABLED", "").lower() == "true")
    # The local word-ASR model of the brand-asr container (read-only mount).
    BRAND_ASR_MODEL_DIR: str = os.getenv("BRAND_ASR_MODEL_DIR", "/model")
    REFRAME_QA_JOB_TIMEOUT: int = int(os.getenv("REFRAME_QA_JOB_TIMEOUT", "900"))
    REFRAME_RENDER_JOB_TIMEOUT: int = int(os.getenv("REFRAME_RENDER_JOB_TIMEOUT", "1800"))

    # Base URL used to build absolute run/result links in webhook payloads,
    # e.g. "https://reskin.example.com:8847". Empty → links are omitted.
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    # Extra browser origins allowed to POST/PUT/PATCH/DELETE (comma-separated,
    # e.g. "http://localhost:8847" for an SSH tunnel). The request's own Host
    # and PUBLIC_BASE_URL are always trusted; see app/csrf.py.
    CSRF_TRUSTED_ORIGINS: str = os.getenv("CSRF_TRUSTED_ORIGINS", "")

    # Web access
    BASIC_AUTH_USER: str = os.getenv("BASIC_AUTH_USER", "reskin")
    BASIC_AUTH_PASS: str = os.getenv("BASIC_AUTH_PASS", "")

    # AICP-151: dedicated secret for the unauthenticated /public/ links. There
    # is NO fallback to BASIC_AUTH_PASS (a leaked link would let anyone brute-
    # force the master password offline). Empty or shorter than
    # PUBLIC_LINK_SECRET_MIN_LEN -> public links are disabled (endpoints 503,
    # tokens are not minted) but the app still boots. Generate one with
    # `python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`.
    PUBLIC_LINK_SECRET: str = os.getenv("PUBLIC_LINK_SECRET", "").strip()
    PUBLIC_LINK_SECRET_MIN_LEN: int = 32
    # Optional, comma-separated OLD secrets whose tokens are still ACCEPTED
    # (verify-only, never used to sign) during a rotation grace period. Unset
    # = no transition. See app/public.py.
    PUBLIC_LINK_SECRET_PREVIOUS: str = os.getenv("PUBLIC_LINK_SECRET_PREVIOUS", "")

    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # DATABASE_URL — two supported forms:
    #   SQLite (default / tests):
    #     sqlite:///./data/app.db  (relative, resolved against BASE_DIR above)
    #     sqlite:///:memory:       (in-memory, used by the test suite)
    #   PostgreSQL (production):
    #     postgresql+psycopg2://<user>:<password>@<host>:5432/<dbname>
    #     e.g. postgresql+psycopg2://reskin:secret@db:5432/reskin
    #     Use the psycopg2 driver — it is pinned in requirements.txt.
    DATABASE_URL: str = _resolve_db_url(
        os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    )

    # App configuration
    # Resolution a new run gets when none is chosen (run form preselect +
    # create-run API). 720p since AICP-391; 480p is the test tier and stays
    # selectable. Rebuild runs use project_types.REBUILD_DEFAULT_RESOLUTION.
    DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "720p")
    MAX_REFERENCE_IMAGES: int = int(os.getenv("MAX_REFERENCE_IMAGES", "5"))
    # NOTE: SEGMENT_MAX_SECONDS used to live here as the global cap on swap
    # segment length. It is gone: the cap is now per-project
    # (VideoProject.max_segment_sec, NULL = ai_models.UNIVERSAL_MAX_SEGMENT_SEC)
    # and bounded by ai_models.ABSOLUTE_MAX_SEGMENT_SEC, which derives from the
    # model registry rather than an env var. A stale SEGMENT_MAX_SECONDS line in
    # a deployed .env is harmless — Settings reads named vars only.

    # Subtitle-removal projects auto-cut on scene changes (PySceneDetect) instead
    # of running face detection. Minimum scene length in seconds — the
    # `--min-scene-len 4s` default. Shorter cuts are merged into their neighbour.
    SCENE_MIN_LEN_SECONDS: float = float(os.getenv("SCENE_MIN_LEN_SECONDS", "4"))

    # Upload size cap (MiB).  Applies to source video and reference image uploads.
    # Set high enough for 1080p source videos (several hundred MB typical).
    # Nginx client_max_body_size must be >= this value.
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "1024"))

    # When True, large media endpoints return X-Accel-Redirect instead of
    # streaming through Python.  nginx must have the /internal/data/ location
    # configured (see deploy/nginx/nginx.conf) and ./data must be bind-mounted
    # into the nginx container.  Default False so dev/tests keep FileResponse.
    NGINX_ACCEL_ENABLED: bool = os.getenv("NGINX_ACCEL_ENABLED", "false").lower() in (
        "1", "true", "yes"
    )

    # Absolute path where ./data is mounted INSIDE the API container.
    # Used to convert absolute file paths to X-Accel-Redirect URIs.
    DATA_DIR: str = os.getenv("DATA_DIR", "")

    # --- Localisation (see docs/localisation.md, app/localisation.py) ---
    # Video-capable model that transcribes the source and labels speakers by
    # what is visible on screen. Called on the OpenAI-compatible kie.ai route
    # (POST https://api.kie.ai/{model}/v1/chat/completions), so the value is
    # both the model id AND a URL path segment.
    LOCALISATION_TRANSCRIBE_MODEL: str = os.getenv(
        "LOCALISATION_TRANSCRIBE_MODEL", "gemini-3.1-pro"
    )
    # Comma-separated models tried in order, each once with the same upload,
    # when the one before is unavailable (kie.ai 5xx/"being maintained" after
    # retries, 422 "not supported", or 404 "Invalid model name"). On 2026-09-25
    # gemini-2.5-pro was down for hours, then every Gemini Pro (3-pro, 3.1-pro)
    # followed while gemini-3-5-flash-openai still answered. By 2026-10-13
    # kie.ai had retired gemini-2.5-pro (404), so 3.1-pro became the primary,
    # and had deprecated gemini-3-5-flash (2026-10-12); kie.ai named 3.6 and
    # 3.8 Flash as the replacements (3.8 answered 500 on the day, so it's last).
    # All use the same chat route and image_url envelope, so any of them drops
    # in. Empty disables the fallback.
    LOCALISATION_TRANSCRIBE_FALLBACK_MODELS: str = os.getenv(
        "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS",
        "gemini-3-pro,gemini-3-6-flash-openai,gemini-3-8-flash-openai",
    )
    # Text model that rewrites the transcript into the target language. Called
    # on the kie.ai Responses route (POST https://api.kie.ai/codex/v1/responses)
    # where the model id travels in the BODY.
    #
    # gpt-6-1-sol since 2026-10-20: on a six-project A/B it scored 8.3/10 vs
    # 5.6 for the previous gpt-5-6-luna, but only with the v3 prompt's frames
    # and full-transcript context — luna given the same context stayed at 5.4.
    # ~$2/$10 per 1M tokens: cents per call against dollars of Seedance.
    LOCALISATION_TRANSLATE_MODEL: str = os.getenv(
        "LOCALISATION_TRANSLATE_MODEL", "gpt-6-1-sol"
    )
    # Comma-separated models tried in order when the one before is unavailable
    # (same semantics as LOCALISATION_TRANSCRIBE_FALLBACK_MODELS). Empty
    # disables the fallback.
    LOCALISATION_TRANSLATE_FALLBACK_MODELS: str = os.getenv(
        "LOCALISATION_TRANSLATE_FALLBACK_MODELS", "gpt-5-6-luna"
    )
    # reasoning.effort on the Responses route (low | medium | high | xhigh).
    # kie.ai defaults it to low server-side.
    LOCALISATION_TRANSLATE_EFFORT: str = os.getenv(
        "LOCALISATION_TRANSLATE_EFFORT", "high"
    )
    # Per-request timeout for the translate call alone. The endpoint that
    # makes it is synchronous behind nginx's 300s proxy_read_timeout, so the
    # shared 600s LOCALISATION_TIMEOUT_SEC would let one hung call outlive the
    # operator's 504 — and a retried click then pays for a second translation.
    # gpt-6-1-sol at effort high answered in 10-35s on the 2026-10-20 A/B.
    LOCALISATION_TRANSLATE_TIMEOUT_SEC: float = float(
        os.getenv("LOCALISATION_TRANSLATE_TIMEOUT_SEC", "90")
    )
    # AICP-338: the translate now runs as an RQ job on the dedicated
    # "localise" queue (app/localisation_prompt_job.py), not inside the
    # request, so nginx no longer bounds it. Inside that job the per-call
    # timeout above is raised to this value: a whole-video translation of a
    # 79s source took 178-281s per call on 2026-10-28 (project 46cc6443).
    LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC: float = float(
        os.getenv("LOCALISATION_TRANSLATE_JOB_CALL_TIMEOUT_SEC", "420")
    )
    # RQ job_timeout of one translation job. Floored at runtime to cover
    # KieClient's 4 transport attempts at the per-call timeout above plus the
    # shorten pass, frames and backoff (localisation_prompt_job.job_timeout_sec).
    LOCALISATION_PROMPT_JOB_TIMEOUT_SEC: int = int(
        os.getenv("LOCALISATION_PROMPT_JOB_TIMEOUT_SEC", "2700")
    )
    # Budgets of localisation.translate_lines INSIDE the job only (Operator,
    # 2026-10-30): the request-path defaults (120 s re-ask / fallback budget,
    # 60 s shorten admission, 45 s shorten call) were sized for nginx and
    # would deny a 180-280 s whole-video call both the 1:1 re-ask and the
    # shorten pass. LOCALISATION_PROMPT_JOB_TIMEOUT_SEC is floored to cover
    # them (localisation_prompt_job.job_timeout_sec).
    LOCALISATION_JOB_TRANSLATE_RETRY_BUDGET_SEC: float = float(
        os.getenv("LOCALISATION_JOB_TRANSLATE_RETRY_BUDGET_SEC", "300")
    )
    LOCALISATION_JOB_SHORTEN_ADMIT_SEC: float = float(
        os.getenv("LOCALISATION_JOB_SHORTEN_ADMIT_SEC", "600")
    )
    LOCALISATION_JOB_SHORTEN_TIMEOUT_SEC: float = float(
        os.getenv("LOCALISATION_JOB_SHORTEN_TIMEOUT_SEC", "300")
    )
    # How long an admitted job may WAIT in the queue before RQ discards it
    # (the operator then sees "expired" and presses Translate again).
    LOCALISATION_PROMPT_QUEUE_TTL_SEC: int = int(
        os.getenv("LOCALISATION_PROMPT_QUEUE_TTL_SEC", "1800")
    )
    # How long a finished (or failed) job's answer stays readable in Redis.
    LOCALISATION_PROMPT_RESULT_TTL_SEC: int = int(
        os.getenv("LOCALISATION_PROMPT_RESULT_TTL_SEC", "21600")
    )
    # Max still frames of the hook handed to the translator (0 disables).
    LOCALISATION_TRANSLATE_FRAMES: int = int(
        os.getenv("LOCALISATION_TRANSLATE_FRAMES", "8")
    )
    # Default localised-hook length in seconds when a project leaves hook_sec
    # NULL. Consumed at analyze time (same contract as max_segment_sec).
    LOCALISATION_DEFAULT_HOOK_SEC: float = float(
        os.getenv("LOCALISATION_DEFAULT_HOOK_SEC", "10")
    )
    # Longest swap segment a WHOLE-VIDEO localisation analysis proposes
    # (VideoProject.localisation_scope="whole", AICP-340). Not a model limit:
    # Seedance 2.5 accepts 30s clips but intermittently truncates its output to
    # exactly 14.72s, so a segment that fits inside 14.72s never loses its tail
    # to that failure. Never lower a model's max_clip_sec to get the same
    # effect — that blocks submission outright.
    LOCALISATION_WHOLE_VIDEO_MAX_SEC: float = float(
        os.getenv("LOCALISATION_WHOLE_VIDEO_MAX_SEC", "14.72")
    )
    # Per-request timeout for both LLM calls. Transcribing a 10s clip takes
    # ~30-60s; the 60s default on the shared KieClient is too tight once the
    # model has to fetch and decode the video, so these calls override it.
    LOCALISATION_TIMEOUT_SEC: float = float(
        os.getenv("LOCALISATION_TIMEOUT_SEC", "600")
    )


settings = Settings()
