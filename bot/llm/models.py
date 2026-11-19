# Shared ensemble roster, reused across pipelines. MODEL_A, MODEL_B and MODEL_C
# are OpenRouter free-tier models; "google:" entries go to Google's own API
# (see openrouter.call_model_json).
# MODEL_A is tried first in most fallback chains — empirically nemotron has
# succeeded on essentially every call, while gemma hits 429 from Google AI
# Studio's shared free-tier pool on most attempts. Leading with the reliable
# one cuts ~15-20s of wasted retries per call. Gemma stays in the pool
# (tried second) for ensemble diversity.
#
# thinkingmachines/inkling:free was dropped entirely (not just deprioritized):
# it 403s on every call with "only available on agentic harnesses" — it will
# never work for a plain bot integration like this one, confirmed via direct
# API test, not just a transient outage. Replaced with a genuinely
# vision-capable free model (verified via direct API test, including an
# actual image) served by a different provider (Novita, not Google AI
# Studio), so it doesn't share gemma's congestion either.
#
# That replacement, inclusionai/ling-3.0-flash-vl:free, disappeared from
# OpenRouter's free list on or before 2026-10-09 (every call 404s). Its
# successor is dots-3-note-preview: a different lab again, and a live test
# that day showed it answering JSON for both text and an image. It is a
# "preview" model, so expect it to vanish one day too; every chain has
# other entries to fall back on.
MODEL_A = "nvidia/nemotron-3-ultra-550b-a55b:free"
MODEL_B = "google/gemma-4-31b-it:free"
MODEL_C = "dots-studio/dots-3-note-preview:free"

# The same gemma as MODEL_B, called straight on Google's Gemini API with the
# owner's own key. Free-tier limits there (AI Studio, 2026-10-09): 30 requests
# per minute, 16K tokens per minute and 14,400 requests per day per Gemma
# model — against OpenRouter's 50 requests per day for all its free models
# together — and the quota is the owner's own instead of a shared pool. It
# reads images too (verified). Skipped when GEMINI_API_KEY isn't set.
MODEL_B_GOOGLE = "google:gemma-4-31b-it"

# Wherever a chain used gemma it now uses this pair: Google's copy first,
# OpenRouter's as its backup. A chain that contains it survives a spent
# OpenRouter quota, because call_model_json still tries the "google:" entries.
GEMMA = [MODEL_B_GOOGLE, MODEL_B]
