import os

from dotenv import load_dotenv

# The owner's .env is the source of truth: an old variable of the same name in
# the OS environment (GEMINI_API_KEY was one, left over from another tool) must
# not shadow it, which is what load_dotenv does by default.
load_dotenv(override=True)


def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


BOT_TOKEN = _require("BOT_TOKEN")
FIREBASE_CREDENTIALS_PATH = _require("FIREBASE_CREDENTIALS_PATH")
OPENROUTER_API_KEY = _require("OPENROUTER_API_KEY")
# Optional second model provider: Google's Gemini API (its own free quota).
# Without a key the bot runs on OpenRouter alone.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Optional: with a YouTube Data API key /hint attaches real video links.
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# One-time code the owner sends as `/claim <code>` to activate the bot.
CLAIM_CODE = os.environ.get("CLAIM_CODE", "")
# Streak days roll over at local midnight; Telegram doesn't expose a user's
# timezone, so it's a fixed UTC offset (default: UTC+5).
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "5"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
