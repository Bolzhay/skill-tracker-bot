import asyncio
import json
import logging
import re
from collections.abc import Callable

import httpx

from .. import config
from ..net import google_transport

# A roster entry is an OpenRouter model id, unless it starts with "google:" —
# then it goes straight to Google's Gemini API (OpenAI-compatible, with its own
# free quota) and needs GEMINI_API_KEY; without the key those entries are skipped.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GOOGLE_PREFIX = "google:"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 120

# A bad key, no credits left (402), a missing model or a rejected request:
# asking again won't change the answer.
PERMANENT_STATUSES = (400, 401, 402, 403, 404)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
# Gemma on Google's endpoint puts its reasoning into the reply, ahead of the answer.
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


class ModelCallError(Exception):
    pass


class RateLimitExhausted(ModelCallError):
    """OpenRouter's own account limit is spent (free tier: 50 requests/day).
    Retrying or switching to another OpenRouter model can't help."""


QUOTA_MESSAGE = (
    "Достигнут лимит запросов к бесплатным моделям OpenRouter (на бесплатном "
    "тарифе — 50 запросов в сутки на все модели). Подожди и попробуй позже; "
    "ничего не потеряно."
)


def failure_text(exc: Exception, default: str) -> str:
    """What to tell the user when a model call failed."""
    return QUOTA_MESSAGE if isinstance(exc, RateLimitExhausted) else default


def _is_google(model: str) -> bool:
    return model.startswith(GOOGLE_PREFIX)


def _endpoint(model: str) -> tuple[str, str, str]:
    """(url, api key, model id as that provider knows it) for a roster entry."""
    if _is_google(model):
        return GOOGLE_URL, config.GEMINI_API_KEY, model.removeprefix(GOOGLE_PREFIX)
    return OPENROUTER_URL, config.OPENROUTER_API_KEY, model


def _is_available(model: str) -> bool:
    return not _is_google(model) or bool(config.GEMINI_API_KEY)


def _is_permanent(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in PERMANENT_STATUSES


def _quota_exhausted(response: httpx.Response) -> bool:
    """A 429 from an overloaded upstream provider (the usual free-model
    flakiness) carries no rate-limit headers; OpenRouter's own limits do
    (X-RateLimit-*), and Remaining == 0 means the account's quota is gone."""
    return response.status_code == 429 and response.headers.get("x-ratelimit-remaining") == "0"


def _strip_code_fence(text: str) -> str:
    """Some free models ignore the 'return only JSON' instruction and wrap
    the object in a ```json ... ``` fence anyway. Not all providers support
    the response_format=json_object param (confirmed: inclusionai/ling-3.0
    rejects it outright with 400), so we can't rely on that to prevent this
    — strip fences ourselves instead."""
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


async def _call_once(model: str, messages: list[dict]) -> str:
    url, api_key, model_id = _endpoint(model)
    transport = google_transport() if _is_google(model) else None
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, transport=transport) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": messages,
            },
        )
        # Only OpenRouter's headers tell a spent account quota from a busy
        # provider; a 429 from Google is retried like any other failure.
        if not _is_google(model) and _quota_exhausted(response):
            raise RateLimitExhausted(f"OpenRouter rate limit exhausted: {response.text[:200]}")
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ModelCallError(f"Model {model} returned no choices: {data!r}")
        choice = choices[0]
        content = choice["message"].get("content")
        if not content:
            raise ModelCallError(
                f"Model {model} returned empty content (finish_reason={choice.get('finish_reason')})"
            )
        return content


async def _call_with_retries(
    model: str,
    messages: list[dict],
    validate: Callable[[dict], bool] | None,
) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = await _call_once(model, messages)
            result = json.loads(_strip_code_fence(_THOUGHT_RE.sub("", content)))
            if validate is not None and not validate(result):
                raise ModelCallError(f"Model {model} returned malformed JSON: {result!r}")
            return result
        except RateLimitExhausted:
            raise
        except Exception as exc:
            last_error = exc
            logging.warning("Model %s failed (attempt %d/%d): %s", model, attempt, MAX_RETRIES, exc)
            if _is_permanent(exc):
                break
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise ModelCallError(f"Model {model} failed after {attempt} attempts") from last_error


async def call_model_json(
    models: list[str],
    messages: list[dict],
    validate: Callable[[dict], bool] | None = None,
) -> dict:
    """Call the first model in `models` that produces valid, well-shaped JSON.

    `validate` checks the parsed result has the fields the caller actually
    needs (e.g. a non-empty "levels" list) — a model can return syntactically
    valid JSON that's still useless (an empty object), which json.loads alone
    won't catch. Each model gets MAX_RETRIES attempts (with backoff) before
    moving on to the next one in the list; a permanent rejection (bad key,
    unknown model) moves on at once. Raises ModelCallError only if every model
    in the list fails, so the caller can report an honest failure instead of
    hanging silently.
    """
    models = [model for model in models if _is_available(model)]
    last_error: Exception | None = None
    quota_spent = False

    for index, model in enumerate(models):
        if quota_spent and not _is_google(model):
            continue  # one OpenRouter quota covers all its models — only Google entries can still help
        try:
            return await _call_with_retries(model, messages, validate)
        except RateLimitExhausted as exc:
            last_error, quota_spent = exc, True
        except ModelCallError as exc:
            last_error = exc
            if index < len(models) - 1:
                logging.warning("Model %s exhausted retries, trying next option", model)

    if quota_spent:
        raise RateLimitExhausted("OpenRouter quota is spent and no other provider answered") from last_error
    raise ModelCallError(f"All models failed: {models}") from last_error
