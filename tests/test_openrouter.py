import json

import httpx
import pytest

from bot import config
from bot.llm import openrouter


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(openrouter, "RETRY_BACKOFF_SECONDS", 0)


def _client_returning(handler):
    """Make openrouter's AsyncClient talk to `handler` instead of the network."""
    real = httpx.AsyncClient

    def make(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)

    return make


def _completion(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


# --- parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '  ```json\n{"a": 1}```  ',
    ],
)
def test_code_fences_around_json_are_stripped(raw):
    assert json.loads(openrouter._strip_code_fence(raw)) == {"a": 1}


# --- one HTTP call --------------------------------------------------------


async def test_request_carries_the_key_and_no_response_format(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion('{"ok": true}'))

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))
    content = await openrouter._call_once("some/model", [{"role": "user", "content": "hi"}])

    assert content == '{"ok": true}'
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "some/model"
    # Some models (ling-vl did) reject response_format outright, so it must never be sent.
    assert "response_format" not in seen["body"]


async def test_http_errors_raise(monkeypatch):
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(lambda r: httpx.Response(429)))
    with pytest.raises(httpx.HTTPStatusError):
        await openrouter._call_once("m", [])


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "Service temporarily overloaded", "code": 502}},  # 200 OK with an error body
        {"choices": []},
        {"choices": [{"message": {"content": None}, "finish_reason": "error"}]},
        {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
    ],
)
async def test_unusable_200_responses_raise_a_clear_error(monkeypatch, payload):
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(openrouter.ModelCallError):
        await openrouter._call_once("m", [])


# --- quota ----------------------------------------------------------------


async def test_an_exhausted_account_quota_fails_fast_without_retries(monkeypatch):
    hits = []

    def handler(request):
        hits.append(json.loads(request.content)["model"])
        return httpx.Response(429, headers={"x-ratelimit-remaining": "0"}, json={"error": {"message": "Rate limit exceeded"}})

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))

    with pytest.raises(openrouter.RateLimitExhausted):
        await openrouter.call_model_json(["m1", "m2"], [])
    assert hits == ["m1"]  # no retries, and no point trying another model on the same account


async def test_an_upstream_429_without_limit_headers_is_still_retried(monkeypatch):
    # The usual free-model flakiness: an overloaded provider, not our quota.
    hits = []

    def handler(request):
        hits.append(1)
        if len(hits) < 3:
            return httpx.Response(429, json={"error": {"message": "Provider returned error"}})
        return httpx.Response(200, json=_completion('{"ok": true}'))

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))
    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}
    assert len(hits) == 3


async def test_a_429_with_requests_remaining_is_not_treated_as_exhausted(monkeypatch):
    handler = lambda request: httpx.Response(429, headers={"x-ratelimit-remaining": "7"})  # noqa: E731
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))

    with pytest.raises(openrouter.ModelCallError) as caught:
        await openrouter.call_model_json(["m1"], [])
    assert not isinstance(caught.value, openrouter.RateLimitExhausted)


def test_failure_text_explains_a_spent_quota_and_otherwise_uses_the_default():
    assert openrouter.failure_text(openrouter.RateLimitExhausted("x"), "обычный сбой") == openrouter.QUOTA_MESSAGE
    assert openrouter.failure_text(RuntimeError("x"), "обычный сбой") == "обычный сбой"
    assert openrouter.failure_text(openrouter.ModelCallError("x"), "обычный сбой") == "обычный сбой"
    assert "50 запросов в сутки" in openrouter.QUOTA_MESSAGE


# --- retry and fallback ---------------------------------------------------


def _script(monkeypatch, behaviours):
    """Replace _call_once with per-model scripted behaviour: a list of items
    that are either an exception to raise or a string to return."""
    calls = []

    async def fake(model, messages):
        calls.append(model)
        item = behaviours[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(openrouter, "_call_once", fake)
    return calls


async def test_a_flaky_model_is_retried_until_it_answers(monkeypatch):
    calls = _script(monkeypatch, {"m1": [RuntimeError("429"), RuntimeError("429"), '{"ok": true}']})

    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}
    assert calls == ["m1", "m1", "m1"]


async def test_after_three_failures_the_next_model_is_tried(monkeypatch):
    calls = _script(monkeypatch, {"m1": [RuntimeError("x")] * 3, "m2": ['{"from": "m2"}']})

    assert await openrouter.call_model_json(["m1", "m2"], []) == {"from": "m2"}
    assert calls == ["m1", "m1", "m1", "m2"]


async def test_the_first_working_model_wins_and_the_rest_are_not_called(monkeypatch):
    calls = _script(monkeypatch, {"m1": ['{"from": "m1"}'], "m2": ['{"from": "m2"}']})

    assert await openrouter.call_model_json(["m1", "m2"], []) == {"from": "m1"}
    assert calls == ["m1"]


async def test_garbage_that_is_not_json_counts_as_a_failure(monkeypatch):
    calls = _script(monkeypatch, {"m1": ["not json", "still not", "nope"], "m2": ['{"ok": 1}']})

    assert await openrouter.call_model_json(["m1", "m2"], []) == {"ok": 1}
    assert calls.count("m1") == 3


async def test_valid_json_of_the_wrong_shape_counts_as_a_failure(monkeypatch):
    calls = _script(monkeypatch, {"m1": ["{}", '{"levels": []}', '{"levels": [1]}']})

    result = await openrouter.call_model_json(["m1"], [], validate=lambda r: bool(r.get("levels")))
    assert result == {"levels": [1]}
    assert calls == ["m1", "m1", "m1"]


async def test_fenced_json_is_accepted(monkeypatch):
    _script(monkeypatch, {"m1": ['```json\n{"ok": true}\n```']})
    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}


async def test_a_thought_block_ahead_of_the_json_is_ignored(monkeypatch):
    # Gemma on Google's endpoint reasons out loud first, sometimes in JSON-looking text.
    reply = '<thought>\n*   Plan: {"draft": 1}\n*   Now answer.\n</thought>\n```json\n{"ok": true}\n```'
    _script(monkeypatch, {"m1": [reply]})
    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}


async def test_each_thought_block_is_removed_on_its_own(monkeypatch):
    # A greedy match would swallow the answer sitting between two thoughts.
    reply = '<thought>first</thought>\n{"ok": true}\n<thought>second</thought>'
    _script(monkeypatch, {"m1": [reply]})
    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}


async def test_a_reply_cut_off_inside_the_thought_is_a_failure(monkeypatch):
    # Out of tokens while still thinking: there is no answer to parse, so move on.
    cut_off = '<thought>\n*   Input: "Ответь одним словом: привет" (Answer in'
    calls = _script(monkeypatch, {"m1": [cut_off] * 3, "m2": ['{"from": "m2"}']})

    assert await openrouter.call_model_json(["m1", "m2"], []) == {"from": "m2"}
    assert calls == ["m1", "m1", "m1", "m2"]


async def test_when_every_model_fails_the_error_is_honest(monkeypatch):
    _script(monkeypatch, {"m1": [RuntimeError("a")] * 3, "m2": [RuntimeError("b")] * 3})

    with pytest.raises(openrouter.ModelCallError, match="All models failed"):
        await openrouter.call_model_json(["m1", "m2"], [])


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
async def test_a_permanent_rejection_is_not_retried_and_the_next_model_runs(monkeypatch, status):
    # A wrong key or a model the account can't use never starts working on retry.
    hits = []

    def handler(request):
        model = json.loads(request.content)["model"]
        hits.append(model)
        if model == "m1":
            return httpx.Response(status, json={"error": {"message": "nope"}})
        return httpx.Response(200, json=_completion('{"from": "m2"}'))

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))

    assert await openrouter.call_model_json(["m1", "m2"], []) == {"from": "m2"}
    assert hits == ["m1", "m2"]


async def test_server_errors_are_still_retried(monkeypatch):
    hits = []

    def handler(request):
        hits.append(1)
        return httpx.Response(502) if len(hits) < 3 else httpx.Response(200, json=_completion('{"ok": true}'))

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))
    assert await openrouter.call_model_json(["m1"], []) == {"ok": True}
    assert len(hits) == 3


# --- second provider (Google) ---------------------------------------------


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-key")


async def test_google_entries_go_to_google_with_their_own_key_and_plain_model_id(monkeypatch, gemini_key):
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["authorization"], json.loads(request.content)["model"]))
        return httpx.Response(200, json=_completion('{"ok": true}'))

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))
    await openrouter._call_once("google:some-model", [])
    await openrouter._call_once("vendor/other-model:free", [])

    assert seen == [
        ("generativelanguage.googleapis.com", "Bearer gemini-key", "some-model"),
        ("openrouter.ai", "Bearer test-key", "vendor/other-model:free"),
    ]


async def test_google_calls_are_pinned_to_ipv4_and_openrouter_calls_are_not(monkeypatch, gemini_key):
    # Some networks have a dead IPv6 route to Google; unpinned, every call stalls for minutes.
    transports = []
    real = httpx.AsyncClient

    def fake_client(**kwargs):
        transports.append(kwargs.get("transport"))
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json=_completion('{"ok": true}')))
        return real(**kwargs)

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", fake_client)
    await openrouter._call_once("google:m", [])
    await openrouter._call_once("vendor/m:free", [])

    assert isinstance(transports[0], httpx.AsyncHTTPTransport)
    assert transports[1] is None


async def test_google_entries_are_tried_in_roster_order_when_a_key_is_set(monkeypatch, gemini_key):
    calls = _script(monkeypatch, {"google:m1": ['{"from": "google"}'], "m2": ['{"from": "m2"}']})

    assert await openrouter.call_model_json(["google:m1", "m2"], []) == {"from": "google"}
    assert calls == ["google:m1"]


async def test_google_entries_are_skipped_when_there_is_no_gemini_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    calls = _script(monkeypatch, {"m1": ['{"from": "m1"}'], "google:m2": ['{"from": "google"}']})

    assert await openrouter.call_model_json(["google:m2", "m1"], []) == {"from": "m1"}
    assert calls == ["m1"]


async def test_a_roster_of_only_google_entries_without_a_key_is_an_honest_error(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    with pytest.raises(openrouter.ModelCallError, match="All models failed"):
        await openrouter.call_model_json(["google:m1"], [])


async def test_a_spent_openrouter_quota_skips_its_models_but_not_googles(monkeypatch, gemini_key):
    calls = _script(
        monkeypatch,
        {"m1": [openrouter.RateLimitExhausted("429")], "m2": ['{"from": "m2"}'], "google:m3": ['{"from": "google"}']},
    )

    assert await openrouter.call_model_json(["m1", "m2", "google:m3"], []) == {"from": "google"}
    assert calls == ["m1", "google:m3"]  # m2 shares m1's account quota, so it isn't even tried


async def test_a_spent_quota_and_a_failing_google_still_report_the_quota(monkeypatch, gemini_key):
    calls = _script(
        monkeypatch,
        {"m1": [openrouter.RateLimitExhausted("429")], "google:m2": [RuntimeError("x")] * 3},
    )

    with pytest.raises(openrouter.RateLimitExhausted):
        await openrouter.call_model_json(["m1", "google:m2"], [])
    assert calls == ["m1", "google:m2", "google:m2", "google:m2"]


async def test_a_429_from_google_is_a_busy_signal_not_a_spent_account(monkeypatch, gemini_key):
    # The same headers from OpenRouter would mean "the account's quota is gone".
    handler = lambda request: httpx.Response(429, headers={"x-ratelimit-remaining": "0"})  # noqa: E731
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _client_returning(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await openrouter._call_once("google:m1", [])
