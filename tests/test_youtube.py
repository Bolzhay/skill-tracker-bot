import httpx
import pytest

from bot import config, youtube


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(config, "YOUTUBE_API_KEY", "yt-key")


def _answering(monkeypatch, handler):
    """Make youtube's AsyncClient talk to `handler` instead of the network."""
    real = httpx.AsyncClient

    def make(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)

    monkeypatch.setattr(youtube.httpx, "AsyncClient", make)


def _item(video_id, title="Урок", channel="Канал"):
    return {"id": {"kind": "youtube#video", "videoId": video_id}, "snippet": {"title": title, "channelTitle": channel}}


def _results(*items):
    return httpx.Response(200, json={"items": list(items)})


async def test_without_a_key_nothing_is_requested(monkeypatch):
    monkeypatch.setattr(config, "YOUTUBE_API_KEY", "")
    _answering(monkeypatch, lambda request: pytest.fail("no request expected without a key"))

    assert await youtube.find_videos(["пиксель-арт"]) == []


async def test_no_queries_means_no_requests(monkeypatch, key):
    _answering(monkeypatch, lambda request: pytest.fail("no request expected without queries"))

    assert await youtube.find_videos([]) == []


async def test_videos_come_back_with_title_channel_and_a_watch_link(monkeypatch, key):
    _answering(monkeypatch, lambda request: _results(_item("abc123", "Pixel &amp; Art &#39;101&#39;", "Pixel School")))

    assert await youtube.find_videos(["пиксель-арт"]) == [
        {"title": "Pixel & Art '101'", "channel": "Pixel School", "url": "https://www.youtube.com/watch?v=abc123"}
    ]


async def test_the_key_travels_in_a_header_and_never_in_the_url(monkeypatch, key):
    seen = []

    def handler(request):
        seen.append(request)
        return _results()

    _answering(monkeypatch, handler)
    await youtube.find_videos(["пиксель-арт линии"])

    request = seen[0]
    assert request.headers["x-goog-api-key"] == "yt-key"
    assert "yt-key" not in str(request.url)  # httpx logs URLs, and the key must stay out of the log
    assert request.url.params["q"] == "пиксель-арт линии"
    assert request.url.params["type"] == "video"
    assert request.url.params["maxResults"] == str(youtube.VIDEOS_PER_QUERY)


async def test_a_video_found_by_two_queries_is_listed_once_and_the_total_is_capped(monkeypatch, key):
    answers = {
        "a": _results(_item("v1"), _item("v2")),
        "b": _results(_item("v2"), _item("v3")),
        "c": _results(_item("v4"), _item("v5")),
    }
    _answering(monkeypatch, lambda request: answers[request.url.params["q"]])

    videos = await youtube.find_videos(["a", "b", "c"])

    assert [video["url"][-2:] for video in videos] == ["v1", "v2", "v3", "v4"]
    assert len(videos) == youtube.MAX_VIDEOS


async def test_results_that_are_not_videos_are_skipped(monkeypatch, key):
    channel = {"id": {"kind": "youtube#channel", "channelId": "UC1"}, "snippet": {"title": "Канал", "channelTitle": "Канал"}}
    _answering(monkeypatch, lambda request: _results(channel, _item("v1")))

    videos = await youtube.find_videos(["a"])

    assert [video["url"][-2:] for video in videos] == ["v1"]


async def test_a_failed_search_ends_the_search_and_keeps_what_was_found(monkeypatch, key):
    queries = []

    def handler(request):
        queries.append(request.url.params["q"])
        if request.url.params["q"] == "a":
            return _results(_item("v1"))
        return httpx.Response(403, json={"error": {"errors": [{"reason": "quotaExceeded"}]}})

    _answering(monkeypatch, handler)
    videos = await youtube.find_videos(["a", "b", "c"])

    assert [video["url"][-2:] for video in videos] == ["v1"]
    assert queries == ["a", "b"]  # the quota is gone, so "c" isn't even tried


async def test_a_network_error_means_no_links_not_an_exception(monkeypatch, key):
    def handler(request):
        raise httpx.ConnectError("no route to host")

    _answering(monkeypatch, handler)

    assert await youtube.find_videos(["a"]) == []


async def test_calls_to_youtube_are_pinned_to_ipv4(monkeypatch, key):
    # Some networks have a dead IPv6 route to Google; unpinned, every call stalls for minutes.
    transports = []
    real = httpx.AsyncClient

    def fake_client(**kwargs):
        transports.append(kwargs.get("transport"))
        kwargs["transport"] = httpx.MockTransport(lambda request: _results())
        return real(**kwargs)

    monkeypatch.setattr(youtube.httpx, "AsyncClient", fake_client)
    await youtube.find_videos(["a"])

    assert isinstance(transports[0], httpx.AsyncHTTPTransport)
