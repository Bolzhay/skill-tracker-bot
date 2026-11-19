import html
import logging

import httpx

from . import config
from .net import google_transport

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_PER_QUERY = 2
MAX_VIDEOS = 4


async def find_videos(queries: list[str]) -> list[dict]:
    """Real YouTube videos for search queries written by a model:
    [{"title", "channel", "url"}], at most MAX_VIDEOS, no repeats.

    Every query costs one of the project's 100 free searches a day. Any failure
    (no key, spent quota, network) just means fewer or no links: the caller
    still has its hint to send."""
    if not config.YOUTUBE_API_KEY or not queries:
        return []

    videos: list[dict] = []
    # The key goes in a header, not the URL: httpx logs request URLs.
    headers = {"X-Goog-Api-Key": config.YOUTUBE_API_KEY}
    async with httpx.AsyncClient(timeout=20, transport=google_transport(), headers=headers) as client:
        for query in queries:
            try:
                response = await client.get(
                    SEARCH_URL,
                    params={"part": "snippet", "type": "video", "q": query, "maxResults": VIDEOS_PER_QUERY},
                )
                response.raise_for_status()
                items = response.json().get("items", [])
            except Exception as exc:
                logging.warning("YouTube search failed for %r: %s", query, exc)
                break  # a spent quota or a network problem won't fix itself within this hint
            for item in items:
                video_id = (item.get("id") or {}).get("videoId")
                if not video_id or any(video["url"].endswith(video_id) for video in videos):
                    continue
                snippet = item.get("snippet", {})
                videos.append(
                    {
                        "title": html.unescape(snippet.get("title", "")),
                        "channel": html.unescape(snippet.get("channelTitle", "")),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    }
                )
    return videos[:MAX_VIDEOS]
