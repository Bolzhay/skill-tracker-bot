"""Whitelist enforcement.

Until the owner claims the bot, nobody but `/claim` gets through. After that
only users with users/{id}.allowed == true do. Runs as an outer middleware, so
it guards every handler (including FSM ones) without each having to check.
"""

import asyncio
import time

from aiogram import BaseMiddleware

from . import db

_TTL_SECONDS = 30.0
_allowed_cache: dict[int, tuple[bool, float]] = {}
_owner_known = False


def reset() -> None:
    global _owner_known
    _allowed_cache.clear()
    _owner_known = False


def invalidate(user_id: int) -> None:
    _allowed_cache.pop(user_id, None)


def mark_owner_known() -> None:
    global _owner_known
    _owner_known = True


async def _owner_exists() -> bool:
    global _owner_known
    if not _owner_known:
        _owner_known = await asyncio.to_thread(db.get_owner_id) is not None
    return _owner_known


async def is_allowed(user_id: int) -> bool:
    hit = _allowed_cache.get(user_id)
    now = time.monotonic()
    if hit and now - hit[1] < _TTL_SECONDS:
        return hit[0]
    allowed = await asyncio.to_thread(db.is_allowed, user_id)
    _allowed_cache[user_id] = (allowed, now)
    return allowed


def _is_claim(text: str | None) -> bool:
    if not text:
        return False
    return text.split(maxsplit=1)[0].split("@")[0] == "/claim"


class AccessMiddleware(BaseMiddleware):
    """`kind` is "message" or "callback" — they're refused in different ways."""

    def __init__(self, kind: str):
        self.kind = kind

    async def __call__(self, handler, event, data):
        user = event.from_user
        if user is None:
            return None
        if self.kind == "message" and event.chat.type != "private":
            return None

        if not await _owner_exists():
            if self.kind == "message" and _is_claim(event.text):
                return await handler(event, data)
            await self._deny(event, "Бот ещё не активирован. Владелец: отправь /claim <код из файла .env>.")
            return None

        if await is_allowed(user.id):
            return await handler(event, data)

        await self._deny(
            event,
            f"Доступ закрыт. Твой Telegram ID: {user.id} — передай его владельцу бота, "
            f"чтобы он открыл доступ командой /allow {user.id}.",
        )
        return None

    async def _deny(self, event, text: str) -> None:
        if self.kind == "callback":
            await event.answer(text, show_alert=True)
        else:
            await event.answer(text)
