"""Drive the real aiogram Dispatcher (routing, middleware, FSM, filters) with
synthetic updates and a fake Telegram session that records every API call."""

import asyncio
from datetime import datetime
from itertools import count

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, GetFile, SendMediaGroup, SendMessage
from aiogram.types import (
    CallbackQuery,
    Chat,
    Document,
    File,
    Message,
    MessageEntity,
    PhotoSize,
    Sticker,
    Update,
    User,
    Voice,
)

from bot.handlers import onboarding


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.requests = []  # every API call the bot made, in order
        self.files = {}  # file_id -> bytes served to bot.download()
        self._ids = count(5000)

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.requests.append(method)
        if isinstance(method, SendMessage):
            chat = Chat(id=method.chat_id, type="private")
            return Message(message_id=next(self._ids), date=datetime.now(), chat=chat, text=method.text)
        if isinstance(method, SendMediaGroup):
            chat = Chat(id=method.chat_id, type="private")
            return [Message(message_id=next(self._ids), date=datetime.now(), chat=chat) for _ in method.media]
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path=f"files/{method.file_id}")
        return True  # DeleteMessage, AnswerCallbackQuery, ...

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield self.files[url.rsplit("files/", 1)[-1]]


class Client:
    """One Telegram user talking to the bot."""

    def __init__(self, dispatcher, user_id=1):
        self.dp = dispatcher
        self.user_id = user_id
        self.session = FakeSession()
        self.bot = Bot("123456:TEST", session=self.session)
        self._update_ids = count(1)
        self._message_ids = count(1)

    # --- building updates -------------------------------------------------

    def _user(self):
        return User(id=self.user_id, is_bot=False, first_name="Tester")

    def _message(self, **fields):
        return Message(
            message_id=next(self._message_ids),
            date=datetime.now(),
            chat=Chat(id=self.user_id, type="private"),
            from_user=self._user(),
            **fields,
        )

    async def _feed(self, **update_fields):
        update = Update(update_id=next(self._update_ids), **update_fields)
        await self.dp.feed_update(self.bot, update)
        # background work (the level-tree build) runs as asyncio tasks
        await asyncio.gather(*tuple(onboarding._background))

    # --- things a user can do ---------------------------------------------

    async def say(self, text):
        entities = None
        if text.startswith("/"):
            entities = [MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
        await self._feed(message=self._message(text=text, entities=entities))

    async def press(self, data):
        callback = CallbackQuery(
            id=str(next(self._update_ids)),
            from_user=self._user(),
            chat_instance="ci",
            data=data,
            message=self._message(text="(a message with buttons)"),
        )
        await self._feed(callback_query=callback)

    async def send_photo(self, data, file_id="photo1"):
        self.session.files[file_id] = data
        photo = PhotoSize(file_id=file_id, file_unique_id=file_id + "u", width=9, height=9, file_size=len(data))
        await self._feed(message=self._message(photo=[photo]))

    async def send_document(self, data, mime_type, file_id="doc1"):
        self.session.files[file_id] = data
        document = Document(file_id=file_id, file_unique_id=file_id + "u", mime_type=mime_type, file_size=len(data))
        await self._feed(message=self._message(document=document))

    async def send_voice(self, data, file_id="voice1"):
        self.session.files[file_id] = data
        voice = Voice(file_id=file_id, file_unique_id=file_id + "u", duration=1, mime_type="audio/ogg", file_size=len(data))
        await self._feed(message=self._message(voice=voice))

    async def send_sticker(self):
        sticker = Sticker(
            file_id="st", file_unique_id="stu", type="regular", width=1, height=1, is_animated=False, is_video=False
        )
        await self._feed(message=self._message(sticker=sticker))

    # --- what the bot did -------------------------------------------------

    def sent(self):
        return [r for r in self.session.requests if isinstance(r, SendMessage)]

    def texts(self):
        return [r.text for r in self.sent()]

    def last(self):
        return self.sent()[-1].text

    def last_buttons(self):
        markup = self.sent()[-1].reply_markup
        rows = getattr(markup, "inline_keyboard", None) or []
        return [button.callback_data for row in rows for button in row]

    def alerts(self):
        return [r for r in self.session.requests if isinstance(r, AnswerCallbackQuery) and r.show_alert]

    def deleted_messages(self):
        return [r for r in self.session.requests if isinstance(r, DeleteMessage)]

    def clear(self):
        self.session.requests.clear()
