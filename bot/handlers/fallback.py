import logging

from aiogram import Router
from aiogram.types import CallbackQuery, Message

from .start import HELP_TEXT

router = Router()


@router.message()
async def unmatched_message(message: Message) -> None:
    logging.warning("Unmatched message from chat %s: %r", message.chat.id, message.text)
    await message.answer(f"Не понял. {HELP_TEXT}")


@router.callback_query()
async def unmatched_callback(callback: CallbackQuery) -> None:
    # Stale buttons (from before a restart/update) land here; just stop the spinner.
    await callback.answer("Эта кнопка устарела — открой меню заново.", show_alert=True)
