import logging

from aiogram import Router
from aiogram.types import ErrorEvent

router = Router()


@router.errors()
async def handle_error(event: ErrorEvent) -> None:
    logging.error("Unhandled error while processing update", exc_info=event.exception)

    update = event.update
    if update.message is not None:
        await update.message.answer("Что-то пошло не так. Попробуйте ещё раз.")
    elif update.callback_query is not None:
        await update.callback_query.answer("Что-то пошло не так. Попробуйте ещё раз.", show_alert=True)
