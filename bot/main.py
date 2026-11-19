import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from . import config, db, transcribe
from .access import AccessMiddleware
from .handlers import admin, errors, fallback, onboarding, progress, skills, start, submissions, tasks

COMMANDS = [
    BotCommand(command="task", description="Получить задание"),
    BotCommand(command="hint", description="Развёрнутая подсказка"),
    BotCommand(command="retry", description="Прислать пруф ещё раз"),
    BotCommand(command="history", description="Последние задания"),
    BotCommand(command="progress", description="Тогда vs сейчас"),
    BotCommand(command="profile", description="Мои скиллы, ранг, стрик"),
    BotCommand(command="new_skill", description="Создать скилл"),
    BotCommand(command="cancel", description="Отменить диалог"),
]


async def _warm_up_whisper() -> None:
    """Load (and on the very first run, download) the speech model in the
    background so the first voice proof doesn't wait minutes for it."""
    try:
        await asyncio.to_thread(transcribe.warmup)
        logging.info("Whisper model ready")
    except Exception:
        logging.exception("Whisper warm-up failed; voice/video proofs will retry on first use")


def build_dispatcher() -> Dispatcher:
    """The whole routing setup. Routers are module-level singletons and aiogram
    allows attaching each only once, so call this once per process."""
    dp = Dispatcher()
    dp.message.outer_middleware(AccessMiddleware("message"))
    dp.callback_query.outer_middleware(AccessMiddleware("callback"))

    # Order matters: the first router whose filters match handles the update.
    dp.include_router(start.router)  # menu buttons must beat the FSM text handlers below
    dp.include_router(admin.router)
    dp.include_router(onboarding.router)
    dp.include_router(skills.router)
    dp.include_router(tasks.router)
    dp.include_router(progress.router)
    dp.include_router(submissions.router)
    # Catch-all must stay last: it matches any message, so anything above
    # it would never be reached if it came first.
    dp.include_router(fallback.router)
    dp.include_router(errors.router)
    return dp


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    db.init_db(config.FIREBASE_CREDENTIALS_PATH)

    if db.get_owner_id() is None:
        if config.CLAIM_CODE:
            logging.warning("Bot is not activated yet: the owner must send /claim <CLAIM_CODE from .env>")
        else:
            logging.error("Bot is not activated and CLAIM_CODE is not set in .env — nobody can claim it")

    if not config.GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY is not set: running on OpenRouter alone (50 requests a day)")

    bot = Bot(token=config.BOT_TOKEN)
    dp = build_dispatcher()

    await bot.set_my_commands(COMMANDS)
    warm_up = asyncio.create_task(_warm_up_whisper())  # keep a reference so it isn't garbage-collected

    try:
        await dp.start_polling(bot)
    finally:
        warm_up.cancel()


if __name__ == "__main__":
    asyncio.run(main())
