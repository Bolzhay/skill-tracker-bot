import logging

from aiogram import F, Bot, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, CallbackQuery, InputMediaPhoto, Message

from .. import db
from ..llm import eval_pipeline
from ..llm.openrouter import failure_text
from .common import callback_arg, skill_or_prompt

router = Router()


async def _load_image(bot: Bot, user_id: int, skill_id: str, item: dict) -> bytes | None:
    """The archived bytes if we kept them, else ask Telegram for the original."""
    attempt, task = item["attempt"], item["task"]
    if attempt.get("has_blob"):
        data = db.get_attempt_blob(user_id, skill_id, task["id"], attempt["id"])
        if data:
            return data

    file_id = (attempt.get("proof") or {}).get("file_id")
    if file_id:
        try:
            return (await bot.download(file_id)).read()
        except Exception:
            logging.warning("Couldn't fetch archived proof %s from Telegram", file_id)
    return None


async def do_progress(target: Message, user_id: int, skill: dict) -> None:
    found = db.successful_photo_attempts(user_id, skill["id"])
    if len(found) < 2:
        await target.answer(
            "Для сравнения нужно минимум два засчитанных задания с фото — "
            f"пока их {len(found)}. Выполни ещё пару заданий."
        )
        return

    first, last = found[0], found[-1]
    await target.answer("Сравниваю твою первую и последнюю работы...")

    before = await _load_image(target.bot, user_id, skill["id"], first)
    after = await _load_image(target.bot, user_id, skill["id"], last)
    if before is None or after is None:
        await target.answer("Не смог достать старую работу из архива.")
        return

    await target.bot.send_media_group(
        target.chat.id,
        [
            InputMediaPhoto(media=BufferedInputFile(before, filename="then"), caption="Тогда"),
            InputMediaPhoto(media=BufferedInputFile(after, filename="now"), caption="Сейчас"),
        ],
    )

    try:
        comment = await eval_pipeline.compare_progress(skill["name"], first["task"], last["task"], before, after)
    except Exception as exc:
        logging.exception("Progress comparison failed for skill_id=%s", skill["id"])
        await target.answer(
            failure_text(exc, "Фото показал, а вот сравнить словами не получилось — модели перегружены. Попробуй позже.")
        )
        return
    await target.answer(comment)


@router.message(Command("progress"))
async def cmd_progress(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    skill = await skill_or_prompt(message, user_id, command.args, "prog")
    if skill:
        await do_progress(message, user_id, skill)


@router.callback_query(F.data.startswith("prog:"))
async def cb_progress(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill:
        await do_progress(callback.message, user_id, skill)
