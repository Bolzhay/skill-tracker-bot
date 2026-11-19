import hmac
import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from .. import access, config, db

router = Router()


@router.message(Command("claim"))
async def cmd_claim(message: Message, command: CommandObject) -> None:
    """The first person to send the right CLAIM_CODE (from .env) becomes the
    owner; the bot is locked to everyone else until then."""
    if db.get_owner_id() is not None:
        await message.answer("Бот уже активирован.")
        return

    if not config.CLAIM_CODE:
        await message.answer("На сервере не задан CLAIM_CODE (файл .env) — активировать бота нечем.")
        return

    code = (command.args or "").strip()
    if not hmac.compare_digest(code.encode(), config.CLAIM_CODE.encode()):
        await message.answer("Неверный код.")
        return

    migrated = db.claim_owner(message.from_user.id, message.from_user.full_name)
    access.mark_owner_known()
    access.invalidate(message.from_user.id)

    try:
        await message.delete()  # keep the code out of the chat history
    except Exception:
        logging.info("Couldn't delete the /claim message")

    text = "Готово — ты владелец бота. Жми /start."
    if migrated:
        text += f"\nПеренёс скиллов из старой базы: {migrated}."
    await message.answer(text)


def _parse_id(args: str | None) -> int | None:
    try:
        return int((args or "").strip())
    except ValueError:
        return None


@router.message(Command("allow"))
async def cmd_allow(message: Message, command: CommandObject) -> None:
    if db.get_owner_id() != message.from_user.id:
        await message.answer("Эта команда только для владельца.")
        return

    target = _parse_id(command.args)
    if target is None:
        await message.answer("Использование: /allow <telegram_id>")
        return

    db.set_user(target, True)
    access.invalidate(target)
    await message.answer(f"Доступ открыт для {target}. Человек сможет пользоваться ботом — у него будут свои скиллы.")


@router.message(Command("deny"))
async def cmd_deny(message: Message, command: CommandObject) -> None:
    owner_id = db.get_owner_id()
    if owner_id != message.from_user.id:
        await message.answer("Эта команда только для владельца.")
        return

    target = _parse_id(command.args)
    if target is None:
        await message.answer("Использование: /deny <telegram_id>")
        return
    if target == owner_id:
        await message.answer("Нельзя закрыть доступ владельцу.")
        return

    db.set_user(target, False)
    access.invalidate(target)
    await message.answer(f"Доступ закрыт для {target}. Его данные остались в базе.")
