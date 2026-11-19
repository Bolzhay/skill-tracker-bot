from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from .. import db
from ..keyboards import no_tree_kb, skill_card_kb, skills_kb
from .common import callback_arg, profile_entry, render_card

router = Router()


async def show_profile(message: Message, user_id: int) -> None:
    skills = db.list_skills(user_id)
    if not skills:
        await message.answer("Скиллов пока нет. Создай первый: /new_skill")
        return

    lines = ["Твои скиллы:", ""] + [profile_entry(user_id, skill) for skill in skills]
    await message.answer("\n".join(lines), reply_markup=skills_kb(skills, "card"))


@router.message(Command("profile", "skills"))
async def cmd_profile(message: Message) -> None:
    await show_profile(message, message.from_user.id)


@router.callback_query(F.data.startswith("card:"))
async def cb_card(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill is None:
        return

    keyboard = skill_card_kb(skill["id"]) if skill.get("current_level_id") else no_tree_kb(skill["id"])
    await callback.message.answer(render_card(user_id, skill), reply_markup=keyboard)
