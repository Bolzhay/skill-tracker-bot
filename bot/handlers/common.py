"""Helpers shared by the handlers: picking a skill and rendering it."""

from aiogram.types import Message

from .. import config, db, gamification
from ..keyboards import skills_kb


def callback_arg(data: str) -> str:
    """'task:<skillId>' -> '<skillId>'"""
    return data.split(":", 1)[1]


async def skill_or_prompt(message: Message, user_id: int, args: str | None, action: str) -> dict | None:
    """Work out which skill a command is about: by name if one was given, the
    only skill if there's just one, otherwise ask with buttons. Returns None
    when the caller should stop (a reply has already been sent)."""
    skills = db.list_skills(user_id)
    if not skills:
        await message.answer("Скиллов пока нет. Создай первый: /new_skill")
        return None

    name = (args or "").strip()
    if name:
        match = next((skill for skill in skills if skill["name"].casefold() == name.casefold()), None)
        if match:
            return match
        await message.answer(f"Скилл «{name}» не найден. Выбери из списка:", reply_markup=skills_kb(skills, action))
        return None

    if len(skills) == 1:
        return skills[0]

    await message.answer("Выбери скилл:", reply_markup=skills_kb(skills, action))
    return None


def level_line(user_id: int, skill: dict) -> str:
    level_id = skill.get("current_level_id")
    if not level_id:
        return "Дерево уровней ещё не построено"
    level = db.get_level(user_id, skill["id"], level_id)
    if not level:
        return "Уровень не найден"
    total = skill.get("levels_total")
    position = f"{level['order']}/{total}" if total else str(level["order"])
    return (
        f"Уровень {position}: «{level['title']}» — засчитано "
        f"{level.get('successful_attempts', 0)}/{gamification.ATTEMPTS_TO_LEVEL_UP}"
    )


def _streak(skill: dict) -> int:
    today = gamification.local_today(config.TZ_OFFSET_HOURS)
    return gamification.current_streak(skill.get("streak_count", 0), skill.get("streak_last_date"), today)


def rank_line(skill: dict) -> str:
    xp = skill.get("xp_total", 0)
    line = f"Ранг: {gamification.rank_for(xp)} · {xp} XP"
    upcoming = gamification.next_rank(xp)
    return line + (f" (до «{upcoming[0]}» ещё {upcoming[1]})" if upcoming else "")


def streak_line(skill: dict) -> str:
    streak = _streak(skill)
    return f"Стрик: {streak} дн." if streak else "Стрик: 0 — начни серию сегодня"


def render_card(user_id: int, skill: dict) -> str:
    lines = [f"«{skill['name']}»", level_line(user_id, skill), rank_line(skill), streak_line(skill)]
    badges = ", ".join(gamification.badge_title(badge) for badge in skill.get("badges", []))
    if badges:
        lines.append(f"Бейджи: {badges}")
    return "\n".join(lines)


def profile_entry(user_id: int, skill: dict) -> str:
    streak = _streak(skill)
    return (
        f"• {skill['name']}\n"
        f"  {level_line(user_id, skill)}\n"
        f"  {rank_line(skill)}" + (f" · 🔥 {streak} дн." if streak else "")
    )
