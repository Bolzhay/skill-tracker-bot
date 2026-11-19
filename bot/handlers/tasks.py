import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from .. import db, youtube
from ..keyboards import after_success_kb, after_task_kb, no_tree_kb
from ..llm import task_pipeline
from ..llm.openrouter import failure_text
from .common import callback_arg, skill_or_prompt
from .submissions import TaskSubmission

router = Router()

# Generation takes a while; a double tap on a button must not start it twice.
_busy: set[tuple[str, int, str]] = set()

_PROOF_PROMPTS = {
    "photo": "Пришли фото результата (лучше файлом — без сжатия).",
    "video": "Пришли видео.",
    "audio": "Пришли голосовое сообщение или аудио.",
    "text": "Пришли ответ текстом.",
}


async def do_task(target: Message, user_id: int, skill: dict, state: FSMContext) -> None:
    level_id = skill.get("current_level_id")
    if not level_id:
        await target.answer(
            f"У скилла «{skill['name']}» ещё не построено дерево уровней.", reply_markup=no_tree_kb(skill["id"])
        )
        return

    level = db.get_level(user_id, skill["id"], level_id)
    if not level:
        await target.answer("Не нашёл текущий уровень — что-то не так с данными.")
        return

    key = ("task", user_id, skill["id"])
    if key in _busy:
        await target.answer("Уже придумываю задание — подожди немного.")
        return
    _busy.add(key)

    try:
        await target.answer("Придумываю задание...")
        recent_tasks = db.get_recent_tasks(user_id, skill["id"], limit=10)
        task_count = db.count_tasks(user_id, skill["id"])

        try:
            candidate = await task_pipeline.generate_task(skill, level, recent_tasks, task_count)
        except Exception as exc:
            logging.exception("Failed to generate task for skill_id=%s", skill["id"])
            await target.answer(
                failure_text(
                    exc, "Не получилось придумать задание — бесплатные модели иногда перегружены. Попробуй ещё раз."
                ),
                reply_markup=after_success_kb(skill["id"]),
            )
            return

        description = candidate.get("description", "")
        hint = candidate.get("hint", "")
        media_type = candidate.get("media_type")
        media_type = media_type if media_type in _PROOF_PROMPTS else "text"
        is_wildcard = bool(candidate.get("is_wildcard"))

        task_id = db.create_task(user_id, skill["id"], level_id, description, hint, media_type, is_wildcard)
        await state.set_state(TaskSubmission.awaiting_proof)
        await state.update_data(skill_id=skill["id"], task_id=task_id, level_id=level_id)

        text = "🃏 Wildcard-задание!\n\n" if is_wildcard else ""
        text += description
        if hint:
            text += f"\n\nПодсказка: {hint}"
        text += f"\n\n{_PROOF_PROMPTS[media_type]}"
        await target.answer(text, reply_markup=after_task_kb(skill["id"]))
    finally:
        _busy.discard(key)


def _hint_text(hint: dict, videos: list[dict]) -> str:
    """The hint plus real YouTube links; with no links (no key, spent quota)
    the queries themselves, so the learner can search by hand."""
    text = hint["hint"]
    if videos:
        links = "\n".join(f"• {video['title']} — {video['channel']}\n{video['url']}" for video in videos)
        text += f"\n\n🎬 Видео по теме:\n{links}"
    elif hint["search_queries"]:
        text += "\n\n🔎 Что поискать на YouTube: " + "; ".join(f"«{query}»" for query in hint["search_queries"])
    return text


async def do_hint(target: Message, user_id: int, skill: dict) -> None:
    recent = db.get_recent_tasks(user_id, skill["id"], limit=1)
    if not recent:
        await target.answer(f"У «{skill['name']}» ещё нет заданий. Начни с /task")
        return

    last_task = recent[0]
    if last_task.get("completed"):
        await target.answer(
            f"Последнее задание по «{skill['name']}» уже выполнено — подсказка не нужна.",
            reply_markup=after_success_kb(skill["id"]),
        )
        return

    level = db.get_level(user_id, skill["id"], last_task["level_id"])
    if not level:
        await target.answer("Не нашёл уровень — что-то не так с данными.")
        return

    key = ("hint", user_id, skill["id"])
    if key in _busy:
        await target.answer("Уже думаю над подсказкой — подожди немного.")
        return
    _busy.add(key)

    try:
        await target.answer("Думаю над подсказкой...")
        try:
            hint = await task_pipeline.generate_detailed_hint(skill, level, last_task)
        except Exception as exc:
            logging.exception("Failed to generate hint for skill_id=%s", skill["id"])
            await target.answer(failure_text(exc, "Не получилось придумать подсказку. Попробуй ещё раз."))
            return
        videos = await youtube.find_videos(hint["search_queries"])
        await target.answer(_hint_text(hint, videos))
    finally:
        _busy.discard(key)


async def do_history(target: Message, user_id: int, skill: dict) -> None:
    tasks = db.get_recent_tasks(user_id, skill["id"], limit=5)
    if not tasks:
        await target.answer(f"У «{skill['name']}» пока нет заданий. Начни с /task")
        return

    lines = [f"Последние задания по «{skill['name']}»:", ""]
    for task in tasks:
        status = "✅ " if task.get("completed") else "⏳ "
        wildcard = "🃏 " if task.get("is_wildcard") else ""
        lines.append(f"{status}{wildcard}{task.get('description', '')}")
    await target.answer("\n".join(lines))


@router.message(Command("task"))
async def cmd_task(message: Message, command: CommandObject, state: FSMContext) -> None:
    user_id = message.from_user.id
    skill = await skill_or_prompt(message, user_id, command.args, "task")
    if skill:
        await do_task(message, user_id, skill, state)


@router.message(Command("hint"))
async def cmd_hint(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    skill = await skill_or_prompt(message, user_id, command.args, "hint")
    if skill:
        await do_hint(message, user_id, skill)


@router.message(Command("history"))
async def cmd_history(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    skill = await skill_or_prompt(message, user_id, command.args, "hist")
    if skill:
        await do_history(message, user_id, skill)


@router.callback_query(F.data.startswith("task:"))
async def cb_task(callback: CallbackQuery, state: FSMContext) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill:
        await do_task(callback.message, user_id, skill, state)


@router.callback_query(F.data.startswith("hint:"))
async def cb_hint(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill:
        await do_hint(callback.message, user_id, skill)


@router.callback_query(F.data.startswith("hist:"))
async def cb_history(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill:
        await do_history(callback.message, user_id, skill)
