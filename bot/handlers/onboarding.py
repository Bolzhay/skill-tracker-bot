import asyncio
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .. import db, gamification
from ..keyboards import no_tree_kb, tree_failed_kb, tree_ready_kb
from ..llm import tree_pipeline
from ..llm.openrouter import failure_text
from .common import callback_arg

router = Router()

MAX_NAME_LENGTH = 60

# A plain-text answer; commands fall through to their own handlers.
_ANSWER = F.text & ~F.text.startswith("/")

# Tree builds run in the background (minutes with free models). asyncio keeps
# only weak references to tasks, so hold them here until they finish.
_background: set[asyncio.Task] = set()
_building: set[str] = set()


class SkillOnboarding(StatesGroup):
    name = State()
    self_assessment = State()
    experience = State()
    goal = State()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def begin_new_skill(message: Message, state: FSMContext, name: str = "") -> None:
    user_id = message.from_user.id
    name = name.strip()

    if not name:
        await state.set_state(SkillOnboarding.name)
        await message.answer("Как назвать скилл? Например: пиксель-арт, риторика, шахматы.")
        return

    if len(name) > MAX_NAME_LENGTH:
        await state.set_state(SkillOnboarding.name)
        await message.answer(f"Слишком длинно — до {MAX_NAME_LENGTH} символов. Придумай название покороче.")
        return

    if db.find_skill_by_name(user_id, name):
        await state.set_state(SkillOnboarding.name)
        await message.answer(f"Скилл «{name}» у тебя уже есть. Придумай другое название.")
        return

    await state.update_data(skill_name=name)
    await state.set_state(SkillOnboarding.self_assessment)
    await message.answer(
        f"Скилл «{name}». Отвечу на 3 коротких вопроса, чтобы построить дерево уровней под тебя "
        "(отменить — /cancel).\n\n"
        "1) Оцени себя в этом навыке от 1 до 10 и коротко объясни, почему"
    )


@router.message(Command("new_skill"))
async def cmd_new_skill(message: Message, command: CommandObject, state: FSMContext) -> None:
    await begin_new_skill(message, state, command.args or "")


@router.message(StateFilter(SkillOnboarding), ~F.text)
async def onboarding_not_text(message: Message) -> None:
    await message.answer("Ответь текстом, пожалуйста.")


@router.message(SkillOnboarding.name, _ANSWER)
async def onboarding_name(message: Message, state: FSMContext) -> None:
    await begin_new_skill(message, state, message.text)


@router.message(SkillOnboarding.self_assessment, _ANSWER)
async def onboarding_self_assessment(message: Message, state: FSMContext) -> None:
    await state.update_data(self_assessment=message.text)
    await state.set_state(SkillOnboarding.experience)
    await message.answer("2) Что уже пробовал в этом направлении раньше?")


@router.message(SkillOnboarding.experience, _ANSWER)
async def onboarding_experience(message: Message, state: FSMContext) -> None:
    await state.update_data(experience=message.text)
    await state.set_state(SkillOnboarding.goal)
    await message.answer("3) Какая цель — чего хочешь достичь?")


@router.message(SkillOnboarding.goal, _ANSWER)
async def onboarding_goal(message: Message, state: FSMContext) -> None:
    data = await state.update_data(goal=message.text)
    await state.clear()

    # The skill is created only now, with its answers, so an abandoned
    # onboarding leaves nothing behind and a failed tree build can be retried.
    user_id = message.from_user.id
    skill_id = db.create_skill(user_id, data["skill_name"])
    db.save_onboarding_context(user_id, skill_id, data["self_assessment"], data["experience"], data["goal"])

    await message.answer(
        "Строю дерево уровней — это ансамбль из нескольких моделей, может "
        "занять несколько минут. Напишу, когда будет готово."
    )
    _spawn(_build_tree_and_notify(message.bot, message.chat.id, user_id, db.get_skill(user_id, skill_id)))


async def _build_tree_and_notify(bot: Bot, chat_id: int, user_id: int, skill: dict) -> None:
    skill_id = skill["id"]
    if skill_id in _building:
        return
    _building.add(skill_id)

    try:
        answers = skill["onboarding"]
        levels = await tree_pipeline.build_skill_tree(
            skill["name"], answers["self_assessment"], answers["experience"], answers["goal"]
        )
        db.save_level_tree(user_id, skill_id, levels)
    except Exception as exc:
        logging.exception("Failed to build skill tree for skill_id=%s", skill_id)
        await bot.send_message(
            chat_id,
            failure_text(
                exc,
                f"Не получилось построить дерево уровней для «{skill['name']}». Бесплатные "
                "модели иногда перегружены — попробуй ещё раз.",
            ),
            reply_markup=tree_failed_kb(skill_id),
        )
        return
    finally:
        _building.discard(skill_id)

    levels_sorted = sorted(levels, key=lambda entry: entry.get("order") or 0)
    first = levels_sorted[0]

    checks = first.get("criteria_checks") or []
    checks_block = ""
    if checks:
        checks_block = "\nКонкретные критерии:\n" + "\n".join(f"– {check}" for check in checks) + "\n"

    await bot.send_message(
        chat_id,
        f"Дерево уровней для «{skill['name']}» готово! Всего уровней: {len(levels_sorted)}.\n\n"
        f"Уровень 1 — «{first.get('title', '')}»\n"
        "Это цель уровня (что нужно уметь, чтобы перейти на следующий), а не задание на сегодня:\n"
        f"{first.get('criteria_summary', '')}\n"
        f"{checks_block}\n"
        "Жми «Первое задание» — получишь конкретное упражнение. Чтобы перейти на следующий "
        f"уровень, нужно получить зачёт по {gamification.ATTEMPTS_TO_LEVEL_UP} заданиям.",
        reply_markup=tree_ready_kb(skill_id),
    )


@router.callback_query(F.data.startswith("tree:"))
async def cb_tree(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill is None:
        return

    target = callback.message
    if skill.get("current_level_id"):
        await target.answer("Дерево уровней уже построено.")
    elif not skill.get("onboarding"):
        await target.answer("Не нашёл сохранённых ответов — удали этот скилл и создай заново.", reply_markup=no_tree_kb(skill["id"]))
    elif skill["id"] in _building:
        await target.answer("Дерево уже строится — подожди немного.")
    else:
        await target.answer("Строю дерево заново, может занять несколько минут.")
        _spawn(_build_tree_and_notify(target.bot, target.chat.id, user_id, skill))


@router.callback_query(F.data.startswith("del:"))
async def cb_delete(callback: CallbackQuery) -> None:
    deleted = db.delete_skill_if_empty(callback.from_user.id, callback_arg(callback.data))
    await callback.answer()
    await callback.message.answer(
        "Скилл удалён." if deleted else "Этот скилл удалить нельзя — у него уже есть уровни."
    )
