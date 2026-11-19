from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot import db

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 24


def make_message(user_id=1, text=None, **extra):
    """A stand-in for aiogram's Message: only what the handlers touch."""
    bot = SimpleNamespace(
        download=AsyncMock(),
        send_message=AsyncMock(),
        send_media_group=AsyncMock(),
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, full_name="Tester"),
        chat=SimpleNamespace(id=user_id, type="private"),
        text=text,
        answer=AsyncMock(),
        delete=AsyncMock(),
        bot=bot,
        photo=None,
        document=None,
        voice=None,
        audio=None,
        video=None,
        video_note=None,
    )
    for key, value in extra.items():
        setattr(message, key, value)
    return message


def make_callback(user_id=1, data="", message=None):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, full_name="Tester"),
        data=data,
        message=message or make_message(user_id),
        answer=AsyncMock(),
    )


def last_text(message) -> str:
    """Text of the most recent message.answer(...) call."""
    return message.answer.call_args.args[0]


def all_text(message) -> str:
    return "\n".join(call.args[0] for call in message.answer.call_args_list)


def last_markup(message):
    return message.answer.call_args.kwargs.get("reply_markup")


def callback_data(markup) -> list[str]:
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def levels(count: int) -> list[dict]:
    return [
        {
            "order": index,
            "title": f"Уровень {index}",
            "criteria_summary": f"Цель {index}",
            "criteria_checks": [f"Критерий {index}"],
        }
        for index in range(1, count + 1)
    ]


def seed_skill(user_id=1, name="Пиксель-арт", level_count=3, onboarding=True) -> dict:
    skill_id = db.create_skill(user_id, name)
    if onboarding:
        db.save_onboarding_context(user_id, skill_id, "3 из 10", "немного рисовал", "нарисовать персонажа")
    if level_count:
        db.save_level_tree(user_id, skill_id, levels(level_count))
    return db.get_skill(user_id, skill_id)


def seed_task(user_id, skill, completed=False, description="Нарисуй квадрат", is_wildcard=False) -> dict:
    task_id = db.create_task(
        user_id, skill["id"], skill["current_level_id"], description, "подсказка", "photo", is_wildcard
    )
    if completed:
        db.mark_task_completed(user_id, skill["id"], task_id)
    return db.get_task(user_id, skill["id"], task_id)


async def arm(state, skill, task):
    """Put the FSM where /task leaves it: waiting for a proof for `task`."""
    from bot.handlers.submissions import TaskSubmission

    await state.set_state(TaskSubmission.awaiting_proof)
    await state.update_data(skill_id=skill["id"], task_id=task["id"], level_id=task["level_id"])


def verdict(passed=True) -> dict:
    return {
        "feedback": "Честный разбор работы.",
        "task_completed": passed,
        "issues": [] if passed else ["Линии неровные"],
        "strengths": ["Чистый контур"] if passed else [],
    }
