import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.filters import CommandObject

from bot import config, db, gamification, youtube
from bot.handlers import onboarding, skills, start, tasks
from bot.handlers.submissions import TaskSubmission
from bot.keyboards import MENU_NEW, MENU_SKILLS
from bot.llm import task_pipeline, tree_pipeline
from tests.helpers import (
    all_text,
    callback_data,
    last_markup,
    last_text,
    levels,
    make_callback,
    make_message,
    seed_skill,
    seed_task,
)

USER = 1


def command(name, args=None):
    return CommandObject(command=name, args=args)


# --- onboarding -----------------------------------------------------------


async def test_new_skill_without_a_name_asks_for_one(fake_db, state):
    message = make_message(USER)
    await onboarding.cmd_new_skill(message, command("new_skill"), state)

    assert await state.get_state() == onboarding.SkillOnboarding.name.state
    assert "Как назвать" in last_text(message)


async def test_a_name_typed_in_reply_starts_the_questions(fake_db, state):
    await onboarding.cmd_new_skill(make_message(USER), command("new_skill"), state)
    message = make_message(USER, "Риторика")
    await onboarding.onboarding_name(message, state)

    assert await state.get_state() == onboarding.SkillOnboarding.self_assessment.state
    assert "1)" in last_text(message) and "«Риторика»" in last_text(message)


@pytest.mark.parametrize("taken", ["Риторика", "  риторика "])
async def test_duplicate_names_are_refused(fake_db, state, taken):
    seed_skill(USER, "Риторика")
    message = make_message(USER)
    await onboarding.cmd_new_skill(message, command("new_skill", taken), state)

    assert "уже есть" in last_text(message)
    assert await state.get_state() == onboarding.SkillOnboarding.name.state  # waits for another name


async def test_overlong_names_are_refused(fake_db, state):
    message = make_message(USER)
    await onboarding.cmd_new_skill(message, command("new_skill", "я" * 61), state)

    assert "покороче" in last_text(message)
    assert await state.get_state() == onboarding.SkillOnboarding.name.state


async def test_non_text_answers_are_turned_down(fake_db, state):
    message = make_message(USER)
    await onboarding.onboarding_not_text(message)
    assert "текстом" in last_text(message)


async def test_full_onboarding_stores_the_answers_and_builds_the_tree(fake_db, state, monkeypatch):
    build = AsyncMock(return_value=levels(3))
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", build)

    await onboarding.cmd_new_skill(make_message(USER), command("new_skill", "Пиксель-арт"), state)
    assert db.list_skills(USER) == []  # nothing is stored until the answers are in

    await onboarding.onboarding_self_assessment(make_message(USER, "3 из 10"), state)
    await onboarding.onboarding_experience(make_message(USER, "рисовал"), state)
    goal = make_message(USER, "персонаж")
    await onboarding.onboarding_goal(goal, state)
    assert "Строю дерево" in last_text(goal)
    await asyncio.gather(*onboarding._background)

    skill = db.list_skills(USER)[0]
    assert skill["onboarding"] == {"self_assessment": "3 из 10", "experience": "рисовал", "goal": "персонаж"}
    assert skill["current_level_id"] and skill["levels_total"] == 3
    build.assert_awaited_once_with("Пиксель-арт", "3 из 10", "рисовал", "персонаж")
    assert await state.get_state() is None

    chat_id, text = goal.bot.send_message.call_args.args
    assert "готово" in text and "Уровень 1" in text and "Критерий 1" in text
    assert callback_data(goal.bot.send_message.call_args.kwargs["reply_markup"]) == [f"task:{skill['id']}"]


async def test_failed_tree_build_keeps_the_skill_and_offers_a_retry(fake_db, state, monkeypatch):
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", AsyncMock(side_effect=RuntimeError("models down")))

    await onboarding.cmd_new_skill(make_message(USER), command("new_skill", "Химия"), state)
    await onboarding.onboarding_self_assessment(make_message(USER, "5"), state)
    await onboarding.onboarding_experience(make_message(USER, "учил"), state)
    goal = make_message(USER, "олимпиада")
    await onboarding.onboarding_goal(goal, state)
    await asyncio.gather(*onboarding._background)

    skill = db.list_skills(USER)[0]
    assert skill["current_level_id"] is None and skill["onboarding"]["goal"] == "олимпиада"
    text = goal.bot.send_message.call_args.args[1]
    assert "Не получилось" in text
    assert callback_data(goal.bot.send_message.call_args.kwargs["reply_markup"]) == [f"tree:{skill['id']}"]
    assert skill["id"] not in onboarding._building  # the guard is released even on failure


async def test_retry_button_rebuilds_from_the_saved_answers(fake_db, monkeypatch):
    skill = seed_skill(USER, level_count=0)
    build = AsyncMock(return_value=levels(2))
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", build)

    callback = make_callback(USER, f"tree:{skill['id']}")
    await onboarding.cb_tree(callback)
    await asyncio.gather(*onboarding._background)

    callback.answer.assert_awaited_once()
    build.assert_awaited_once_with("Пиксель-арт", "3 из 10", "немного рисовал", "нарисовать персонажа")
    assert db.get_skill(USER, skill["id"])["levels_total"] == 2


async def test_retry_button_does_nothing_for_a_finished_tree(fake_db, monkeypatch):
    skill = seed_skill(USER)
    build = AsyncMock()
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", build)

    callback = make_callback(USER, f"tree:{skill['id']}")
    await onboarding.cb_tree(callback)

    assert "уже построено" in last_text(callback.message)
    build.assert_not_called()


async def test_retry_button_without_saved_answers_offers_deletion(fake_db):
    skill = seed_skill(USER, level_count=0, onboarding=False)
    callback = make_callback(USER, f"tree:{skill['id']}")
    await onboarding.cb_tree(callback)

    assert "создай заново" in last_text(callback.message)
    assert f"del:{skill['id']}" in callback_data(last_markup(callback.message))


async def test_a_build_already_in_progress_is_not_started_twice(fake_db, monkeypatch):
    skill = seed_skill(USER, level_count=0)
    build = AsyncMock()
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", build)
    onboarding._building.add(skill["id"])

    callback = make_callback(USER, f"tree:{skill['id']}")
    await onboarding.cb_tree(callback)

    assert "уже строится" in last_text(callback.message)
    build.assert_not_called()


async def test_delete_button_removes_only_treeless_skills(fake_db):
    empty = seed_skill(USER, "Пустой", level_count=0)
    built = seed_skill(USER, "Готовый")

    gone = make_callback(USER, f"del:{empty['id']}")
    await onboarding.cb_delete(gone)
    assert "удалён" in last_text(gone.message) and db.get_skill(USER, empty["id"]) is None

    kept = make_callback(USER, f"del:{built['id']}")
    await onboarding.cb_delete(kept)
    assert "нельзя" in last_text(kept.message) and db.get_skill(USER, built["id"]) is not None


# --- start, menu, cancel --------------------------------------------------


async def test_start_greets_and_shows_the_menu(fake_db):
    message = make_message(USER)
    await start.cmd_start(message)

    keyboard = last_markup(message).keyboard
    assert [button.text for button in keyboard[0]] == [MENU_SKILLS, MENU_NEW]
    assert "/task" in last_text(message)


async def test_menu_buttons_open_the_list_and_the_new_skill_dialog(fake_db, state):
    seed_skill(USER, "Риторика")
    listing = make_message(USER, MENU_SKILLS)
    await start.menu_skills(listing)
    assert "Риторика" in last_text(listing)

    new = make_message(USER, MENU_NEW)
    await start.menu_new_skill(new, state)
    assert await state.get_state() == onboarding.SkillOnboarding.name.state


async def test_cancel_clears_whatever_dialog_is_open(fake_db, state):
    nothing = make_message(USER)
    await start.cmd_cancel(nothing, state)
    assert "нечего" in last_text(nothing)

    await state.set_state(TaskSubmission.awaiting_proof)
    message = make_message(USER)
    await start.cmd_cancel(message, state)
    assert await state.get_state() is None
    assert "Отменил" in last_text(message)


async def test_cancel_button_clears_the_state(fake_db, state):
    await state.set_state(TaskSubmission.awaiting_proof)
    callback = make_callback(USER, "cancel")
    await start.cb_cancel(callback, state)

    assert await state.get_state() is None
    callback.answer.assert_awaited_once()


# --- profile and skill card -----------------------------------------------


async def test_profile_without_skills_points_to_creating_one(fake_db):
    message = make_message(USER)
    await skills.cmd_profile(message)
    assert "/new_skill" in last_text(message)


async def test_profile_lists_every_skill_with_rank_level_and_buttons(fake_db):
    first = seed_skill(USER, "Риторика")
    second = seed_skill(USER, "Химия", level_count=0)
    message = make_message(USER)
    await skills.cmd_profile(message)

    text = last_text(message)
    assert "Риторика" in text and "Химия" in text
    assert "Уровень 1/3: «Уровень 1» — засчитано 0/3" in text
    assert "Ранг: Новичок · 0 XP" in text
    assert "Дерево уровней ещё не построено" in text
    assert callback_data(last_markup(message)) == [f"card:{first['id']}", f"card:{second['id']}"]


async def test_skill_card_shows_progress_streak_and_badges(fake_db):
    today = gamification.local_today(config.TZ_OFFSET_HOURS)  # the same "today" the bot uses

    skill = seed_skill(USER)
    for _ in range(3):
        db.record_attempt_result(USER, skill["id"], skill["current_level_id"], True, False, today)
    skill = db.get_skill(USER, skill["id"])

    callback = make_callback(USER, f"card:{skill['id']}")
    await skills.cb_card(callback)

    text = last_text(callback.message)
    assert "Уровень 2/3" in text  # three passes moved the learner up
    assert "Ранг: Новичок · 30 XP (до «Ученик» ещё 20)" in text
    assert "Стрик: 1 дн." in text
    assert "Первое засчитанное задание" in text and "Первый левелап" in text
    assert {"task:" + skill["id"], "hint:" + skill["id"], "prog:" + skill["id"]} <= set(
        callback_data(last_markup(callback.message))
    )


async def test_card_of_a_treeless_skill_offers_build_and_delete(fake_db):
    skill = seed_skill(USER, level_count=0)
    callback = make_callback(USER, f"card:{skill['id']}")
    await skills.cb_card(callback)

    assert set(callback_data(last_markup(callback.message))) == {f"tree:{skill['id']}", f"del:{skill['id']}"}


async def test_stale_card_button_is_ignored(fake_db):
    callback = make_callback(USER, "card:gone")
    await skills.cb_card(callback)
    callback.answer.assert_awaited_once()
    callback.message.answer.assert_not_called()


# --- /task ----------------------------------------------------------------

CANDIDATE = {"description": "Нарисуй квадрат", "hint": "Возьми Линию", "media_type": "photo", "is_wildcard": False}


@pytest.fixture
def generate(monkeypatch):
    mock = AsyncMock(return_value=dict(CANDIDATE))
    monkeypatch.setattr(task_pipeline, "generate_task", mock)
    return mock


async def test_task_without_any_skills(fake_db, state, generate):
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert "/new_skill" in last_text(message)
    generate.assert_not_called()


async def test_task_with_a_single_skill_needs_no_name(fake_db, state, generate):
    skill = seed_skill(USER)
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    text = last_text(message)
    assert "Нарисуй квадрат" in text and "Подсказка: Возьми Линию" in text and "фото результата" in text
    assert callback_data(last_markup(message)) == [f"hint:{skill['id']}", "cancel"]

    stored = db.get_recent_tasks(USER, skill["id"])[0]
    assert stored["description"] == "Нарисуй квадрат" and stored["completed"] is False
    assert await state.get_state() == TaskSubmission.awaiting_proof.state
    assert await state.get_data() == {
        "skill_id": skill["id"],
        "task_id": stored["id"],
        "level_id": skill["current_level_id"],
    }
    # the pipeline got the skill (with onboarding), its current level and the history
    args = generate.await_args.args
    assert args[0]["id"] == skill["id"] and args[1]["order"] == 1 and args[2] == [] and args[3] == 0


async def test_task_by_name_picks_that_skill_case_insensitively(fake_db, state, generate):
    seed_skill(USER, "Риторика")
    chemistry = seed_skill(USER, "Химия")
    message = make_message(USER)
    await tasks.cmd_task(message, command("task", "химия"), state)

    assert len(db.get_recent_tasks(USER, chemistry["id"])) == 1


async def test_task_with_several_skills_and_no_name_shows_buttons(fake_db, state, generate):
    first = seed_skill(USER, "Риторика")
    second = seed_skill(USER, "Химия")
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert callback_data(last_markup(message)) == [f"task:{first['id']}", f"task:{second['id']}"]
    generate.assert_not_called()


async def test_task_for_an_unknown_name_lists_the_options(fake_db, state, generate):
    skill = seed_skill(USER)
    message = make_message(USER)
    await tasks.cmd_task(message, command("task", "нет такого"), state)

    assert "не найден" in last_text(message)
    assert callback_data(last_markup(message)) == [f"task:{skill['id']}"]


async def test_task_for_a_treeless_skill_offers_to_build_it(fake_db, state, generate):
    skill = seed_skill(USER, level_count=0)
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert "не построено дерево" in last_text(message)
    assert f"tree:{skill['id']}" in callback_data(last_markup(message))
    generate.assert_not_called()


async def test_task_generation_failure_is_reported_and_stores_nothing(fake_db, state, generate):
    skill = seed_skill(USER)
    generate.side_effect = RuntimeError("models down")
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert "Не получилось" in last_text(message)
    assert db.get_recent_tasks(USER, skill["id"]) == []
    assert await state.get_state() is None
    assert tasks._busy == set()  # released, so the user can simply tap again


@pytest.mark.parametrize(
    "media, expected",
    [("video", "Пришли видео"), ("audio", "голосовое"), ("text", "текстом"), ("hologram", "текстом")],
)
async def test_the_proof_prompt_matches_the_task_type(fake_db, state, generate, media, expected):
    seed_skill(USER)
    generate.return_value = {**CANDIDATE, "media_type": media}
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert expected in last_text(message)


async def test_wildcard_tasks_are_flagged(fake_db, state, generate):
    skill = seed_skill(USER)
    generate.return_value = {**CANDIDATE, "is_wildcard": True}
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert last_text(message).startswith("🃏 Wildcard-задание!")
    assert db.get_recent_tasks(USER, skill["id"])[0]["is_wildcard"] is True


async def test_a_double_tap_does_not_generate_twice(fake_db, state, generate):
    skill = seed_skill(USER)
    tasks._busy.add(("task", USER, skill["id"]))
    message = make_message(USER)
    await tasks.cmd_task(message, command("task"), state)

    assert "Уже придумываю" in last_text(message)
    generate.assert_not_called()


async def test_task_button_works_like_the_command(fake_db, state, generate):
    skill = seed_skill(USER)
    callback = make_callback(USER, f"task:{skill['id']}")
    await tasks.cb_task(callback, state)

    callback.answer.assert_awaited_once()
    assert "Нарисуй квадрат" in all_text(callback.message)
    assert await state.get_state() == TaskSubmission.awaiting_proof.state


# --- /hint and /history ---------------------------------------------------


LINK = {"title": "Урок про линии", "channel": "Pixel School", "url": "https://www.youtube.com/watch?v=abc123"}


@pytest.fixture
def hint(monkeypatch):
    mock = AsyncMock(return_value={"hint": "Начни с сетки", "search_queries": ["пиксель-арт для начинающих"]})
    monkeypatch.setattr(task_pipeline, "generate_detailed_hint", mock)
    return mock


@pytest.fixture
def videos(monkeypatch):
    mock = AsyncMock(return_value=[LINK])
    monkeypatch.setattr(youtube, "find_videos", mock)
    return mock


async def test_hint_for_the_open_task(fake_db, hint, videos):
    skill = seed_skill(USER)
    task = seed_task(USER, skill)
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    text = last_text(message)
    assert text.startswith("Начни с сетки")
    assert "🎬 Видео по теме:" in text and "• Урок про линии — Pixel School\nhttps://www.youtube.com/watch?v=abc123" in text
    videos.assert_awaited_once_with(["пиксель-арт для начинающих"])
    skill_arg, level_arg, task_arg = hint.await_args.args
    assert task_arg["id"] == task["id"] and level_arg["order"] == 1 and skill_arg["id"] == skill["id"]


async def test_hint_without_found_videos_offers_the_queries_to_search_by_hand(fake_db, hint, videos):
    videos.return_value = []
    skill = seed_skill(USER)
    seed_task(USER, skill)
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    assert last_text(message) == "Начни с сетки\n\n🔎 Что поискать на YouTube: «пиксель-арт для начинающих»"


async def test_hint_with_neither_videos_nor_queries_is_just_the_hint(fake_db, hint, videos):
    hint.return_value = {"hint": "Начни с сетки", "search_queries": []}
    videos.return_value = []
    skill = seed_skill(USER)
    seed_task(USER, skill)
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    assert last_text(message) == "Начни с сетки"


async def test_hint_when_there_is_no_task_yet(fake_db, hint):
    seed_skill(USER)
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    assert "/task" in last_text(message)
    hint.assert_not_called()


async def test_hint_for_a_finished_task_is_declined_with_a_next_step(fake_db, hint):
    skill = seed_skill(USER)
    seed_task(USER, skill, completed=True)
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    assert "подсказка не нужна" in last_text(message)
    assert f"task:{skill['id']}" in callback_data(last_markup(message))
    hint.assert_not_called()


async def test_hint_failure_is_reported(fake_db, hint):
    skill = seed_skill(USER)
    seed_task(USER, skill)
    hint.side_effect = RuntimeError("models down")
    message = make_message(USER)
    await tasks.cmd_hint(message, command("hint"))

    assert "Не получилось" in last_text(message)
    assert tasks._busy == set()


async def test_hint_button(fake_db, hint, videos):
    skill = seed_skill(USER)
    seed_task(USER, skill)
    callback = make_callback(USER, f"hint:{skill['id']}")
    await tasks.cb_hint(callback)

    assert "Начни с сетки" in last_text(callback.message)


async def test_history_shows_status_and_wildcards_newest_first(fake_db):
    skill = seed_skill(USER)
    seed_task(USER, skill, completed=True, description="первое")
    seed_task(USER, skill, completed=False, description="второе", is_wildcard=True)
    message = make_message(USER)
    await tasks.cmd_history(message, command("history"))

    lines = last_text(message).splitlines()
    assert lines[2:] == ["⏳ 🃏 второе", "✅ первое"]


async def test_history_without_tasks_and_via_button(fake_db):
    skill = seed_skill(USER)
    empty = make_message(USER)
    await tasks.cmd_history(empty, command("history"))
    assert "пока нет заданий" in last_text(empty)

    seed_task(USER, skill, description="единственное")
    callback = make_callback(USER, f"hist:{skill['id']}")
    await tasks.cb_history(callback)
    assert "единственное" in last_text(callback.message)
