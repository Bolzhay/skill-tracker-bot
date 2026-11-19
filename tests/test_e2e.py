"""End-to-end through the real Dispatcher: routing order, access middleware,
FSM states, filters and callback queries — with only the LLM pipelines, the
speech model and Telegram's HTTP API stubbed."""

from datetime import datetime  # noqa: F401
from unittest.mock import AsyncMock

import pytest
from aiogram.methods import SendMediaGroup

from bot import db, transcribe, youtube
from bot.keyboards import MENU_NEW, MENU_SKILLS
from bot.llm import eval_pipeline, openrouter, task_pipeline, tree_pipeline
from tests.fake_telegram import Client
from tests.helpers import PNG, levels, seed_skill, seed_task, verdict


@pytest.fixture
def llm(monkeypatch):
    """Stub every model call; tests assert on what the bot did with the answers."""
    stubs = SimpleStubs = type("Stubs", (), {})()
    stubs.tree = AsyncMock(return_value=levels(3))
    stubs.task = AsyncMock(
        return_value={"description": "Нарисуй квадрат", "hint": "Возьми Линию", "media_type": "photo", "is_wildcard": False}
    )
    stubs.photo = AsyncMock(return_value=verdict(True))
    stubs.text = AsyncMock(return_value=verdict(True))
    stubs.hint = AsyncMock(return_value={"hint": "Начни с сетки", "search_queries": []})
    stubs.videos = AsyncMock(return_value=[])
    stubs.compare = AsyncMock(return_value="Линии стали ровнее")
    stubs.transcribe = AsyncMock(return_value="привет мир")
    monkeypatch.setattr(tree_pipeline, "build_skill_tree", stubs.tree)
    monkeypatch.setattr(task_pipeline, "generate_task", stubs.task)
    monkeypatch.setattr(task_pipeline, "generate_detailed_hint", stubs.hint)
    monkeypatch.setattr(youtube, "find_videos", stubs.videos)
    monkeypatch.setattr(eval_pipeline, "evaluate_photo", stubs.photo)
    monkeypatch.setattr(eval_pipeline, "evaluate_text", stubs.text)
    monkeypatch.setattr(eval_pipeline, "compare_progress", stubs.compare)
    monkeypatch.setattr(transcribe, "transcribe_bytes", stubs.transcribe)
    return stubs


# --- activation and access ------------------------------------------------


async def test_a_fresh_bot_accepts_only_the_claim_code(fake_db, dispatcher):
    user = Client(dispatcher, 1)

    await user.say("/start")
    assert "не активирован" in user.last()

    await user.say("/claim wrong-code")
    assert "Неверный код" in user.last() and db.get_owner_id() is None

    await user.say("/claim test-claim-code")
    assert "владелец" in user.last()
    assert user.deleted_messages(), "the message with the code should be removed from the chat"
    assert db.get_owner_id() == 1

    await user.say("/start")
    assert "Skill Tracker Bot" in user.last()


async def test_a_stranger_is_locked_out_until_the_owner_lets_them_in(owner, dispatcher):
    stranger = Client(dispatcher, 99)

    await stranger.say("/profile")
    assert "Твой Telegram ID: 99" in stranger.last()
    await stranger.press("task:whatever")
    assert stranger.alerts(), "buttons are refused with an alert"

    await owner.say("/allow 99")
    assert "Доступ открыт для 99" in owner.last()

    await stranger.say("/profile")
    assert "Скиллов пока нет" in stranger.last()  # they get their own, empty, space


async def test_users_never_see_each_others_skills(owner, dispatcher):
    seed_skill(1, "Скилл владельца")
    await owner.say("/allow 2")
    guest = Client(dispatcher, 2)

    await guest.say("/profile")
    assert "Скиллов пока нет" in guest.last()
    await owner.say("/profile")
    assert "Скилл владельца" in owner.last()


async def test_only_the_owner_can_manage_access(owner, dispatcher):
    await owner.say("/allow 2")
    guest = Client(dispatcher, 2)

    await guest.say("/allow 3")
    assert "только для владельца" in guest.last()
    await guest.say("/deny 1")
    assert "только для владельца" in guest.last()
    assert db.is_allowed(1)


# --- the whole journey ----------------------------------------------------


async def test_from_a_new_skill_to_a_level_up(owner, llm):
    await owner.say(MENU_NEW)
    assert "Как назвать" in owner.last()
    await owner.say("Пиксель-арт")
    assert "1)" in owner.last()
    await owner.say("3 из 10")
    assert "2)" in owner.last()
    await owner.say("рисовал")
    assert "3)" in owner.last()
    await owner.say("нарисовать персонажа")

    assert "Строю дерево" in owner.texts()[-2]
    assert "готово" in owner.last()  # the background build has finished
    skill = db.list_skills(1)[0]
    assert owner.last_buttons() == [f"task:{skill['id']}"]

    for attempt in range(1, 4):
        await owner.press(f"task:{skill['id']}")
        assert "Нарисуй квадрат" in owner.last()
        assert owner.last_buttons() == [f"hint:{skill['id']}", "cancel"]
        await owner.send_photo(PNG, file_id=f"p{attempt}")
        assert "Честный разбор работы." in owner.last() and "+10 XP" in owner.last()

    assert "🎉 Левелап! Новый уровень 2" in owner.last()
    llm.photo.assert_awaited()
    assert llm.photo.await_args.args[3] == PNG  # the downloaded bytes reached the evaluator

    await owner.say("/profile")
    assert "Уровень 2/3" in owner.last() and "30 XP" in owner.last()


async def test_a_single_skill_needs_no_name_and_two_skills_ask(owner, llm):
    first = seed_skill(1, "Риторика")
    await owner.say("/task")
    assert "Нарисуй квадрат" in owner.last()

    second = seed_skill(1, "Химия")
    await owner.say("/task")
    assert owner.last_buttons() == [f"task:{first['id']}", f"task:{second['id']}"]

    await owner.press(f"task:{second['id']}")
    assert "Нарисуй квадрат" in owner.last()
    assert len(db.get_recent_tasks(1, second["id"])) == 1


# --- routing order --------------------------------------------------------


async def test_while_a_proof_is_awaited_menu_buttons_and_commands_are_not_graded(owner, llm):
    seed_skill(1)
    await owner.say("/task")

    await owner.say(MENU_SKILLS)
    assert "Твои скиллы" in owner.last()
    await owner.say("/history")
    assert "Нарисуй квадрат" in owner.last()
    await owner.say("/foo")
    assert "Не понял" in owner.last()
    llm.text.assert_not_called()

    await owner.say("а вот это уже ответ")  # a plain message still counts as the proof
    llm.text.assert_awaited_once()
    assert llm.text.await_args.args[3] == "а вот это уже ответ"


async def test_commands_typed_during_onboarding_are_commands_not_answers(owner, llm):
    seed_skill(1, "Старый скилл")
    await owner.say("/new_skill Риторика")
    assert "1)" in owner.last()

    await owner.say("/profile")
    assert "Старый скилл" in owner.last()

    await owner.say("3 из 10")  # the dialog was not disturbed
    assert "2)" in owner.last()


async def test_cancel_drops_a_half_finished_dialog(owner, llm):
    await owner.say("/new_skill Риторика")
    await owner.say("/cancel")
    assert "Отменил" in owner.last()

    await owner.say("просто текст")
    assert "Не понял" in owner.last()  # no longer treated as an answer
    assert db.list_skills(1) == []


async def test_unsupported_content_while_a_proof_is_awaited(owner, llm):
    seed_skill(1)
    await owner.say("/task")

    await owner.send_sticker()
    assert "Такой формат оценить не могу" in owner.last()
    await owner.send_document(b"%PDF-1.4", "application/pdf")
    assert "Такой формат оценить не могу" in owner.last()
    llm.photo.assert_not_called()


async def test_stale_buttons_get_an_alert_instead_of_a_spinner(owner):
    await owner.press("garbage:123")
    assert owner.alerts()


async def test_a_spent_quota_is_explained_everywhere_a_model_is_used(owner, llm):
    skill = seed_skill(1)
    quota = openrouter.RateLimitExhausted("429")
    llm.task.side_effect = quota
    llm.hint.side_effect = quota
    llm.photo.side_effect = quota
    llm.compare.side_effect = quota

    await owner.press(f"task:{skill['id']}")
    assert openrouter.QUOTA_MESSAGE == owner.last()

    seed_task(1, skill)
    await owner.press(f"hint:{skill['id']}")
    assert openrouter.QUOTA_MESSAGE == owner.last()

    await owner.press(f"retry:{skill['id']}")
    await owner.send_photo(PNG)
    assert openrouter.QUOTA_MESSAGE == owner.last()  # the proof stays valid, just resend later

    llm.tree.side_effect = quota
    await owner.say("/new_skill Химия")
    for answer in ("5", "учил", "олимпиада"):
        await owner.say(answer)
    assert openrouter.QUOTA_MESSAGE == owner.last()


async def test_a_crash_inside_a_handler_still_answers(owner, monkeypatch):
    monkeypatch.setattr(db, "list_skills", lambda user_id: 1 / 0)
    await owner.say("/profile")
    assert "Что-то пошло не так" in owner.last()


# --- proofs of every kind -------------------------------------------------


async def test_an_image_sent_as_a_file_is_evaluated_like_a_photo(owner, llm):
    seed_skill(1)
    await owner.say("/task")
    await owner.send_document(PNG, "image/png")

    llm.photo.assert_awaited_once()
    assert llm.photo.await_args.args[3] == PNG
    assert "✅" in owner.last()


async def test_a_voice_message_is_transcribed_and_graded(owner, llm):
    seed_skill(1)
    await owner.say("/task")
    await owner.send_voice(b"OGG-BYTES")

    llm.transcribe.assert_awaited_once_with(b"OGG-BYTES", ".ogg")
    assert llm.text.await_args.kwargs["source"] == "voice"
    assert "🎙 Расшифровка: «привет мир»" in owner.last()


async def test_a_failed_proof_can_be_resent_straight_away(owner, llm):
    seed_skill(1)
    await owner.say("/task")
    llm.text.side_effect = [verdict(False), verdict(True)]

    await owner.say("слабый ответ")
    assert "❌ Не засчитано" in owner.last()
    await owner.say("лучше")
    assert "✅" in owner.last()


async def test_retry_works_after_a_restart_wiped_the_dialog_state(owner, llm, dispatcher):
    skill = seed_skill(1)
    task = seed_task(1, skill, description="Нарисуй круг")

    dispatcher.storage.storage.clear()  # what a bot restart does to in-memory FSM state
    await owner.say("/retry")
    assert "Нарисуй круг" in owner.last()

    await owner.say("мой ответ")
    assert "✅" in owner.last()
    assert db.get_task(1, skill["id"], task["id"])["completed"] is True


async def test_hint_and_history_buttons(owner, llm):
    skill = seed_skill(1)
    await owner.press(f"task:{skill['id']}")
    await owner.press(f"hint:{skill['id']}")
    assert owner.last() == "Начни с сетки"

    await owner.press(f"hist:{skill['id']}")
    assert "⏳ Нарисуй квадрат" in owner.last()


async def test_the_hint_button_attaches_real_youtube_links(owner, llm):
    skill = seed_skill(1)
    seed_task(1, skill)
    llm.hint.return_value = {"hint": "Начни с сетки", "search_queries": ["пиксель-арт линии"]}
    llm.videos.return_value = [{"title": "Урок", "channel": "Канал", "url": "https://www.youtube.com/watch?v=abc"}]

    await owner.press(f"hint:{skill['id']}")

    assert "Начни с сетки" in owner.last() and "https://www.youtube.com/watch?v=abc" in owner.last()
    llm.videos.assert_awaited_once_with(["пиксель-арт линии"])


async def test_progress_compares_the_first_and_last_photo(owner, llm):
    skill = seed_skill(1)
    for n in range(2):
        await owner.press(f"task:{skill['id']}")
        await owner.send_photo(PNG + bytes([n]), file_id=f"p{n}")

    owner.clear()
    await owner.press(f"prog:{skill['id']}")

    assert any(isinstance(r, SendMediaGroup) for r in owner.session.requests)
    assert owner.last() == "Линии стали ровнее"


async def test_a_failed_tree_build_can_be_retried_with_the_button(owner, llm):
    llm.tree.side_effect = [RuntimeError("models down"), levels(2)]
    await owner.say("/new_skill Химия")
    for answer in ("5", "учил", "олимпиада"):
        await owner.say(answer)

    assert "Не получилось" in owner.last()
    skill = db.list_skills(1)[0]
    assert owner.last_buttons() == [f"tree:{skill['id']}"]

    await owner.press(f"tree:{skill['id']}")
    assert "готово" in owner.last()
    assert db.get_skill(1, skill["id"])["levels_total"] == 2
