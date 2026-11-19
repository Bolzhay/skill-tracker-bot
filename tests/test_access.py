from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.filters import CommandObject

from bot import access, config, db
from bot.handlers import admin
from tests.helpers import all_text, last_text, make_callback, make_message


def _middleware(kind="message"):
    return access.AccessMiddleware(kind)


async def _run(kind, event):
    handler = AsyncMock(return_value="handled")
    result = await _middleware(kind)(handler, event, {})
    return handler, result


# --- middleware -----------------------------------------------------------


async def test_before_the_bot_is_claimed_everything_but_claim_is_refused(fake_db):
    message = make_message(1, "/start")
    handler, result = await _run("message", message)

    handler.assert_not_called()
    assert result is None
    assert "не активирован" in last_text(message)


async def test_claim_gets_through_while_the_bot_is_unclaimed(fake_db):
    for text in ("/claim secret", "/claim@some_bot secret"):
        handler, result = await _run("message", make_message(1, text))
        handler.assert_awaited_once()
        assert result == "handled"


async def test_buttons_are_refused_while_the_bot_is_unclaimed(fake_db):
    callback = make_callback(1, "task:abc")
    handler, _ = await _run("callback", callback)

    handler.assert_not_called()
    assert callback.answer.call_args.kwargs["show_alert"] is True


async def test_whitelisted_user_gets_through(fake_db):
    db.claim_owner(1, "Владелец")
    handler, result = await _run("message", make_message(1, "/task"))

    handler.assert_awaited_once()
    assert result == "handled"


async def test_stranger_is_refused_and_told_their_id(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(42, "/task")
    handler, result = await _run("message", message)

    handler.assert_not_called()
    assert result is None
    assert "Твой Telegram ID: 42" in last_text(message)
    assert "/allow 42" in last_text(message)


async def test_stranger_pressing_a_button_gets_an_alert(fake_db):
    db.claim_owner(1, "Владелец")
    callback = make_callback(42, "task:abc")
    handler, _ = await _run("callback", callback)

    handler.assert_not_called()
    assert callback.answer.call_args.kwargs["show_alert"] is True


async def test_group_chats_are_ignored_silently(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(42, "/task")
    message.chat.type = "supergroup"
    handler, result = await _run("message", message)

    handler.assert_not_called()
    assert result is None
    message.answer.assert_not_called()


async def test_events_without_a_sender_are_dropped(fake_db):
    message = make_message(1, "/task")
    message.from_user = None
    handler, result = await _run("message", message)

    handler.assert_not_called()
    assert result is None


async def test_access_changes_apply_once_the_cache_is_invalidated(fake_db):
    db.claim_owner(1, "Владелец")
    handler, _ = await _run("message", make_message(2, "/task"))
    handler.assert_not_called()

    db.set_user(2, True)
    handler, _ = await _run("message", make_message(2, "/task"))
    handler.assert_not_called()  # still cached as denied (30 s TTL)

    access.invalidate(2)
    handler, _ = await _run("message", make_message(2, "/task"))
    handler.assert_awaited_once()


# --- /claim ---------------------------------------------------------------


async def test_wrong_code_does_not_claim(fake_db):
    message = make_message(5, "/claim nope")
    await admin.cmd_claim(message, CommandObject(command="claim", args="nope"))

    assert "Неверный код" in last_text(message)
    assert db.get_owner_id() is None


async def test_right_code_makes_the_sender_the_owner_and_hides_the_code(fake_db):
    message = make_message(5, f"/claim {config.CLAIM_CODE}")
    await admin.cmd_claim(message, CommandObject(command="claim", args=config.CLAIM_CODE))

    assert db.get_owner_id() == 5
    assert db.is_allowed(5)
    message.delete.assert_awaited_once()
    assert "владелец" in last_text(message)


async def test_claim_reports_migrated_legacy_skills(fake_db):
    fake_db.collection("skills").document("s1").set({"name": "Старый"})
    message = make_message(5)
    await admin.cmd_claim(message, CommandObject(command="claim", args=config.CLAIM_CODE))

    assert "Перенёс скиллов из старой базы: 1" in last_text(message)
    assert db.get_skill(5, "s1")["name"] == "Старый"


async def test_second_claim_is_rejected(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(9)
    await admin.cmd_claim(message, CommandObject(command="claim", args=config.CLAIM_CODE))

    assert "уже активирован" in last_text(message)
    assert db.get_owner_id() == 1


async def test_claim_without_a_configured_code_explains_what_to_do(fake_db, monkeypatch):
    monkeypatch.setattr(config, "CLAIM_CODE", "")
    message = make_message(5)
    await admin.cmd_claim(message, CommandObject(command="claim", args=""))

    assert "CLAIM_CODE" in last_text(message)
    assert db.get_owner_id() is None


async def test_failed_message_delete_does_not_break_the_claim(fake_db):
    message = make_message(5)
    message.delete = AsyncMock(side_effect=RuntimeError("can't delete"))
    await admin.cmd_claim(message, CommandObject(command="claim", args=config.CLAIM_CODE))

    assert db.get_owner_id() == 5


# --- /allow and /deny -----------------------------------------------------


async def test_owner_can_open_and_close_access(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(1)

    await admin.cmd_allow(message, CommandObject(command="allow", args="222"))
    assert db.is_allowed(222)
    assert "222" in last_text(message)

    await admin.cmd_deny(message, CommandObject(command="deny", args="222"))
    assert not db.is_allowed(222)


async def test_allow_takes_effect_immediately_despite_the_cache(fake_db):
    db.claim_owner(1, "Владелец")
    handler, _ = await _run("message", make_message(222, "/task"))
    handler.assert_not_called()  # now cached as denied

    await admin.cmd_allow(make_message(1), CommandObject(command="allow", args="222"))
    handler, _ = await _run("message", make_message(222, "/task"))
    handler.assert_awaited_once()


async def test_only_the_owner_can_change_access(fake_db):
    db.claim_owner(1, "Владелец")
    db.set_user(2, True)
    message = make_message(2)

    await admin.cmd_allow(message, CommandObject(command="allow", args="333"))
    assert "только для владельца" in last_text(message)
    assert not db.is_allowed(333)

    await admin.cmd_deny(message, CommandObject(command="deny", args="1"))
    assert "только для владельца" in last_text(message)
    assert db.is_allowed(1)


async def test_the_owner_cannot_lock_themselves_out(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(1)
    await admin.cmd_deny(message, CommandObject(command="deny", args="1"))

    assert "Нельзя" in last_text(message)
    assert db.is_allowed(1)


async def test_allow_and_deny_validate_the_id(fake_db):
    db.claim_owner(1, "Владелец")
    message = make_message(1)

    for command, handler in (("allow", admin.cmd_allow), ("deny", admin.cmd_deny)):
        for bad in (None, "", "abc", "12x"):
            message.answer.reset_mock()
            await handler(message, CommandObject(command=command, args=bad))
            assert "Использование" in last_text(message)


async def test_users_have_separate_data(fake_db):
    db.claim_owner(1, "Владелец")
    await admin.cmd_allow(make_message(1), CommandObject(command="allow", args="2"))

    first = db.create_skill(1, "Навык владельца")
    second = db.create_skill(2, "Навык гостя")
    assert [s["id"] for s in db.list_skills(1)] == [first]
    assert [s["id"] for s in db.list_skills(2)] == [second]
    assert db.get_skill(2, first) is None
