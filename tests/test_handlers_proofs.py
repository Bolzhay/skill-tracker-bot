import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.filters import CommandObject

from bot import config, db, gamification, transcribe
from bot.handlers import progress, submissions
from bot.llm import eval_pipeline
from tests.helpers import (
    PNG,
    arm,
    callback_data,
    last_markup,
    last_text,
    make_callback,
    make_message,
    seed_skill,
    seed_task,
    verdict,
)

USER = 1
TODAY = lambda: gamification.local_today(config.TZ_OFFSET_HOURS)  # noqa: E731


def command(name, args=None):
    return CommandObject(command=name, args=args)


def attempts_of(fake_db, skill, task):
    ref = (
        fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"])
        .collection("tasks").document(task["id"]).collection("attempts")
    )
    return [doc.to_dict() for doc in ref.order_by("created_at").stream()]


@pytest.fixture
def open_task(fake_db):
    skill = seed_skill(USER)
    return skill, seed_task(USER, skill)


@pytest.fixture
def evaluate_text(monkeypatch):
    mock = AsyncMock(return_value=verdict(True))
    monkeypatch.setattr(eval_pipeline, "evaluate_text", mock)
    return mock


@pytest.fixture
def evaluate_photo(monkeypatch):
    mock = AsyncMock(return_value=verdict(True))
    monkeypatch.setattr(eval_pipeline, "evaluate_photo", mock)
    return mock


# --- text proofs ----------------------------------------------------------


async def test_a_passing_text_proof_completes_the_task(fake_db, state, open_task, evaluate_text):
    skill, task = open_task
    await arm(state, skill, task)
    message = make_message(USER, "мой ответ")
    await submissions.on_text(message, state)

    assert evaluate_text.await_args.args[3] == "мой ответ"
    text = last_text(message)
    assert "Честный разбор работы." in text
    assert "Что хорошо:\n+ Чистый контур" in text
    assert "+10 XP" in text
    assert "✅ Задание засчитано (1/3 для перехода на следующий уровень)" in text
    assert "Первое засчитанное задание — новый бейдж!" in text

    assert db.get_task(USER, skill["id"], task["id"])["completed"] is True
    assert await state.get_state() is None
    [attempt] = attempts_of(fake_db, skill, task)
    assert attempt["xp_awarded"] == 10 and attempt["media_type"] == "text"
    assert attempt["proof"] == {"kind": "text", "text": "мой ответ"}
    assert attempt["ensemble_feedback"]["task_completed"] is True
    assert callback_data(last_markup(message)) == [f"task:{skill['id']}"]


async def test_a_failed_attempt_leaves_the_task_open_and_the_next_message_is_the_retake(
    fake_db, state, open_task, evaluate_text
):
    skill, task = open_task
    await arm(state, skill, task)
    evaluate_text.side_effect = [verdict(False), verdict(True)]

    first = make_message(USER, "слабый ответ")
    await submissions.on_text(first, state)
    text = last_text(first)
    assert "❌ Не засчитано" in text and "+2 XP" in text and "– Линии неровные" in text
    assert callback_data(last_markup(first)) == [f"hint:{skill['id']}"]
    assert db.get_task(USER, skill["id"], task["id"])["completed"] is False
    assert await state.get_state() == submissions.TaskSubmission.awaiting_proof.state

    second = make_message(USER, "лучше")
    await submissions.on_text(second, state)
    assert "✅ Задание засчитано" in last_text(second)
    assert await state.get_state() is None
    assert len(attempts_of(fake_db, skill, task)) == 2
    assert db.get_skill(USER, skill["id"])["xp_total"] == 2 + 10


async def test_when_evaluation_breaks_nothing_is_lost(fake_db, state, open_task, evaluate_text):
    skill, task = open_task
    await arm(state, skill, task)
    evaluate_text.side_effect = RuntimeError("models down")
    message = make_message(USER, "ответ")
    await submissions.on_text(message, state)

    assert "никуда не делось" in last_text(message)
    assert attempts_of(fake_db, skill, task) == []
    assert db.get_skill(USER, skill["id"])["xp_total"] == 0
    assert await state.get_state() == submissions.TaskSubmission.awaiting_proof.state  # just resend


async def test_a_task_that_vanished_from_the_database_resets_the_dialog(fake_db, state, open_task, evaluate_text):
    skill, _ = open_task
    await state.set_state(submissions.TaskSubmission.awaiting_proof)
    await state.update_data(skill_id=skill["id"], task_id="gone", level_id=skill["current_level_id"])
    message = make_message(USER, "ответ")
    await submissions.on_text(message, state)

    assert "Не нашёл это задание" in last_text(message)
    assert await state.get_state() is None
    evaluate_text.assert_not_called()


async def test_very_long_feedback_still_fits_in_one_telegram_message(fake_db, state, open_task, evaluate_text):
    skill, task = open_task
    await arm(state, skill, task)
    evaluate_text.return_value = {**verdict(True), "feedback": "очень подробно " * 600}
    message = make_message(USER, "ответ")
    await submissions.on_text(message, state)

    assert len(last_text(message)) <= 4000


async def test_unsupported_formats_are_declined(fake_db):
    message = make_message(USER)
    await submissions.on_unsupported(message)
    assert "фото, голосовое, видео или текст" in last_text(message)


# --- level-up, rank, tree end ---------------------------------------------


async def test_the_third_pass_levels_the_learner_up(fake_db, state, evaluate_text):
    skill = seed_skill(USER, level_count=3)
    message = None
    for _ in range(3):
        task = seed_task(USER, skill)
        await arm(state, skill, task)
        message = make_message(USER, "ответ")
        await submissions.on_text(message, state)

    text = last_text(message)
    assert "🎉 Левелап! Новый уровень 2: «Уровень 2»" in text
    assert "для перехода" not in text
    assert db.get_skill(USER, skill["id"])["current_level_id"] != skill["current_level_id"]


async def test_finishing_the_last_level_congratulates(fake_db, state, evaluate_text):
    skill = seed_skill(USER, level_count=1)
    message = None
    for _ in range(3):
        task = seed_task(USER, skill)
        await arm(state, skill, task)
        message = make_message(USER, "ответ")
        await submissions.on_text(message, state)

    assert "🏆" in last_text(message)


async def test_crossing_an_xp_threshold_announces_the_new_rank(fake_db, state, open_task, evaluate_text):
    skill, task = open_task
    fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"]).update({"xp_total": 45})
    await arm(state, skill, task)
    message = make_message(USER, "ответ")
    await submissions.on_text(message, state)

    assert "⭐ Новый ранг: Ученик" in last_text(message)


async def test_streak_shows_up_in_the_result(fake_db, state, open_task, evaluate_text):
    skill, task = open_task
    from datetime import timedelta

    yesterday = (TODAY() - timedelta(days=1)).isoformat()
    fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"]).update(
        {"streak_count": 4, "streak_last_date": yesterday}
    )
    await arm(state, skill, task)
    message = make_message(USER, "ответ")
    await submissions.on_text(message, state)

    assert "🔥 стрик 5 дн." in last_text(message)
    assert "+14 XP" in last_text(message)  # 10 base + 4 streak bonus


# --- photo proofs ---------------------------------------------------------


def photo_message(data=PNG, file_id="big", file_size=None):
    message = make_message(
        USER,
        photo=[SimpleNamespace(file_id="small", file_size=10), SimpleNamespace(file_id=file_id, file_size=file_size or len(data))],
    )
    message.bot.download = AsyncMock(return_value=io.BytesIO(data))
    return message


async def test_a_photo_is_evaluated_and_archived(fake_db, state, open_task, evaluate_photo):
    skill, task = open_task
    await arm(state, skill, task)
    message = photo_message()
    await submissions.on_photo(message, state)

    message.bot.download.assert_awaited_once_with("big")  # the largest size
    assert evaluate_photo.await_args.args[3] == PNG
    [attempt] = attempts_of(fake_db, skill, task)
    assert attempt["proof"] == {"kind": "photo", "file_id": "big", "mime": "image/png", "size": len(PNG)}
    assert attempt["has_blob"] is True
    assert db.get_attempt_blob(USER, skill["id"], task["id"], _attempt_id(fake_db, skill, task)) == PNG
    assert "✅" in last_text(message)


def _attempt_id(fake_db, skill, task):
    ref = (
        fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"])
        .collection("tasks").document(task["id"]).collection("attempts")
    )
    return next(iter(ref.stream())).id


async def test_a_photo_sent_as_a_file_is_treated_the_same(fake_db, state, open_task, evaluate_photo):
    skill, task = open_task
    await arm(state, skill, task)
    message = make_message(USER, document=SimpleNamespace(file_id="doc1", file_size=len(PNG), mime_type="image/png"))
    message.bot.download = AsyncMock(return_value=io.BytesIO(PNG))
    await submissions.on_image_file(message, state)

    message.bot.download.assert_awaited_once_with("doc1")
    assert attempts_of(fake_db, skill, task)[0]["proof"]["file_id"] == "doc1"


async def test_a_photo_too_big_for_a_document_is_kept_by_file_id_only(fake_db, state, open_task, evaluate_photo):
    skill, task = open_task
    await arm(state, skill, task)
    big = b"\x89PNG" + b"0" * (submissions.MAX_BLOB_BYTES + 1)
    message = photo_message(big)
    await submissions.on_photo(message, state)

    [attempt] = attempts_of(fake_db, skill, task)
    assert attempt["has_blob"] is False
    assert attempt["proof"]["file_id"] == "big" and attempt["proof"]["size"] == len(big)
    assert evaluate_photo.await_args.args[3] == big  # it was still evaluated in full


async def test_a_file_telegram_will_not_let_the_bot_download_is_refused(fake_db, state, open_task, evaluate_photo):
    skill, task = open_task
    await arm(state, skill, task)
    message = photo_message(file_size=submissions.MAX_FILE_BYTES + 1)
    await submissions.on_photo(message, state)

    assert "больше 20 МБ" in last_text(message)
    message.bot.download.assert_not_called()
    evaluate_photo.assert_not_called()
    assert await state.get_state() == submissions.TaskSubmission.awaiting_proof.state


async def test_photo_evaluation_failure_keeps_the_task_open(fake_db, state, open_task, evaluate_photo):
    skill, task = open_task
    await arm(state, skill, task)
    evaluate_photo.side_effect = RuntimeError("vision models down")
    message = photo_message()
    await submissions.on_photo(message, state)

    assert "никуда не делось" in last_text(message)
    assert attempts_of(fake_db, skill, task) == []


# --- voice and video proofs -----------------------------------------------


@pytest.fixture
def transcribe_bytes(monkeypatch):
    mock = AsyncMock(return_value="привет мир")
    monkeypatch.setattr(transcribe, "transcribe_bytes", mock)
    return mock


async def test_a_voice_message_is_transcribed_then_evaluated_as_text(
    fake_db, state, open_task, evaluate_text, transcribe_bytes
):
    skill, task = open_task
    await arm(state, skill, task)
    message = make_message(USER, voice=SimpleNamespace(file_id="v1", file_size=500, mime_type="audio/ogg"))
    message.bot.download = AsyncMock(return_value=io.BytesIO(b"OGGDATA"))
    await submissions.on_audio_video(message, state)

    transcribe_bytes.assert_awaited_once_with(b"OGGDATA", ".ogg")
    assert evaluate_text.await_args.args[3] == "привет мир"
    assert evaluate_text.await_args.kwargs["source"] == "voice"
    assert "🎙 Расшифровка: «привет мир»" in last_text(message)

    [attempt] = attempts_of(fake_db, skill, task)
    assert attempt["proof"] == {
        "kind": "voice", "file_id": "v1", "mime": "audio/ogg", "size": 7, "transcript": "привет мир",
    }
    assert attempt["has_blob"] is True


async def test_a_round_video_note_is_handled_as_video(fake_db, state, open_task, evaluate_text, transcribe_bytes):
    skill, task = open_task
    await arm(state, skill, task)
    message = make_message(USER, video_note=SimpleNamespace(file_id="n1", file_size=900))  # no mime_type attribute
    message.bot.download = AsyncMock(return_value=io.BytesIO(b"MP4"))
    await submissions.on_audio_video(message, state)

    transcribe_bytes.assert_awaited_once_with(b"MP4", ".mp4")
    assert evaluate_text.await_args.kwargs["source"] == "video"
    assert attempts_of(fake_db, skill, task)[0]["proof"]["mime"] == ""


async def test_audio_files_and_videos_pick_the_right_decoder_hint(fake_db, state, open_task, evaluate_text, transcribe_bytes):
    skill, task = open_task
    for kind, suffix in (("audio", ".mp3"), ("video", ".mp4")):
        await arm(state, skill, task)
        message = make_message(USER, **{kind: SimpleNamespace(file_id="x", file_size=10, mime_type="x/y")})
        message.bot.download = AsyncMock(return_value=io.BytesIO(b"DATA"))
        transcribe_bytes.reset_mock()
        await submissions.on_audio_video(message, state)
        assert transcribe_bytes.await_args.args[1] == suffix
        await arm(state, skill, seed_task(USER, skill))  # the previous one passed and closed


async def test_silence_is_not_evaluated(fake_db, state, open_task, evaluate_text, transcribe_bytes):
    skill, task = open_task
    await arm(state, skill, task)
    transcribe_bytes.return_value = ""
    message = make_message(USER, voice=SimpleNamespace(file_id="v", file_size=10, mime_type="audio/ogg"))
    message.bot.download = AsyncMock(return_value=io.BytesIO(b"x"))
    await submissions.on_audio_video(message, state)

    assert "Не услышал речи" in last_text(message)
    evaluate_text.assert_not_called()
    assert attempts_of(fake_db, skill, task) == []


async def test_a_transcription_crash_is_reported(fake_db, state, open_task, evaluate_text, transcribe_bytes):
    skill, task = open_task
    await arm(state, skill, task)
    transcribe_bytes.side_effect = RuntimeError("decoder failed")
    message = make_message(USER, voice=SimpleNamespace(file_id="v", file_size=10, mime_type="audio/ogg"))
    message.bot.download = AsyncMock(return_value=io.BytesIO(b"x"))
    await submissions.on_audio_video(message, state)

    assert "Не получилось расшифровать" in last_text(message)
    evaluate_text.assert_not_called()


# --- /retry ---------------------------------------------------------------


async def test_retry_rearms_the_last_open_task(fake_db, state):
    skill = seed_skill(USER)
    seed_task(USER, skill, description="старое", completed=True)
    open_task = seed_task(USER, skill, description="открытое")
    message = make_message(USER)
    await submissions.cmd_retry(message, command("retry"), state)

    assert (await state.get_data())["task_id"] == open_task["id"]
    assert await state.get_state() == submissions.TaskSubmission.awaiting_proof.state
    assert "открытое" in last_text(message) and "Подсказка: подсказка" in last_text(message)


async def test_retry_for_a_finished_task_points_to_a_new_one(fake_db, state):
    skill = seed_skill(USER)
    seed_task(USER, skill, completed=True)
    message = make_message(USER)
    await submissions.cmd_retry(message, command("retry"), state)

    assert "уже засчитано" in last_text(message)
    assert await state.get_state() is None
    assert f"task:{skill['id']}" in callback_data(last_markup(message))


async def test_retry_without_any_task(fake_db, state):
    seed_skill(USER)
    message = make_message(USER)
    await submissions.cmd_retry(message, command("retry"), state)

    assert "/task" in last_text(message)
    assert await state.get_state() is None


async def test_retry_button(fake_db, state):
    skill = seed_skill(USER)
    seed_task(USER, skill)
    callback = make_callback(USER, f"retry:{skill['id']}")
    await submissions.cb_retry(callback, state)

    callback.answer.assert_awaited_once()
    assert await state.get_state() == submissions.TaskSubmission.awaiting_proof.state


# --- /progress ------------------------------------------------------------


def photo_pass(skill, description, data, blob=True, file_id="fid"):
    task = seed_task(USER, skill, completed=True, description=description)
    proof = {"kind": "photo", "file_id": file_id, "mime": "image/png", "size": len(data), "bytes": data if blob else None}
    db.create_attempt(USER, skill["id"], task["id"], "photo", verdict(True), 10, proof)
    return task


@pytest.fixture
def compare(monkeypatch):
    mock = AsyncMock(return_value="Линии стали ровнее")
    monkeypatch.setattr(eval_pipeline, "compare_progress", mock)
    return mock


async def test_progress_needs_at_least_two_photo_passes(fake_db, compare):
    skill = seed_skill(USER)
    photo_pass(skill, "одно", PNG)
    message = make_message(USER)
    await progress.cmd_progress(message, command("progress"))

    assert "минимум два" in last_text(message) and "пока их 1" in last_text(message)
    compare.assert_not_called()


async def test_progress_shows_first_and_last_work_with_a_comment(fake_db, compare):
    skill = seed_skill(USER)
    first = photo_pass(skill, "первая", PNG + b"1")
    photo_pass(skill, "средняя", PNG + b"2")
    last = photo_pass(skill, "последняя", PNG + b"3")
    message = make_message(USER)
    await progress.cmd_progress(message, command("progress"))

    media = message.bot.send_media_group.await_args.args[1]
    assert [item.caption for item in media] == ["Тогда", "Сейчас"]
    skill_name, before_task, after_task, before, after = compare.await_args.args
    assert (before_task["id"], after_task["id"]) == (first["id"], last["id"])
    assert (before, after) == (PNG + b"1", PNG + b"3")
    assert last_text(message) == "Линии стали ровнее"


async def test_progress_falls_back_to_telegram_when_the_bytes_were_not_archived(fake_db, compare):
    skill = seed_skill(USER)
    photo_pass(skill, "первая", PNG, blob=False, file_id="old-file")
    photo_pass(skill, "последняя", PNG + b"3")
    message = make_message(USER)
    message.bot.download = AsyncMock(return_value=io.BytesIO(b"from-telegram"))
    await progress.cmd_progress(message, command("progress"))

    message.bot.download.assert_awaited_once_with("old-file")
    assert compare.await_args.args[3] == b"from-telegram"


async def test_progress_reports_an_unrecoverable_archive(fake_db, compare):
    skill = seed_skill(USER)
    photo_pass(skill, "первая", PNG, blob=False, file_id="")
    photo_pass(skill, "последняя", PNG)
    message = make_message(USER)
    await progress.cmd_progress(message, command("progress"))

    assert "Не смог достать" in last_text(message)
    message.bot.send_media_group.assert_not_called()


async def test_progress_still_shows_the_photos_if_the_comment_fails(fake_db, compare):
    skill = seed_skill(USER)
    photo_pass(skill, "первая", PNG)
    photo_pass(skill, "последняя", PNG + b"1")
    compare.side_effect = RuntimeError("vision models down")
    message = make_message(USER)
    await progress.cmd_progress(message, command("progress"))

    message.bot.send_media_group.assert_awaited_once()
    assert "сравнить словами не получилось" in last_text(message)


async def test_progress_button(fake_db, compare):
    skill = seed_skill(USER)
    callback = make_callback(USER, f"prog:{skill['id']}")
    await progress.cb_progress(callback)

    callback.answer.assert_awaited_once()
    assert "минимум два" in last_text(callback.message)
