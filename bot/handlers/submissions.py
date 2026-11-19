import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from .. import config, db, gamification, transcribe
from ..keyboards import after_fail_kb, after_success_kb, after_task_kb
from ..llm import eval_pipeline
from ..llm.openrouter import failure_text
from .common import callback_arg, skill_or_prompt

router = Router()

MAX_FILE_BYTES = 20 * 1024 * 1024  # the Bot API won't let a bot download more
MAX_BLOB_BYTES = 900_000  # a Firestore document is capped at 1 MiB
MAX_TRANSCRIPT_SHOWN = 500

_EVAL_FAILED = (
    "Не получилось оценить пруф (бесплатные модели сейчас перегружены). Задание "
    "никуда не делось — пришли пруф ещё раз, когда захочешь повторить попытку."
)


class TaskSubmission(StatesGroup):
    awaiting_proof = State()


async def _prepare(message: Message, state: FSMContext):
    data = await state.get_data()
    user_id = message.from_user.id
    skill = db.get_skill(user_id, data["skill_id"])
    level = db.get_level(user_id, data["skill_id"], data["level_id"])
    task = db.get_task(user_id, data["skill_id"], data["task_id"])
    if not (skill and level and task):
        await state.clear()
        await message.answer("Не нашёл это задание в базе. Получи новое: /task")
        return None
    return user_id, skill, level, task


async def _download(message: Message, file_id: str, size: int | None) -> bytes | None:
    if size and size > MAX_FILE_BYTES:
        await message.answer("Файл больше 20 МБ — Telegram не даёт боту его скачать. Пришли поменьше.")
        return None
    buffer = await message.bot.download(file_id)
    return buffer.read()


def _proof(kind: str, file_id: str, data: bytes, mime: str = "", **extra) -> dict:
    """What gets archived with the attempt. The bytes are kept only if they
    fit in a Firestore doc; file_id lets Telegram re-send the original."""
    return {
        "kind": kind,
        "file_id": file_id,
        "mime": mime,
        "size": len(data),
        "bytes": data if len(data) <= MAX_BLOB_BYTES else None,
        **extra,
    }


# --- proofs ---------------------------------------------------------------


@router.message(TaskSubmission.awaiting_proof, F.photo)
async def on_photo(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    await _on_image(message, state, photo.file_id, photo.file_size)


@router.message(TaskSubmission.awaiting_proof, F.document.mime_type.startswith("image/"))
async def on_image_file(message: Message, state: FSMContext) -> None:
    # Sent "as a file": Telegram doesn't recompress it, which matters for pixel art.
    await _on_image(message, state, message.document.file_id, message.document.file_size)


async def _on_image(message: Message, state: FSMContext, file_id: str, size: int | None) -> None:
    prepared = await _prepare(message, state)
    if prepared is None:
        return
    user_id, skill, level, task = prepared

    await message.answer("Оцениваю...")
    image = await _download(message, file_id, size)
    if image is None:
        return

    try:
        verdict = await eval_pipeline.evaluate_photo(skill["name"], level, task, image)
    except Exception as exc:
        logging.exception("Evaluation failed for skill_id=%s task_id=%s", skill["id"], task["id"])
        await message.answer(failure_text(exc, _EVAL_FAILED))
        return  # state stays awaiting_proof — nothing lost, just resend

    proof = _proof("photo", file_id, image, mime=eval_pipeline.sniff_mime(image))
    await _finish(message, state, user_id, skill, task, verdict, proof)


@router.message(TaskSubmission.awaiting_proof, F.voice | F.audio | F.video | F.video_note)
async def on_audio_video(message: Message, state: FSMContext) -> None:
    if message.voice:
        media, kind, suffix = message.voice, "voice", ".ogg"
    elif message.audio:
        media, kind, suffix = message.audio, "audio", ".mp3"
    elif message.video:
        media, kind, suffix = message.video, "video", ".mp4"
    else:
        media, kind, suffix = message.video_note, "video", ".mp4"

    prepared = await _prepare(message, state)
    if prepared is None:
        return
    user_id, skill, level, task = prepared

    await message.answer("Расшифровываю и оцениваю...")
    data = await _download(message, media.file_id, media.file_size)
    if data is None:
        return

    try:
        transcript = await transcribe.transcribe_bytes(data, suffix)
    except Exception:
        logging.exception("Transcription failed for skill_id=%s task_id=%s", skill["id"], task["id"])
        await message.answer("Не получилось расшифровать запись. Пришли ещё раз или ответь текстом.")
        return
    if not transcript:
        await message.answer("Не услышал речи в записи. Запиши ещё раз (ближе к микрофону, без шума) или ответь текстом.")
        return

    try:
        verdict = await eval_pipeline.evaluate_text(skill["name"], level, task, transcript, source=kind)
    except Exception as exc:
        logging.exception("Evaluation failed for skill_id=%s task_id=%s", skill["id"], task["id"])
        await message.answer(failure_text(exc, _EVAL_FAILED))
        return

    proof = _proof(kind, media.file_id, data, mime=getattr(media, "mime_type", None) or "", transcript=transcript)
    await _finish(message, state, user_id, skill, task, verdict, proof)


@router.message(TaskSubmission.awaiting_proof, F.text & ~F.text.startswith("/"))
async def on_text(message: Message, state: FSMContext) -> None:
    prepared = await _prepare(message, state)
    if prepared is None:
        return
    user_id, skill, level, task = prepared

    await message.answer("Оцениваю...")
    try:
        verdict = await eval_pipeline.evaluate_text(skill["name"], level, task, message.text)
    except Exception as exc:
        logging.exception("Evaluation failed for skill_id=%s task_id=%s", skill["id"], task["id"])
        await message.answer(failure_text(exc, _EVAL_FAILED))
        return

    await _finish(message, state, user_id, skill, task, verdict, {"kind": "text", "text": message.text})


@router.message(TaskSubmission.awaiting_proof, ~F.text)
async def on_unsupported(message: Message) -> None:
    await message.answer("Такой формат оценить не могу. Пришли фото, голосовое, видео или текст.")


# --- retry ----------------------------------------------------------------


async def do_retry(target: Message, user_id: int, skill: dict, state: FSMContext) -> None:
    """Re-arm submission for the skill's last task without generating a new
    one. Covers a failed evaluation retried later, and the FSM state lost to
    a bot restart (it lives in memory)."""
    recent = db.get_recent_tasks(user_id, skill["id"], limit=1)
    if not recent:
        await target.answer(f"У «{skill['name']}» ещё нет заданий. Начни с /task")
        return

    last_task = recent[0]
    if last_task.get("completed"):
        await target.answer(
            f"Последнее задание по «{skill['name']}» уже засчитано ✅ — присылать пруф для него больше не нужно.",
            reply_markup=after_success_kb(skill["id"]),
        )
        return

    await state.set_state(TaskSubmission.awaiting_proof)
    await state.update_data(skill_id=skill["id"], task_id=last_task["id"], level_id=last_task["level_id"])

    text = f"Жду пруф для последнего задания:\n\n{last_task.get('description', '')}"
    if last_task.get("hint"):
        text += f"\n\nПодсказка: {last_task['hint']}"
    text += "\n\nПришли фото, голосовое, видео или текст."
    await target.answer(text, reply_markup=after_task_kb(skill["id"]))


@router.message(Command("retry"))
async def cmd_retry(message: Message, command: CommandObject, state: FSMContext) -> None:
    user_id = message.from_user.id
    skill = await skill_or_prompt(message, user_id, command.args, "retry")
    if skill:
        await do_retry(message, user_id, skill, state)


@router.callback_query(F.data.startswith("retry:"))
async def cb_retry(callback: CallbackQuery, state: FSMContext) -> None:
    user_id = callback.from_user.id
    skill = db.get_skill(user_id, callback_arg(callback.data))
    await callback.answer()
    if skill:
        await do_retry(callback.message, user_id, skill, state)


# --- result ---------------------------------------------------------------


async def _finish(
    message: Message,
    state: FSMContext,
    user_id: int,
    skill: dict,
    task: dict,
    verdict: dict,
    proof: dict,
) -> None:
    """Record the attempt. Only a genuine pass closes the task and counts
    toward level-up; a fail leaves submission open, so the next message is
    simply the retake."""
    passed = bool(verdict.get("task_completed"))
    today = gamification.local_today(config.TZ_OFFSET_HOURS)
    result = db.record_attempt_result(
        user_id, skill["id"], task["level_id"], passed, bool(task.get("is_wildcard")), today
    )
    db.create_attempt(user_id, skill["id"], task["id"], proof["kind"], verdict, result["xp"], proof)

    if passed:
        db.mark_task_completed(user_id, skill["id"], task["id"])
        await state.clear()

    keyboard = after_success_kb(skill["id"]) if passed else after_fail_kb(skill["id"])
    await message.answer(_render_result(verdict, proof, result, passed)[:4000], reply_markup=keyboard)


def _render_result(verdict: dict, proof: dict, result: dict, passed: bool) -> str:
    lines = []

    transcript = proof.get("transcript")
    if transcript:
        shown = transcript if len(transcript) <= MAX_TRANSCRIPT_SHOWN else transcript[:MAX_TRANSCRIPT_SHOWN] + "…"
        lines += [f"🎙 Расшифровка: «{shown}»", ""]

    lines.append(verdict.get("feedback", ""))

    strengths = verdict.get("strengths") or []
    if strengths:
        lines.append("\nЧто хорошо:")
        lines.extend(f"+ {item}" for item in strengths)

    issues = verdict.get("issues") or []
    if issues:
        lines.append("\nЧто стоит улучшить:")
        lines.extend(f"– {item}" for item in issues)

    xp_line = f"\n+{result['xp']} XP"
    if result["streak"] > 1:
        xp_line += f" · 🔥 стрик {result['streak']} дн."
    lines.append(xp_line)

    if result["leveled_up"]:
        level = result["new_level"]
        lines.append(f"\n🎉 Левелап! Новый уровень {level.get('order', '')}: «{level.get('title', '')}»")
    elif result["tree_complete"]:
        lines.append("\n🏆 Это был последний уровень дерева — поздравляю с прохождением скилла!")
    elif passed and result["successful_attempts"]:
        lines.append(
            f"\n✅ Задание засчитано ({result['successful_attempts']}/{gamification.ATTEMPTS_TO_LEVEL_UP} "
            "для перехода на следующий уровень)"
        )
    elif passed:
        lines.append("\n✅ Задание засчитано.")
    else:
        lines.append("\n❌ Не засчитано — пришли исправленный вариант, когда будешь готов.")

    if result["rank_after"] != result["rank_before"]:
        lines.append(f"⭐ Новый ранг: {result['rank_after']}")
    for badge in result["new_badges"]:
        lines.append(f"{gamification.badge_title(badge)} — новый бейдж!")

    return "\n".join(lines)
