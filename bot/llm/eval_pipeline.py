import asyncio
import base64
import json

from . import openrouter
from .models import GEMMA, MODEL_A, MODEL_C

# Only gemma (both copies) and MODEL_C are confirmed multimodal (accept image
# input, all verified with a real test image); nemotron is text-only. The two
# independent vision readings are each led by a different model. The skeptic
# judge stage is text-only reasoning over the vision models' descriptions, so
# the full roster works there.
VISION_CHAIN_A = [*GEMMA, MODEL_C]
VISION_CHAIN_B = [MODEL_C, *GEMMA]
SKEPTIC_MODEL_CHAIN = [MODEL_A, *GEMMA, MODEL_C]

ANALYSIS_JSON_SHAPE = (
    '{"observations": str, "task_assessment": str}'
)
VERDICT_JSON_SHAPE = (
    '{"feedback": str, "task_completed": bool, "issues": [str, ...], '
    '"strengths": [str, ...]}'
)

# Tasks are deliberately scoped to ONE focused thing (see task_pipeline's
# CALIBRATION_NOTE — combining everything a level requires into one task was
# exactly what made early tasks overwhelming). That means no single task is
# meant to demonstrate a level's full criteria list, and grading a task
# against the whole level (as an earlier version of this prompt did) made
# every focused task look like a fail. Judge the TASK; the level's criteria
# are background context for calibrating relevance and strictness, not a
# checklist this one submission must fully clear.
SCOPE_NOTE = (
    "Важно: задание намеренно охватывает только ЧАСТЬ критериев уровня — "
    "остальное отрабатывается другими заданиями. Не требуй, чтобы один пруф "
    "демонстрировал вообще все критерии уровня сразу. Оценивай, выполнено "
    "ли именно ЭТО задание — критерии уровня даны только для контекста, "
    "чтобы понимать общий уровень строгости и что вообще отрабатывается."
)

VISION_SYSTEM_PROMPT = (
    "Ты анализируешь фото-пруф выполнения задания по развитию навыка. Опиши, "
    "что видишь на изображении, максимально конкретно и честно, без "
    "приукрашивания. Оцени, выполнено ли именно то, что просило задание, с "
    "конкретным обоснованием — что именно на фото подтверждает или "
    "опровергает это. Не будь вежливым из вежливости: если что-то не так, "
    f"прямо скажи об этом. {SCOPE_NOTE} "
    f"Верни только JSON в формате {ANALYSIS_JSON_SHAPE}, без пояснений вне JSON."
)

TEXT_ASSESS_SYSTEM_PROMPT = (
    "Ты анализируешь текстовый пруф выполнения задания по развитию навыка. "
    "Оцени, выполнено ли именно то, что просило задание, судя по присланному "
    f"тексту — конкретно и честно, без приукрашивания. {SCOPE_NOTE} "
    f"Верни только JSON в формате {ANALYSIS_JSON_SHAPE}, без пояснений вне JSON."
)

SKEPTIC_SYSTEM_PROMPT = (
    "Ты — строгий скептик в ансамбле оценщиков прогресса по навыку. Твоя "
    "роль — искать любые недочёты, не хвалить без железных оснований, быть "
    "придирчивым: ИИ-оценщики склонны переоценивать прогресс из вежливости, "
    "а это убивает ощущение реального роста. Тебе даны задание и независимые "
    "наблюдения о присланном пруфе. Вынеси финальный вердикт: выполнено ли "
    "именно ЭТО задание настолько честно и качественно, чтобы засчитать "
    f"попытку. {SCOPE_NOTE} При сомнении — не засчитывай, а объясни, чего не "
    "хватает. Дай содержательный фидбек пользователю: что хорошо, что нет, "
    "что стоит улучшить в следующий раз. "
    f"Верни только JSON в формате {VERDICT_JSON_SHAPE}, без пояснений вне JSON."
)

COMPARE_SYSTEM_PROMPT = (
    "Тебе даны две работы ученика по одному навыку: первая сделана давно "
    "('тогда'), вторая недавно ('сейчас'). Честно и конкретно сравни их: что "
    "реально улучшилось, какие ошибки из первой работы ушли, что осталось "
    "слабым или появилось нового. Не приукрашивай, но и не занижай — подмечай "
    "настоящий прогресс. В конце одним предложением скажи, над чем стоит "
    "поработать дальше. Пиши простым языком, без профессионального жаргона. "
    'Верни только JSON: {"comment": str}, без пояснений вне JSON.'
)

# Transcripts come from speech recognition: don't punish recognition errors,
# and say what can't be judged from words alone.
_PROOF_LABELS = {
    "text": "Присланный текст:",
    "voice": (
        "Присланное голосовое сообщение — автоматическая расшифровка (возможны "
        "ошибки распознавания, не снижай оценку за них; интонацию и паузы по "
        "тексту оценить нельзя):"
    ),
    "audio": (
        "Присланная аудиозапись — автоматическая расшифровка (возможны ошибки "
        "распознавания, не снижай оценку за них; интонацию и паузы по тексту "
        "оценить нельзя):"
    ),
    "video": (
        "Присланное видео — автоматическая расшифровка звука (возможны ошибки "
        "распознавания, не снижай оценку за них; изображение не анализировалось):"
    ),
}


def _has_analysis(result: dict) -> bool:
    return bool(result.get("observations"))


def _has_verdict(result: dict) -> bool:
    return "task_completed" in result and bool(result.get("feedback"))


def _has_comment(result: dict) -> bool:
    return bool(result.get("comment"))


def sniff_mime(data: bytes) -> str:
    """Telegram photos are JPEG but "send as file" keeps the original format,
    and pixel art is usually a PNG — the vision API needs the right type."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return "image/jpeg"


def _image_part(data: bytes) -> dict:
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{sniff_mime(data)};base64,{b64}"}}


def _task_context(skill_name: str, level: dict, task: dict) -> str:
    criteria = level.get("criteria", {})
    checks = criteria.get("checks") or []
    checks_text = "\n".join(f"- {check}" for check in checks)
    return (
        f"Навык: {skill_name}\n"
        f"Уровень: {level.get('title', '')}\n"
        f"Критерии уровня (общий контекст, не чек-лист для одного задания):\n{checks_text}\n\n"
        f"Задание, которое нужно оценить: {task.get('description', '')}\n"
        f"Подсказка, которая была дана к заданию: {task.get('hint', '')}"
    )


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


async def _analyze_photo(image: bytes, context: str, models: list[str]) -> dict:
    return await openrouter.call_model_json(
        models,
        [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": context}, _image_part(image)]},
        ],
        validate=_has_analysis,
    )


async def _analyze_text(proof_text: str, context: str, source: str) -> dict:
    label = _PROOF_LABELS.get(source, _PROOF_LABELS["text"])
    return await openrouter.call_model_json(
        SKEPTIC_MODEL_CHAIN,
        [
            {"role": "system", "content": TEXT_ASSESS_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n{label}\n{proof_text}"},
        ],
        validate=_has_analysis,
    )


async def _skeptic_verdict(context: str, analyses: list[dict]) -> dict:
    analyses_text = "\n\n".join(
        f"Независимое наблюдение {i + 1}:\n{_dump(analysis)}" for i, analysis in enumerate(analyses)
    )
    return await openrouter.call_model_json(
        SKEPTIC_MODEL_CHAIN,
        [
            {"role": "system", "content": SKEPTIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n{analyses_text}"},
        ],
        validate=_has_verdict,
    )


async def evaluate_photo(skill_name: str, level: dict, task: dict, image: bytes) -> dict:
    context = _task_context(skill_name, level, task)
    # Two independent vision readings, each led by a different model; they
    # don't depend on each other, so run them side by side.
    analysis_a, analysis_b = await asyncio.gather(
        _analyze_photo(image, context, VISION_CHAIN_A),
        _analyze_photo(image, context, VISION_CHAIN_B),
    )
    return await _skeptic_verdict(context, [analysis_a, analysis_b])


async def evaluate_text(skill_name: str, level: dict, task: dict, proof_text: str, source: str = "text") -> dict:
    """`source` is "text", or "voice"/"audio"/"video" when proof_text is a
    speech transcript."""
    context = _task_context(skill_name, level, task)
    analysis = await _analyze_text(proof_text, context, source)
    return await _skeptic_verdict(context, [analysis])


async def compare_progress(skill_name: str, before_task: dict, after_task: dict, before: bytes, after: bytes) -> str:
    text = (
        f"Навык: {skill_name}\n"
        f"Первая работа ('тогда') — задание: {before_task.get('description', '')}\n"
        f"Вторая работа ('сейчас') — задание: {after_task.get('description', '')}\n"
        "Ниже сначала работа 'тогда', потом работа 'сейчас'."
    )
    result = await openrouter.call_model_json(
        VISION_CHAIN_B,
        [
            {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": text}, _image_part(before), _image_part(after)]},
        ],
        validate=_has_comment,
    )
    return result["comment"]
