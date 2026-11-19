import json

from . import openrouter
from .models import GEMMA, MODEL_A, MODEL_C

# Draft and revision lead with nemotron, the critique with gemma, the final
# judgement with MODEL_C; each falls back to the others.
MODEL_DRAFT = MODEL_A
MODEL_JUDGE = MODEL_C

LEVELS_JSON_SHAPE = (
    '{"levels": [{"order": int, "title": str, "criteria_summary": str, '
    '"criteria_checks": [str, ...]}, ...]}'
)

CALIBRATION_NOTE = (
    "Критически важно про уровень сложности: ориентируйся на реальную "
    "самооценку и опыт пользователя из контекста, а не на профессиональный "
    "стандарт индустрии. Ранние уровни должны быть по-настоящему доступны "
    "новичку — избегай профессионального жаргона (термины уровня "
    "'изометрия', 'фаски', конкретные пропорции/шаги, названия инструментов) "
    "без крайней необходимости; если термин обязателен — объясни его простыми "
    "словами прямо в тексте. Не предполагай, что пользователь знает "
    "конкретный софт или его инструменты, если он сам их не упоминал."
)

DRAFT_SYSTEM_PROMPT = (
    "Ты помогаешь построить продуманное дерево уровней для развития навыка. "
    "Дерево должно быть адаптировано под стартовый уровень человека, с плавным "
    "нарастанием сложности без резких скачков. Обычно это 8-15 уровней, но "
    "выбери количество, подходящее именно этому случаю. Каждый уровень — "
    "конкретный, проверяемый прогресс, а не общие фразы вроде 'рисуй лучше'. "
    f"{CALIBRATION_NOTE} "
    f"Верни только JSON в формате {LEVELS_JSON_SHAPE}, без пояснений вне JSON."
)

CRITIQUE_SYSTEM_PROMPT = (
    "Ты — придирчивый критик планов обучения. Твоя задача — найти реальные "
    "проблемы в дереве уровней: пробелы в прогрессии, слишком резкие скачки "
    "сложности, шаблонность формулировок между уровнями, расплывчатые или "
    "непроверяемые критерии перехода. Отдельно ищи несоответствие реальному "
    "стартовому уровню пользователя: профессиональный жаргон без объяснений, "
    "критерии, написанные языком туториала для опытных, а не для новичка с "
    "такой самооценкой и опытом, как в контексте. Не хвали без оснований. "
    'Верни только JSON: {"issues": [{"level_order": int, "problem": str, '
    '"suggestion": str}, ...]}, без пояснений вне JSON.'
)

REVISE_SYSTEM_PROMPT = (
    "Ты дорабатываешь дерево уровней с учётом критики. Исправь названные "
    "проблемы по существу, сохрани структуру там, где критики не было. "
    f"{CALIBRATION_NOTE} "
    f"Верни только исправленный план в формате {LEVELS_JSON_SHAPE}, без "
    "пояснений вне JSON."
)

JUDGE_SYSTEM_PROMPT = (
    "Ты — финальный судья дерева уровней для развития навыка. Проверь план на "
    "цельность: порядок уровней логичен, названия не повторяются, критерии "
    "конкретны и проверяемы. Сведи всё в чистовую финальную версию. "
    f"{CALIBRATION_NOTE} "
    f"Верни только JSON в формате {LEVELS_JSON_SHAPE}, без пояснений вне JSON."
)


def _user_context(skill_name: str, self_assessment: str, experience: str, goal: str) -> str:
    return (
        f"Навык: {skill_name}\n"
        f"Самооценка пользователя (1-10) и пояснение: {self_assessment}\n"
        f"Что уже пробовал: {experience}\n"
        f"Цель: {goal}"
    )


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _has_levels(result: dict) -> bool:
    levels = result.get("levels")
    return isinstance(levels, list) and len(levels) > 0


def _has_issues(result: dict) -> bool:
    return isinstance(result.get("issues"), list)


async def build_skill_tree(
    skill_name: str,
    self_assessment: str,
    experience: str,
    goal: str,
) -> list[dict]:
    context = _user_context(skill_name, self_assessment, experience, goal)

    draft = await openrouter.call_model_json(
        [MODEL_DRAFT, *GEMMA, MODEL_JUDGE],
        [
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        validate=_has_levels,
    )

    critique = await openrouter.call_model_json(
        [*GEMMA, MODEL_JUDGE, MODEL_DRAFT],
        [
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{context}\n\nЧерновой план:\n{_dump(draft)}",
            },
        ],
        validate=_has_issues,
    )

    revised = await openrouter.call_model_json(
        [MODEL_DRAFT, MODEL_JUDGE, *GEMMA],
        [
            {"role": "system", "content": REVISE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Черновой план:\n{_dump(draft)}\n\n"
                    f"Критика:\n{_dump(critique)}"
                ),
            },
        ],
        validate=_has_levels,
    )

    final = await openrouter.call_model_json(
        [MODEL_JUDGE, MODEL_DRAFT, *GEMMA],
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{context}\n\nФинальный черновик:\n{_dump(revised)}",
            },
        ],
        validate=_has_levels,
    )

    levels = final["levels"]

    return levels
