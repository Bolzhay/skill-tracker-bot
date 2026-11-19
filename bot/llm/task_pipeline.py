import json

from .. import similarity
from . import openrouter
from .models import GEMMA, MODEL_A, MODEL_C

WILDCARD_INTERVAL = 6
SIMILARITY_THRESHOLD = 0.5
MAX_REGENERATION_ATTEMPTS = 3
MAX_SEARCH_QUERIES = 3

TASK_JSON_SHAPE = '{"description": str, "hint": str, "media_type": "photo"|"video"|"audio"|"text"}'

CALIBRATION_NOTE = (
    "Важно про сложность и язык: ориентируйся на самооценку и опыт ученика "
    "из контекста — язык и сложность должны соответствовать его реальному "
    "стартовому уровню, а не профессиональному стандарту. Не используй "
    "непояснённый жаргон и не полагайся на знание конкретного софта или "
    "инструментов, если ученик не упоминал, что им пользуется; если термин "
    "необходим — поясни его одним коротким предложением. Задание должно "
    "фокусироваться на ОДНОЙ конкретной вещи для отработки, а не совмещать "
    "сразу много разных техник и правил одновременно."
)

PROPOSE_SYSTEM_PROMPT = (
    "Ты придумываешь конкретное практическое задание для развития навыка, "
    "подходящее под текущий уровень ученика. Задание должно быть выполнимо "
    "за один присест, конкретным (не расплывчатым вроде 'потренируйся ещё'). "
    "Добавь краткую подсказку по технике/подходу — 1-2 предложения, "
    "достаточно, чтобы не пришлось искать с нуля, но не полноценный урок. "
    "Выбери media_type — какой тип пруфа подходит именно этому заданию. "
    f"{CALIBRATION_NOTE} "
    f"Верни только JSON в формате {TASK_JSON_SHAPE}, без пояснений вне JSON."
)

WILDCARD_SYSTEM_PROMPT = (
    "Ты придумываешь НЕОЖИДАННОЕ 'wildcard'-задание для развития навыка — "
    "оно должно нарочно ломать привычный паттерн последних заданий: "
    "неожиданное сочетание техник, непривычное ограничение или контекст. "
    "При этом всё ещё соответствовать текущему уровню ученика и быть "
    "выполнимым за один присест. Добавь краткую подсказку по подходу — "
    "1-2 предложения. Выбери подходящий media_type. "
    f"{CALIBRATION_NOTE} "
    f"Верни только JSON в формате {TASK_JSON_SHAPE}, без пояснений вне JSON."
)

REGENERATE_SYSTEM_PROMPT = (
    "Ты придумываешь задание для развития навыка, но предыдущая попытка "
    "оказалась слишком похожа на недавние задания. Придумай другое, явно "
    "меняя хотя бы одну ось вариативности: материал, контекст, ограничение "
    "по времени или неожиданное сочетание техник. Добавь краткую подсказку "
    "по подходу — 1-2 предложения. Выбери подходящий media_type. "
    f"{CALIBRATION_NOTE} "
    f"Верни только JSON в формате {TASK_JSON_SHAPE}, без пояснений вне JSON."
)

NOVELTY_SYSTEM_PROMPT = (
    "Ты проверяешь, не похоже ли новое задание на недавние по сути (а не "
    "только по формулировке) — та же механика, тот же навык отрабатывается "
    "тем же способом, просто другими словами. Есть ли в задании элемент "
    "неожиданности по сравнению с недавней историей? "
    'Верни только JSON: {"too_similar": bool, "reason": str}, без пояснений вне JSON.'
)

HINT_SYSTEM_PROMPT = (
    "Ученик застрял на задании и просит более подробную подсказку, чем уже "
    "дана. Объясни конкретнее: какие инструменты или техники использовать, "
    "из каких шагов состоит выполнение, на что обратить внимание. Это не "
    "должно быть решением задания за него — цель дать достаточно конкретики, "
    "чтобы он сам справился, а не просто пересказать задание другими "
    "словами. Кроме подсказки дай в search_queries от 1 до 3 коротких "
    "поисковых запросов для YouTube — так, как человек ввёл бы их в строку "
    "поиска (например, 'пиксель-арт ровные линии для начинающих'): по ним бот "
    "сам найдёт настоящие обучающие видео и пришлёт ссылки. В самой подсказке "
    "не пиши 'поищи' и не вставляй ссылки. НЕ придумывай ссылки и названия "
    "роликов: у тебя нет доступа к живому поиску. Если видео тут не поможет, "
    "верни пустой список. "
    f"{CALIBRATION_NOTE} "
    'Верни только JSON: {"hint": str, "search_queries": [str, ...]}, без пояснений вне JSON.'
)


def _level_context(skill: dict, level: dict) -> str:
    criteria = level.get("criteria", {})
    checks = criteria.get("checks") or []
    checks_text = "\n".join(f"- {check}" for check in checks)

    student_context = ""
    onboarding = skill.get("onboarding") or {}
    if onboarding:
        student_context = (
            "\nСамооценка ученика при старте (1-10) и пояснение: "
            f"{onboarding.get('self_assessment', '')}\n"
            f"Что уже пробовал: {onboarding.get('experience', '')}\n"
            f"Цель: {onboarding.get('goal', '')}"
        )

    return (
        f"Навык: {skill.get('name', '')}\n"
        f"Текущий уровень: {level.get('title', '')}\n"
        f"Цель уровня: {criteria.get('summary', '')}\n"
        f"Критерии:\n{checks_text}"
        f"{student_context}"
    )


def _recent_descriptions(recent_tasks: list[dict]) -> list[str]:
    return [task["description"] for task in recent_tasks if task.get("description")]


def _has_description(result: dict) -> bool:
    return bool(result.get("description"))


def _has_verdict(result: dict) -> bool:
    return "too_similar" in result


def _has_hint(result: dict) -> bool:
    return bool(result.get("hint"))


def _search_queries(raw) -> list[str]:
    """The model's YouTube queries, cleaned: strings only, no blanks, no
    repeats, at most MAX_SEARCH_QUERIES."""
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    for item in raw:
        query = item.strip() if isinstance(item, str) else ""
        if query and query not in queries:
            queries.append(query)
    return queries[:MAX_SEARCH_QUERIES]


async def _propose(system_prompt: str, user_content: str) -> dict:
    return await openrouter.call_model_json(
        [MODEL_A, *GEMMA, MODEL_C],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        validate=_has_description,
    )


async def _check_novelty(candidate: dict, recent_descriptions: list[str]) -> dict:
    history_text = "\n".join(f"- {desc}" for desc in recent_descriptions)
    return await openrouter.call_model_json(
        [*GEMMA, MODEL_C, MODEL_A],
        [
            {"role": "system", "content": NOVELTY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Недавние задания:\n{history_text}\n\n"
                    f"Новое задание: {candidate.get('description', '')}"
                ),
            },
        ],
        validate=_has_verdict,
    )


async def generate_task(skill: dict, level: dict, recent_tasks: list[dict], task_count: int) -> dict:
    """Propose a task, guarding against repetition via a cheap local filter
    plus an LLM novelty check, and occasionally forcing a wildcard.

    Gives up after MAX_REGENERATION_ATTEMPTS and returns the last candidate
    anyway — an imperfect task beats no task at all.
    """
    context = _level_context(skill, level)
    recent_descriptions = _recent_descriptions(recent_tasks)
    is_wildcard = (task_count + 1) % WILDCARD_INTERVAL == 0

    system_prompt = WILDCARD_SYSTEM_PROMPT if is_wildcard else PROPOSE_SYSTEM_PROMPT
    user_content = context
    if recent_descriptions:
        history_text = "\n".join(f"- {desc}" for desc in recent_descriptions)
        user_content = f"{context}\n\nНедавние задания (не повторять):\n{history_text}"

    candidate = await _propose(system_prompt, user_content)

    for _ in range(MAX_REGENERATION_ATTEMPTS):
        if not recent_descriptions:
            break

        if similarity.is_too_similar(candidate.get("description", ""), recent_descriptions, SIMILARITY_THRESHOLD):
            candidate = await _propose(
                REGENERATE_SYSTEM_PROMPT,
                f"{context}\n\nНедавние задания (не повторять):\n"
                + "\n".join(f"- {desc}" for desc in recent_descriptions),
            )
            continue

        novelty = await _check_novelty(candidate, recent_descriptions)
        if not novelty.get("too_similar"):
            break

        candidate = await _propose(
            REGENERATE_SYSTEM_PROMPT,
            f"{context}\n\nНедавние задания (не повторять):\n"
            + "\n".join(f"- {desc}" for desc in recent_descriptions),
        )

    candidate["is_wildcard"] = is_wildcard
    return candidate


async def generate_detailed_hint(skill: dict, level: dict, task: dict) -> dict:
    """{"hint": str, "search_queries": [str, ...]}. The queries are for YouTube
    (see youtube.find_videos): the model has no live search, so it only words
    them and never invents links."""
    context = _level_context(skill, level)
    user_content = (
        f"{context}\n\n"
        f"Задание: {task.get('description', '')}\n"
        f"Уже данная краткая подсказка: {task.get('hint', '')}"
    )

    result = await openrouter.call_model_json(
        [MODEL_A, *GEMMA, MODEL_C],
        [
            {"role": "system", "content": HINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        validate=_has_hint,
    )
    return {"hint": result["hint"], "search_queries": _search_queries(result.get("search_queries"))}
