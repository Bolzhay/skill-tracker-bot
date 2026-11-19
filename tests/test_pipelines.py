import pytest

from bot.llm import eval_pipeline, openrouter, task_pipeline, tree_pipeline
from bot.llm.models import MODEL_A, MODEL_B_GOOGLE, MODEL_C
from tests.helpers import PNG


class Script:
    """Stands in for openrouter.call_model_json: returns canned answers in
    order and refuses any whose shape the real validator would reject, so a
    test can't pass on a response the pipeline would actually discard."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, models, messages, validate=None):
        self.calls.append({"models": models, "messages": messages})
        response = self.responses.pop(0)
        if validate is not None:
            assert validate(response), f"canned response fails the pipeline's own validator: {response}"
        return response

    def user_text(self, index):
        content = self.calls[index]["messages"][-1]["content"]
        if isinstance(content, list):
            return " ".join(part["text"] for part in content if part["type"] == "text")
        return content

    def system(self, index):
        return self.calls[index]["messages"][0]["content"]


@pytest.fixture
def script(monkeypatch):
    def install(*responses):
        fake = Script(*responses)
        monkeypatch.setattr(openrouter, "call_model_json", fake)
        return fake

    return install


LEVEL = {"title": "Чистая линия", "criteria": {"summary": "Ровные линии", "checks": ["1 пиксель"]}}
SKILL = {"name": "Пиксель-арт", "onboarding": {"self_assessment": "3 из 10", "experience": "рисовал", "goal": "персонаж"}}


# --- level tree -----------------------------------------------------------


async def test_tree_runs_draft_critique_revision_and_judge(script):
    draft = {"levels": [{"order": 1, "title": "Черновик"}]}
    critique = {"issues": [{"level_order": 1, "problem": "слишком сложно", "suggestion": "проще"}]}
    revised = {"levels": [{"order": 1, "title": "Исправленный"}]}
    final = {"levels": [{"order": 1, "title": "Итог", "criteria_summary": "s", "criteria_checks": []}]}
    fake = script(draft, critique, revised, final)

    levels = await tree_pipeline.build_skill_tree("Пиксель-арт", "3 из 10", "рисовал", "персонаж")

    assert levels == final["levels"]
    assert len(fake.calls) == 4
    assert "Пиксель-арт" in fake.user_text(0) and "3 из 10" in fake.user_text(0)
    assert "Черновик" in fake.user_text(1)  # the critic sees the draft
    assert "слишком сложно" in fake.user_text(2) and "Черновик" in fake.user_text(2)  # reviser sees both
    assert "Исправленный" in fake.user_text(3)  # the judge sees the revision
    # each stage leads with a different lab's model; gemma's lead copy is Google's own
    assert [call["models"][0] for call in fake.calls] == [MODEL_A, MODEL_B_GOOGLE, MODEL_A, MODEL_C]
    # ...and every stage can still run once OpenRouter's daily quota is spent
    assert all(MODEL_B_GOOGLE in call["models"] for call in fake.calls)


async def test_every_tree_stage_is_told_to_match_the_learners_level(script):
    fake = script(
        {"levels": [{"order": 1}]}, {"issues": []}, {"levels": [{"order": 1}]}, {"levels": [{"order": 1}]}
    )
    await tree_pipeline.build_skill_tree("x", "1", "2", "3")

    for index in (0, 2, 3):
        assert "реальную самооценку" in fake.system(index)
    assert "жаргон" in fake.system(1)  # the critic hunts for it


# --- task generation ------------------------------------------------------


def _task(description, hint="подсказка", media="photo"):
    return {"description": description, "hint": hint, "media_type": media}


async def test_first_task_needs_a_single_call(script):
    fake = script(_task("Нарисуй квадрат"))
    result = await task_pipeline.generate_task(SKILL, LEVEL, [], task_count=0)

    assert result["description"] == "Нарисуй квадрат"
    assert result["is_wildcard"] is False
    assert len(fake.calls) == 1
    assert "Самооценка ученика" in fake.user_text(0)  # onboarding context reaches the model
    assert "ОДНОЙ конкретной вещи" in fake.system(0)


@pytest.mark.parametrize("count, wildcard", [(0, False), (4, False), (5, True), (6, False), (11, True)])
async def test_every_sixth_task_is_a_wildcard(script, count, wildcard):
    fake = script(_task("Что-то необычное"))
    result = await task_pipeline.generate_task(SKILL, LEVEL, [], task_count=count)

    assert result["is_wildcard"] is wildcard
    assert ("НЕОЖИДАННОЕ" in fake.system(0)) is wildcard


async def test_a_lookalike_is_regenerated_without_asking_a_model_first(script):
    recent = [{"description": "Нарисуй квадрат инструментом Линия"}]
    fake = script(
        _task("Нарисуй квадрат инструментом Линия."),  # caught by the free local filter
        _task("Собери палитру из четырёх цветов и закрась шарик"),
        {"too_similar": False, "reason": ""},
    )
    result = await task_pipeline.generate_task(SKILL, LEVEL, recent, task_count=1)

    assert result["description"].startswith("Собери палитру")
    assert len(fake.calls) == 3
    assert "слишком похожа" in fake.system(1)  # the regeneration prompt
    assert "слишком похожа" not in fake.system(0)
    assert "Нарисуй квадрат инструментом Линия" in fake.user_text(1)  # told what to avoid


async def test_the_novelty_check_catches_what_the_local_filter_misses(script):
    recent = [{"description": "Нарисуй квадрат инструментом Линия"}]
    fake = script(
        _task("Изобрази прямоугольник прямыми штрихами без кисти"),  # different words, same idea
        {"too_similar": True, "reason": "та же механика"},
        _task("Собери палитру из четырёх цветов и закрась шарик"),
        {"too_similar": False, "reason": ""},
    )
    result = await task_pipeline.generate_task(SKILL, LEVEL, recent, task_count=1)

    assert result["description"].startswith("Собери палитру")
    assert len(fake.calls) == 4
    assert fake.calls[1]["models"][0] == MODEL_B_GOOGLE  # the novelty check is gemma's job
    assert all(MODEL_B_GOOGLE in call["models"] for call in fake.calls)


async def test_it_gives_up_after_the_regeneration_limit_and_still_returns_a_task(script):
    recent = [{"description": "Нарисуй квадрат инструментом Линия"}]
    same = "Нарисуй квадрат инструментом Линия"
    fake = script(*[_task(same) for _ in range(1 + task_pipeline.MAX_REGENERATION_ATTEMPTS)])

    result = await task_pipeline.generate_task(SKILL, LEVEL, recent, task_count=1)

    assert result["description"] == same
    assert len(fake.calls) == 1 + task_pipeline.MAX_REGENERATION_ATTEMPTS


async def test_detailed_hint_sees_the_task_and_asks_for_search_queries_not_links(script):
    queries = ["пиксель-арт для начинающих", "ровные линии"]
    fake = script({"hint": "Начни с сетки.", "search_queries": queries})
    task = {"description": "Нарисуй квадрат", "hint": "Используй Линию"}

    hint = await task_pipeline.generate_detailed_hint(SKILL, LEVEL, task)

    assert hint == {"hint": "Начни с сетки.", "search_queries": queries}
    assert "Нарисуй квадрат" in fake.user_text(0) and "Используй Линию" in fake.user_text(0)
    assert "НЕ придумывай ссылки" in fake.system(0)
    assert "search_queries" in fake.system(0)
    assert MODEL_B_GOOGLE in fake.calls[0]["models"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["a", " ", "a", "b ", 5, None, "c", "d"], ["a", "b", "c"]),  # blanks, repeats, non-strings; max 3
        ([], []),
        (None, []),
        ("пиксель-арт", []),  # a bare string is not a list of queries
    ],
)
async def test_the_models_search_queries_are_cleaned(script, raw, expected):
    script({"hint": "Начни с сетки", "search_queries": raw})
    hint = await task_pipeline.generate_detailed_hint(SKILL, LEVEL, {"description": "x"})
    assert hint["search_queries"] == expected


async def test_a_hint_without_search_queries_is_still_a_hint(script):
    script({"hint": "Начни с сетки"})
    hint = await task_pipeline.generate_detailed_hint(SKILL, LEVEL, {"description": "x"})
    assert hint == {"hint": "Начни с сетки", "search_queries": []}


# --- evaluation -----------------------------------------------------------

ANALYSIS = {"observations": "Видно квадрат", "task_assessment": "выполнено"}
VERDICT = {"feedback": "Хорошо", "task_completed": True, "issues": [], "strengths": ["ровно"]}
TASK = {"description": "Нарисуй квадрат", "hint": "Линия"}


async def test_photo_goes_to_two_vision_models_then_a_skeptic(script):
    fake = script(ANALYSIS, ANALYSIS, VERDICT)

    result = await eval_pipeline.evaluate_photo("Пиксель-арт", LEVEL, TASK, PNG)

    assert result == VERDICT
    assert len(fake.calls) == 3
    first, second, judge = fake.calls
    assert first["models"] == eval_pipeline.VISION_CHAIN_A
    assert second["models"] == eval_pipeline.VISION_CHAIN_B
    assert first["models"][0] != second["models"][0]  # a different model leads
    assert judge["models"] == eval_pipeline.SKEPTIC_MODEL_CHAIN
    for call in (first, second):
        assert MODEL_A not in call["models"]  # nemotron is text-only and can't read a photo
    assert all(MODEL_B_GOOGLE in call["models"] for call in fake.calls)
    assert "Видно квадрат" in fake.user_text(2)
    assert "СКЕПТИК".lower() in fake.system(2).lower()


async def test_photo_is_sent_with_its_real_image_type(script):
    fake = script(ANALYSIS, ANALYSIS, VERDICT)
    await eval_pipeline.evaluate_photo("x", LEVEL, TASK, PNG)

    image = fake.calls[0]["messages"][-1]["content"][1]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")


async def test_the_evaluator_judges_the_task_not_the_whole_level(script):
    fake = script(ANALYSIS, ANALYSIS, VERDICT)
    await eval_pipeline.evaluate_photo("x", LEVEL, TASK, PNG)

    for index in range(3):
        assert "ЧАСТЬ критериев уровня" in fake.system(index)
    assert "Нарисуй квадрат" in fake.user_text(0)


async def test_text_proof_gets_a_single_analysis_and_a_verdict(script):
    fake = script(ANALYSIS, VERDICT)
    result = await eval_pipeline.evaluate_text("x", LEVEL, TASK, "Вот мой ответ")

    assert result == VERDICT
    assert len(fake.calls) == 2
    assert all(MODEL_B_GOOGLE in call["models"] for call in fake.calls)
    assert "Вот мой ответ" in fake.user_text(0)
    assert "автоматическая расшифровка" not in fake.user_text(0)


@pytest.mark.parametrize("source", ["voice", "audio", "video"])
async def test_transcripts_are_labelled_so_recognition_errors_arent_punished(script, source):
    fake = script(ANALYSIS, VERDICT)
    await eval_pipeline.evaluate_text("x", LEVEL, TASK, "привет мир", source=source)

    assert "автоматическая расшифровка" in fake.user_text(0)
    assert "привет мир" in fake.user_text(0)


async def test_progress_comparison_sends_both_images(script):
    fake = script({"comment": "Линии стали ровнее"})
    comment = await eval_pipeline.compare_progress(
        "Пиксель-арт", {"description": "старое"}, {"description": "новое"}, PNG, PNG + b"1"
    )

    assert comment == "Линии стали ровнее"
    assert MODEL_A not in fake.calls[0]["models"] and MODEL_B_GOOGLE in fake.calls[0]["models"]
    content = fake.calls[0]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "image_url"]
    assert "старое" in content[0]["text"] and "новое" in content[0]["text"]


@pytest.mark.parametrize(
    "data, mime",
    [
        (b"\x89PNG\r\n\x1a\n....", "image/png"),
        (b"\xff\xd8\xff\xe0....", "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"GIF89a....", "image/gif"),
        (b"unknown", "image/jpeg"),
    ],
)
def test_image_type_is_detected_from_the_bytes(data, mime):
    assert eval_pipeline.sniff_mime(data) == mime
