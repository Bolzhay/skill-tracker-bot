import pytest

from bot import similarity as s


def test_identical_text_is_fully_similar():
    assert s.max_similarity("Нарисуй квадрат", ["Нарисуй квадрат"]) == pytest.approx(1.0)


def test_inflected_forms_count_as_the_same_word():
    # Without stemming these share 1 of 2 words (0.5); with it, both.
    assert s.max_similarity("Нарисуй квадраты", ["Нарисуй квадрат"]) == pytest.approx(1.0)


def test_a_reworded_lookalike_is_still_caught():
    new = "Нарисуй три квадрата инструментом Линия"
    old = "Нарисуй квадрат инструментом Линия"
    assert s.is_too_similar(new, [old])


def test_different_tasks_are_not_similar():
    a = "Нарисуй силуэт профиля головы персонажа размером 32x32 пикселя"
    b = "Собери палитру из четырёх цветов и закрась шарик с тенью"
    assert not s.is_too_similar(a, [b])


def test_no_history_means_not_similar():
    assert s.max_similarity("что угодно", []) == 0.0
    assert not s.is_too_similar("что угодно", [])


def test_text_made_of_short_words_has_no_signal():
    assert s.max_similarity("в на и по", ["в на и по"]) == 0.0


def test_yo_and_ye_are_the_same_letter():
    assert s.max_similarity("чёрный контур", ["черный контур"]) == pytest.approx(1.0)


def test_the_closest_of_many_wins():
    others = ["Сыграй гамму на гитаре", "Нарисуй квадрат инструментом Линия", "Спой песню"]
    assert s.max_similarity("Нарисуй квадрат инструментом Линия", others) == pytest.approx(1.0)


def test_threshold_is_respected():
    a, b = "Нарисуй квадрат ручкой", "Нарисуй круг карандашом"
    similarity = s.max_similarity(a, [b])
    assert 0 < similarity < 1
    assert s.is_too_similar(a, [b], threshold=similarity)
    assert not s.is_too_similar(a, [b], threshold=similarity + 0.01)
