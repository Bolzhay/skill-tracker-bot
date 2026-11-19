from datetime import date

import pytest

from bot import gamification as g

TODAY = date(2026, 10, 5)


def test_first_attempt_starts_the_streak_at_one():
    assert g.next_streak(0, None, TODAY) == 1


def test_second_attempt_the_same_day_keeps_the_streak():
    assert g.next_streak(3, "2026-10-05", TODAY) == 3


def test_attempt_the_next_day_extends_the_streak():
    assert g.next_streak(3, "2026-10-04", TODAY) == 4


def test_a_skipped_day_resets_the_streak():
    assert g.next_streak(9, "2026-10-02", TODAY) == 1


def test_same_day_with_a_zero_counter_still_counts_as_one():
    assert g.next_streak(0, "2026-10-05", TODAY) == 1


@pytest.mark.parametrize(
    "last_date, expected",
    [("2026-10-05", 4), ("2026-10-04", 4), ("2026-10-03", 0), (None, 0)],
)
def test_displayed_streak_breaks_after_a_skipped_day(last_date, expected):
    assert g.current_streak(4, last_date, TODAY) == expected


def test_xp_for_a_pass_grows_with_the_streak_and_is_capped():
    assert g.xp_for_attempt(True, 1) == g.BASE_XP
    assert g.xp_for_attempt(True, 4) == g.BASE_XP + 3
    assert g.xp_for_attempt(True, 500) == g.BASE_XP + g.MAX_STREAK_BONUS


def test_a_failed_attempt_gets_only_participation_xp():
    assert g.xp_for_attempt(False, 50) == g.FAIL_XP


@pytest.mark.parametrize(
    "xp, rank",
    [(0, "Новичок"), (49, "Новичок"), (50, "Ученик"), (400, "Практик"), (99999, "Легенда")],
)
def test_rank_thresholds(xp, rank):
    assert g.rank_for(xp) == rank


def test_next_rank_tells_how_much_is_left():
    assert g.next_rank(0) == ("Ученик", 50)
    assert g.next_rank(60) == ("Упорный", 90)
    assert g.next_rank(1500) is None


def test_badges_are_awarded_once():
    assert g.new_badges(1, 3, False, False, owned=[]) == ["first_task", "streak_3"]
    assert g.new_badges(1, 3, False, False, owned=["first_task", "streak_3"]) == []


def test_levelup_wildcard_and_volume_badges():
    assert g.new_badges(10, 1, True, True, owned=[]) == ["first_task", "ten_tasks", "first_levelup", "wildcard"]


def test_streak_badges_stack_up():
    assert g.new_badges(0, 30, False, False, owned=[]) == ["streak_3", "streak_7", "streak_30"]


def test_every_badge_has_a_title():
    for badge_id in g.BADGES:
        assert g.badge_title(badge_id)
    assert g.badge_title("unknown") == "unknown"


def test_local_today_is_a_date():
    assert isinstance(g.local_today(5), date)
