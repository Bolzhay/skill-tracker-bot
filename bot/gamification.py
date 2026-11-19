"""Pure gamification rules: streak, XP, ranks, badges. No I/O, easy to test."""

from datetime import date, datetime, timedelta, timezone

ATTEMPTS_TO_LEVEL_UP = 3

BASE_XP = 10
FAIL_XP = 2
MAX_STREAK_BONUS = 10

RANKS = [
    (0, "Новичок"),
    (50, "Ученик"),
    (150, "Упорный"),
    (400, "Практик"),
    (800, "Мастер"),
    (1500, "Легенда"),
]

BADGES = {
    "first_task": "🏅 Первое засчитанное задание",
    "ten_tasks": "🏅 10 заданий засчитано",
    "first_levelup": "🏅 Первый левелап",
    "wildcard": "🃏 Wildcard покорён",
    "streak_3": "🔥 Серия 3 дня",
    "streak_7": "🔥 Серия 7 дней",
    "streak_30": "🔥 Серия 30 дней",
}


def local_today(offset_hours: int) -> date:
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).date()


def next_streak(count: int, last_date: str | None, today: date) -> int:
    """Streak after an attempt made `today` (ISO date strings in storage)."""
    if last_date == today.isoformat():
        return max(count, 1)
    if last_date == (today - timedelta(days=1)).isoformat():
        return count + 1
    return 1


def current_streak(count: int, last_date: str | None, today: date) -> int:
    """Streak to display: it's broken once a full day has been skipped."""
    if last_date in (today.isoformat(), (today - timedelta(days=1)).isoformat()):
        return count
    return 0


def xp_for_attempt(success: bool, streak: int) -> int:
    if not success:
        return FAIL_XP
    return BASE_XP + min(max(streak - 1, 0), MAX_STREAK_BONUS)


def rank_for(xp: int) -> str:
    name = RANKS[0][1]
    for threshold, rank in RANKS:
        if xp >= threshold:
            name = rank
    return name


def next_rank(xp: int) -> tuple[str, int] | None:
    """(name, xp still needed) of the next rank, or None at the top."""
    for threshold, rank in RANKS:
        if xp < threshold:
            return rank, threshold - xp
    return None


def badge_title(badge_id: str) -> str:
    return BADGES.get(badge_id, badge_id)


def new_badges(tasks_completed: int, streak: int, leveled_up: bool, wildcard_done: bool, owned: list[str]) -> list[str]:
    earned = []
    if tasks_completed >= 1:
        earned.append("first_task")
    if tasks_completed >= 10:
        earned.append("ten_tasks")
    if leveled_up:
        earned.append("first_levelup")
    if wildcard_done:
        earned.append("wildcard")
    for days in (3, 7, 30):
        if streak >= days:
            earned.append(f"streak_{days}")
    return [badge for badge in earned if badge not in owned]
