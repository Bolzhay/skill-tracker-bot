from datetime import date

from bot import db, gamification
from tests.helpers import PNG, levels, seed_skill, seed_task, verdict

USER = 111
OTHER = 222
DAY = date(2026, 10, 5)


# --- skills ---------------------------------------------------------------


def test_skills_are_isolated_per_user(fake_db):
    seed_skill(USER, "Риторика")
    seed_skill(OTHER, "Химия")

    assert [s["name"] for s in db.list_skills(USER)] == ["Риторика"]
    assert [s["name"] for s in db.list_skills(OTHER)] == ["Химия"]
    assert db.find_skill_by_name(OTHER, "Риторика") is None


def test_find_skill_by_name_ignores_case_and_spaces(fake_db):
    seed_skill(USER, "Пиксель-арт")
    assert db.find_skill_by_name(USER, "  пиксель-АРТ ")["name"] == "Пиксель-арт"
    assert db.find_skill_by_name(USER, "нет такого") is None


def test_new_skill_starts_with_zeroed_counters(fake_db):
    skill = seed_skill(USER, level_count=0, onboarding=False)
    assert skill["xp_total"] == 0
    assert skill["streak_count"] == 0
    assert skill["tasks_completed"] == 0
    assert skill["badges"] == []
    assert skill["onboarding"] is None
    assert skill["current_level_id"] is None


def test_skills_list_in_creation_order(fake_db):
    for name in ("Первый", "Второй", "Третий"):
        seed_skill(USER, name, level_count=0)
    assert [s["name"] for s in db.list_skills(USER)] == ["Первый", "Второй", "Третий"]


def test_delete_skill_if_empty_only_removes_treeless_skills(fake_db):
    built = seed_skill(USER, "С деревом")
    empty = seed_skill(USER, "Без дерева", level_count=0)

    assert db.delete_skill_if_empty(USER, empty["id"]) is True
    assert db.get_skill(USER, empty["id"]) is None

    assert db.delete_skill_if_empty(USER, built["id"]) is False
    assert db.get_skill(USER, built["id"]) is not None

    assert db.delete_skill_if_empty(USER, "missing") is False


# --- level tree -----------------------------------------------------------


def test_level_tree_is_renumbered_and_first_level_activated(fake_db):
    skill_id = db.create_skill(USER, "Навык")
    messy = [
        {"order": 9, "title": "Третий", "criteria_summary": "c", "criteria_checks": []},
        {"order": 2, "title": "Первый", "criteria_summary": "a", "criteria_checks": ["x"]},
        {"order": 2, "title": "Второй", "criteria_summary": "b", "criteria_checks": []},
    ]
    db.save_level_tree(USER, skill_id, messy)
    skill = db.get_skill(USER, skill_id)

    first = db.get_level(USER, skill_id, skill["current_level_id"])
    assert (first["order"], first["status"], first["title"]) == (1, "active", "Первый")
    assert first["criteria"] == {"summary": "a", "checks": ["x"]}
    assert skill["levels_total"] == 3

    stored = sorted(
        (doc.to_dict() for doc in fake_db.collection("users").document(str(USER))
         .collection("skills").document(skill_id).collection("levels").stream()),
        key=lambda level: level["order"],
    )
    assert [level["order"] for level in stored] == [1, 2, 3]
    assert [level["status"] for level in stored] == ["active", "locked", "locked"]
    assert all(level["successful_attempts"] == 0 for level in stored)


# --- tasks and attempts ---------------------------------------------------


def test_recent_tasks_come_newest_first_and_are_counted(fake_db):
    skill = seed_skill(USER)
    for text in ("раз", "два", "три"):
        seed_task(USER, skill, description=text)

    recent = db.get_recent_tasks(USER, skill["id"], limit=2)
    assert [t["description"] for t in recent] == ["три", "два"]
    assert db.count_tasks(USER, skill["id"]) == 3


def test_new_task_is_open_until_marked_completed(fake_db):
    skill = seed_skill(USER)
    task = seed_task(USER, skill)
    assert task["completed"] is False

    db.mark_task_completed(USER, skill["id"], task["id"])
    assert db.get_task(USER, skill["id"], task["id"])["completed"] is True
    assert db.get_task(USER, skill["id"], "missing") is None


def test_attempt_keeps_proof_bytes_in_a_separate_document(fake_db):
    skill = seed_skill(USER)
    task = seed_task(USER, skill)
    proof = {"kind": "photo", "file_id": "f1", "mime": "image/png", "size": len(PNG), "bytes": PNG}

    attempt_id = db.create_attempt(USER, skill["id"], task["id"], "photo", verdict(), 10, proof)

    attempts = fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"]) \
        .collection("tasks").document(task["id"]).collection("attempts")
    stored = attempts.document(attempt_id).get().to_dict()
    assert stored["has_blob"] is True
    assert "bytes" not in stored["proof"]
    assert stored["proof"]["file_id"] == "f1"
    assert stored["xp_awarded"] == 10
    assert db.get_attempt_blob(USER, skill["id"], task["id"], attempt_id) == PNG


def test_attempt_without_bytes_has_no_blob(fake_db):
    skill = seed_skill(USER)
    task = seed_task(USER, skill)
    proof = {"kind": "text", "text": "мой ответ"}

    attempt_id = db.create_attempt(USER, skill["id"], task["id"], "text", verdict(), 10, proof)

    assert db.get_attempt_blob(USER, skill["id"], task["id"], attempt_id) is None
    assert db.create_attempt(USER, skill["id"], task["id"], "text", verdict(), 2)  # proof is optional


def test_successful_photo_attempts_pick_the_passing_photo_of_each_completed_task(fake_db):
    skill = seed_skill(USER)
    photo = {"kind": "photo", "file_id": "f", "mime": "image/png", "size": 1, "bytes": PNG}

    first = seed_task(USER, skill, completed=True, description="первое")
    db.create_attempt(USER, skill["id"], first["id"], "photo", verdict(False), 2, photo)
    passing_first = db.create_attempt(USER, skill["id"], first["id"], "photo", verdict(True), 10, photo)

    text_task = seed_task(USER, skill, completed=True, description="текстовое")
    db.create_attempt(USER, skill["id"], text_task["id"], "text", verdict(True), 10, {"kind": "text", "text": "x"})

    open_task = seed_task(USER, skill, completed=False, description="открытое")
    db.create_attempt(USER, skill["id"], open_task["id"], "photo", verdict(True), 10, photo)

    last = seed_task(USER, skill, completed=True, description="последнее")
    passing_last = db.create_attempt(USER, skill["id"], last["id"], "photo", verdict(True), 10, photo)

    found = db.successful_photo_attempts(USER, skill["id"])
    assert [(f["task"]["description"], f["attempt"]["id"]) for f in found] == [
        ("первое", passing_first),
        ("последнее", passing_last),
    ]


# --- XP, streak, level-up, badges -----------------------------------------


def _record(skill, passed=True, wildcard=False, day=DAY, level_id=None):
    return db.record_attempt_result(
        USER, skill["id"], level_id or skill["current_level_id"], passed, wildcard, day
    )


def test_xp_streak_and_pass_counter_after_a_pass(fake_db):
    skill = seed_skill(USER)
    result = _record(skill)

    assert (result["xp"], result["streak"], result["successful_attempts"]) == (10, 1, 1)
    stored = db.get_skill(USER, skill["id"])
    assert (stored["xp_total"], stored["streak_count"], stored["streak_last_date"]) == (10, 1, "2026-10-05")
    assert stored["tasks_completed"] == 1


def test_streak_grows_across_days_and_resets_after_a_gap(fake_db):
    skill = seed_skill(USER)
    _record(skill, day=date(2026, 10, 5))
    second_day = _record(skill, day=date(2026, 10, 6))
    assert second_day["streak"] == 2 and second_day["xp"] == 11

    after_gap = _record(skill, day=date(2026, 10, 9))
    assert after_gap["streak"] == 1


def test_a_failed_attempt_earns_participation_xp_but_no_level_progress(fake_db):
    skill = seed_skill(USER)
    result = _record(skill, passed=False)

    assert result["xp"] == gamification.FAIL_XP
    assert result["successful_attempts"] == 0
    stored = db.get_skill(USER, skill["id"])
    assert stored["tasks_completed"] == 0
    assert db.get_level(USER, skill["id"], skill["current_level_id"])["successful_attempts"] == 0
    assert stored["streak_count"] == 1  # a failed attempt still counts as practice


def test_three_passes_move_to_the_next_level(fake_db):
    skill = seed_skill(USER, level_count=3)
    level_one = skill["current_level_id"]

    first = _record(skill)
    second = _record(skill)
    assert (first["successful_attempts"], second["successful_attempts"]) == (1, 2)
    assert not first["leveled_up"] and not second["leveled_up"]

    third = _record(skill)
    assert third["leveled_up"] is True
    assert third["new_level"]["order"] == 2

    stored = db.get_skill(USER, skill["id"])
    assert stored["current_level_id"] == third["new_level"]["id"] != level_one
    assert db.get_level(USER, skill["id"], level_one)["status"] == "done"
    assert db.get_level(USER, skill["id"], stored["current_level_id"])["status"] == "active"
    assert "first_levelup" in third["new_badges"]


def test_passing_a_task_from_an_older_level_does_not_count_toward_the_current_one(fake_db):
    skill = seed_skill(USER)
    result = _record(skill, level_id="some-old-level")

    assert result["successful_attempts"] == 0
    assert db.get_level(USER, skill["id"], skill["current_level_id"])["successful_attempts"] == 0
    assert db.get_skill(USER, skill["id"])["tasks_completed"] == 1  # still a completed task


def test_finishing_the_last_level_completes_the_tree(fake_db):
    skill = seed_skill(USER, level_count=1)
    results = [_record(skill) for _ in range(3)]

    assert [r["tree_complete"] for r in results] == [False, False, True]
    assert not results[-1]["leveled_up"]
    assert db.get_skill(USER, skill["id"])["current_level_id"] == skill["current_level_id"]


def test_badges_are_awarded_once(fake_db):
    skill = seed_skill(USER)
    first = _record(skill)
    second = _record(skill)

    assert first["new_badges"] == ["first_task"]
    assert second["new_badges"] == []
    assert db.get_skill(USER, skill["id"])["badges"] == ["first_task"]


def test_wildcard_badge_needs_a_passed_wildcard_task(fake_db):
    skill = seed_skill(USER)
    assert "wildcard" not in _record(skill, passed=False, wildcard=True)["new_badges"]
    assert "wildcard" in _record(skill, passed=True, wildcard=True)["new_badges"]


def test_rank_change_is_reported(fake_db):
    skill = seed_skill(USER)
    db_skill = fake_db.collection("users").document(str(USER)).collection("skills").document(skill["id"])
    db_skill.update({"xp_total": 45})

    result = _record(skill)  # +10 -> 55
    assert (result["rank_before"], result["rank_after"]) == ("Новичок", "Ученик")


# --- access control -------------------------------------------------------


def test_owner_claim_and_whitelist(fake_db):
    assert db.get_owner_id() is None
    assert db.is_allowed(1) is False

    assert db.claim_owner(1, "Аня") == 0
    assert db.get_owner_id() == 1
    assert db.is_allowed(1) is True
    assert db.is_allowed(2) is False

    db.set_user(2, True)
    assert db.is_allowed(2) is True
    db.set_user(2, False)
    assert db.is_allowed(2) is False


def test_set_user_updates_in_place_and_keeps_the_original_creation_time(fake_db):
    db.set_user(5, True, name="Боря")
    created = fake_db.collection("users").document("5").get().to_dict()["created_at"]
    db.set_user(5, False)

    stored = fake_db.collection("users").document("5").get().to_dict()
    assert stored["allowed"] is False and stored["name"] == "Боря"
    assert stored["created_at"] == created


def test_claiming_moves_legacy_top_level_skills_under_the_owner(fake_db):
    legacy = fake_db.collection("skills").document("old1")
    legacy.set({"name": "Пиксель-арт", "xp_total": 5})
    legacy.collection("levels").document("l1").set({"order": 1, "title": "Старт"})
    task = legacy.collection("tasks").document("t1")
    task.set({"description": "задание", "completed": True})
    attempt = task.collection("attempts").document("a1")
    attempt.set({"ensemble_feedback": {"task_completed": True}})
    attempt.collection("media").document("proof").set({"data": PNG, "mime": "image/png"})

    assert db.claim_owner(7, "Владелец") == 1

    assert list(fake_db.collection("skills").stream()) == []
    moved = db.get_skill(7, "old1")
    assert moved["name"] == "Пиксель-арт" and moved["xp_total"] == 5
    assert db.get_level(7, "old1", "l1")["title"] == "Старт"
    assert db.get_task(7, "old1", "t1")["completed"] is True
    assert db.get_attempt_blob(7, "old1", "t1", "a1") == PNG
    assert db.get_owner_id() == 7


def test_migration_with_no_legacy_data_is_a_no_op(fake_db):
    seed_skill(7, "Уже под владельцем")
    assert db.migrate_legacy_skills(7) == 0
    assert len(db.list_skills(7)) == 1
