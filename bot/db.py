"""Firestore access. Everything a user owns lives under users/{telegram_id}:

    meta/owner                                  {user_id}
    users/{uid}                                 {allowed, role, name}
    users/{uid}/skills/{skillId}                skill + counters (xp, streak, badges)
    .../skills/{id}/levels/{levelId}            the level tree
    .../skills/{id}/tasks/{taskId}              generated tasks
    .../tasks/{id}/attempts/{attemptId}         evaluated proofs (metadata)
    .../attempts/{id}/media/proof               archived proof bytes, if small enough
"""

import firebase_admin
from firebase_admin import credentials, firestore

from . import gamification

_db_client = None


def init_db(credentials_path: str):
    global _db_client
    cred = credentials.Certificate(credentials_path)
    firebase_admin.initialize_app(cred)
    _db_client = firestore.client()
    return _db_client


def get_db():
    if _db_client is None:
        raise RuntimeError("DB not initialized — call init_db() first")
    return _db_client


def _with_id(doc) -> dict:
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def _user(user_id: int):
    return get_db().collection("users").document(str(user_id))


def _skills(user_id: int):
    return _user(user_id).collection("skills")


def _skill(user_id: int, skill_id: str):
    return _skills(user_id).document(skill_id)


def _tasks(user_id: int, skill_id: str):
    return _skill(user_id, skill_id).collection("tasks")


# --- access control -------------------------------------------------------


def get_owner_id() -> int | None:
    doc = get_db().collection("meta").document("owner").get()
    return doc.to_dict()["user_id"] if doc.exists else None


def is_allowed(user_id: int) -> bool:
    doc = _user(user_id).get()
    return bool(doc.exists and doc.to_dict().get("allowed"))


def set_user(user_id: int, allowed: bool, role: str = "member", name: str = "") -> None:
    ref = _user(user_id)
    data = {"allowed": allowed, "role": role}
    if name:
        data["name"] = name
    if not ref.get().exists:
        data["created_at"] = firestore.SERVER_TIMESTAMP
    ref.set(data, merge=True)


def claim_owner(user_id: int, name: str = "") -> int:
    """Make `user_id` the owner and move any pre-multi-user data under them.
    Returns how many legacy skills were migrated."""
    get_db().collection("meta").document("owner").set(
        {"user_id": user_id, "claimed_at": firestore.SERVER_TIMESTAMP}
    )
    set_user(user_id, True, role="owner", name=name)
    return migrate_legacy_skills(user_id)


def _copy_tree(src, dst) -> None:
    dst.set(src.get().to_dict())
    for sub in src.collections():
        for child in sub.stream():
            _copy_tree(child.reference, dst.collection(sub.id).document(child.id))


def migrate_legacy_skills(user_id: int) -> int:
    """One-time move of the old top-level `skills` collection (from before
    the bot was multi-user) under the owner. Each skill is copied in full and
    only then deleted, so a failure midway never loses data."""
    client = get_db()
    moved = 0
    for legacy in list(client.collection("skills").stream()):
        _copy_tree(legacy.reference, _skills(user_id).document(legacy.id))
        client.recursive_delete(legacy.reference)
        moved += 1
    return moved


# --- skills and levels ----------------------------------------------------


def create_skill(user_id: int, name: str) -> str:
    doc_ref = _skills(user_id).document()
    doc_ref.set(
        {
            "name": name,
            "created_at": firestore.SERVER_TIMESTAMP,
            "current_level_id": None,
            "levels_total": 0,
            "xp_total": 0,
            "streak_count": 0,
            "streak_last_date": None,
            "tasks_completed": 0,
            "badges": [],
            "onboarding": None,
        }
    )
    return doc_ref.id


def save_onboarding_context(user_id: int, skill_id: str, self_assessment: str, experience: str, goal: str) -> None:
    """Persist the onboarding answers so later pipelines (tasks, evaluation)
    can calibrate to the user's real starting point, and so a failed tree
    build can be retried without asking the questions again."""
    _skill(user_id, skill_id).update(
        {"onboarding": {"self_assessment": self_assessment, "experience": experience, "goal": goal}}
    )


def list_skills(user_id: int) -> list[dict]:
    return [_with_id(doc) for doc in _skills(user_id).order_by("created_at").stream()]


def find_skill_by_name(user_id: int, name: str) -> dict | None:
    wanted = name.strip().casefold()
    return next((skill for skill in list_skills(user_id) if skill["name"].casefold() == wanted), None)


def get_skill(user_id: int, skill_id: str) -> dict | None:
    doc = _skill(user_id, skill_id).get()
    return _with_id(doc) if doc.exists else None


def delete_skill_if_empty(user_id: int, skill_id: str) -> bool:
    """Delete a skill whose tree was never built (a failed onboarding).
    Refuses to touch a skill that already has levels."""
    skill = get_skill(user_id, skill_id)
    if skill is None or skill.get("current_level_id"):
        return False
    get_db().recursive_delete(_skill(user_id, skill_id))
    return True


def get_level(user_id: int, skill_id: str, level_id: str) -> dict | None:
    doc = _skill(user_id, skill_id).collection("levels").document(level_id).get()
    return _with_id(doc) if doc.exists else None


def save_level_tree(user_id: int, skill_id: str, levels: list[dict]) -> None:
    """Persist the ensemble's level tree and activate the first level.

    Re-derives `order` from sort position rather than trusting the model's
    raw values, since models occasionally miscount despite instructions.
    """
    client = get_db()
    skill_ref = _skill(user_id, skill_id)
    batch = client.batch()

    levels_sorted = sorted(levels, key=lambda entry: entry.get("order") or 0)
    first_level_id = None

    for index, level in enumerate(levels_sorted, start=1):
        level_ref = skill_ref.collection("levels").document()
        if first_level_id is None:
            first_level_id = level_ref.id

        batch.set(
            level_ref,
            {
                "order": index,
                "title": level.get("title", ""),
                "criteria": {
                    "summary": level.get("criteria_summary", ""),
                    "checks": level.get("criteria_checks", []),
                },
                "status": "active" if index == 1 else "locked",
                "successful_attempts": 0,
            },
        )

    batch.update(skill_ref, {"current_level_id": first_level_id, "levels_total": len(levels_sorted)})
    batch.commit()


# --- tasks and attempts ---------------------------------------------------


def count_tasks(user_id: int, skill_id: str) -> int:
    result = _tasks(user_id, skill_id).count().get()
    return int(result[0][0].value)


def get_recent_tasks(user_id: int, skill_id: str, limit: int = 10) -> list[dict]:
    docs = _tasks(user_id, skill_id).order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
    return [_with_id(doc) for doc in docs]


def create_task(
    user_id: int,
    skill_id: str,
    level_id: str,
    description: str,
    hint: str,
    media_type: str,
    is_wildcard: bool,
) -> str:
    doc_ref = _tasks(user_id, skill_id).document()
    doc_ref.set(
        {
            "level_id": level_id,
            "description": description,
            "hint": hint,
            "media_type": media_type,
            "is_wildcard": is_wildcard,
            "completed": False,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
    )
    return doc_ref.id


def mark_task_completed(user_id: int, skill_id: str, task_id: str) -> None:
    _tasks(user_id, skill_id).document(task_id).update({"completed": True})


def get_task(user_id: int, skill_id: str, task_id: str) -> dict | None:
    doc = _tasks(user_id, skill_id).document(task_id).get()
    return _with_id(doc) if doc.exists else None


def create_attempt(
    user_id: int,
    skill_id: str,
    task_id: str,
    media_type: str,
    verdict: dict,
    xp_awarded: int,
    proof: dict | None = None,
) -> str:
    """Store an evaluated attempt. `proof` describes what the user sent
    ({"kind", "file_id", "text"/"transcript", "mime", "size", "bytes"}); the
    raw bytes go to a separate doc so listing attempts stays cheap."""
    doc_ref = _tasks(user_id, skill_id).document(task_id).collection("attempts").document()
    meta = {
        "media_type": media_type,
        "ensemble_feedback": verdict,
        "xp_awarded": xp_awarded,
        "has_blob": False,
        "created_at": firestore.SERVER_TIMESTAMP,
    }

    blob = None
    if proof:
        proof = dict(proof)
        blob = proof.pop("bytes", None)
        meta["proof"] = proof
        meta["has_blob"] = blob is not None

    doc_ref.set(meta)
    if blob is not None:
        doc_ref.collection("media").document("proof").set({"data": blob, "mime": proof.get("mime", "")})
    return doc_ref.id


def get_attempt_blob(user_id: int, skill_id: str, task_id: str, attempt_id: str) -> bytes | None:
    doc = (
        _tasks(user_id, skill_id)
        .document(task_id)
        .collection("attempts")
        .document(attempt_id)
        .collection("media")
        .document("proof")
        .get()
    )
    return doc.to_dict()["data"] if doc.exists else None


def successful_photo_attempts(user_id: int, skill_id: str) -> list[dict]:
    """The passing photo attempt of each completed task, oldest first —
    the raw material for the "then vs now" comparison."""
    found = []
    for task_doc in _tasks(user_id, skill_id).order_by("created_at").stream():
        task = _with_id(task_doc)
        if not task.get("completed"):
            continue
        attempts = task_doc.reference.collection("attempts").order_by(
            "created_at", direction=firestore.Query.DESCENDING
        )
        for attempt_doc in attempts.stream():
            attempt = _with_id(attempt_doc)
            passed = attempt.get("ensemble_feedback", {}).get("task_completed")
            if passed and (attempt.get("proof") or {}).get("kind") == "photo":
                found.append({"task": task, "attempt": attempt})
                break
    return found


def _count_toward_level(skill_ref, level_id: str, updates: dict, result: dict) -> None:
    """Count one passed task toward the level; advance when the threshold is
    hit. Tasks are scoped to one focused thing each, so no single task proves
    a level — progress is "ATTEMPTS_TO_LEVEL_UP passed tasks", not one verdict."""
    levels = skill_ref.collection("levels")
    level_ref = levels.document(level_id)
    level = level_ref.get().to_dict()

    successful = level.get("successful_attempts", 0) + 1
    result["successful_attempts"] = successful
    if successful < gamification.ATTEMPTS_TO_LEVEL_UP:
        level_ref.update({"successful_attempts": successful})
        return

    wanted_order = level.get("order", 0) + 1
    next_doc = next((doc for doc in levels.stream() if doc.to_dict().get("order") == wanted_order), None)
    if next_doc is None:
        level_ref.update({"successful_attempts": successful})
        result["tree_complete"] = True
        return

    level_ref.update({"status": "done", "successful_attempts": successful})
    levels.document(next_doc.id).update({"status": "active"})
    updates["current_level_id"] = next_doc.id
    result["leveled_up"] = True
    result["new_level"] = _with_id(next_doc)


def record_attempt_result(
    user_id: int,
    skill_id: str,
    task_level_id: str,
    task_completed: bool,
    is_wildcard: bool,
    today,
) -> dict:
    """Apply one evaluated attempt to the skill: XP, streak, level progress,
    badges. Level progress is only counted when the task belongs to the
    current level (a late pass of an old task shouldn't push the new one).

    Returns xp, xp_total, streak, successful_attempts, leveled_up, new_level,
    tree_complete, new_badges, rank_before, rank_after.
    """
    skill_ref = _skill(user_id, skill_id)
    skill = skill_ref.get().to_dict()

    streak = gamification.next_streak(skill.get("streak_count", 0), skill.get("streak_last_date"), today)
    xp = gamification.xp_for_attempt(task_completed, streak)
    xp_before = skill.get("xp_total", 0)
    xp_total = xp_before + xp

    updates = {"xp_total": xp_total, "streak_count": streak, "streak_last_date": today.isoformat()}
    result = {
        "xp": xp,
        "xp_total": xp_total,
        "streak": streak,
        "successful_attempts": 0,
        "leveled_up": False,
        "new_level": None,
        "tree_complete": False,
        "new_badges": [],
        "rank_before": gamification.rank_for(xp_before),
        "rank_after": gamification.rank_for(xp_total),
    }

    tasks_completed = skill.get("tasks_completed", 0)
    if task_completed:
        tasks_completed += 1
        updates["tasks_completed"] = tasks_completed
        current_level_id = skill.get("current_level_id")
        if current_level_id and current_level_id == task_level_id:
            _count_toward_level(skill_ref, current_level_id, updates, result)

    owned = skill.get("badges", [])
    earned = gamification.new_badges(
        tasks_completed, streak, result["leveled_up"], task_completed and is_wildcard, owned
    )
    if earned:
        updates["badges"] = owned + earned
        result["new_badges"] = earned

    skill_ref.update(updates)
    return result
