from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

MENU_SKILLS = "📋 Мои скиллы"
MENU_NEW = "➕ Новый скилл"


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=MENU_SKILLS), KeyboardButton(text=MENU_NEW)]],
        resize_keyboard=True,
    )


def _button(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _kb(*rows: list[InlineKeyboardButton]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=list(rows))


def skills_kb(skills: list[dict], action: str) -> InlineKeyboardMarkup:
    return _kb(*([_button(skill["name"], f"{action}:{skill['id']}")] for skill in skills))


def skill_card_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb(
        [_button("📝 Задание", f"task:{skill_id}"), _button("💡 Подсказка", f"hint:{skill_id}")],
        [_button("📎 Прислать пруф", f"retry:{skill_id}"), _button("🕘 История", f"hist:{skill_id}")],
        [_button("📈 Тогда vs сейчас", f"prog:{skill_id}")],
    )


def no_tree_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("🌳 Построить дерево", f"tree:{skill_id}"), _button("🗑 Удалить", f"del:{skill_id}")])


def after_task_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("💡 Подсказка", f"hint:{skill_id}"), _button("✖ Отмена", "cancel")])


def after_fail_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("💡 Подсказка", f"hint:{skill_id}")])


def after_success_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("📝 Новое задание", f"task:{skill_id}")])


def tree_failed_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("🌳 Попробовать снова", f"tree:{skill_id}")])


def tree_ready_kb(skill_id: str) -> InlineKeyboardMarkup:
    return _kb([_button("📝 Первое задание", f"task:{skill_id}")])
