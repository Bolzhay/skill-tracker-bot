from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from ..keyboards import MENU_NEW, MENU_SKILLS, main_menu
from . import onboarding, skills

router = Router()

HELP_TEXT = (
    "Команды (скилл можно указать названием или выбрать кнопкой):\n"
    "/new_skill — создать новый скилл\n"
    "/task — получить задание\n"
    "/hint — развёрнутая подсказка к текущему заданию\n"
    "/retry — прислать пруф для последнего задания ещё раз\n"
    "/history — последние задания\n"
    "/progress — сравнить первую и последнюю работы (тогда vs сейчас)\n"
    "/profile — мои скиллы, ранг, стрик, бейджи\n"
    "/cancel — отменить текущий диалог\n\n"
    "Пруф можно прислать фото (лучше файлом — без сжатия), голосовым, видео или текстом."
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я Skill Tracker Bot: строю план уровней под тебя, даю задания и честно оцениваю пруфы.\n\n"
        + HELP_TEXT,
        reply_markup=main_menu(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, reply_markup=main_menu())


@router.message(F.text == MENU_SKILLS)
async def menu_skills(message: Message) -> None:
    await skills.show_profile(message, message.from_user.id)


@router.message(F.text == MENU_NEW)
async def menu_new_skill(message: Message, state: FSMContext) -> None:
    await onboarding.begin_new_skill(message, state)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    if await state.get_state() is None:
        await message.answer("Отменять нечего.")
        return
    await state.clear()
    await message.answer("Отменил. Вернуться к заданию — /retry.")


@router.callback_query(F.data == "cancel")
async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer("Отменено")
    await callback.message.answer("Ок, пруф ждать не буду. Вернуться к заданию — /retry.")
