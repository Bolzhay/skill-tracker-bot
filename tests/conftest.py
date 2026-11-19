import os

# Set before any `bot` import so tests never see the real .env secrets:
# config loads .env with override=True, so the file is switched off here.
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False

os.environ["BOT_TOKEN"] = "test-token"
os.environ["FIREBASE_CREDENTIALS_PATH"] = "unused.json"
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = ""  # tests that need the second provider switch it on themselves
os.environ["YOUTUBE_API_KEY"] = ""  # ...and likewise YouTube search
os.environ["CLAIM_CODE"] = "test-claim-code"
os.environ["TZ_OFFSET_HOURS"] = "5"

import pytest  # noqa: E402
from aiogram.fsm.context import FSMContext  # noqa: E402
from aiogram.fsm.storage.base import StorageKey  # noqa: E402
from aiogram.fsm.storage.memory import MemoryStorage  # noqa: E402

from bot import access, db  # noqa: E402
from bot.handlers import onboarding, tasks  # noqa: E402
from tests.fake_firestore import FakeClient  # noqa: E402


@pytest.fixture
def fake_db(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(db, "_db_client", client)
    return client


@pytest.fixture(autouse=True)
def clean_module_state():
    """Module-level caches/guards must not leak between tests."""
    access.reset()
    onboarding._building.clear()
    tasks._busy.clear()
    yield
    access.reset()


@pytest.fixture
def state():
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=1, user_id=1))


_dispatcher = None


@pytest.fixture
def dispatcher():
    """The real dispatcher, built exactly as in production. aiogram lets each
    module-level router be attached once, so it's built once per test run and
    every dialog state is wiped between tests."""
    global _dispatcher
    if _dispatcher is None:
        from bot.main import build_dispatcher

        _dispatcher = build_dispatcher()
    _dispatcher.storage.storage.clear()
    return _dispatcher


@pytest.fixture
async def owner(fake_db, dispatcher):
    """User 1, already activated as the owner via /claim."""
    from tests.fake_telegram import Client

    client = Client(dispatcher, user_id=1)
    await client.say("/claim test-claim-code")
    client.clear()
    return client
