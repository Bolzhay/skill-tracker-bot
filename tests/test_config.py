import importlib

import dotenv

from bot import config


def test_the_env_file_wins_over_variables_already_set_in_the_os(monkeypatch):
    # A stale GEMINI_API_KEY in the Windows environment once shadowed the key in .env.
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: calls.append(kwargs) or False)
    try:
        importlib.reload(config)
        assert calls == [{"override": True}]
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # back to the test setup: the real .env stays switched off
