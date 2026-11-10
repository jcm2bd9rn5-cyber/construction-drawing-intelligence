"""Unit tests must never consume a developer's live Gemini quota."""
import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def no_live_gemini(monkeypatch):
    monkeypatch.setattr(get_settings(), "gemini_api_key", None)
    # Embedding/generation adapter tests inject fake clients. Fail immediately
    # if a new test accidentally tries to instantiate the real SDK client.
    import google.genai
    def blocked_client(*args, **kwargs):
        raise AssertionError("Tests must inject a fake Gemini client")
    monkeypatch.setattr(google.genai, "Client", blocked_client)
