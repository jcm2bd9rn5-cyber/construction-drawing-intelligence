from types import SimpleNamespace

import pytest

from app.ai.client import build_http_options, call_with_retry, structured_output_config
from app.ai.gemini import VisionOutput
from app.core.config import Settings


class ApiFailure(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"{code} API failure")


def retry_settings(attempts: int = 3) -> Settings:
    return Settings(
        gemini_retry_attempts=attempts,
        gemini_retry_initial_delay=1,
        gemini_retry_max_delay=8,
        gemini_retry_jitter=0,
    )


def test_transient_503_retries_then_succeeds():
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ApiFailure(503)
        return "ok"

    assert call_with_retry(
        operation,
        "test",
        settings=retry_settings(),
        sleep=sleeps.append,
        jitter=lambda _low, _high: 0,
    ) == "ok"
    assert calls == 3
    assert sleeps == [1, 2]


def test_retry_limit_is_bounded():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise ApiFailure(503)

    with pytest.raises(ApiFailure):
        call_with_retry(
            operation,
            "test",
            settings=retry_settings(2),
            sleep=lambda _delay: None,
            jitter=lambda _low, _high: 0,
        )
    assert calls == 2


def test_permanent_404_is_not_retried():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise ApiFailure(404)

    with pytest.raises(ApiFailure):
        call_with_retry(operation, "test", settings=retry_settings())
    assert calls == 1


def test_client_timeout_and_sdk_retry_are_configured_once():
    options = build_http_options(Settings(gemini_timeout_seconds=12.5))
    assert options.timeout == 12_500
    assert options.retry_options.attempts == 1


def test_structured_output_explicitly_disables_unused_afc():
    config = structured_output_config(VisionOutput)
    assert config.response_schema is VisionOutput
    assert config.automatic_function_calling.disable is True
    assert config.temperature == 0.0
