"""Shared Gemini client, timeout, and bounded retry policy."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import random
import re
import time
from typing import TypeVar

from google import genai
from google.genai import types

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class QuotaDetails:
    retry_after_seconds: float | None = None
    metric: str | None = None
    daily: bool = False


class DailyQuotaExhausted(RuntimeError):
    def __init__(self, operation_name: str, details: QuotaDetails, cause: Exception, attempts: int):
        self.operation_name = operation_name
        self.retry_after_seconds = details.retry_after_seconds
        self.quota_metric = details.metric
        self.http_status = _status_code(cause)
        self.attempts = attempts
        super().__init__(str(cause))


def build_http_options(settings: Settings | None = None) -> types.HttpOptions:
    settings = settings or get_settings()
    return types.HttpOptions(
        timeout=int(settings.gemini_timeout_seconds * 1000),
        # Application-level retry below owns logging and timing. Explicitly
        # disable the SDK retry to avoid multiplying the attempt count.
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def create_client(settings: Settings | None = None) -> genai.Client:
    settings = settings or get_settings()
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=build_http_options(settings),
    )


def _status_code(exc: Exception) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def quota_details(exc: Exception) -> QuotaDetails:
    """Extract Google quota metadata without depending on SDK private fields."""
    message = str(exc)
    retry_after: float | None = None
    patterns = (
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
        r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?",
        r"retry after\s+(\d+(?:\.\d+)?)\s*seconds?",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        if pattern.startswith(r"retry in"):
            hours, minutes, seconds = match.groups()
            retry_after = (
                int(hours or 0) * 3600
                + int(minutes or 0) * 60
                + float(seconds or 0)
            )
        else:
            retry_after = float(match.group(1))
        break
    metric_match = re.search(
        r"Quota exceeded for metric:\s*([\w./-]+)", message, re.IGNORECASE
    )
    metric = metric_match.group(1) if metric_match else None
    daily = bool(
        re.search(r"(?:per[_ ]?day|requests/day|free[_ ]?tier.*(?:day|daily))", message, re.I)
        or (retry_after is not None and retry_after >= 3600)
    )
    return QuotaDetails(retry_after, metric, daily)


def call_with_retry(
    operation: Callable[[], T],
    operation_name: str,
    *,
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    settings = settings or get_settings()
    attempts = max(1, settings.gemini_retry_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            code = _status_code(exc)
            details = quota_details(exc) if code == 429 else QuotaDetails()
            if details.daily:
                logger.error(
                    "Gemini %s daily quota exhausted (metric=%s, retry_after=%s); "
                    "deferring without short retries",
                    operation_name,
                    details.metric or "unknown",
                    details.retry_after_seconds,
                )
                raise DailyQuotaExhausted(operation_name, details, exc, attempt) from exc
            transient = code in TRANSIENT_STATUS_CODES or isinstance(exc, TimeoutError)
            if not transient or attempt == attempts:
                try:
                    exc.gemini_attempts = attempt
                except Exception:
                    pass
                logger.error(
                    "Gemini %s failed on attempt %d/%d (status=%s): %s",
                    operation_name,
                    attempt,
                    attempts,
                    code or "timeout" if isinstance(exc, TimeoutError) else code or "unknown",
                    exc,
                )
                raise
            base_delay = min(
                settings.gemini_retry_max_delay,
                settings.gemini_retry_initial_delay * (2 ** (attempt - 1)),
            )
            delay = base_delay + jitter(0, settings.gemini_retry_jitter)
            logger.warning(
                "Gemini %s transient failure on attempt %d/%d "
                "(status=%s); retrying in %.2fs",
                operation_name,
                attempt,
                attempts,
                code or "timeout",
                delay,
            )
            sleep(delay)
    raise AssertionError("retry loop must return or raise")


def structured_output_config(
    schema: type, *, temperature: float = 0.0, top_p: float | None = None
) -> types.GenerateContentConfig:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        # This application does not expose callable tools. google-genai 2.28
        # otherwise enters its AFC path by default even for structured output.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if top_p is not None:
        config.top_p = top_p
    return config
