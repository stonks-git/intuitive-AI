"""LLM client wrapper with retry logic for all providers.

Exponential backoff with jitter on transient failures (rate limits,
timeouts, 5xx). Permanent errors (auth, bad request) fail immediately.

Usage:
    from .llm import retry_llm_call
    from .config import RetryConfig

    result = await retry_llm_call(
        some_async_api_call,
        prompt="hello",
        config=config.retry,   # from runtime.yaml
        label="system1",
    )
"""

import asyncio
import logging
import random
import time
from typing import Any

from .config import RetryConfig

logger = logging.getLogger("agent.llm")


def _is_transient(exc: Exception) -> bool:
    """Decide if an error is worth retrying.

    Transient: rate limits (429), server errors (5xx), timeouts, connection drops.
    Permanent: auth errors (401/403), bad request (400), not found (404).
    """
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    # Rate limits
    if "429" in exc_str or "ResourceExhausted" in exc_name:
        return True

    # Server errors
    if any(code in exc_str for code in ("500", "502", "503", "504")):
        return True
    if any(name in exc_name for name in ("ServiceUnavailable", "InternalServerError")):
        return True

    # Timeouts
    if "timeout" in exc_str or "DeadlineExceeded" in exc_name:
        return True

    # HTTP status codes (openai/anthropic SDKs)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 or status >= 500:
            return True

    # Connection errors (httpx, aiohttp, requests, urllib3)
    if any(
        t in exc_name
        for t in ("ConnectionError", "ConnectError", "ReadTimeout",
                  "ConnectTimeout", "RemoteProtocolError")
    ):
        return True

    # Ollama connection issues
    if "connection" in exc_str and ("refused" in exc_str or "reset" in exc_str):
        return True

    return False


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header hint if the API sent one."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _compute_delay(attempt: int, config: RetryConfig, exc: Exception) -> float:
    """Exponential backoff with jitter, respecting Retry-After if present."""
    retry_after = _get_retry_after(exc)
    if retry_after is not None:
        delay = retry_after
    else:
        delay = min(
            config.base_delay_seconds * (2 ** attempt),
            config.max_delay_seconds,
        )

    # Jitter: ±jitter fraction
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    return max(0.1, delay)


async def retry_llm_call(
    fn,
    *args,
    config: RetryConfig | None = None,
    label: str = "",
    **kwargs,
) -> Any:
    """Call an async LLM function with exponential backoff retry.

    Args:
        fn: async callable (the actual API call)
        config: RetryConfig from runtime.yaml (uses defaults if None)
        label: tag for log messages (e.g. "system1", "embeddings")

    Raises:
        Immediately on permanent errors.
        Last exception after all retries exhausted.
    """
    if config is None:
        config = RetryConfig()
    label = label or getattr(fn, "__qualname__", str(fn))

    last_exc = None
    for attempt in range(1 + config.max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not _is_transient(exc):
                logger.error(f"[{label}] permanent error: {exc}")
                raise

            if attempt >= config.max_retries:
                logger.error(
                    f"[{label}] all {config.max_retries} retries exhausted. "
                    f"Last error: {exc}"
                )
                raise

            delay = _compute_delay(attempt, config, exc)
            logger.warning(
                f"[{label}] transient error (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)

    raise last_exc  # unreachable, but satisfies type checkers


def retry_llm_call_sync(
    fn,
    *args,
    config: RetryConfig | None = None,
    label: str = "",
    **kwargs,
) -> Any:
    """Synchronous version for blocking contexts (e.g. Ollama embeddings)."""
    if config is None:
        config = RetryConfig()
    label = label or getattr(fn, "__qualname__", str(fn))

    last_exc = None
    for attempt in range(1 + config.max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not _is_transient(exc):
                logger.error(f"[{label}] permanent error: {exc}")
                raise

            if attempt >= config.max_retries:
                logger.error(
                    f"[{label}] all {config.max_retries} retries exhausted. "
                    f"Last error: {exc}"
                )
                raise

            delay = _compute_delay(attempt, config, exc)
            logger.warning(
                f"[{label}] transient error (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {exc}"
            )
            time.sleep(delay)

    raise last_exc
