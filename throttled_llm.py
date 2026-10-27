"""GeneralLlm with a per-model request-rate limit and patient retries.

The Google AI Studio free tier allows only a few requests per minute per model
(5/min for Flash at the time of writing) and often answers 503 "high demand".
Every request goes through `_mockable_direct_call_to_model`, so throttling and
retrying there covers forecasting, research and structured-output parsing alike.
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

# Requests allowed per rolling 60 s, keyed by model name; kept one below the
# published free-tier limit to leave headroom for clock skew.
RPM_LIMITS = {
    "gemini/gemini-3.5-flash": 4,
    "gemini/gemini-3.5-flash-lite": 9,
    "gemini/gemma-4-26b-a4b-it": 8,
}
DEFAULT_RPM = 4
MAX_ATTEMPTS = 8

_windows: dict[str, deque] = defaultdict(deque)
# main.py calls asyncio.run() more than once, and a Lock is bound to the loop it
# first waits on, so keep one lock per (event loop, model).
_locks: dict[tuple[int, str], asyncio.Lock] = defaultdict(asyncio.Lock)


async def _acquire(model: str) -> None:
    limit = RPM_LIMITS.get(model, DEFAULT_RPM)
    async with _locks[(id(asyncio.get_running_loop()), model)]:
        window = _windows[model]
        while True:
            now = time.monotonic()
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) < limit:
                window.append(now)
                return
            await asyncio.sleep(60 - (now - window[0]) + 0.5)


def _is_transient(err: Exception) -> bool:
    text = f"{type(err).__name__} {err}"
    if "PerDay" in text:  # daily quota: retrying today won't help
        return False
    return any(
        s in text
        for s in ("RateLimit", "429", "503", "ServiceUnavailable", "high demand", "Timeout", "500", "InternalServerError")
    )


class ThrottledLlm(GeneralLlm):
    async def _mockable_direct_call_to_model(self, prompt):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await _acquire(self.model)
            try:
                return await super()._mockable_direct_call_to_model(prompt)
            except Exception as e:
                if attempt == MAX_ATTEMPTS or not _is_transient(e):
                    raise
                wait = min(90, 15 * attempt) + random.uniform(0, 5)
                logger.warning(
                    f"{self.model} transient error ({type(e).__name__}), retry {attempt}/{MAX_ATTEMPTS - 1} in {wait:.0f}s"
                )
                await asyncio.sleep(wait)
