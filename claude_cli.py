"""Call Claude through the local Claude Code CLI (`claude -p`).

This runs on the owner's Claude subscription instead of a paid API key, so it
only works on the machine where Claude Code is logged in, never on GitHub
Actions. Every call is a fresh, non-persisted session in an empty working
directory, with no MCP servers, no user/local settings and no tools except the
two web tools when research is requested.

When the subscription's usage limit is hit, `ClaudeUnavailable` is raised and
the CLI is switched off for the rest of the process, so callers fall back to
the free Gemma model instead of failing the question.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
WORK_DIR = HERE / ".claude_work"
USAGE_LOG = HERE / "claude_usage.log"
CLAUDE_EXE = os.environ.get("CLAUDE_EXE") or str(
    Path(os.environ.get("APPDATA", ""))
    / "npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe"
)
MODEL = os.environ.get("CLAUDE_FORECAST_MODEL", "sonnet")
MAX_PARALLEL = int(os.environ.get("CLAUDE_MAX_PARALLEL", "3"))

_LIMIT_MARKERS = ("usage limit", "limit reached", "rate limit", "rate_limit", "quota", "credit balance")


class ClaudeUnavailable(Exception):
    """Claude can't answer right now; use the fallback model."""


_disabled_reason: str | None = None
_semaphores: dict[int, asyncio.Semaphore] = {}


def available() -> bool:
    return _disabled_reason is None and Path(CLAUDE_EXE).exists()


def _semaphore() -> asyncio.Semaphore:
    # main.py calls asyncio.run() more than once; keep one semaphore per loop.
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(MAX_PARALLEL)
    return _semaphores[loop_id]


def _child_env() -> dict[str, str]:
    # Drop variables that would make the child think it is nested inside
    # another Claude Code session.
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(("CLAUDE", "ANTHROPIC"))}


def _log_usage(kind: str, data: dict, seconds: float) -> None:
    usage = data.get("usage") or {}
    row = {
        "t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "ok": not data.get("is_error"),
        "sec": round(seconds, 1),
        "turns": data.get("num_turns"),
        "api_equiv_usd": data.get("total_cost_usd"),
        "out_tokens": usage.get("output_tokens"),
        "models": list((data.get("modelUsage") or {}).keys()),
    }
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


async def claude_ask(prompt: str, *, web: bool = False, timeout: float = 900) -> str:
    """Return Claude's final text answer to `prompt`.

    web=True lets Claude use WebSearch and WebFetch; otherwise it has no tools.
    """
    global _disabled_reason
    if not available():
        raise ClaudeUnavailable(_disabled_reason or f"claude CLI not found at {CLAUDE_EXE}")
    WORK_DIR.mkdir(exist_ok=True)
    tools = "WebSearch,WebFetch" if web else ""
    cmd = [
        CLAUDE_EXE, "-p",
        "--output-format", "json",
        "--model", MODEL,
        "--tools", tools,
        "--strict-mcp-config",
        "--setting-sources", "project",
        "--no-session-persistence",
        "--disable-slash-commands",
    ]
    if web:
        cmd += ["--allowedTools", tools]
    async with _semaphore():
        if not available():
            raise ClaudeUnavailable(_disabled_reason)
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WORK_DIR,
            env=_child_env(),
            creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ClaudeUnavailable(f"claude -p timed out after {timeout:.0f}s")
        seconds = time.monotonic() - start

    text_out = out.decode("utf-8", errors="replace")
    try:
        data = json.loads(text_out)
    except json.JSONDecodeError:
        data = {"is_error": True, "result": (text_out + err.decode("utf-8", errors="replace"))[-500:]}
    _log_usage("research" if web else "forecast", data, seconds)

    result = data.get("result") or ""
    if data.get("is_error") or proc.returncode != 0 or not result.strip():
        message = (result or err.decode("utf-8", errors="replace"))[-300:]
        if any(m in message.lower() for m in _LIMIT_MARKERS):
            _disabled_reason = f"Claude usage limit: {message}"
            logger.warning(f"Claude switched off for this run: {message}")
        raise ClaudeUnavailable(f"claude -p failed (rc={proc.returncode}): {message}")
    return result


if __name__ == "__main__":
    # Self-test: python claude_cli.py -> writes the answer to local_runs/selftest.log
    out = HERE / "local_runs" / "selftest.log"
    out.parent.mkdir(exist_ok=True)
    try:
        answer = asyncio.run(claude_ask("Reply with exactly: OK", timeout=120))
    except Exception as e:  # noqa: BLE001
        answer = f"ERROR {type(e).__name__}: {e}"
    with out.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ} {answer.strip()[:200]}\n")
