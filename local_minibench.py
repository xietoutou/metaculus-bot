"""Local MiniBench runner, started every 20 minutes by a Windows scheduled task.

MiniBench runs here instead of on GitHub Actions so numeric and discrete
questions can use Claude through the logged-in Claude Code CLI
(main.py --claude-numeric); other question types still use Gemma. The GitHub
workflow passes --minibench skip so the two never race for the same question.

A lock file keeps runs from overlapping; each run's output goes to
local_runs/<timestamp>.log and a one-line summary to local_runs/runs.log.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "local_runs"
LOCK = RUNS / "lock"
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
TIMEOUT = 150 * 60
KEEP_LOGS = 300


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout
    return str(pid) in out


def summary(msg: str) -> None:
    with (RUNS / "runs.log").open("a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\n")


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except ValueError:
            pid = 0
        if pid and pid_alive(pid):
            summary(f"skip: run pid {pid} still going")
            return
    LOCK.write_text(str(os.getpid()))
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        log_path = RUNS / f"{stamp}.log"
        start = time.monotonic()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with log_path.open("w", encoding="utf-8") as log:
            try:
                rc = subprocess.run(
                    [str(PYTHON), "main.py", "--mode", "tournament",
                     "--minibench", "only", "--claude-numeric", *sys.argv[1:]],
                    cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env,
                    timeout=TIMEOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        forecasted = text.count("Forecasted URL")
        fallbacks = text.count("Claude research failed") + text.count("Claude forecast failed")
        summary(
            f"rc={rc} {time.monotonic() - start:.0f}s predictions={forecasted} "
            f"claude_fallbacks={fallbacks} log={log_path.name}"
        )
        logs = sorted(p for p in RUNS.glob("*.log") if p.name != "runs.log")
        for old in logs[:-KEEP_LOGS]:
            old.unlink(missing_ok=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
