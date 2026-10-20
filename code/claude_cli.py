"""Call Claude as a plain chat model through the local `claude -p` CLI (subscription, no API key).

Every call reads the subscription's 5-hour utilization from the CLI's rate_limit_event.
Once it reaches PAUSE_AT, new calls wait until the window resets, then continue.
Calls already in flight finish, so utilization can overshoot PAUSE_AT by up to
max_workers calls.

Schedule (local time): PAUSE_AT by day, NIGHT_PAUSE_AT from NIGHT_START, and no new calls
from QUIET_START to QUIET_END, so the window a night job opens has reset by morning.
"""

import json
from pathlib import Path
import re
import subprocess
import threading
import time
from datetime import datetime

PAUSE_AT = 0.80
NIGHT_PAUSE_AT = 0.95  # just under 100%, so in-flight calls don't hit the hard limit
NIGHT_START, QUIET_START, QUIET_END = 21, 4, 9  # hours


# Optional temporary override, read at every call: {"pause_at": 0.98, "until": "2026-10-21T04:00"}.
OVERRIDE = Path(__file__).resolve().parent.parent / "claude_pause_override.json"


def _override():
    try:
        o = json.loads(OVERRIDE.read_text())
        return float(o["pause_at"]) if datetime.now() < datetime.fromisoformat(o["until"]) else None
    except (OSError, ValueError, KeyError):
        return None


def _pause_at():
    hour = datetime.now().hour
    if QUIET_START <= hour < QUIET_END:
        return None
    if _override() is not None:
        return _override()
    if hour >= NIGHT_START or hour < QUIET_START:
        return NIGHT_PAUSE_AT
    return PAUSE_AT


def _wait_quiet_hours():
    now = datetime.now()
    if _pause_at() is None:
        end = now.replace(hour=QUIET_END, minute=0, second=0, microsecond=0)
        print(f"[claude_cli] quiet hours; pausing until {end:%H:%M}.", flush=True)
        time.sleep((end - now).total_seconds() + 60)
_lock = threading.Lock()
_state = {"utilization": None, "resets_at": None}


def _run(content, model, system_msg, timeout, tools=""):
    # tools: "" disables every tool; e.g. "WebSearch,WebFetch" enables (and pre-approves) web search.
    cmd = [
        "claude", "-p",
        "--model", model,
        "--system-prompt", system_msg,
        "--tools", tools,
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
    ]
    if tools:
        cmd += ["--allowedTools", *tools.split(",")]
    proc = subprocess.run(cmd, input=content, capture_output=True, text=True, timeout=timeout)
    result = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "rate_limit_event":
            window = event.get("rate_limit_info", {}).get("unifiedWindows", {}).get("five_hour", {})
            if "utilization" in window:
                with _lock:
                    _state["utilization"] = window["utilization"]
                    _state["resets_at"] = window.get("resetsAt")
        elif event.get("type") == "result":
            result = event
    if result is None or result.get("is_error") or proc.returncode != 0:
        detail = result.get("result", "") if result else proc.stderr[-2000:]
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {str(detail)[:2000]}")
    return result


def _wait_if_over_limit():
    _wait_quiet_hours()
    with _lock:
        if _state["utilization"] is None:
            probe = True
        else:
            probe = False
    if probe:
        # Learn the current utilization before the first real call goes out.
        _run("Reply with OK.", "claude-haiku-5-5", "Reply with OK.", 120)
    while True:
        with _lock:
            util, resets_at = _state["utilization"], _state["resets_at"]
            pause_at = _pause_at()
            if pause_at is not None and (util is None or util < pause_at):
                return
            if pause_at is None:
                wait = None
            else:
                wait = max(60, (resets_at or time.time() + 600) - time.time() + 60)
            # Assume the window has reset after waiting; the next call re-reads it.
            _state["utilization"] = None
        if wait is None:
            _wait_quiet_hours()
        else:
            print(f"[claude_cli] 5-hour usage at {util:.0%} (pause at {pause_at:.0%}); "
                  f"pausing until {datetime.fromtimestamp(time.time() + wait):%H:%M}.", flush=True)
            time.sleep(wait)
            _wait_quiet_hours()
        _run("Reply with OK.", "claude-haiku-5-5", "Reply with OK.", 120)


def call_claude(content, model, system_msg, timeout=3600, tools=""):
    """Return (raw_text, in_tokens, out_tokens, api_equivalent_cost_usd)."""
    _wait_if_over_limit()
    out = _run(content, model, system_msg, timeout, tools)
    raw_text = out["result"].strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_text, re.S)
    if fence:
        raw_text = fence.group(1)
    usage = out.get("usage", {})
    in_tokens = (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                 + usage.get("cache_creation_input_tokens", 0))
    return raw_text, in_tokens, usage.get("output_tokens", 0), out.get("total_cost_usd", 0.0)
