"""Call Claude as a plain chat model through the local `claude -p` CLI (subscription, no API key).

Every call reads the subscription's 5-hour utilization from the CLI's rate_limit_event.
Once it reaches PAUSE_AT, new calls wait until the window resets, then continue.
Calls already in flight finish, so utilization can overshoot PAUSE_AT by up to
max_workers calls.
"""

import json
import re
import subprocess
import threading
import time
from datetime import datetime

PAUSE_AT = 0.80
_lock = threading.Lock()
_state = {"utilization": None, "resets_at": None}


def _run(content, model, system_msg, timeout):
    cmd = [
        "claude", "-p",
        "--model", model,
        "--system-prompt", system_msg,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
    ]
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
            if util is None or util < PAUSE_AT:
                return
            wait = max(60, (resets_at or time.time() + 600) - time.time() + 60)
            # Assume the window has reset after waiting; the next call re-reads it.
            _state["utilization"] = None
        print(f"[claude_cli] 5-hour usage at {util:.0%} (pause at {PAUSE_AT:.0%}); "
              f"pausing until {datetime.fromtimestamp(time.time() + wait):%H:%M}.", flush=True)
        time.sleep(wait)
        _run("Reply with OK.", "claude-haiku-5-5", "Reply with OK.", 120)


def call_claude(content, model, system_msg, timeout=3600):
    """Return (raw_text, in_tokens, out_tokens, api_equivalent_cost_usd)."""
    _wait_if_over_limit()
    out = _run(content, model, system_msg, timeout)
    raw_text = out["result"].strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_text, re.S)
    if fence:
        raw_text = fence.group(1)
    usage = out.get("usage", {})
    in_tokens = (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                 + usage.get("cache_creation_input_tokens", 0))
    return raw_text, in_tokens, usage.get("output_tokens", 0), out.get("total_cost_usd", 0.0)
