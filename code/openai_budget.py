"""Hard spending cap for OpenAI chat calls, shared across processes.

Importing this module wraps openai's Completions.create: every call's token usage is priced and
appended to a JSON ledger (file-locked, so parallel films share one total), and a new call is
refused with BudgetExceeded once the ledger total reaches the stop level.

Environment (all optional):
    OPENAI_BUDGET_USD     cap in dollars (default 20)
    OPENAI_BUDGET_MARGIN  stop this far below the cap, for calls already in flight (default 2)
    OPENAI_BUDGET_LEDGER  ledger path (default ~/.cache/armen-subtitulos/openai-spend.json)
"""
import fcntl
import json
import os
import time
from pathlib import Path

from openai.resources.chat.completions import Completions

CAP = float(os.environ.get("OPENAI_BUDGET_USD", "20"))
MARGIN = float(os.environ.get("OPENAI_BUDGET_MARGIN", "2"))
LEDGER = Path(os.environ.get("OPENAI_BUDGET_LEDGER",
                             Path.home() / ".cache/armen-subtitulos/openai-spend.json"))

# USD per 1M tokens (input, output); output includes reasoning tokens.
PRICES = {
    "gpt-6-sol": (2.00, 10.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-mini": (0.25, 2.00),
}
UNKNOWN = (10.00, 50.00)  # conservative for any model not listed


class BudgetExceeded(RuntimeError):
    pass


def _locked(update):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        data = json.loads(raw) if raw.strip() else {"total_usd": 0.0, "calls": 0, "by_model": {}}
        out = update(data)
        f.seek(0); f.truncate()
        json.dump(data, f, indent=2)
        return out


def spent():
    return _locked(lambda d: d["total_usd"])


def _check(_):
    def upd(d):
        if d["total_usd"] >= CAP - MARGIN:
            raise BudgetExceeded(f"OpenAI spend ${d['total_usd']:.2f} has reached the stop level "
                                 f"${CAP - MARGIN:.2f} (cap ${CAP:.2f}); ledger {LEDGER}")
    _locked(upd)


def _record(model, usage):
    pin, pout = PRICES.get(model, UNKNOWN)
    if usage is None:  # no usage returned: charge a pessimistic flat amount
        cost, tin, tout = 0.5, None, None
    else:
        tin, tout = usage.prompt_tokens, usage.completion_tokens
        cost = (tin * pin + tout * pout) / 1e6

    def upd(d):
        d["total_usd"] += cost
        d["calls"] += 1
        m = d["by_model"].setdefault(model, {"usd": 0.0, "calls": 0, "in": 0, "out": 0})
        m["usd"] += cost; m["calls"] += 1; m["in"] += tin or 0; m["out"] += tout or 0
        d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _locked(upd)


_orig = Completions.create
if not getattr(_orig, "_budgeted", False):
    def create(self, *args, **kwargs):
        _check(None)
        r = _orig(self, *args, **kwargs)
        _record(kwargs.get("model", "?"), getattr(r, "usage", None))
        return r
    create._budgeted = True
    Completions.create = create
