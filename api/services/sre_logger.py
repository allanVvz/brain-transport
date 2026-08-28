# -*- coding: utf-8 -*-
"""
SRE Logger — structured error/warn/info logging persisted in agent_logs.
Errors appear at GET /logs/agents. Never raises — always falls back to stderr.
"""
from __future__ import annotations

import atexit
import sys
from queue import Empty, SimpleQueue
from threading import Event, Thread
import traceback as tb
from datetime import datetime, timezone
from typing import Optional

_DB_LOG_QUEUE: "SimpleQueue[dict]" = SimpleQueue()
_DB_LOG_STOP = Event()


def _db_log_worker() -> None:
    while not _DB_LOG_STOP.is_set():
        try:
            payload = _DB_LOG_QUEUE.get(timeout=0.5)
        except Empty:
            continue
        try:
            from services import supabase_client
            supabase_client.insert_agent_log(payload)
        except Exception:
            pass


_DB_LOG_THREAD = Thread(target=_db_log_worker, name="sre-db-log-writer", daemon=True)
_DB_LOG_THREAD.start()
atexit.register(_DB_LOG_STOP.set)


def _write(
    level: str,
    component: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> None:
    tb_short = ""
    if exc:
        lines = tb.format_exception(type(exc), exc, exc.__traceback__)
        full = "".join(lines)
        parts = [ln for ln in full.splitlines() if ln.strip()]
        # Keep last 5 lines to stay concise but informative
        tb_short = "\n".join(parts[-5:]) if len(parts) > 5 else full.strip()

    ts = datetime.now(timezone.utc).isoformat()
    label = f"[{ts}] [{level}] [{component}]"
    print(f"{label} {message}" + (f"\n{tb_short}" if tb_short else ""), file=sys.stderr, flush=True)

    try:
        _DB_LOG_QUEUE.put({
            "agent_type": component,
            "action": f"[{level}] {message[:200]}",
            "decision": (tb_short or message)[:500],
            "metadata": {
                "level": level,
                "component": component,
                "message": message,
                "traceback": tb_short,
                "ts": ts,
            },
        })
    except Exception:
        # Queueing failed — stderr already has the record
        pass


def error(component: str, message: str, exc: Optional[BaseException] = None) -> None:
    _write("ERROR", component, message, exc)


def warn(component: str, message: str, exc: Optional[BaseException] = None) -> None:
    _write("WARN", component, message, exc)


def info(component: str, message: str) -> None:
    _write("INFO", component, message)
