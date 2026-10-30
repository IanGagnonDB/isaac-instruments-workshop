import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator


TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def valid_trace_id(value: str | None) -> str:
    candidate = (value or "").strip()
    return candidate if TRACE_ID_PATTERN.fullmatch(candidate) else new_trace_id()


def current_trace_id() -> str:
    trace_id = _trace_id.get()
    if trace_id:
        return trace_id
    trace_id = new_trace_id()
    _trace_id.set(trace_id)
    return trace_id


@contextmanager
def trace_context(trace_id: str) -> Iterator[None]:
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1_000)


def emit(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "trace_id": current_trace_id(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    logger.log(level, json.dumps(payload, separators=(",", ":"), default=str))
