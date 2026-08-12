"""Log-ID context and shared config for the logging package.

``log_id`` identifies one execution log and is carried implicitly through a
ContextVar set by ``sample_logging``.
"""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass

# Fixed-width logger-name column so the ``|`` separators line up; _AlignedFormatter
# trims each name to _NAME_WIDTH and fills ``shortname``.
_NAME_WIDTH = 22
_LOG_FORMAT = f"%(asctime)s | %(shortname)-{_NAME_WIDTH}s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    # DEBUG_MODE surfaces per-sample INFO on stdout; it does NOT lower the level to DEBUG.
    return _env_flag("DEBUG_MODE")


# Flush every line to disk (slower). Off by default: files flush on buffer-fill/close.
_FLUSH_EACH_LINE = _env_flag("LOG_FLUSH_EACH_LINE")


@dataclass(frozen=True)
class LogContext:
    """Explicit routing information for one logical execution log."""

    log_id: str
    log_path: str | None = None


_current_log_context: contextvars.ContextVar[LogContext | None] = contextvars.ContextVar(
    "uni_agent_log_context",
    default=None,
)


def get_current_log_context() -> LogContext | None:
    """Return the logging context bound to the current execution."""
    return _current_log_context.get()


def _resolve_log_id(record: logging.LogRecord) -> str | None:
    context = get_current_log_context()
    return context.log_id if context is not None else None
