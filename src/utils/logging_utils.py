"""Structured logging + fail-loud config validation.

One configured logger factory so every module logs with consistent levels and
format to stdout (which Hugging Face Spaces captures). `require_env` makes the
app fail LOUDLY at startup if required config is missing, instead of dying
deep in a request handler later.
"""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


def require_env(keys: list[str]) -> None:
    """Raise RuntimeError naming every missing required env var."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
