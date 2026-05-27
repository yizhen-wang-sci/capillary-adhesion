"""
Logging configuration.

Single source of truth for log format and handler setup.

Library modules just do:
    import logging
    logger = logging.getLogger(__name__)

Only application entry points (CLI main, conftest.py, scripts) should
call `setup_logging`. Never call it from library code at import time.
"""

import logging
import sys
from pathlib import Path

_FORMAT_MESSAGE_ONLY = "%(message)s"
_FORMAT_VERBOSE = "[%(levelname)s][%(name)s::%(funcName)s#L%(lineno)d] %(message)s"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None, *,
                  console_verbose: bool | None = None, file_verbose: bool = True, file_mode: str = "w", ):
    """Configure the root logger. Safe to call multiple times.

    Default format:
      - console: just the message; bumps to verbose when level=DEBUG
      - file: verbose

    Parameters
    ----------
    level
        Log level for the root logger. Accepts int (logging.DEBUG) or
        str ("DEBUG", "info", ...).
    log_file
        If given, also write to this file. None = console only.
    console_verbose
        If None (default), use verbose format on console iff level <= DEBUG.
        Pass True/False to force.
    file_verbose
        Format for the file handler. Default True (always detailed).
    file_mode
        "w" to overwrite each run, "a" to append.
    """
    # Reset the root logger
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    # Set level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown log level: {level!r}")
        level = resolved
    root.setLevel(level)

    # Add stream handler
    if console_verbose is None:
        console_verbose = level <= logging.DEBUG
    console_fmt = logging.Formatter(_FORMAT_VERBOSE if console_verbose else _FORMAT_MESSAGE_ONLY)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # Add file handler if provided
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
        file_fmt = logging.Formatter(_FORMAT_VERBOSE if file_verbose else _FORMAT_MESSAGE_ONLY)
        file.setFormatter(file_fmt)
        root.addHandler(file)


# A utility for cranking one subsystem's log level without touching the rest.
def set_module_level(name: str, level: int | str) -> None:
    """Set log level for one logger (and its descendants) for debugging.

    Example:
        set_module_level("mypkg.solver", logging.DEBUG)
    """
    logging.getLogger(name).setLevel(level)
