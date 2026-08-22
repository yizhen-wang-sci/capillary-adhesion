"""Logging configuration, called by scripts (including conftest.py)."""

import contextlib
import logging
import sys
from pathlib import Path

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_FORMAT_BRIEF = "[%(asctime)s] %(message)s"
_FORMAT_VERBOSE = "[%(asctime)s][%(name)s::%(funcName)s#L%(lineno)d] %(levelname)s: %(message)s"

# Logger names this module has personally touched (level and/or a file handler)
# via `modules`, so a later `setup_logging` call can undo exactly what a
# previous one did before applying its own.
_adjusted_loggers: set[str] = set()


class _StreamToLogger:
    """File-like shim that redirects writes (e.g. from `print`) into a logger."""

    def __init__(self, logger: logging.Logger, level: int):
        """Bind the shim to a logger and a level.

        Args:
            logger: Where the writes are sent.
            level: Level each write is logged at.
        """
        self._logger = logger
        self._level = level

    def write(self, message: str) -> None:
        """Log `message`, dropping it if it is blank once stripped.

        Args:
            message: The text to log.
        """
        message = message.rstrip()
        if message:
            # stacklevel=2: attribute the record to print()'s caller, not to write() itself
            self._logger.log(self._level, message, stacklevel=2)

    def flush(self) -> None:
        """Accept `flush` to complete the file-like interface."""


def setup_logging(test: bool = False, file: str | Path | None = None, *, modules: list[str] | None = None):
    """Configure logging. Safe to call multiple times.

    Args:
        test: Verbose console format and level DEBUG. Default False: brief format, level
            INFO.
        file: Path to append everything to, always verbose. Appended to, never truncated.
            Default None: console only.
        modules: Names of the loggers that DEBUG level and the file are scoped to. Default
            None: the whole app. A scope set by a previous call and not repeated here is
            undone first.
    """
    root = logging.getLogger()
    print_logger = logging.getLogger("print")

    # Reset root's, the print-logger's, and any previously-touched module
    # loggers' handlers (their file handler may differ from this call's).
    for logger in (root, print_logger, *(logging.getLogger(name) for name in _adjusted_loggers)):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
    print_logger.propagate = False

    # Undo any previous call's per-module DEBUG level overrides
    for name in _adjusted_loggers:
        logging.getLogger(name).setLevel(logging.NOTSET)

    # Level: `modules` only matters here when `test` is True — it scopes
    # DEBUG to those loggers instead of the whole app.
    if test and modules:
        root.setLevel(logging.INFO)
        for name in modules:
            logging.getLogger(name).setLevel(logging.DEBUG)
    elif test:
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # Console handlers: ordinary loggers -> stderr, print() -> the real stdout
    console_fmt = logging.Formatter(_FORMAT_VERBOSE if test else _FORMAT_BRIEF, datefmt=_TIMESTAMP_FORMAT)

    stderr_console = logging.StreamHandler(sys.stderr)
    stderr_console.setFormatter(console_fmt)
    root.addHandler(stderr_console)

    stdout_console = logging.StreamHandler(sys.__stdout__)
    stdout_console.setFormatter(console_fmt)
    print_logger.addHandler(stdout_console)

    # File: always verbose, always appends. `print()` always gets it; the
    # rest of the app gets it either via root (whole app) or, when `modules`
    # is given, only via those named loggers (a scoped file).
    if file is not None:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_fmt = logging.Formatter(_FORMAT_VERBOSE, datefmt=_TIMESTAMP_FORMAT)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(file_fmt)
        print_logger.addHandler(file_handler)
        if modules:
            for name in modules:
                logging.getLogger(name).addHandler(file_handler)
        else:
            root.addHandler(file_handler)

    # Make `print` show up formatted on stdout (and in the file, if any)
    sys.stdout = _StreamToLogger(print_logger, logging.INFO)

    _adjusted_loggers.clear()
    _adjusted_loggers.update(modules or [])
