"""The console a pass writes to: log records on stderr, `print` on stdout."""

import logging
import sys

from a_package.dataset import brief_format

_PRINT_LOGGER = "print"

CONSOLE_LOGGERS = ("", _PRINT_LOGGER)
"""Names of the loggers the console writes."""


class _StreamToLogger:
    """File-like shim that sends writes into a logger."""

    def __init__(self, logger: logging.Logger, level: int):
        """Bind the shim to a logger and a level.

        Args:
            logger: Where the writes are sent.
            level: Level each write is logged at.
        """
        self._logger = logger
        self._level = level

    def write(self, message: str) -> None:
        """Log `message`, dropping it if it is blank once stripped."""
        message = message.rstrip()
        if message:
            # stacklevel=2: attribute the record to print()'s caller, not to write() itself
            self._logger.log(self._level, message, stacklevel=2)

    def flush(self) -> None:
        """Part of the file-like interface. The logger flushes."""


def setup_console():
    """Send log records at INFO and above to stderr, and `print` to stdout.

    Binds `sys.stdout` to the `print` logger, which does not propagate. Replaces the
    handlers a previous call added.
    """
    root = logging.getLogger()
    printed = logging.getLogger(_PRINT_LOGGER)

    for logger in (root, printed):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    root.setLevel(logging.INFO)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(brief_format())
    root.addHandler(stderr)

    printed.propagate = False
    stdout = logging.StreamHandler(sys.__stdout__)
    stdout.setFormatter(brief_format())
    printed.addHandler(stdout)

    sys.stdout = _StreamToLogger(printed, logging.INFO)
