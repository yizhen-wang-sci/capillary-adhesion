"""A record's log kept as a text file."""

import contextlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def brief_format() -> logging.Formatter:
    """Build the formatter writing when a record was made, and what it says."""
    return logging.Formatter("[%(asctime)s] %(message)s", datefmt=_TIMESTAMP_FORMAT)


def verbose_format() -> logging.Formatter:
    """Build the formatter writing where a record came from and how severe it is as well."""
    return logging.Formatter(
        "[%(asctime)s][%(name)s::%(funcName)s#L%(lineno)d] %(levelname)s: %(message)s",
        datefmt=_TIMESTAMP_FORMAT,
    )


def open_log_file(path: str | os.PathLike, *, formatter: logging.Formatter | None = None) -> logging.FileHandler:
    """Open a log file for appending, creating the directories above it.

    Args:
        path: The file. Appended to, never truncated.
        formatter: How each record is written. Defaults to `verbose_format()`.

    Returns:
        The handler, for the caller to add to the loggers it chooses and to close.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(verbose_format() if formatter is None else formatter)
    return handler


@contextlib.contextmanager
def log_into(path: str | os.PathLike, *, loggers: Sequence[str] = ("",), formatter: logging.Formatter | None = None):
    """Send what the named loggers emit inside the block into a log file.

    Args:
        path: The file. Appended to, never truncated.
        loggers: Names of the loggers writing into it. Defaults to the root logger alone.
        formatter: How each record is written. Defaults to `verbose_format()`.

    Yields:
        The handler, so the block can raise or lower its level.

    Raises:
        ValueError: If one of the loggers propagates into another, which would write every
            record of it twice.
    """
    _refuse_propagating_into_each_other(loggers)
    handler = open_log_file(path, formatter=formatter)
    attached = [logging.getLogger(name) for name in loggers]
    for logger in attached:
        logger.addHandler(handler)
    try:
        yield handler
    finally:
        for logger in attached:
            logger.removeHandler(handler)
        handler.close()


def _refuse_propagating_into_each_other(names: Sequence[str]):
    """Check that no logger of the given names reaches another one by propagation.

    Args:
        names: Names of the loggers about to share a handler.

    Raises:
        ValueError: If one propagates into another.
    """
    named = {logging.getLogger(name) for name in names}
    for name in names:
        logger = logging.getLogger(name)
        while logger.propagate and logger.parent is not None:
            logger = logger.parent
            if logger in named:
                raise ValueError(f"Logger {name!r} propagates into logger {logger.name!r}.")
