"""
Logging configuration.

Used in scripts (including conftest.py) to configure logging via
`setup_logging`. Library modules should simply do:

    import logging
    logger = logging.getLogger(__name__)
"""

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
    """File-like shim that redirects writes (e.g. from ``print``) into a logger.

    Needed because ``print`` bypasses the logging module entirely, so its
    output never reaches a file handler set up via `setup_logging`.
    """

    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level

    def write(self, message: str) -> None:
        message = message.rstrip()
        if message:
            # stacklevel=2: attribute the record to print()'s caller, not to write() itself
            self._logger.log(self._level, message, stacklevel=2)

    def flush(self) -> None:
        pass


def setup_logging(test: bool = False, file: str | Path | None = None, *,
                  modules: list[str] | None = None):
    """Configure logging. Safe to call multiple times. See the behavior matrix
    below for exactly what each argument combination does.

    Parameters
    ----------
    test
        Verbose console format and raises level to DEBUG (scoped to `modules`
        if given). Default False: brief format, level INFO.
    file
        Path to also append everything to, always verbose. Default None:
        console only. Attached to root, or to `modules` if given (scoped).
    modules
        Names of loggers to scope DEBUG level and/or file attachment to.
        Any override from a *previous* call that isn't repeated here is
        undone first, so each call's effect is self-contained.

    Notes
    -----
    The file handler always appends (never truncates), since the same `file`
    path may be opened by more than one process (e.g. every MPI rank calling
    `setup_logging`, not just rank 0): truncating on open would let whichever
    process opens last silently wipe out what earlier ones already wrote.
    Remove the file (or use a new path) first if a fresh log is wanted.

    `print()` always lands on the real stdout, formatted per the console
    format, in every combination below — `file`/`modules` only ever add a
    *second* destination (the file) for it, never redirect it away from
    stdout.

    Behavior matrix (test / file / modules, each given or not):

        test   file  modules | level                          | console | file contents
        -----  ----  ------- | ------------------------------ | ------- | --------------------------------------------
        false   -      -     | root INFO                      | brief   | (no file)
        false   -      yes   | root INFO                      | brief   | (no file)
        false  yes     -     | root INFO                      | brief   | everything (INFO+) + print, verbose
        false  yes     yes   | root INFO                      | brief   | only named modules' INFO+ + print, verbose
        true    -      -     | root DEBUG                     | verbose | (no file)
        true    -      yes   | named modules DEBUG, root INFO | verbose | (no file)
        true   yes     -     | root DEBUG                     | verbose | everything (DEBUG+) + print, verbose
        true   yes     yes   | named modules DEBUG, root INFO | verbose | only named modules' DEBUG+ + print, verbose
    """
    root = logging.getLogger()
    print_logger = logging.getLogger("print")

    # Reset root's, the print-logger's, and any previously-touched module
    # loggers' handlers (their file handler may differ from this call's).
    for logger in (root, print_logger, *(logging.getLogger(name) for name in _adjusted_loggers)):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
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
