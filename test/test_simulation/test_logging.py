"""
Tests for `setup_logging` (see its docstring for the full behavior matrix).

Covers 5 of the 8 (test, file, modules) combinations; `test=True` + `file`
isn't tested separately since it's the union of what (test, modules) and
(file, modules) each verify independently.
"""

import logging
import re

from a_package.simulation.logging import setup_logging

_BRIEF_LINE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] .+$", re.MULTILINE)
_VERBOSE_LINE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[[\w.]+::\w+#L\d+\] \w+: .+$", re.MULTILINE)


def _assert_print_only_on_stdout(out, err, needle="print msg"):
    """print()'s usual stdout stream must never be hijacked, in any configuration."""
    assert needle in out
    assert needle not in err


def test_no_arguments_only_affects_print_formatting(capfd):
    """No arguments: print() is formatted, but stays on stdout; nothing else changes."""
    setup_logging()
    logging.getLogger("mod").info("logger info msg")
    print("print msg")

    out, err = capfd.readouterr()

    _assert_print_only_on_stdout(out, err)
    assert _BRIEF_LINE.search(out)

    assert "logger info msg" in err
    assert _BRIEF_LINE.search(err)


def test_test_true_no_modules_sets_root_debug_and_verbose_console(capfd):
    """`test=True`, no `modules`: root goes to DEBUG, both consoles go verbose."""
    setup_logging(test=True)
    logging.getLogger("mod").debug("logger debug msg")
    print("print msg")

    out, err = capfd.readouterr()

    _assert_print_only_on_stdout(out, err)
    assert _VERBOSE_LINE.search(out)

    assert "logger debug msg" in err
    assert _VERBOSE_LINE.search(err)


def test_test_true_with_modules_scopes_debug_to_named_loggers(capfd):
    """`test=True` + `modules`: only the named loggers go to DEBUG; root stays INFO."""
    setup_logging(test=True, modules=["scoped.module"])
    logging.getLogger("scoped.module").debug("scoped debug msg")
    logging.getLogger("other.module").debug("other debug msg (must not show)")
    print("print msg")

    out, err = capfd.readouterr()

    _assert_print_only_on_stdout(out, err)
    assert _VERBOSE_LINE.search(out)  # console format still follows `test`, regardless of `modules`

    assert "scoped debug msg" in err
    assert "other debug msg" not in err


def test_file_given_no_modules_attaches_file_to_root(tmp_path, capfd):
    """`file`, no `modules`: file handler on root captures everything, verbose."""
    log_path = tmp_path / "run.log"
    setup_logging(file=log_path)
    logging.getLogger("mod").info("logger info msg")
    print("print msg")

    out, err = capfd.readouterr()
    file_content = log_path.read_text()

    _assert_print_only_on_stdout(out, err)
    assert _BRIEF_LINE.search(out)  # console format unaffected by `file`

    assert "logger info msg" in err
    assert file_content.count("logger info msg") == 1
    assert file_content.count("print msg") == 1
    assert _VERBOSE_LINE.search(file_content)


def test_file_given_with_modules_scopes_file_to_named_loggers(tmp_path, capfd):
    """`file` + `modules`: file handler attaches only to the named loggers, not root."""
    log_path = tmp_path / "run.log"
    setup_logging(file=log_path, modules=["scoped.module"])
    logging.getLogger("scoped.module").info("scoped info msg")
    logging.getLogger("other.module").info("other info msg (console only, not in file)")
    print("print msg")

    out, err = capfd.readouterr()
    file_content = log_path.read_text()

    _assert_print_only_on_stdout(out, err)
    assert _BRIEF_LINE.search(out)

    # both still show on console (stderr), since root's console handler sees everything
    assert "scoped info msg" in err
    assert "other info msg" in err

    # only the named module's messages (and print) land in the file — not "other.module"'s
    assert "scoped info msg" in file_content
    assert "print msg" in file_content
    assert "other info msg" not in file_content
    assert _VERBOSE_LINE.search(file_content)
