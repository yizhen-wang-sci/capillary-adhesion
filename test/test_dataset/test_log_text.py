"""Tests for a record's log kept as a text file."""

import logging

import pytest

from a_package.dataset.log_text import log_into, open_log_file


@pytest.fixture
def logger(request):
    log = logging.getLogger(f"test_log_text.{request.node.name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    yield log
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


def test_a_handler_uses_the_formatter_it_was_given(tmp_path, logger):
    formatter = logging.Formatter("%(levelname)s says %(message)s")
    path = tmp_path / "log"
    logger.addHandler(open_log_file(path, formatter=formatter))
    logger.info("what happened")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "what happened", None, None)
    assert path.read_text() == f"{formatter.format(record)}\n"


def test_a_log_file_is_appended_to_never_truncated(tmp_path, logger):
    path = tmp_path / "log"
    for message in ("first", "second"):
        handler = open_log_file(path)
        logger.addHandler(handler)
        logger.info(message)
        logger.removeHandler(handler)
        handler.close()
    assert [line.split(": ")[-1] for line in path.read_text().splitlines()] == ["first", "second"]


def test_the_directories_above_a_log_file_are_created(tmp_path):
    handler = open_log_file(tmp_path / "run" / "theta=30" / "log")
    handler.close()
    assert (tmp_path / "run" / "theta=30" / "log").is_file()


def test_a_scope_takes_only_what_the_block_emits(tmp_path, logger):
    logger.info("before")
    with log_into(tmp_path / "log", loggers=(logger.name,)):
        logger.info("inside")
    logger.info("after")
    assert [line.split(": ")[-1] for line in (tmp_path / "log").read_text().splitlines()] == ["inside"]


def test_a_scope_hands_the_logger_back_as_it_found_it(tmp_path, logger):
    before = list(logger.handlers)
    with pytest.raises(RuntimeError), log_into(tmp_path / "log", loggers=(logger.name,)):
        logger.info("inside")
        raise RuntimeError("the simulation blew up")
    assert logger.handlers == before
    assert (tmp_path / "log").read_text().endswith("inside\n")


def test_two_scopes_keep_their_records_apart(tmp_path, logger):
    for name in ("theta=30", "theta=60"):
        with log_into(tmp_path / name, loggers=(logger.name,)):
            logger.info(f"solving {name}")
    for name in ("theta=30", "theta=60"):
        assert [line.split(": ")[-1] for line in (tmp_path / name).read_text().splitlines()] == [f"solving {name}"]


def test_loggers_that_reach_each_other_are_refused(tmp_path, logger):
    child = logging.getLogger(f"{logger.name}.child")
    child.propagate = True
    with pytest.raises(ValueError), log_into(tmp_path / "log", loggers=(logger.name, child.name)):
        pass
    assert not (tmp_path / "log").exists()
