import logging

import pytest

from a_package.simulation import setup_logging


@pytest.fixture(autouse=True)
def _configure_logging(tmp_path):
    setup_logging(level=logging.DEBUG)
