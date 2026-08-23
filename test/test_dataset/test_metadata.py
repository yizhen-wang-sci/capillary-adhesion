"""Tests for the provenance metadata helpers."""

import subprocess
from unittest import mock

import pytest

from a_package.dataset import metadata


def test_git_hash_of_a_modified_package_differs_from_a_clean_one():
    rev_parse = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr="")
    status = {
        "clean": subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        "dirty": subprocess.CompletedProcess(args=[], returncode=0, stdout=" M a_package/dataset/dirs.py\n", stderr=""),
    }

    hashes = {}
    for state, reported in status.items():
        with mock.patch.object(subprocess, "run", side_effect=[reported, rev_parse]):
            hashes[state] = metadata.get_git_hash()

    assert hashes["clean"] != hashes["dirty"]


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(returncode=128, cmd=["git"]),
        subprocess.TimeoutExpired(cmd=["git"], timeout=30),
        FileNotFoundError("git"),
    ],
)
def test_git_hash_is_none_when_a_command_fails(error, caplog):
    with mock.patch.object(subprocess, "run", side_effect=error):
        assert metadata.get_git_hash() is None
    assert "Failed to retrieve Git commit hash" in caplog.text
