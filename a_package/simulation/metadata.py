"""Metadata utilities for tracking simulation provenance."""

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_script_hash(script_path: Path | str):
    """Compute SHA256 hash of script file content.

    Args:
        script_path: The script to hash, read as bytes.

    Returns:
        The hash as a hex string.
    """
    content = Path(script_path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def compute_config_hash(config: dict):
    """Compute SHA256 hash of config dict.

    Args:
        config: The configuration to hash. Keys are sorted first, so two dicts that differ
            only in key order hash alike.

    Returns:
        The hash as a hex string.
    """
    # sort keys for a consistent hash
    content = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def get_timestamp():
    """Generate timestamp string (YYMMDD-HHMMSS).

    Returns:
        The local time, in a form usable within a file or directory name.
    """
    return time.strftime("%y%m%d-%H%M%S", time.localtime())


def get_iso_time():
    """Generate ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS).

    Returns:
        The local time, carrying no timezone offset.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def get_git_hash():
    """Retrieve the current Git commit hash of the repository.

    Returns:
        The hash of the latest commit, with "-dirty" appended if the package directory has
        uncommitted changes, which is also logged as a warning. None if `git` is missing or
        a command failed.
    """
    package_root = Path(__file__).parent.parent.resolve()
    extra_args = {
        "capture_output": True,  # capture stdout and stderr
        "text": True,  # decode to str
        "cwd": package_root,  # run in package root
        "timeout": 30,  # seconds
    }

    try:
        # Print information if there are uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain", "."], check=True, **extra_args)
        is_dirty = len(result.stdout.splitlines()) > 0
        if is_dirty:
            logger.warning(f"Uncommitted changes in the package\n{result.stdout}\n")
        # Get hash of the latest commit
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, **extra_args)
        commit_hash = result.stdout.strip()
        if is_dirty:
            commit_hash += "-dirty"
        return commit_hash
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to retrieve Git commit hash.")
        return None
