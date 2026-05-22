"""
Metadata utilities for tracking simulation provenance.

Provides hashing and timestamping.
"""

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path


logger = logging.getLogger(__name__)


def compute_script_hash(script_path: Path | str):
    """Compute SHA256 hash of script file content."""
    content = Path(script_path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def compute_config_hash(config: dict):
    """Compute SHA256 hash of config dict.

    Note: TOML datetime values are not JSON serializable. Use string dates in config.
    """
    # sort keys for a consistent hash
    content = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def get_timestamp():
    """Generate timestamp string (YYMMDD-HHMMSS)."""
    return time.strftime("%y%m%d-%H%M%S", time.localtime())


def get_iso_time():
    """Generate ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def get_git_hash():
    """Get current git commit hash, or None if not in a git repo."""
    package_root = Path(__file__).parent.parent.resolve()
    extra_args = dict(capture_output=True,  # capture stdout and stderr
                      text=True,            # decode to str
                      cwd=package_root,     # run in package root
                      check=True)           # raise error if command fails

    try:
        # Print warning information if there are uncommitted changes
        result = subprocess.run(["git", "status", "--porcelain"], **extra_args)
        if result.stdout:
            logger.warning("WARNING: Uncommitted changes detected. This may affect reproducibility.")
        # Get current git commit hash
        result = subprocess.run(["git", "rev-parse", "HEAD"], **extra_args)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning("WARNING: Not in a git repository.")
        return None
