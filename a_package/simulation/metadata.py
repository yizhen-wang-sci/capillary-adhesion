"""
Metadata utilities for tracking simulation provenance.

Provides hashing and timestamping.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path


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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
