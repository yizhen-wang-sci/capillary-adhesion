"""
Directory management for organizing simulations.

Structure:
ROOT
├─ case1
│ ├─ run1
│ │ ├─ metadata.json
│ │ ├─ *.* (Files for simulation, can be script, config, figure, animation, etc.)
│ │ ├─ record1
│ │ │ ├─ input.cfg
│ │ │ ├─ data/
│ │ │ ├─ log.txt
│ │ │ └─ ...
│ │ ├─ recordX/
│ │ └─ ...
│ ├─ runX/
│ └─ ...
├─ caseX/
└─ ...

Usage:
    case = CaseDir("cases/my_setup")
    run = case.new_run("to_do", notes="first try")
    # or run = case.continue_run("to_do", attempt=1)
    run.add_metadata({"cmd": sys.argv})
    run.copy_file(__file__)
    rin, rdata, rlog = work.access_record("run-01", create_new=True)
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

from .metadata import get_git_hash, get_iso_time


logger = logging.getLogger(__name__)


class _Dir:
    """Just a directory path with some convenience methods.

    Creates the directory if not exist. Supports using '/' like a path.
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        self._path = Path(path).resolve()
        if self._path.is_file():
            raise FileExistsError(f"{self._path} is occupied by a file.")
        try:
            self._path.mkdir(parents=True)
            logger.info(f"Create directory at {self._path}")
        except FileExistsError as err:
            if not exist_ok:
                raise err

    def __truediv__(self, other: str | Path):
        return self._path / other

    def __fspath__(self):
        return str(self._path)


class CaseDir(_Dir):
    """
    The top-level directories. Each case shall have a distinct physical setup. It
    contains multiple subdirectories organized by task name and attempt number
    to keep track of progress.
    """

    name_pattern = re.compile(r"([\w-]+)--(\d+)(--[\S ]+)?")
    """
    Pattern to match work directory names: '{task}--{attempt}--{notes}', where '--{notes}' is optional.
    """

    def new_run(self, task: str, notes: str = ""):
        task = self._format_str(task)

        attempts = []
        for entry in self._path.iterdir():
            match = self.name_pattern.fullmatch(entry.name)
            if match and match.group(1) == task:
                attempt = int(match.group(2))
                attempts.append(attempt)
        nb_attempts = max(attempts, default=0)
        next_attempt = nb_attempts + 1

        dir_name = "--".join([task, self._format_num(next_attempt), notes])
        return RunDir(self._path / dir_name, exist_ok=False)

    def continue_run(self, task: str, attempt: int):
        task = self._format_str(task)

        for entry in self._path.iterdir():
            match = self.name_pattern.fullmatch(entry.name)
            if match and match.group(1) == task and int(match.group(2)) == attempt:
                return RunDir(entry, exist_ok=True)

        raise FileNotFoundError(f"No directory found for task '{task}' and attempt {self._format_num(attempt)}")

    @staticmethod
    def _format_str(s: str) -> str:
        return s.strip().casefold().replace(" ", "-")

    @staticmethod
    def _format_num(n: int) -> str:
        return f"{n:02d}"


class RunDir(_Dir):
    """
    A directory to store self-contained simulation, including files (scripts, configs, run dumps, etc.),
    necessary metadata (git hash of the code, etc.) and run records (as subdirectories).
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        super().__init__(path, exist_ok)

    def add_metadata(self, new: dict):
        metadata_file = self._path / "metadata.json"

        try:
            with open(metadata_file, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        except FileNotFoundError:
            metadata = {}

        metadata.update(new)

        with open(metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=False)

    def copy_file(self, src):
        src = os.path.abspath(src)
        _, file_name = os.path.split(src)
        dst = self._path / f"{file_name}"
        return shutil.copy2(src, dst)

    def access_record(self, name: str, create_new: bool = False):
        record = RecordDir(self._path / name, exist_ok=not create_new)
        return record.input_file, record.data_dir, record.log_file


class RecordDir(_Dir):
    """
    A directory to record the standard IO (inputs, data, logs) during simulation runs.
    It defines the layout and naming convention for those records.
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        super().__init__(path, exist_ok)
        self.data_dir.mkdir(exist_ok=exist_ok)

    @property
    def input_file(self):
        return self._path / "input.cfg"

    @property
    def data_dir(self):
        return self._path / "data"

    @property
    def log_file(self):
        return self._path / "log.txt"
