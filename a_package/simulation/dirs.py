"""
Directory management.

Provides CaseDir and RunDir for organizing simulation. Because one script with different parameter values can genearte
different results, one CaseDir contains multiple RunDir.
"""

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

from .metadata import compute_script_hash, get_iso_time


logger = logging.getLogger(__name__)


class CaseDir:
    """
    Container for all runs in a case directory.

    Directory structure:
        case_dir/
        ├── script.py
        ├── INDEX.json
        ├── run1_dir
        ├── run2_dir
        └── ...
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        self._path = Path(path).absolute()
        if self._path.is_file():
            raise FileExistsError(f"{self._path} is occupied by a file.")
        self._path.mkdir(parents=True, exist_ok=exist_ok)
        logger.info(f"Case directory path {self._path}")

    def __truediv__(self, other: str | Path):
        return self._path / other

    @property
    def index_file(self):
        return self._path / "INDEX.json"

    @property
    def script_file(self):
        return self._path / "script.py"

    @classmethod
    def alongside(
        cls,
        script_path: str | Path,
        case_name: str | None = None,
        parent_dir: str | Path | None = None,
        append_hash: bool = True,
    ):
        """
        Create or reuse a CaseDir alongside the given script.

        Places the case directory in the same folder as the script, named
        "script_name--hash". The hash ensures different script versions
        get separate directories. Copies the script into the case directory
        for reproducibility.

        Parameters
        ----------
        script_path : str | Path
            Path to the script file.
        case_name : str | None
            Override directory name. Defaults to script name (without extension).
        parent_dir : str | Path | None
            Override parent directory. Defaults to script's directory.
        append_hash : bool
            Append script content hash to case_name. Defaults to True.

        Returns
        -------
        CaseDir
            New or existing case directory.
        """
        script_path = os.path.abspath(script_path)
        script_dir, script_file = os.path.split(script_path)
        if case_name is None:
            case_name, _ = os.path.splitext(script_file)
        if parent_dir is None:
            parent_dir = script_dir
        if append_hash:
            nb_hex_digits = 6  # shall be fine for ~1000 versions
            case_name += f"--{compute_script_hash(script_path)[:nb_hex_digits]}"

        case_dir = cls(Path(parent_dir) / case_name)
        if not case_dir.script_file.exists():
            shutil.copy2(script_path, case_dir.script_file)
        return case_dir

    @contextlib.contextmanager
    def bookkeep(self):
        """
        Yield a dict for user to add information about specific runs. Time is automatically recorded.
        """
        entry = {}
        try:
            entry["time_start"] = get_iso_time()
            yield entry
        finally:
            entry["time_stop"] = get_iso_time()
            index = self._load_index()
            index.append(entry)
            self._save_index(index)

    def _load_index(self):
        """Load the index file."""
        try:
            with open(self.index_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return []

    def _save_index(self, index):
        """Save the index file."""
        with open(self.index_file, "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2, sort_keys=False)


class RunDir:
    """
    Working directory of a single run.

    Directory structure:
        run_dir/
        ├── input       Parameters and configuration 
        ├── data/       Simulation output data
        ├── visuals/    Generated plots and animations
        ├── log.txt     Execution log
        └── ...
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        self._path = Path(path).absolute()
        if self._path.is_file():
            raise FileExistsError(f"{self._path} is occupied by a file.")
        self._path.mkdir(parents=True, exist_ok=exist_ok)
        self.data_dir.mkdir(exist_ok=exist_ok)
        self.visuals_dir.mkdir(exist_ok=exist_ok)
        logger.info(f"Run directory path {self._path}")

    def __truediv__(self, other: str | Path):
        return self._path / other

    @property
    def input_file(self):
        return self._path / "input.cfg"

    @property
    def data_dir(self):
        return self._path / "data"

    @property
    def visuals_dir(self):
        return self._path / "visuals"

    @property
    def log_file(self):
        return self._path / "log.txt"
