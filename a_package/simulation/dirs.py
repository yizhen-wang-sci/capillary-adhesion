"""
Directory management.

Provides CaseDir and RunDir for organizing simulation.
"""

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

from NuMPI import MPI
_comm = MPI.COMM_WORLD

from .metadata import compute_script_hash, get_iso_time


logger = logging.getLogger(__name__)


class _Dir:
    """Just a directory.
    
    Creates the directory if not exist. Supports using '/' like a path.
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        self._path = Path(path).absolute()
        if _comm.rank == 0:
            if self._path.is_file():
                raise FileExistsError(f"{self._path} is occupied by a file.")
            try:
                self._path.mkdir(parents=True)
                logger.info(f"Create directory at {self._path}")
            except FileExistsError as err:
                if not exist_ok:
                    raise err
        _comm.barrier()

    def __truediv__(self, other: str | Path):
        return self._path / other

    @property
    def path(self):
        return self._path


class CaseDir(_Dir):
    """
    A directory to hold scripts. Supports dumping information of a run and script 
    versioning.

    Directory structure:
        case_dir/
        ├── script.py, ...  One or more script files.
        ├── INDEX.json      A file for script to dump information.
        ├── version.json    A file for use with script versioning.
        └── ...             Any other stuff
    """

    @property
    def run_index(self):
        return self._path / "INDEX.json"

    @property
    def version_index(self):
        return self._path / "version.json"

    @contextlib.contextmanager
    def bookkeep(self):
        """
        Yield a dict to add information during a run. Time is automatically recorded.
        """
        entry = {}
        try:
            entry["time_start"] = get_iso_time()
            yield entry
        finally:
            entry["time_stop"] = get_iso_time()
            index = self._load_run_index()
            index.append(entry)
            self._save_run_index(index)

    def _load_run_index(self):
        try:
            with open(self.run_index, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return []

    def _save_run_index(self, index):
        with open(self.run_index, "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2, sort_keys=False)


    def copy_script(self, src, version: bool = False):
        """Copy script to case directory.

        If version=True, appends version suffix (e.g. --v1, --v2) based on
        script content hash. Same content gets same version number. Scripts with
        different filenames are versioned separately.
        """
        src = os.path.abspath(src)
        _, script_file = os.path.split(src)
        script_name, script_ext = os.path.splitext(script_file)

        if version:
            nb_hex_digits = 8
            hash_hex = compute_script_hash(src)[:nb_hex_digits]
            version_number = self._get_or_create_version(script_file, hash_hex)
            script_name += f"--v{version_number}"

        dst = self._path / f"{script_name}{script_ext}"
        return shutil.copy2(src, dst)

    def _get_or_create_version(self, script_file: str, hash_hex: str):
        index = self._load_version_index()
        versions = index.setdefault(script_file, {})
        if hash_hex in versions:
            return versions[hash_hex]["version"]
        # if new, create an incremental one
        version_number = len(versions) + 1
        versions[hash_hex] = {"version": version_number, "created": get_iso_time()}
        self._save_version_index(index)
        return version_number

    def _load_version_index(self):
        try:
            with open(self.version_index, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return {}

    def _save_version_index(self, index):
        with open(self.version_index, "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2)


class RunDir(_Dir):
    """
    A directory to work with simulation runs. Specifies some convention about file or
    folder names to hold input, output (separate data & visuals) and log.

    Directory structure:
        run_dir/
        ├── input.cfg   Parameters and configuration 
        ├── data/       Simulation output data
        ├── visuals/    Generated plots and animations
        ├── log.txt     Execution log
        └── ...         Any other stuff
    """

    def __init__(self, path: str | Path, exist_ok: bool = True):
        super().__init__(path, exist_ok)
        if _comm.rank == 0:
            self.data_dir.mkdir(exist_ok=exist_ok)
            self.visuals_dir.mkdir(exist_ok=exist_ok)
        _comm.barrier()

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
