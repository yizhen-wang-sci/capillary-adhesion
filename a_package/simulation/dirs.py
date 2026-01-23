"""
Directory management.

Provides a flat container for runs with an index for sweep tracking.
"""

import json
import os
import subprocess
import shutil
import time
from pathlib import Path


def _get_git_hash() -> str | None:
    """Get current git commit hash."""
    try:
        hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return hash
    except Exception:
        return None


def _generate_run_id(with_hash: bool = True) -> str:
    """Generate a unique run ID based on timestamp and git hash."""
    current_time = time.localtime()
    run_id = time.strftime("%y%m%d-%H%M%S", current_time)

    if with_hash:
        git_hash = _get_git_hash()
        if git_hash:
            run_id += f"-{git_hash[:6]}"

    return run_id


class RunDir:
    """
    Structure of a single run directory.

    A run directory contains:
    - parameters/   Configuration files
    - results/      Simulation output data
    - visuals/      Generated plots and animations
    - log.txt       Execution log
    - METADATA.json Run metadata
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.run_id = self.path.name

    @property
    def results_dir(self) -> Path:
        return self.path / "results"

    @property
    def parameters_dir(self) -> Path:
        return self.path / "parameters"

    @property
    def visuals_dir(self) -> Path:
        return self.path / "visuals"

    @property
    def log_file(self) -> Path:
        return self.path / "log.txt"

    @property
    def metadata_file(self) -> Path:
        return self.path / "METADATA.json"

    def setup(self) -> None:
        """Create the directory structure."""
        self.path.mkdir(parents=True, exist_ok=False)
        self.results_dir.mkdir()
        self.parameters_dir.mkdir()
        self.visuals_dir.mkdir()
        self.log_file.touch()
        with open(self.metadata_file, "w", encoding="utf-8") as fp:
            json.dump({}, fp)

    def load_metadata(self) -> dict:
        """Load metadata from file."""
        with open(self.metadata_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def update_metadata(self, new_info: dict) -> None:
        """Update metadata file with new information."""
        metadata = self.load_metadata()
        metadata.update(new_info)
        with open(self.metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)

    def add_parameter_file(self, file_path: str | Path) -> None:
        """Copy a parameter file into the parameters directory."""
        shutil.copy2(file_path, self.parameters_dir)


class CaseDir:
    """
    Container for all runs in a case directory.

    Manages a flat collection of runs with an index file for sweep tracking.

    Directory structure:
        case_dir/
        ├── INDEX.json          # Tracks runs and sweeps
        ├── run-id-1/           # RunDir
        ├── run-id-2/           # RunDir
        └── ...

    Index format:
        {
            "runs": {
                "run-id-1": {},
                "run-id-2": {"sweep_id": "sweep-001", "sweep_index": 0}
            },
            "sweeps": {
                "sweep-001": {
                    "created": "2024-12-18T12:34:56",
                    "run_ids": ["run-id-2", "run-id-3"]
                }
            }
        }
    """

    def __init__(self, path: str | Path, data_root: str | Path | None = None):
        """
        Initialize a CaseDir.

        Parameters
        ----------
        path : str | Path
            Relative path for the case (e.g., "load_unload/tip-on-flat").
        data_root : str | Path | None
            Root directory for all simulation data. Defaults to "data/" in repo root.
        """
        if data_root is None:
            repo_root = os.getenv("PYTHONPATH", os.getcwd())
            data_root = Path(repo_root) / "data"

        self.case_dir = Path(data_root) / path
        self.index_file = self.case_dir / "INDEX.json"

    def _ensure_case_dir(self) -> None:
        """Create case directory if it doesn't exist."""
        self.case_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._save_index({"runs": {}, "sweeps": {}})

    def _load_index(self) -> dict:
        """Load the index file."""
        if not self.index_file.exists():
            return {"runs": {}, "sweeps": {}}
        with open(self.index_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _save_index(self, index: dict) -> None:
        """Save the index file."""
        with open(self.index_file, "w", encoding="utf-8") as fp:
            json.dump(index, fp, indent=2, sort_keys=True)

    def create_run(
        self,
        script_path: str | Path,
        *param_paths: str | Path,
        with_hash: bool = True,
    ) -> RunDir:
        """
        Create a new standalone run.

        Parameters
        ----------
        script_path : str | Path
            Path to the script creating this run (for metadata).
        *param_paths : str | Path
            Parameter files to copy into the run.
        with_hash : bool
            Whether to include git hash in run ID.

        Returns
        -------
        RunDir
            The created run directory.
        """
        self._ensure_case_dir()

        run_id = _generate_run_id(with_hash)
        run_dir = RunDir(self.case_dir / run_id)
        run_dir.setup()

        # Copy parameter files
        for param_path in param_paths:
            run_dir.add_parameter_file(param_path)

        # Set metadata
        metadata = {
            "run_id": run_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": str(Path(script_path).absolute()),
        }
        if param_paths:
            metadata["parameters"] = [str(Path(p).absolute()) for p in param_paths]
        if with_hash:
            git_hash = _get_git_hash()
            if git_hash:
                metadata["git_hash"] = git_hash
        run_dir.update_metadata(metadata)

        # Update index
        index = self._load_index()
        index["runs"][run_id] = {}
        self._save_index(index)

        return run_dir

    def create_sweep(
        self,
        script_path: str | Path,
        nb_runs: int,
        with_hash: bool = True,
    ) -> list[RunDir]:
        """
        Create multiple runs as part of a sweep.

        Note: Requires count upfront (from sweep.count_sweep_combinations).
        This coupling enables atomic sweep creation (all-or-nothing) and
        proper sweep grouping in INDEX.json.

        Parameters
        ----------
        script_path : str | Path
            Path to the script creating this sweep.
        nb_runs : int
            Number of runs in the sweep.
        with_hash : bool
            Whether to include git hash in run IDs.

        Returns
        -------
        list[RunDir]
            List of created run directories, in order.
        """
        self._ensure_case_dir()

        sweep_id = _generate_run_id(with_hash)
        sweep_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        git_hash = _get_git_hash() if with_hash else None

        run_dirs = []
        run_ids = []

        for sweep_index in range(nb_runs):
            # Create unique run_id by appending sweep index
            run_id = f"{sweep_id}--{sweep_index}"
            run_dir = RunDir(self.case_dir / run_id)
            run_dir.setup()

            # Set metadata
            metadata = {
                "run_id": run_id,
                "time": sweep_time,
                "script": str(Path(script_path).absolute()),
                "sweep_id": sweep_id,
                "sweep_index": sweep_index,
            }
            if git_hash:
                metadata["git_hash"] = git_hash
            run_dir.update_metadata(metadata)

            run_dirs.append(run_dir)
            run_ids.append(run_id)

        # Update index
        index = self._load_index()
        for i, run_id in enumerate(run_ids):
            index["runs"][run_id] = {"sweep_id": sweep_id, "sweep_index": i}
        index["sweeps"][sweep_id] = {
            "created": sweep_time,
            "run_ids": run_ids,
        }
        self._save_index(index)

        return run_dirs

    def get_run(self, run_id: str) -> RunDir:
        """
        Retrieve a run by ID.

        Parameters
        ----------
        run_id : str
            The run identifier.

        Returns
        -------
        RunDir
            The run directory.

        Raises
        ------
        KeyError
            If run_id is not found in the index.
        """
        index = self._load_index()
        if run_id not in index["runs"]:
            raise KeyError(f"Run '{run_id}' not found in index")
        return RunDir(self.case_dir / run_id)

    def get_sweep_runs(self, sweep_id: str) -> list[RunDir]:
        """
        Retrieve all runs in a sweep, ordered by sweep_index.

        Parameters
        ----------
        sweep_id : str
            The sweep identifier.

        Returns
        -------
        list[RunDir]
            List of run directories, sorted by sweep_index.

        Raises
        ------
        KeyError
            If sweep_id is not found.
        """
        index = self._load_index()
        if sweep_id not in index["sweeps"]:
            raise KeyError(f"Sweep '{sweep_id}' not found in index")

        run_ids = index["sweeps"][sweep_id]["run_ids"]
        return [RunDir(self.case_dir / rid) for rid in run_ids]

    def list_runs(self) -> list[RunDir]:
        """List all runs in this case directory."""
        index = self._load_index()
        return [RunDir(self.case_dir / rid) for rid in index["runs"]]

    def list_sweeps(self) -> list[str]:
        """List all sweep IDs in this case directory."""
        index = self._load_index()
        return list(index["sweeps"].keys())
