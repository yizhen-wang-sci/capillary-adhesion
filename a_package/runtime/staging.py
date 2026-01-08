"""
Run staging: prepare environment before simulation execution.

Owns infrastructure setup that must happen before each run:
- Log file switching (redirect output to run directory)
- Config persistence (via injected callback)
- Progress reporting (log run index in sweeps)

No config or simulation dependency - receives pre-expanded data from caller.

Usage:
    for run_dir in prepare_sweep(case_dir, nb_runs, __file__):
        persist_config(config, run_dir)
        run_simulation(config, run_dir)
"""

import logging
from pathlib import Path
from typing import Iterator

from .dirs import CaseDir, RunDir
from .logging import switch_log_file


logger = logging.getLogger(__name__)


def prepare_run(run_dir: RunDir) -> None:
    """
    Prepare infrastructure for a single run.

    Switches logging to the run's log file.

    Parameters
    ----------
    run_dir : RunDir
        The run directory.
    """
    switch_log_file(run_dir.log_file)


def prepare_sweep(
    case_dir: CaseDir,
    nb_runs: int,
    script_path: str | Path,
) -> Iterator[RunDir]:
    """
    Create and prepare all run directories for a sweep.

    Handles both single-run and multi-run cases.
    Yields RunDir objects ready for use.

    Parameters
    ----------
    case_dir : CaseDir
        Case directory for this sweep.
    nb_runs : int
        Number of runs to create.
    script_path : str | Path
        Path to the calling script.

    Yields
    ------
    RunDir
        Run directory for each run.
    """
    if nb_runs == 1:
        # Single run
        run_dir = case_dir.create_run(script_path, with_hash=True)
        prepare_run(run_dir)
        yield run_dir
        return

    # Parameter sweep - create all run dirs upfront
    run_dirs = case_dir.create_sweep(script_path, nb_runs, with_hash=True)

    for index, run_dir in enumerate(run_dirs):
        logger.info(f"Run #{index + 1} of {nb_runs}")
        prepare_run(run_dir)
        yield run_dir
