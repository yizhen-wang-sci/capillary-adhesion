"""
Run staging: prepare environment before simulation execution.

Owns infrastructure setup that must happen before each run:
- Log file switching (redirect output to run directory)
- Config persistence (save expanded config for reproducibility)
- Progress reporting (log run index in sweeps)

Usage:
    for run_dir, config in prepare_sweep(case_dir, config, __file__):
        run_simulation(config, run_dir)
"""

import logging
from pathlib import Path
from typing import Iterator

from a_package.config import Config, save_config
from a_package.sweep import expand_sweeps, count_sweep_combinations

from .dirs import CaseDir, RunDir
from .logging import switch_log_file


logger = logging.getLogger(__name__)


def prepare_run(run_dir: RunDir, config: Config) -> None:
    """
    Prepare infrastructure for a single run.

    Parameters
    ----------
    run_dir : RunDir
        The run directory.
    config : Config
        Configuration to save.
    """
    switch_log_file(run_dir.log_file)
    save_config(config, run_dir.parameters_dir / "config.toml")


def prepare_sweep(
    case_dir: CaseDir,
    config: Config,
    script_path: str | Path,
) -> Iterator[tuple[RunDir, Config]]:
    """
    Create and prepare all runs for a sweep.

    Handles both single-run and multi-run cases.
    Yields (run_dir, config) pairs ready for execution.

    Parameters
    ----------
    case_dir : CaseDir
        Case directory for this sweep.
    config : Config
        Configuration, possibly with sweep definitions.
    script_path : str | Path
        Path to the calling script.

    Yields
    ------
    tuple[RunDir, Config]
        Pairs of (run_dir, expanded_config) for each run.
    """
    nb_configs = count_sweep_combinations(config)

    if nb_configs == 1:
        # Single run
        run_dir = case_dir.create_run(script_path, with_hash=True)
        prepare_run(run_dir, config)
        yield run_dir, config
        return

    # Parameter sweep - create all run dirs upfront
    run_dirs = case_dir.create_sweep(script_path, nb_configs, with_hash=True)

    for index, (run_dir, expanded_config) in enumerate(zip(run_dirs, expand_sweeps(config))):
        logger.info(f"Run #{index + 1} of {nb_configs}")
        prepare_run(run_dir, expanded_config)
        yield run_dir, expanded_config
