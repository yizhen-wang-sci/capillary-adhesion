"""
Sweep data extraction utilities.

Extract and aggregate data from multiple runs in a parameter sweep.
"""

from typing import Callable, Any

import numpy as np

from a_package.config import load_config
from a_package.runtime import CaseDir, RunDir
from a_package.simulation import SimulationIO, Term


def extract_from_sweep(
    case_dir: CaseDir,
    sweep_id: str,
    extractor: Callable[[RunDir], Any],
) -> list[Any]:
    """
    Extract data from each run in a sweep.

    Parameters
    ----------
    case_dir : CaseDir
        The case directory containing the sweep.
    sweep_id : str
        The sweep identifier.
    extractor : Callable[[RunDir], Any]
        Function that takes a RunDir and returns extracted data.

    Returns
    -------
    list[Any]
        List of extracted values, one per run.
    """
    run_dirs = case_dir.get_sweep_runs(sweep_id)
    return [extractor(run_dir) for run_dir in run_dirs]


def get_config_value(run_dir: RunDir, path: str) -> Any:
    """
    Get a value from a run's saved config using dot notation.

    Parameters
    ----------
    run_dir : RunDir
        The run directory.
    path : str
        Dot-separated path to the config value.
        Example: "problem.capillary.contact_angle_degree"

    Returns
    -------
    Any
        The config value.
    """
    config = load_config(run_dir.parameters_dir / "config.toml")
    parts = path.split(".")
    obj = getattr(config, parts[0])
    for part in parts[1:]:
        obj = obj[part]
    return obj


def get_trajectory_value(
    run_dir: RunDir,
    grid,
    term: Term,
    step_index: int = -1,
) -> float:
    """
    Get a single trajectory value from a run's results.

    Parameters
    ----------
    run_dir : RunDir
        The run directory.
    grid : Grid
        The computational grid.
    term : Term
        The data term to extract.
    step_index : int
        Which step to extract (-1 for last).

    Returns
    -------
    float
        The trajectory value.
    """
    io = SimulationIO(grid, run_dir.results_dir)
    data = io.load_trajectory(single_value_names=[term])
    return data[term][step_index]


def collect_sweep_data(
    case_dir: CaseDir,
    sweep_id: str,
    grid,
    result_term: Term,
    config_path: str,
    step_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect result values and config parameters from a sweep.

    Common pattern: extract one result value and one config parameter
    from each run in a sweep.

    Parameters
    ----------
    case_dir : CaseDir
        The case directory containing the sweep.
    sweep_id : str
        The sweep identifier.
    grid : Grid
        The computational grid.
    result_term : Term
        The result term to extract (e.g., Term.energy).
    config_path : str
        Dot-separated path to config parameter.
    step_index : int
        Which step to extract results from (-1 for last).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (result_values, config_values) arrays.
    """
    run_dirs = case_dir.get_sweep_runs(sweep_id)

    results = np.empty(len(run_dirs))
    params = np.empty(len(run_dirs))

    for i, run_dir in enumerate(run_dirs):
        # Get result value
        io = SimulationIO(grid, run_dir.results_dir)
        data = io.load_trajectory(single_value_names=[result_term])
        results[i] = data[result_term][step_index]

        # Get config parameter
        params[i] = get_config_value(run_dir, config_path)

    return results, params
