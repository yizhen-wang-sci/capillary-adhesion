
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from a_package.model import NodalFormCapillary, RigidContact, Term
from a_package.model.roughness import SelfAffineRoughness, psd_to_height
from a_package.simulation import (
    SimulationIO, CaseDir, _RecordDir, reset_logging, switch_log_file,
    load_config, save_config, unroll_sweep, get_timestamp
)

from config_helper import (
    create_grid_from_config,
    generate_roughness_from_config,
    build_capillary_args,
    build_trajectory,
)


logger = logging.getLogger(__name__)


def main():
    # CLI
    if len(sys.argv) < 2:
        print(f"At least one config file is required via CLI.")
        sys.exit(1)
    if len(sys.argv) > 2:
        print("Only the first config file is loaded")
    config_file = sys.argv[1]

    # setup
    reset_logging()
    case_dir = CaseDir(os.path.dirname(__file__))
    config = load_config(config_file)

    # run simulation now
    run_name = "--".join([get_timestamp(), os.path.splitext(config_file)[0]])
    run_dir = _RecordDir(case_dir / run_name, exist_ok=False)
    switch_log_file(run_dir.log_file)
    save_config(config, run_dir.input_file)
    run_level_set_approach(config, run_dir.data)

    # Update latest symlink
    latest_link = Path(case_dir / "latest")
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_name)
    logger.info(f"Updated 'latest' symlink -> {run_name}")


def run_level_set_approach(config: dict, output_path: Path):
    """Run simulation with level set approach."""

    # Build everything
    grid = create_grid_from_config(config)
    upper = generate_roughness_from_config(grid, config["problem"]["upper"])
    lower = generate_roughness_from_config(grid, config["problem"]["lower"])
    capillary_args = build_capillary_args(config)
    capillary = NodalFormCapillary(grid, capillary_args)
    trajectory = np.round(build_trajectory(config), 6)

    # Contact
    contact = RigidContact(upper, lower)

    # Compute target volume from percentage at minimum separation
    constraint_cfg = config["simulation"]["constraint"]
    z_min = np.amin(trajectory)
    contact.set_mean_separation(z_min)
    gap_at_min = contact.get_gap()
    capillary.set_gap(gap_at_min)
    max_volume = capillary.get_max_volume()
    volume = constraint_cfg["liquid_volume_percent"] / 100 * max_volume

    # IO
    io = SimulationIO(output_path, grid)

    for idx, separation in enumerate(trajectory):
        contact.set_mean_separation(separation)
        gap = contact.get_gap()
        phase = solve_phase_by_level_set(capillary, gap, volume, fill_below=True)
        io.save_step(idx, fields={Term.phase: phase, Term.gap: gap}, single_values={Term.separation: separation})


def solve_phase_by_level_set(capillary: NodalFormCapillary, gap: np.ndarray, volume: float, fill_below: bool=True):
    """

    :param capillary:
    :param gap:
    :param volume:
    :param fill_below: True if fill below the given height, match with hydrophilic, otherwise hydrophobic.
    :return:
    """
    capillary.set_gap(gap)
    phase = np.zeros_like(gap)

    def fill_phase_at(height):
        if fill_below:
            to_fill = gap < height
        else:
            to_fill = gap > height
        phase[to_fill] = 1.
        phase[~to_fill] = 0.
        capillary.set_phase(phase)

    def compute_volume_deviation(height):
        fill_phase_at(height)
        return capillary.get_volume() - volume

    height = bisection(compute_volume_deviation, gap.min(), gap.max())
    fill_phase_at(height)
    return capillary.get_phase()


def bisection(f, xa, xb, xtol=1e-6, ftol=1e-6, max_iter=100):
    """
    Find a root of f(x) = 0 using the bisection method.

    Args:
        f: Function to find root of
        xa: Left endpoint of interval
        xb: Right endpoint of interval
        xtol: Absolute tolerance for interval width (xb-xa)/2
        ftol: Tolerance for function value abs(f(xc))
        max_iter: Maximum number of iterations

    Returns:
        Approximate root location

    Raises:
        ValueError: If f(xa) and f(xb) have the same sign
    """
    if f(xa) * f(xb) >= 0:
        raise ValueError("f(xa) and f(xb) must have opposite signs")

    for _ in range(max_iter):
        xc = (xa + xb) / 2
        if abs(f(xc)) < ftol:
            print(f"ftol achieved. Root={xc}")
            return xc
        if (xb - xa) / 2 < xtol:
            print(f"xtol achieved. Root={xc}")
            return xc
        if f(xc) * f(xa) < 0:
            xb = xc
        else:
            xa = xc

    return (xa + xb) / 2


if __name__ == "__main__":
    main()
