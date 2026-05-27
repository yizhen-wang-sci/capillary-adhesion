import os
import sys

import numpy as np

from a_package.model import CapillaryBridge, RigidContact, Term
from a_package.simulation import SimulationIO, RunDir, RecordDir, setup_logging, load_config, save_config

from config_helper import *


def main():
    # CLI
    if len(sys.argv) < 2:
        print(f"At least one config file is required via CLI.")
        sys.exit(1)
    if len(sys.argv) > 2:
        print("Only the first config file is loaded")
    config_file = sys.argv[1]
    config = load_config(config_file)

    for fill in ["fill-above", "fill-below"]:
        # setup
        run = RunDir(os.path.dirname(__file__))
        record = RecordDir(run / fill)
        setup_logging(log_file=record.log)
        save_config(config, record.input)
        io = SimulationIO(record.data)

        # Build everything
        grid = build_grid(config)
        # Need ghost layers
        grid.decompose([1, 1], nb_ghost_layers=[1, 1])
        upper_surface = np.load(run / f"{Term.upper_solid}.npy")
        lower_surface = np.load(run / f"{Term.lower_solid}.npy")
        contact = RigidContact(upper_surface, lower_surface)
        phase_mixture = build_phase_mixture(config)
        capillary = CapillaryBridge(grid, phase_mixture)
        trajectory = np.round(build_trajectory(config), 6)

        # concrete liquid volume
        z_min = np.amin(trajectory)
        contact.set_mean_separation(z_min)
        gap_at_min = contact.get_gap()
        capillary.set_gap(gap_at_min)
        volume_percent = config['constraint']['liquid_volume_percent']
        liquid_volume = capillary.get_max_volume() * (volume_percent / 100.0)

        for idx, separation in enumerate(trajectory):
            contact.set_mean_separation(separation)
            gap = contact.get_gap()
            phase = solve_phase_by_level_set(capillary, gap, liquid_volume, fill_below=fill == "fill-below")
            io.save_step(idx, single_values={Term.separation: separation}, fields={Term.gap: gap, Term.phase: phase})


def solve_phase_by_level_set(capillary: CapillaryBridge, gap: np.ndarray, volume: float, fill_below: bool = True):
    """
    Solve for the liquid phase distribution using a level-set approach.

    Parameters
    ----------
    capillary : CapillaryBridge
        The capillary bridge model.
    gap : np.ndarray
        The local gap between surfaces.
    volume : float
        Target liquid volume.
    fill_below : bool, optional
        True if filling below the given height (hydrophilic),
        otherwise hydrophobic. Default is True.

    Returns
    -------
    np.ndarray
        The computed liquid phase distribution.
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

    Parameters
    ----------
    f : callable
        Function to find root of.
    xa : float
        Left endpoint of interval.
    xb : float
        Right endpoint of interval.
    xtol : float, optional
        Absolute tolerance for interval width (xb-xa)/2. Default is 1e-6.
    ftol : float, optional
        Tolerance for function value abs(f(xc)). Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    float
        Approximate root location.

    Raises
    ------
    ValueError
        If f(xa) and f(xb) have the same sign.
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
