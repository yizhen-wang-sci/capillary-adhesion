"""Check the solved record's liquid volume against the level-set filling, and across its steps.

Reads the last cycle of the `theta=5` record and the `fill-below` record beside it.
"""

import os

import numpy as np
from a_package.dataset import NpyBack, QuantityFront, RecordDir, RunDir
from a_package.model import CapillaryBridge

from case import *


def main():
    # This file resides in RunDir
    run = RunDir(os.path.dirname(__file__))

    # Find records
    [record] = run.find_records(theta=5)
    if not (run / "fill-below").exists():
        raise FileNotFoundError("No reocrds of level-set filling below.")
    record_fill_below = RecordDir(run / "fill-below")

    # Build capillary
    config = load_config(record.input)
    grid = build_grid(config)
    grid.decompose([1, 1], nb_ghost_layers=[1, 1])
    phase_mixture = build_phase_mixture(config)
    capillary = CapillaryBridge(grid, phase_mixture)

    # The solved record spans cycles and steps; the level-set one is a single trajectory of
    # the same length. Both records state their own bases, so the shape comes from them.
    solved = QuantityFront(NpyBack(record.data))
    filled = QuantityFront(NpyBack(record_fill_below.data))

    # steps to be checked against: select the last cycle
    i_cycle = len(solved.load_value("cycle")) - 1
    nb_steps = len(solved.load_value("step"))
    volumes = np.zeros((2, nb_steps))

    # Compute volume
    for i_step in range(nb_steps):
        volumes[0, i_step] = volume_of(capillary, solved, {"cycle": i_cycle, "step": i_step})
        volumes[1, i_step] = volume_of(capillary, filled, {"step": i_step})

    # Check difference between two methods
    volume_diffs = volumes[1] - volumes[0]
    assert np.any(np.isclose(volume_diffs, 0))
    # Check difference between steps
    volume_diffs = np.diff(volumes, axis=1)
    assert np.any(np.isclose(volume_diffs, 0))


def volume_of(capillary: CapillaryBridge, quantities: QuantityFront, at: dict) -> float:
    """The liquid volume of the phase field one record holds at one point of its frame.

    Args:
        capillary: What computes the volume.
        quantities: The record's quantities.
        at: The point of the frame to read the phase field at.

    Returns:
        float: The volume.
    """
    # The volume is the phase over the gap, and the gap belongs to the step alone: the same
    # separation trajectory is walked in every cycle.
    capillary.set_gap(quantities.load_value(Term.gap, at={"step": at["step"]}))
    capillary.set_phase(quantities.load_value(Term.phase, at=at).squeeze())
    return capillary.get_volume()


if __name__ == "__main__":
    main()
