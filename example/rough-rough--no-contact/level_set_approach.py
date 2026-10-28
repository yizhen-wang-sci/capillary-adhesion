"""Fill the gap by level set alone, as a reference for the minimisation-based runs.

Two records per run: `fill-above` treats the liquid as sitting above the cut-off height
(hydrophobic) and `fill-below` below it (hydrophilic).

Typical usage example:
    python level_set_approach.py params.toml [overlay.toml ...]
"""

import logging
import os
import sys

import numpy as np
from a_package.dataset import NpyBack, NpyIO, QuantityFront, RecordDir, RunDir, log_into, write_input
from a_package.domain import factorize_closest
from a_package.model import CapillaryBridge, RigidContact
from NuMPI import MPI

from case import *

logger = logging.getLogger(__name__)
comm_world = MPI.COMM_WORLD


def main(*config_files: str):
    # CLI: later config files override earlier ones, so a small overlay such as
    # params--test.toml can shrink the problem without editing params.toml
    if not config_files:
        sys.exit("No config file given.")
    config = load_config(*config_files)

    setup_console()

    run = RunDir(os.path.dirname(os.path.abspath(__file__)))
    for fill in ["fill-above", "fill-below"]:
        record = None
        if comm_world.rank == 0:
            record = RecordDir(run / fill)
            write_input(record.input, config)
        record = comm_world.bcast(record)
        with log_into(record.log, loggers=CONSOLE_LOGGERS):
            fill_record(record, config, fill)


def fill_record(record, config, fill: str) -> None:
    """Fill the gap at every separation of the trajectory, at a fixed liquid volume.

    Args:
        record: The record to write into.
        config: The recipe this run was expanded from.
        fill: ``"fill-above"`` or ``"fill-below"``, which side of the cut-off height the
            liquid sits on.
    """
    grid = build_grid(config)

    # Decomposed grid and have a parallel IO
    grid.decompose(
        factorize_closest(comm_world.size, 2),
        nb_ghost_layers=[1, 1],
        communicator=comm_world,
    )
    quantities = QuantityFront(
        NpyBack(
            record.data,
            NpyIO(**grid.owned_layout(), communicator=comm_world),
            decomposed=SPATIAL_BASES,
        )
    )
    ref_length = quantities.define("L")
    quantities.save_value("L", grid.domain_lengths[0])
    for basis_name in SPATIAL_BASES:
        quantities.define(basis_name, unit=ref_length, is_basis=True)
    save_grid(quantities, SPATIAL_BASES, grid)

    trajectory = np.round(build_trajectory(config), 6)

    # Surface generation is in serial. The frame goes down with it, on the same
    # SPATIAL_BASES declaration the parallel back below uses -- whichever back opens the
    # record first settles that convention, and a later one cannot change it.
    quantities.define(Term.upper_solid, unit=ref_length, frame=SPATIAL_BASES)
    quantities.define(Term.lower_solid, unit=ref_length, frame=SPATIAL_BASES)
    if comm_world.rank == 0:
        serial_front = QuantityFront(NpyBack(record.data, NpyIO()))
        upper, lower = build_surface(config)
        serial_front.save_value(Term.upper_solid, upper)
        serial_front.save_value(Term.lower_solid, lower)
        print(f"Saved surface data in {record.name}")
        del upper, lower, serial_front
    comm_world.barrier()

    # Build everything
    upper_surface = quantities.load_value(Term.upper_solid)
    lower_surface = quantities.load_value(Term.lower_solid)
    contact = RigidContact(upper_surface, lower_surface)
    phase_mixture = build_phase_mixture(config)
    capillary = CapillaryBridge(grid, phase_mixture, communicator=comm_world)

    # Separation is swept; the liquid volume is fixed, set from the smallest gap
    contact.set_mean_separation(np.amin(trajectory))
    capillary.set_gap(contact.get_gap())
    liquid_volume = build_liquid_volume(capillary, config)

    quantities.define("step", unit="", is_basis=True)
    quantities.save_value("step", np.arange(len(trajectory)))
    quantities.define(Term.separation, unit=quantities["L"], frame=("step",))
    quantities.define(Term.gap, unit=quantities["L"], frame=("step", *SPATIAL_BASES))
    quantities.define(Term.phase, unit="", frame=("step", *SPATIAL_BASES))

    quantities.save_value(Term.separation, trajectory)
    for i_step, separation in enumerate(trajectory):
        contact.set_mean_separation(separation)
        gap = contact.get_gap()
        phase = solve_phase_by_level_set(capillary, gap, liquid_volume, fill_below=fill == "fill-below")
        at = {"step": i_step}
        quantities.save_value(Term.gap, element_values(gap), at=at)
        quantities.save_value(Term.phase, element_values(phase), at=at)


if __name__ == "__main__":
    cli_config(main, __doc__)
