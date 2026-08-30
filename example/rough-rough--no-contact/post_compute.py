"""Derive the quantities the run did not save, from the fields it did.

Which quantities are constants and which are trajectories follows this case's simulation loop.

`simulate.py` calls `run_post_compute` at the end of a run; this script repeats it over
whatever records already exist.

Typical usage example:
    mpiexec -np N python post_compute.py
"""

import os

import numpy as np
from a_package.dataset import NpyBack, NpyIO, QuantityFront, RunDir, read_input
from a_package.domain import factorize_closest
from a_package.model import CapillaryBridge
from NuMPI import MPI

from case import *

comm_world = MPI.COMM_WORLD


def main():
    run = RunDir(os.path.dirname(os.path.abspath(__file__)))
    run_post_compute(run.find_records(), comm_world)


def run_post_compute(records, comm_world) -> None:
    """Add force, volume, max volume, liquid area, perimeter, work and hysteresis.

    Args:
        records: The same records, in the same order, on every rank.
        comm_world: The communicator the records are decomposed over.
    """
    for record in records:
        # Build instances from config
        config = read_input(record.input)
        grid = build_grid(config)
        decomposition = grid.decompose(factorize_closest(comm_world.size, 2), [1, 1], communicator=comm_world)
        phase_mixture = build_phase_mixture(config)
        eps, curv = phase_mixture._epsilon, phase_mixture._curv
        capillary = CapillaryBridge(grid, phase_mixture, communicator=comm_world)

        quantities = QuantityFront(
            NpyBack(
                record.data,
                NpyIO(**grid.owned_layout(), communicator=comm_world),
                decomposed=SPATIAL_BASES,
            )
        )
        separation_traj = quantities.load_value(Term.separation)
        pressure_traj = quantities.load_value(Term.pressure)
        nb_cycles, nb_steps = pressure_traj.shape
        ref_length = quantities["L"]

        # `unit` names a reference scalar and carries no exponent, so only the quantities
        # normalising by one power of L state one.
        quantities.define(Term.perimeter, unit=ref_length, frame=("cycle", "step"))
        quantities.define("area_ls", frame=("cycle", "step"))
        quantities.define("area_lv", frame=("cycle", "step"))
        quantities.define(Term.volume, frame=("cycle", "step"))
        quantities.define(Term.max_volume, frame=("step",))
        quantities.define("force", unit=ref_length, frame=("cycle", "step"))

        # The max volume depends on the gap alone, which is the same for each cycle.
        max_volume_traj = np.empty(nb_steps)
        for i_step in range(nb_steps):
            capillary.set_gap(quantities.load_value(Term.gap, at={"step": i_step}))
            max_volume_traj[i_step] = capillary.get_max_volume()

        # The rest follow the phase, hence the cycle.
        force_traj = np.empty((nb_cycles, nb_steps))
        volume_traj = np.empty((nb_cycles, nb_steps))
        area_ls_traj = np.empty((nb_cycles, nb_steps))
        area_lv_traj = np.empty((nb_cycles, nb_steps))
        perimeter_traj = np.empty((nb_cycles, nb_steps))
        for i_cycle in range(nb_cycles):
            for i_step in range(nb_steps):
                capillary.set_gap(quantities.load_value(Term.gap, at={"step": i_step}))
                capillary.set_phase(quantities.load_value(Term.phase, at={"cycle": i_cycle, "step": i_step}))
                volume_traj[i_cycle, i_step] = capillary.get_volume()
                area_ls_traj[i_cycle, i_step] = capillary.get_liquid_solid_area()
                area_lv_traj[i_cycle, i_step] = capillary.get_liquid_vapor_area()
                perimeter_traj[i_cycle, i_step] = capillary.get_perimeter()
                force_traj[i_cycle, i_step] = (
                    pressure_traj[i_cycle, i_step] * area_ls_traj[i_cycle, i_step]
                    - eps * curv * perimeter_traj[i_cycle, i_step]
                )
        quantities.save_value(Term.max_volume, max_volume_traj)
        quantities.save_value(Term.volume, volume_traj)
        quantities.save_value("force", force_traj)
        quantities.save_value("area_ls", area_ls_traj)
        quantities.save_value("area_lv", area_lv_traj)
        quantities.save_value(Term.perimeter, perimeter_traj)

        save_cycle_work(quantities, separation_traj, force_traj, perimeter_traj, phase_mixture)


def save_cycle_work(quantities, separation, force, perimeter, mixture) -> None:
    """Add the per-trip work and the per-cycle hysteresis of a cycled record.

    One quantity per force formulation: ``work1`` from ``f = p * A``, ``work2`` from
    ``f = p * A - eps * curv * P``.

    Args:
        quantities: The record's front, which gains `trip`, the works and the hystereses.
        separation: The separations of one cycle.
        force: The force of every point, shaped `(nb_cycles, nb_steps)`.
        perimeter: The perimeter of every point, shaped `(nb_cycles, nb_steps)`.
        mixture: Supplies the interface thickness and curvature.
    """
    force = np.asarray(force)
    nb_cycles, nb_steps = force.shape
    if nb_cycles == 0:
        return

    eps, curv = mixture._epsilon, mixture._curv
    # the stored `force` is the second formulation, so the first adds the interface term back
    forces = {
        "work1": force + eps * curv * np.asarray(perimeter),
        "work2": force,
    }

    quantities.define("trip", is_basis=True)
    quantities.save_value("trip", np.arange(2))

    # a cycle is two trips sharing the turning point, so both carry the middle step
    half = (nb_steps - 1) // 2
    for name, f in forces.items():
        quantities.define(name, frame=("cycle", "trip"))
        quantities.define(name.replace("work", "hysteresis"), frame=("cycle",))

        work = np.empty((nb_cycles, 2))
        for k in range(nb_cycles):
            work[k, 0] = np.trapezoid(f[k, : half + 1], separation[: half + 1])
            work[k, 1] = np.trapezoid(f[k, half:], separation[half:])
        quantities.save_value(name, work)

        # The trips traverse the separation in opposite directions, so the sum of their
        # integrals is the area the loop encloses.
        hysteresis = abs(work[:, 0] + work[:, 1])
        quantities.save_value(name.replace("work", "hysteresis"), hysteresis)


if __name__ == "__main__":
    main()
