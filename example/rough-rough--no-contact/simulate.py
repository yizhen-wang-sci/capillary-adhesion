"""Cycle rough surfaces through approach and retraction, at one contact angle per record.

Typical usage example:
    mpiexec -n <N> python simulate.py
    mpiexec -n <N> python simulate.py --theta 90.0
"""

import os

import numpy as np
from a_package.dataset import NpyBack, NpyIO, QuantityFront, RunDir, get_iso_time, log_into, read_input
from a_package.domain import factorize_closest
from a_package.model import (
    CapillaryBridge,
    RigidContact,
    extract_pressure_in_constant_volume_solution,
    formulate_constant_volume_phase_problem,
)
from mpi4py import MPI

from case import *
from post_compute import run_post_compute

comm_world = MPI.COMM_WORLD

POINT_SCALARS = [Term.pressure]


def main(**query):
    run = RunDir(os.path.dirname(os.path.abspath(__file__)))
    setup_console()

    records = None
    if comm_world.rank == 0:
        records = sorted(run.find_records(**query), key=lambda record: record.name)
        records = [record for record in records if not is_complete(record)]
    records = comm_world.bcast(records)
    if comm_world.rank == 0:
        print(f"{len(records)} record(s) to solve: {', '.join(record.name for record in records)}")

    for record in records:
        with log_into(record.log, loggers=CONSOLE_LOGGERS):
            solve_record(record)

    run_post_compute(records, comm_world)

    if comm_world.rank == 0:
        run.add_metadata({"modified": get_iso_time()})


def trajectory_shape(config) -> tuple[int, int]:
    """The number of cycles and steps one record holds.

    Args:
        config: The record's own request.

    Returns:
        tuple: Cycles, then steps per cycle.
    """
    return config["trajectory"]["nb_cycles"], len(build_trajectory(config))


def is_complete(record) -> bool:
    """Whether every point of a record's cycles is written.

    Args:
        record: The record to read.

    Returns:
        bool: True where the frontier has reached the last point.
    """
    nb_cycles, nb_steps = trajectory_shape(read_input(record.input))
    return count_complete_points(record, POINT_SCALARS) >= nb_cycles * nb_steps


def write_bases_and_surfaces(record, config) -> None:
    """Write a record's bases and its two surfaces.

    Args:
        record: The record to write into.
        config: The record's own request.
    """
    grid = build_grid(config)
    nb_cycles, nb_steps = trajectory_shape(config)
    upper, lower = build_surface(config)

    quantities = QuantityFront(NpyBack(record.data, decomposed=SPATIAL_BASES))
    ref_length = quantities.define("L")
    quantities.save_value("L", grid.domain_lengths[0])
    for basis_name in SPATIAL_BASES:
        quantities.define(basis_name, unit=ref_length, is_basis=True)
    save_grid(quantities, SPATIAL_BASES, grid)
    quantities.define("cycle", is_basis=True)
    quantities.save_value("cycle", np.arange(nb_cycles))
    quantities.define("step", is_basis=True)
    quantities.save_value("step", np.arange(nb_steps))
    quantities.define(Term.upper_solid, unit=ref_length, frame=SPATIAL_BASES)
    quantities.save_value(Term.upper_solid, upper)
    quantities.define(Term.lower_solid, unit=ref_length, frame=SPATIAL_BASES)
    quantities.save_value(Term.lower_solid, lower)


def solve_record(record) -> None:
    """Solve one record, from wherever it got to.

    Args:
        record: The record to solve, holding the request in its `input`.
    """
    config = read_input(record.input)
    trajectory = build_trajectory(config)
    nb_cycles, nb_steps = trajectory_shape(config)

    nb_done = None
    if comm_world.rank == 0:
        nb_done = count_complete_points(record, POINT_SCALARS)
        if nb_done == 0:
            write_bases_and_surfaces(record, config)
    nb_done = comm_world.bcast(nb_done)
    if comm_world.rank == 0:
        print(f"{record.name}: {nb_done} of {nb_cycles * nb_steps} point(s) already done")
    comm_world.barrier()

    grid = build_grid(config)
    decomposition = grid.decompose(factorize_closest(comm_world.size, 2), (1, 1), communicator=comm_world)
    quantities = QuantityFront(
        NpyBack(
            record.data,
            NpyIO(**grid.owned_layout(), communicator=comm_world),
            decomposed=SPATIAL_BASES,
        )
    )
    ref_length = quantities["L"]

    upper_surface_local = quantities.load_value(Term.upper_solid)
    lower_surface_local = quantities.load_value(Term.lower_solid)

    contact = RigidContact(upper_surface_local, lower_surface_local)
    mixture = build_phase_mixture(config)
    capillary = CapillaryBridge(grid, mixture, communicator=comm_world)
    optimizer = build_optimizer(config)

    # concrete liquid volume
    z_min = np.amin(trajectory)
    contact.set_mean_separation(z_min)
    gap_at_min = contact.get_gap()
    capillary.set_gap(gap_at_min)
    liquid_volume = build_liquid_volume(capillary, config)

    # inform
    print(
        f"[rank{comm_world.rank}] at ({','.join(str(loc) for loc in decomposition.subdomain_locations_with_ghosts)}),"
        f" local domain: {'x'.join(str(dim) for dim in decomposition.nb_subdomain_grid_pts_with_ghosts)}."
    )
    if comm_world.rank == 0:
        print(f"Global domain: {'x'.join(str(dim) for dim in grid.nb_domain_grid_pts)}.")
        print(f"volume={liquid_volume}({config['constraint']['liquid_volume_percent']}%)")
        print(f"mean separation: min={trajectory.min()}, max={trajectory.max()}")

    # The gap is solved only for one cycle, as it is the same for all cycles
    quantities.define(Term.separation, unit=ref_length, frame=("step",))
    quantities.define(Term.gap, unit=ref_length, frame=("step", *SPATIAL_BASES))
    if nb_done == 0:
        solve_gap(quantities, contact, trajectory)

    # Get initial guess of phases
    phase_init = square_init_guess(grid, liquid_volume, np.amin(trajectory))
    phase_init_local = grid.get_local(phase_init)

    # Save constant (the surfaces are already stored in this record)
    quantities.define(Term.phase_init, unit="", frame=SPATIAL_BASES)
    if nb_done == 0:
        quantities.save_value(Term.phase_init, phase_init_local)

    # Solve the phases per step per cycle
    quantities.define(Term.phase, unit="", frame=("cycle", "step", *SPATIAL_BASES))
    # FIXME: pressure unit needs some power and multiplication
    quantities.define(Term.pressure, frame=("cycle", "step"))
    if nb_done == 0:
        seed_point_scalars(quantities, POINT_SCALARS, (nb_cycles, nb_steps))

    solve_constant_volume(
        quantities,
        decomposition.nb_subdomain_grid_pts,
        capillary,
        optimizer,
        trajectory,
        nb_cycles,
        liquid_volume,
        phase_init_local,
        nb_done,
    )


def solve_gap(quantities, contact, trajectory) -> None:
    """Write the separation and the gap of every step of one cycle.

    Args:
        quantities: The record's front.
        contact: The two surfaces, whose separation is set here.
        trajectory: The separations to walk.
    """
    quantities.save_value(Term.separation, trajectory)
    for i_step, separation in enumerate(trajectory):
        contact.set_mean_separation(separation)
        at = {"step": i_step}
        quantities.save_value(Term.gap, contact.get_gap().squeeze(), at=at)


def solve_constant_volume(
    quantities,
    original_shape,
    capillary,
    optimizer,
    trajectory,
    nb_cycles,
    liquid_volume,
    phase_init_local,
    nb_done,
):
    """Walk the trajectory `nb_cycles` times, writing one solved point at a time.

    Args:
        quantities: The record's front. The gap is read back from it.
        original_shape: The local field shape a solution is reshaped to.
        capillary: The bridge to solve.
        optimizer: The minimiser.
        trajectory: The separations of one cycle.
        nb_cycles: How many times to walk it.
        liquid_volume: The volume held fixed.
        phase_init_local: The phase the first point starts from.
        nb_done: How many leading points are already written.
    """
    nb_steps = len(trajectory)
    if nb_done == 0:
        phase_local = phase_init_local.copy()
    else:
        i_cycle, i_step = divmod(nb_done - 1, nb_steps)
        phase_local = quantities.load_value(Term.phase, at={"cycle": i_cycle, "step": i_step})

    for i_point in range(nb_done, nb_cycles * nb_steps):
        i_cycle, i_step = divmod(i_point, nb_steps)
        separation = trajectory[i_step]
        if comm_world.rank == 0:
            print(f"cycle {i_cycle}, step {i_step}: separation={separation}")

        capillary.set_gap(quantities.load_value(Term.gap, at={"step": i_step}))
        problem = formulate_constant_volume_phase_problem(capillary, liquid_volume, explicit_phase_bounds=False)
        solution = optimizer.solve_minimisation(problem, x0=phase_local)
        if comm_world.rank == 0:
            print(f"After {solution['nit']} iterations, {solution['message']}")

        phase_local = solution["x"].reshape(original_shape)
        pressure = extract_pressure_in_constant_volume_solution(solution)

        at = {"cycle": i_cycle, "step": i_step}
        quantities.save_value(Term.phase, phase_local, at=at)
        quantities.save_value(Term.pressure, pressure, at=at)


if __name__ == "__main__":
    cli_records(main, __doc__, RECORD_NAMING_TYPES)
