"""
Run constant-volume simulation from config file.

Usage:
    python simulate.py config.toml
"""

import logging
import os
import sys

import numpy as np
from mpi4py import MPI

from a_package.domain import Grid, factorize_closest
from a_package.model import CapillaryBridge, RigidContact, Term, formulate_constant_volume_phase_problem
from a_package.simulation import (SimulationIO, SourceDir, RunDir, setup_logging, load_config,
                                  save_config, unroll_sweep, get_iso_time, get_git_hash)

from config_helper import *


visual_check = False
comm_world = MPI.COMM_WORLD


def main():
    # CLI
    try:
        config_file = sys.argv[1]
    except IndexError:
        print(f"One config file is required via CLI.")
        sys.exit(1)
    config_origin = load_config(config_file)

    # setup
    run = None
    if comm_world.rank == 0:
        # make a snapshot of scripts & configs
        cwd = SourceDir(os.path.dirname(__file__))
        cwd._suffixes = (*cwd._suffixes, ".npy")  # include surface data in snapshot
        run = RunDir(cwd.snapshot(tag="affinity-variation"))
        run.add_metadata({"created": get_iso_time(), "git-hash": get_git_hash()})
    run = comm_world.bcast(run)

    # parameter sweep loop
    for config in unroll_sweep(config_origin):
        # build instances
        grid = build_grid(config)
        decomposition = grid.decompose(factorize_closest(comm_world.size, 2), (1, 1), communicator=comm_world)

        surface_io = SimulationIO(run, decomposition, communicator=comm_world)
        surface_data = surface_io.load_constant(field_names=[Term.upper_solid, Term.lower_solid])
        upper_surface_local = surface_data[Term.upper_solid]
        lower_surface_local = surface_data[Term.lower_solid]
        del surface_io, surface_data

        contact = RigidContact(upper_surface_local, lower_surface_local)
        mixture = build_phase_mixture(config)
        capillary = CapillaryBridge(grid, mixture, communicator=comm_world)
        optimizer = build_optimizer(config)
        trajectory = build_trajectory(config)

        # concrete liquid volume
        z_min = np.amin(trajectory)
        contact.set_mean_separation(z_min)
        gap_at_min = contact.get_gap()
        capillary.set_gap(gap_at_min)
        volume_percent = config['constraint']['liquid_volume_percent']
        liquid_volume = capillary.get_max_volume() * (volume_percent / 100.0)

        print(f"Problem size: {'x'.join(str(dim) for dim in grid.nb_domain_grid_pts)}. "
              f"Simulating for {len(trajectory)} separation values at volume={liquid_volume}({volume_percent}%)...")

        # records
        record = None
        if comm_world.rank == 0:
            theta = config['capillary']['contact_angle_degree']
            record = run.new_record(theta=theta)
            setup_logging(log_file=record.log)
            save_config(config, record.input)
        record = comm_world.bcast(record)

        # initial guess
        phase_init = square_init_guess(grid, liquid_volume, np.amin(trajectory))
        phase_init_local = phase_init[*decomposition.icoords]

        # IO and save setup
        io = SimulationIO(record.data, decomposition, communicator=comm_world)
        io.save_constant(fields={Term.upper_solid: upper_surface_local, Term.lower_solid: lower_surface_local,
                                 Term.phase_init: phase_init_local})

        # Simulation and save results
        for i_step, separation, gap_local, phase_local, pressure in solve_constant_volume(
                decomposition.nb_subdomain_grid_pts, contact, capillary, optimizer, trajectory, liquid_volume,
                phase_init_local):
            io.save_step(i_step, single_values={Term.separation: separation, Term.pressure: pressure},
                         fields={Term.phase: phase_local, Term.gap: gap_local})


def square_init_guess(grid: Grid, volume, mean_separation):
    half_nb_domain_grid_pts = round(0.5 * np.sqrt(volume / mean_separation / grid.element_area))
    Nx, Ny = grid.nb_domain_grid_pts
    phase = np.zeros(grid.nb_domain_grid_pts)
    phase[
        Nx // 2 - half_nb_domain_grid_pts:Nx // 2 + half_nb_domain_grid_pts, Ny // 2 - half_nb_domain_grid_pts:Ny // 2 + half_nb_domain_grid_pts] = 1.0
    return phase


def solve_constant_volume(original_shape, contact, capillary, optimizer, trajectory, liquid_volume, phase_init_local):
    phase_local = phase_init_local.copy()
    for i_step, separation in enumerate(trajectory):
        print(f"Step {i_step}: separation={separation}")

        # Gap
        contact.set_mean_separation(separation)
        gap_local = contact.get_gap()

        # phase
        capillary.set_gap(gap_local)
        problem = formulate_constant_volume_phase_problem(capillary, liquid_volume)
        solution = optimizer.solve_minimisation(problem, x0=phase_local)
        print(f"It took {solution['nit']} iterations.")

        phase_local = solution['x'].reshape(original_shape)
        pressure = -solution['dual']

        yield i_step, separation, gap_local, phase_local, pressure


if __name__ == "__main__":
    main()
