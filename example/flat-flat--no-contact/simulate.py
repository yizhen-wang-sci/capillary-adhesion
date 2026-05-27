import os
import sys

import numpy as np
from NuMPI import MPI

from a_package.simulation import SourceDir, RunDir, SimulationIO, load_config, unroll_sweep, save_config, \
    setup_logging, get_git_hash
from a_package.domain import Grid, factorize_closest
from a_package.model import CapillaryBridge, RigidContact, formulate_constant_volume_phase_problem, Term

from config_helper import *

comm_world = MPI.COMM_WORLD


def main():
    # CLI arguments
    if len(sys.argv) != 2:
        raise ValueError(f"Provide one config file.")
    config_file = sys.argv[1]
    config_origin = load_config(config_file)

    run = None
    if comm_world.rank == 0:
        # make a snapshot of scripts & configs
        cwd = SourceDir(os.path.dirname(__file__))
        run = RunDir(cwd.snapshot(tag="baseline"))
        run.add_metadata({"git-hash": get_git_hash()})
    run = comm_world.bcast(run)

    # parameter sweep loop
    for config in unroll_sweep(config_origin):
        # build instances
        grid = build_grid(config)
        decomposition = grid.decompose(factorize_closest(comm_world.size, 2), (1, 1), communicator=comm_world)
        upper_surface = np.zeros(decomposition.nb_subdomain_grid_pts)
        lower_surface = np.zeros(decomposition.nb_subdomain_grid_pts)
        contact = RigidContact(upper_surface, lower_surface)
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

        # records
        record = None
        if comm_world.rank == 0:
            params = {'grid': grid.nb_domain_grid_pts[0], 'theta': mixture._theta}
            record = run.new_record(**params)
            save_config(config, record.input)
        record = comm_world.bcast(record)
        setup_logging(log_file=record.log)
        io = SimulationIO(record.data, decomposition, communicator=comm_world)

        # simulation loop
        phase = grid.get_local(square_init_guess(grid, liquid_volume, z_min))
        for i_step, separation in enumerate(trajectory):
            # ideal plastic contact
            contact.set_mean_separation(separation)
            gap = contact.get_gap()
            capillary.set_gap(gap)

            # solve phase problem
            problem = formulate_constant_volume_phase_problem(capillary, liquid_volume)
            print(f"rank={comm_world.rank}, before solve_minimisation")
            solution = optimizer.solve_minimisation(problem, x0=phase)
            print(f"rank={comm_world.rank}, problem solved, it took {solution['nit']} iterations.")

            # subtract quantities and save them
            phase = solution['x'].reshape(decomposition.nb_subdomain_grid_pts)
            pressure = -solution['dual']
            print(f"rank={comm_world.rank}, before io save step")
            io.save_step(i_step, single_values={Term.separation: separation, Term.pressure: pressure},
                         fields={Term.gap: gap, Term.phase: phase})
            print(f"rank={comm_world.rank}, after io save step")


def square_init_guess(grid: Grid, volume, mean_separation):
    half_nb_elements = round(0.5 * np.sqrt(volume / mean_separation / grid.element_area))
    Nx, Ny = grid.nb_domain_grid_pts
    phase = np.zeros(grid.nb_domain_grid_pts)
    phase[
        Nx // 2 - half_nb_elements:Nx // 2 + half_nb_elements, Ny // 2 - half_nb_elements:Ny // 2 + half_nb_elements] = 1.0
    return phase


if __name__ == '__main__':
    main()
