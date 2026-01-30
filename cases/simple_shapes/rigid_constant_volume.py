"""
Run constant-volume simulation from config file.

Usage:
    python -m cases.run_constant_volume config.toml
"""

import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.model import NodalFormCapillary, RigidContact, Term
from a_package.domain import AugmentedLagrangian
from a_package.simulation import load_config, save_config, reset_logging, switch_log_file, RunDir, SimulationIO

from cases.config_helpers import (
    create_grid_from_config,
    build_capillary_args,
    build_solver_args,
    build_trajectory,
)
from .surface_helpers import generate_surface_from_config

from cases.visualisation import (
    latexify_plot,
    plot_combined_topography,
    plot_cross_section_sketch,
    plot_gibbs_free_energy,
    plot_normal_force,
    plot_pressure,
)


logger = logging.getLogger(__name__)


def main():
    # CLI
    if len(sys.argv) < 2:
        print(f"Provide one config file.")
        sys.exit(1)
    config_file = sys.argv[1]

    # setup
    reset_logging()
    config = load_config(config_file)

    script_dir = os.path.dirname(__file__)
    run_name = os.path.splitext(config_file)[0]
    run_dir = RunDir(os.path.join(script_dir, run_name))

    # run simulation
    switch_log_file(run_dir.log_file)
    save_config(config, run_dir.input_file)
    io = _run_constant_volume(config, run_dir.data_dir)

    # post-process
    _visualize_constant_volume(io, run_dir.visuals_dir)

# =============================================================================
# Simulation
# =============================================================================


def _run_constant_volume(config, output_path):
    """Run simulation with constant volume constraint."""
    grid = create_grid_from_config(config)
    upper = generate_surface_from_config(grid, config["problem"]["upper"])
    lower = generate_surface_from_config(grid, config["problem"]["lower"])
    capillary_args = build_capillary_args(config)
    solver_args = build_solver_args(config)
    trajectory = np.round(build_trajectory(config), 6)

    # Create solvers
    contact_solver = RigidContact(upper, lower)
    formulation = NodalFormCapillary(grid, capillary_args)
    optimizer = AugmentedLagrangian(**solver_args)

    # Compute target volume from percentage at minimum separation
    constraint_cfg = config["simulation"]["constraint"]
    z_min = np.amin(trajectory)
    contact_solver.set_mean_separation(z_min)
    gap_at_min = contact_solver.get_gap()
    formulation.set_gap(gap_at_min)
    max_volume = formulation.get_max_volume()
    volume = max_volume * (constraint_cfg["liquid_volume_percent"] / 100.0)

    # Initial phase field
    rng = random.default_rng()
    phase = rng.random((1, 1, *grid.nb_elements))
    pressure = 0.0

    # IO
    io = SimulationIO(grid, output_path)
    io.save_constant(
        fields={Term.phase_init: phase},
        single_values={Term.pressure_init: pressure},
    )

    logger.info(
        f"Problem size: {'x'.join(str(dim) for dim in phase.shape[-2:])}. "
        f"Simulating for {len(trajectory)} separation values at volume={volume}..."
    )

    # Simulation loop
    for index, separation in enumerate(trajectory):
        logger.info(f"Step {index}: separation={separation}")
        contact_solver.set_mean_separation(separation)
        gap = contact_solver.get_gap()
        formulation.set_gap(gap)

        # Build constrained optimization problem
        def volume_constraint():
            return formulation.get_volume() - volume

        volume_constraint_jacobian = formulation.get_volume_jacobian

        problem = SimpleNamespace(
            get_x=formulation.get_phase,
            set_x=formulation.set_phase,
            get_f=formulation.get_energy,
            get_f_Dx=formulation.get_energy_jacobian,
            get_g=volume_constraint,
            get_g_Dx=volume_constraint_jacobian,
            x_lb=formulation.phase_lb,
            x_ub=formulation.phase_ub,
        )

        # Solve
        result = optimizer.solve_minimisation(problem, x0=phase, lam0=pressure)
        phase = np.reshape(result.primal, phase.shape)
        pressure = result.dual
        formulation.set_phase(phase)

        io.save_step(
            index,
            fields={
                Term.upper_solid: contact_solver.upper,
                Term.lower_solid: contact_solver.lower,
                Term.gap: gap,
                Term.phase: phase,
            },
            single_values={
                Term.separation: separation,
                Term.pressure: pressure,
                Term.energy: formulation.get_energy(),
                Term.volume: formulation.get_volume(),
            },
        )

    return io


# =============================================================================
# Visualization
# =============================================================================


def animate_constant_volume(io: SimulationIO):
    """
    Create animation for constant-volume simulation.

    Shows topography evolution and energy/force/pressure curves.
    """
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    sf1, sf2 = fig.subfigures(1, 2, width_ratios=[1, 1])
    axs_rhs = sf2.subplots(3, 1, sharex=True)
    axs_lhs = sf1.subplots(2, 1, sharex=True, height_ratios=[1, 4])

    # Load trajectory data for axis limits
    data = io.load_trajectory(
        field_names=[Term.upper_solid, Term.lower_solid],
        single_value_names=[Term.separation, Term.energy, Term.pressure]
    )
    n_step = len(data[Term.separation])
    idx_row = io.grid.nb_elements[0] // 2
    unit = min(io.grid.element_sizes)

    # Compute view limits
    margin = 0.05

    h_min = np.amin(data[Term.lower_solid][0][0, 0, idx_row, :]) / unit
    h_max = (np.amax(data[Term.upper_solid][0][0, 0, idx_row, :]) + np.amax(data[Term.separation])) / unit
    h_margin = margin * (h_max - h_min)
    h_min -= h_margin
    h_max += 10 * h_margin

    energy = data[Term.energy]
    e_min, e_max = np.amin(energy) / unit**2, np.amax(energy) / unit**2
    e_margin = margin * (e_max - e_min)
    e_min -= e_margin
    e_max += e_margin

    z = data[Term.separation]
    normal_force = -(energy[1:] - energy[:-1]) / (z[1:] - z[:-1])
    F_min, F_max = np.amin(normal_force) / unit, np.amax(normal_force) / unit
    F_margin = margin * (F_max - F_min)
    F_min -= F_margin
    F_max += F_margin

    pressure = data[Term.pressure]
    P_min, P_max = np.amin(pressure) * unit, np.amax(pressure) * unit
    P_margin = margin * (P_max - P_min)
    P_min -= P_margin
    P_max += P_margin

    def update_image(i_frame: int):
        for ax in [*axs_rhs, *axs_lhs]:
            ax.clear()

        # Right side: evolution curves
        plot_gibbs_free_energy(axs_rhs[0], io, i_frame + 1)
        axs_rhs[0].set_ylim(e_min, e_max)
        axs_rhs[0].set_ylabel(r"Energy $E/\gamma_\mathrm{lv} a^2$")
        axs_rhs[0].set_title("Evolution")

        plot_normal_force(axs_rhs[1], io, i_frame + 1)
        axs_rhs[1].set_ylim(F_min, F_max)
        axs_rhs[1].set_ylabel(r"Normal force $F/\gamma_\mathrm{lv} a$")

        plot_pressure(axs_rhs[2], io, i_frame + 1)
        axs_rhs[2].set_ylim(P_min, P_max)
        axs_rhs[2].set_ylabel(r"Pressure $P/\gamma_\mathrm{lv} a^{-1}$")
        axs_rhs[2].set_xlim([0, n_step])
        axs_rhs[2].set_xlabel(r"Step")

        # Left side: cross-section and topography
        plot_cross_section_sketch(axs_lhs[0], io, i_frame, idx_row)
        axs_lhs[0].set_ylim(h_min, h_max)
        axs_lhs[0].set_ylabel(r"Position $z/a$")
        axs_lhs[0].set_title("Cross section")

        plot_combined_topography(axs_lhs[1], io, i_frame)
        axs_lhs[1].axhline(io.grid.form_nodal_axis(0)[idx_row] / unit, color="k")
        axs_lhs[1].set_ylabel(r"Position $y/a$")
        axs_lhs[1].set_xlabel(r"Position $x/a$")
        axs_lhs[1].set_title("Gap & Phase")

        return []

    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


def _visualize_constant_volume(io: SimulationIO, output_dir):
    """Create and save visualization for constant-volume simulation."""
    latexify_plot(15)
    anim = animate_constant_volume(io)
    filename = os.path.join(output_dir, "constant_volume.mp4")
    anim.save(filename, writer="ffmpeg")
    logger.info(f"Saved animation to {filename}")
    return anim


if __name__ == "__main__":
    main()
