"""
Animation creation for simulation results.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.domain import Grid
from a_package.simulation import SimulationIO, Term
from a_package.runtime import RunDir

from .primitives import latexify_plot
from .plots import (
    plot_combined_topography,
    plot_cross_section_sketch,
    plot_gibbs_free_energy,
    plot_normal_force,
    plot_pressure,
)


def create_overview_animation(run_path, grid: Grid):
    """
    Create and save an overview animation for a simulation run.

    Parameters
    ----------
    run_path : str or RunDir or SimulationIO
        Path to run directory, RunDir object, or SimulationIO object.
    grid : Grid
        The computational grid.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    # Handle different input types
    if isinstance(run_path, SimulationIO):
        io = run_path
        run_dir = RunDir(io.results_dir.parent)
    else:
        run_dir = RunDir(run_path) if not isinstance(run_path, RunDir) else run_path
        io = SimulationIO(grid, run_dir.results_dir)

    latexify_plot(15)
    anim = animate_droplet_evolution_with_curves(io)

    # Save animation
    filename_base = os.path.join(run_dir.visuals_dir, "overview")
    anim.save(f"{filename_base}.mp4", writer="ffmpeg")

    return anim


def animate_droplet_evolution(io: SimulationIO):
    """
    Create simple animation of droplet topography evolution.

    Parameters
    ----------
    io : SimulationIO
        Simulation IO object.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    def update_image(i_frame: int):
        ax.clear()
        plot_combined_topography(ax, io, i_frame)
        ax.set_xlabel(r"Position $x/\eta$")
        ax.set_ylabel(r"Position $y/\eta$")
        return ax.images

    data = io.load_trajectory(single_value_names=[Term.separation])
    n_step = len(data[Term.separation])
    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)


def animate_droplet_evolution_with_curves(io: SimulationIO):
    """
    Create comprehensive animation with topography and evolution curves.

    Layout:
    - Left: Cross-section sketch (top) and combined topography (bottom)
    - Right: Energy, normal force, and pressure evolution curves

    Parameters
    ----------
    io : SimulationIO
        Simulation IO object.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    # Figure with two subfigures
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    sf1, sf2 = fig.subfigures(1, 2, width_ratios=[1, 1])
    axs = sf2.subplots(3, 1, sharex=True)
    axs_lhs = sf1.subplots(2, 1, sharex=True, height_ratios=[1, 4])
    axs = np.append(axs, axs_lhs)

    # Load trajectory data for axis limits
    data = io.load_trajectory(
        field_names=[Term.upper_solid, Term.lower_solid],
        single_value_names=[Term.separation, Term.energy, Term.pressure]
    )
    n_step = len(data[Term.separation])
    idx_row = io.grid.nb_elements[0] // 2
    unit = min(io.grid.element_sizes)

    # Compute view limits
    view_margin_scale = 0.05

    h_min = np.amin(data[Term.lower_solid][0][0, 0, idx_row, :]) / unit
    h_max = (np.amax(data[Term.upper_solid][0][0, 0, idx_row, :]) + np.amax(data[Term.separation])) / unit
    h_margin = view_margin_scale * (h_max - h_min)
    h_min = h_min - h_margin
    h_max = h_max + 10 * h_margin  # Extra margin for legend

    energy = data[Term.energy]
    e_min = np.amin(energy) / unit**2
    e_max = np.amax(energy) / unit**2
    e_margin = view_margin_scale * (e_max - e_min)
    e_min = e_min - e_margin
    e_max = e_max + e_margin

    z = data[Term.separation]
    normal_force = -(energy[1:] - energy[:-1]) / (z[1:] - z[:-1])
    F_n_min = np.amin(normal_force) / unit
    F_n_max = np.amax(normal_force) / unit
    F_n_margin = view_margin_scale * (F_n_max - F_n_min)
    F_n_min = F_n_min - F_n_margin
    F_n_max = F_n_max + F_n_margin

    pressure = data[Term.pressure]
    P_min = np.amin(pressure) * unit
    P_max = np.amax(pressure) * unit
    P_margin = view_margin_scale * (P_max - P_min)
    P_min = P_min - P_margin
    P_max = P_max + P_margin

    def update_image(i_frame: int):
        for ax in axs:
            ax.clear()

        # Right side: evolution curves
        plot_gibbs_free_energy(axs[0], io, i_frame + 1)
        axs[0].set_ylim(e_min, e_max)
        axs[0].set_ylabel(r"Energy $E/\gamma_\mathrm{lv} a^2$")
        axs[0].set_title("Evolution")

        plot_normal_force(axs[1], io, i_frame + 1)
        axs[1].set_ylim(F_n_min, F_n_max)
        axs[1].set_ylabel(r"Normal force $F/\gamma_\mathrm{lv} a$")

        plot_pressure(axs[2], io, i_frame + 1)
        axs[2].set_ylim(P_min, P_max)
        axs[2].set_ylabel(r"Pressure $P/\gamma_\mathrm{lv} a^{-1}$")

        axs[2].set_xlim([0, n_step])
        axs[2].set_xlabel(r"Step (size=$0.1 a$)")

        # Left side: cross-section and topography
        plot_cross_section_sketch(axs[-2], io, i_frame, idx_row)
        axs[-2].set_ylim(h_min, h_max)
        axs[-2].set_ylabel(r"Position $z/a$")
        axs[-2].set_title("Cross section")

        plot_combined_topography(axs[-1], io, i_frame)
        axs[-1].axhline(io.grid.form_nodal_axis(0)[idx_row] / unit, color="k")
        axs[-1].set_ylabel(r"Position $y/a$")
        axs[-1].set_title("Gap & Phase")

        axs[-1].set_xlabel(r"Position $x/a$")

        return [axs[0].lines, axs[1].lines, axs[2].lines, axs[-1].images]

    return animation.FuncAnimation(fig, update_image, n_step, interval=200, repeat_delay=3000)
