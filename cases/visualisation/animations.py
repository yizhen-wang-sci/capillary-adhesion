"""
Animation creation for simulation results.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.model import Term
from a_package.simulation import SimulationIO

from .primitives import latexify_plot
from .plots import plot_combined_topography


def create_overview_animation(grid, data_dir, output_dir):
    """
    Create and save an overview animation for a simulation run.

    Parameters
    ----------
    grid : Grid
        Grid object.
    data_dir : str or Path
        Directory containing simulation results.
    output_dir : str or Path
        Directory to save the animation.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    latexify_plot(15)
    anim = animate_droplet_evolution(SimulationIO(grid, data_dir))

    # Save animation
    filename_base = os.path.join(output_dir, "overview")
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
