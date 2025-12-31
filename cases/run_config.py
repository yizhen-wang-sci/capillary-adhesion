"""
Load-unload simulation case.

Usage:
    python -m cases.load_unload config.toml [config2.toml ...]

The first config file provides base parameters. Additional config files
can override specific values (useful for sweep specifications).
"""

import logging
import os
import sys

from a_package.config import Config, load_config
from a_package.runtime import CaseDir, reset_logging
from a_package.run import run_sweep, build_trajectory, create_grid_from_config, generate_surface_from_config, get_surface_shape

from cases.visualise_onerun import create_overview_animation


show_me_preview = False
logger = logging.getLogger(__name__)


def main():
    reset_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m cases.load_unload config.toml")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # visual check
    if show_me_preview:
        preview_surface_and_gap(config)

    # setup case directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    upper_shape = get_surface_shape(config, "upper")
    lower_shape = get_surface_shape(config, "lower")
    shape_name = f'{upper_shape}-on-{lower_shape}'
    base_dir = os.path.join(script_name, shape_name)
    case_dir = CaseDir(base_dir)

    # Run simulation(s) - handles sweeps automatically
    ios = run_sweep(config, case_dir)

    # Create visualisations
    for io in ios:
        create_overview_animation(io, io.grid)


def preview_surface_and_gap(config: Config):
    """A visual check before running simulations."""
    import matplotlib.animation as ani
    import matplotlib.pyplot as plt
    import numpy as np

    from a_package.simulation.visualisation import latexify_plot

    grid = create_grid_from_config(config)
    h1 = generate_surface_from_config(grid, config.problem["upper"])
    h0 = generate_surface_from_config(grid, config.problem["lower"])
    trajectory = build_trajectory(config)

    latexify_plot(12)

    # create the figure and axes
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    def update_frame(i_frame: int):
        # clear content of last frames
        for ax in (ax1, ax2):
            ax.clear()

        [xm, ym] = grid.form_nodal_mesh()
        a = min(grid.element_sizes)
        # 3D surface plot of upper and lower rigid body
        ax1.plot_surface(xm, ym, h0 / a, cmap="berlin")
        ax1.plot_surface(xm, ym, (h1 + trajectory[i_frame]) / a, cmap="plasma")
        ax1.view_init(elev=0, azim=-45)
        ax1.set_xlabel(r"Position $x/a$")
        ax1.set_ylabel(r"Position $y/a$")
        ax1.set_zlabel(r"Position $z/a$")

        # 2D colour map of the gap
        h_diff = h1 - h0 + trajectory[i_frame]
        gap = np.clip(h_diff, 0, None)
        contact = np.ma.masked_where(gap > 0, gap)
        [nx, ny] = grid.nb_elements
        border = (0, nx, 0, ny)
        ax2.imshow(gap / a, vmin=0, interpolation="nearest", cmap="hot", extent=border)
        ax2.imshow(contact, cmap="Greys", vmin=-1, vmax=1, alpha=0.4, interpolation="nearest", extent=border)
        ax2.set_xlabel(r"Position $x/a$")
        ax2.set_ylabel(r"Position $y/a$")

        return *ax1.images, *ax2.images

    # draw the animation
    nb_steps = len(trajectory)
    _ = ani.FuncAnimation(fig, update_frame, nb_steps, interval=200, repeat_delay=3000)

    # allow to exit if it does not look right
    plt.show()
    skip = input("Run simulation [Y/n]? ").strip().lower() in ("n", "no")
    if skip:
        sys.exit(0)


if __name__ == "__main__":
    main()
