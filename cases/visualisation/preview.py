"""
Config preview before running simulations.

Provides visual inspection of surfaces and gap trajectory.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from .primitives import latexify_plot


def preview_surface_and_gap(primitives: dict):
    """
    Visual check of surfaces and gap before running simulations.

    Shows 3D surface plot and 2D gap animation across trajectory.
    Prompts user to continue or abort.

    Parameters
    ----------
    primitives : dict
        Dictionary from inspect_config() containing:
        - grid: Grid object
        - upper: upper surface height array
        - lower: lower surface height array
        - trajectory: separation values array
    """
    grid = primitives["grid"]
    h1 = primitives["upper"]
    h0 = primitives["lower"]
    trajectory = primitives["trajectory"]

    latexify_plot(12)

    # Create figure with 3D surface and 2D gap views
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    def update_frame(i_frame: int):
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

    # Animate through trajectory
    nb_steps = len(trajectory)
    _ = ani.FuncAnimation(fig, update_frame, nb_steps, interval=200, repeat_delay=3000)

    # Show and prompt user
    plt.show()
    skip = input("Run simulation [Y/n]? ").strip().lower() in ("n", "no")
    if skip:
        sys.exit(0)
