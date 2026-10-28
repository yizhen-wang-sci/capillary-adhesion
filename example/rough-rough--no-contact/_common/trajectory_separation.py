"""Separation is the swept quantity: the surfaces approach and optionally retract.

The preview animation is here: one frame per separation.
"""

import numpy as np

from a_package.domain import Grid

from _common.grid import build_grid, build_length_unit


def build_trajectory(config: dict) -> np.ndarray:
    """Build the sequence of mean separations to solve at.

    Recognised keys under ``[trajectory]``:
    - ``length_unit`` (str, required)                 -> unit of the two separations below
    - ``max_separation`` (float, required)            -> where the approach starts
    - ``min_separation`` (float, required)            -> where it turns around
    - ``nb_decrements`` (int, required)               -> steps from max to min, so the approach
      has ``nb_decrements + 1`` points
    - ``round_trip`` (bool, optional, default true)   -> append the retraction, the approach
      reversed without repeating the turning point
    """
    section = config["trajectory"]
    length_scale = build_length_unit(build_grid(config), section["length_unit"])
    d_max = length_scale.to_physical(section["max_separation"])
    d_min = length_scale.to_physical(section["min_separation"])
    separations = np.linspace(d_max, d_min, section["nb_decrements"] + 1)

    if section.get("round_trip", True):
        separations = np.concatenate([separations, np.flip(separations)[1:]])

    return separations


def preview_surface_approaching(grid: Grid, upper: np.ndarray, lower: np.ndarray, separations: np.ndarray):
    """Animate the surfaces and the gap along the trajectory, one frame per separation.

    Args:
        grid: Supplies the spatial axes and the pixel size.
        upper: Whole height profile, as `surface.build_surface` returns it.
        lower: Whole height profile, as `surface.build_surface` returns it.
        separations: One frame each. Pass a one-element array for a case that sweeps something
            else.

    Returns:
        matplotlib.animation.FuncAnimation: Serial and interactive. Keep a reference to it and
            call ``plt.show()``, or it is collected before it draws.
    """
    # in the body, so a `simulate.py` taking `build_trajectory` from this module carries no
    # plotting stack onto a compute node
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Spatial coordinates
    x = grid.form_spatial_axis(1)
    y = grid.form_spatial_axis(0)
    X, Y = np.meshgrid(x, y)
    pixel_size = grid.element_sizes[0]

    # The gaps, for the axis limits
    gap_traj = [np.clip(upper + separation - lower, 0, None) for separation in separations]
    gap_max = max(np.amax(g) for g in gap_traj)

    # The z limits of the 3D plot
    z_min = np.amin(lower.squeeze()) / pixel_size
    z_max = (np.amax(upper.squeeze()) + np.amax(separations)) / pixel_size

    # The figure
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_gap = fig.add_subplot(1, 2, 2)

    def update(i_frame):
        ax_3d.clear()
        ax_gap.clear()

        gap = gap_traj[i_frame]

        # Left: the surfaces
        lower_z = lower.squeeze() / pixel_size
        separation = separations[i_frame]
        upper_z = (upper.squeeze() + separation) / pixel_size

        ax_3d.plot_surface(X / pixel_size, Y / pixel_size, lower_z, alpha=0.7, cmap="Blues", edgecolor="none")
        ax_3d.plot_surface(X / pixel_size, Y / pixel_size, upper_z, alpha=0.7, cmap="Greens", edgecolor="none")
        ax_3d.set_xlabel(r"$x$" + " (pixel)")
        ax_3d.set_ylabel(r"$y$" + " (pixel)")
        ax_3d.set_zlabel(r"$z$" + " (pixel)")
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_title(f"Surfaces (sep={separation / pixel_size:.2f}px)")

        # Right: the gap
        ax_gap.imshow(
            gap / pixel_size,
            extent=[x[0] / pixel_size, x[-1] / pixel_size, y[0] / pixel_size, y[-1] / pixel_size],
            origin="lower",
            cmap="hot",
            vmin=0,
            vmax=gap_max / pixel_size,
        )
        ax_gap.set_xlabel(r"$x$" + " (pixel)")
        ax_gap.set_ylabel(r"$y$" + " (pixel)")
        ax_gap.set_title("Gap")
        ax_gap.set_aspect("equal")

        return []

    return animation.FuncAnimation(fig, update, frames=len(separations), interval=200, repeat_delay=2000)
