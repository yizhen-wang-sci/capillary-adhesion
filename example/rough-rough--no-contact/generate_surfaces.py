import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a_package.domain import Grid
from a_package.model import SelfAffineRoughness, psd_to_height, Term
from a_package.simulation import SourceDir, load_config

from config_helper import *


visual_check = True


def main():
    # CLI arguments
    if len(sys.argv) != 2:
        raise ValueError(f"Provide one config file.")
    config_file = sys.argv[1]
    config = load_config(config_file)

    # setup
    cwd = SourceDir(os.path.dirname(__file__))

    # Generate surfaces
    grid = build_grid(config)
    upper, lower = generate_rough_surfaces(grid, config)

    # Print some info about the surfaces
    upper_max_amp = np.amax(abs(upper - upper.mean()))
    lower_max_amp = np.amax(abs(lower - lower.mean()))
    pixel_size = grid._element_sizes[0]
    print(f"Max amplitude, upper={upper_max_amp / pixel_size:.2e}px, lower={lower_max_amp / pixel_size:.2e}px")

    # visual check
    if visual_check:
        trajectory = build_trajectory(config)
        anime = preview_surface_approaching(grid, upper, lower, trajectory)
        plt.show()

        skip = input("Save [Y/n]? ").strip().lower() in ("n", "no")
        if skip:
            sys.exit(0)

    # save surfaces
    np.save(cwd / f"{Term.upper_solid}.npy", upper)
    np.save(cwd / f"{Term.lower_solid}.npy", lower)
    print(f"Saved surfaces to {cwd}")


def generate_rough_surfaces(grid: Grid, config: dict):
    """Generate rough surface from config dict."""

    surfaces = []
    for key in ["upper", "lower"]:
        section = config["surface"][key]
        # Convert wavelength in pixels to angular wavenumber
        a = grid.element_sizes[0]
        qR = (2 * np.pi) / (a * section["rolloff_wavelength_pixels"])
        qS = (2 * np.pi) / (a * section["cutoff_wavelength_pixels"])

        # Generate PSD and convert to height
        roughness = SelfAffineRoughness(C0=section["prefactor"], H=section["hurst_exponent"], qR=qR, qS=qS)
        qx, qy = grid.form_spectral_mesh()
        wavevector = np.stack([qx, qy], axis=0)
        C_2D = roughness.mapto_isotropic_psd(wavevector, component_axis=0)

        height = psd_to_height(C_2D, seed=config.get("seed"))
        surfaces.append(height)
    return surfaces


def preview_surface_approaching(grid: Grid, upper: np.ndarray, lower: np.ndarray, trajectory: np.ndarray):
    """Create animation previewing surfaces and gap through trajectory."""

    # Spatial coordinates
    x = grid.form_spatial_axis(1)
    y = grid.form_spatial_axis(0)
    X, Y = np.meshgrid(x, y)
    unit = min(grid.element_sizes)

    # Precompute gaps for axis limits
    gap_traj = []
    for sep in trajectory:
        gap = np.clip(upper + sep - lower, 0, None)
        gap_traj.append(gap)

    gap_min = min(np.amin(g) for g in gap_traj)
    gap_max = max(np.amax(g) for g in gap_traj)

    # Z limits for 3D plot
    z_min = np.amin(lower.squeeze()) / unit
    z_max = (np.amax(upper.squeeze()) + np.amax(trajectory)) / unit

    # Create figure
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_gap = fig.add_subplot(1, 2, 2)

    def update(i_frame):
        ax_3d.clear()
        ax_gap.clear()

        gap = gap_traj[i_frame]

        # Left: 3D surfaces
        lower_z = lower.squeeze() / unit
        separation = trajectory[i_frame]
        upper_z = (upper.squeeze() + separation) / unit

        ax_3d.plot_surface(
            X / unit, Y / unit, lower_z,
            alpha=0.7, cmap="Blues", edgecolor="none"
        )
        ax_3d.plot_surface(
            X / unit, Y / unit, upper_z,
            alpha=0.7, cmap="Greens", edgecolor="none"
        )
        ax_3d.set_xlabel(r"$x/a$")
        ax_3d.set_ylabel(r"$y/a$")
        ax_3d.set_zlabel(r"$z/a$")
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_title(f"Surfaces (sep={separation/unit:.2f}a)")

        # Right: Gap topography
        im = ax_gap.imshow(
            gap / unit,
            extent=[x[0] / unit, x[-1] / unit, y[0] / unit, y[-1] / unit],
            origin="lower",
            cmap="hot",
            vmin=gap_min / unit,
            vmax=gap_max / unit,
        )
        ax_gap.set_xlabel(r"$x/a$")
        ax_gap.set_ylabel(r"$y/a$")
        ax_gap.set_title("Gap")
        ax_gap.set_aspect("equal")

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(trajectory), interval=200, repeat_delay=2000
    )
    return anim


if __name__ == '__main__':
    main()
