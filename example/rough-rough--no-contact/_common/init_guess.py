"""Starting phase fields: the cheap analytic alternative to a level-set fill."""

import numpy as np

from a_package.domain import Grid


def square_init_guess(grid: Grid, volume: float, mean_separation: float) -> np.ndarray:
    """A filled square in the middle of the domain, holding roughly `volume`.

    Args:
        grid: Supplies the element area and the pixel counts.
        volume: Target liquid volume.
        mean_separation: Sets the side length, by treating the bridge as a box of this height.

    Returns:
        np.ndarray: The whole domain, not decomposed -- take the local slice with
            ``grid.get_local(...)``.
    """
    half_nb_pixels = round(0.5 * np.sqrt(volume / mean_separation / grid.element_area))
    Nx, Ny = grid.nb_domain_grid_pts
    phase = np.zeros(grid.nb_domain_grid_pts)
    phase[Nx // 2 - half_nb_pixels : Nx // 2 + half_nb_pixels, Ny // 2 - half_nb_pixels : Ny // 2 + half_nb_pixels] = (
        1.0
    )
    return phase
