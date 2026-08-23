"""Fill the gap to a target volume by bisecting for a cut-off height.

The sharp-interface answer: every point is wet or dry by which side of the height its gap
falls on. Serves both as a case and as the initial guess for the minimisation ones. Collective.
"""

import logging

import numpy as np
from NuMPI import MPI

from a_package.model import CapillaryBridge


logger = logging.getLogger(__name__)


def solve_phase_by_level_set(
    capillary: CapillaryBridge, gap: np.ndarray, volume: float, fill_below: bool = True
) -> np.ndarray:
    """Solve for the phase field that fills `gap` to `volume`.

    Args:
        capillary: The bridge to fill. Its communicator decides the reduction, so it must be
            the one `gap` was decomposed over.
        gap: The local slice of the gap between the surfaces.
        volume: Target liquid volume.
        fill_below: True wets below the cut-off height (hydrophilic), False above it
            (hydrophobic).

    Returns:
        np.ndarray: The local phase field.
    """
    capillary.set_gap(gap)
    phase = np.zeros_like(gap)

    def fill_phase_at(height):
        if fill_below:
            to_fill = gap < height
        else:
            to_fill = gap > height
        phase[to_fill] = 1.0
        phase[~to_fill] = 0.0
        capillary.set_phase(phase)

    def compute_volume_deviation(height):
        fill_phase_at(height)
        return capillary.get_volume() - volume

    # `gap` is local, so propagate to find the global min and max.
    comm = capillary.communicator
    gap_min = comm.allreduce(gap.min(), op=MPI.MIN)
    gap_max = comm.allreduce(gap.max(), op=MPI.MAX)

    height = bisection(compute_volume_deviation, gap_min, gap_max, comm=comm)
    fill_phase_at(height)
    return capillary.get_phase()


def bisection(f, xa, xb, xtol=1e-6, ftol=1e-6, max_iter=100, comm=None):
    """Find a root of ``f(x) = 0`` by bisection.

    Args:
        f: The function to find a root of. Evaluated on every rank, and must agree on all.
        xa: Lower end of the bracket.
        xb: Upper end of the bracket. Must satisfy ``xa < xb``, with ``f`` changing sign across.
        xtol: Stop once the half-width ``(xb - xa) / 2`` is below this.
        ftol: Stop once ``abs(f(xc))`` is below this.
        max_iter: Warn and return the midpoint after this many iterations.
        comm: Selects the rank that logs; default COMM_WORLD.

    Returns:
        float: The approximate root.

    Raises:
        ValueError: If the bracket is empty, or ``f`` has the same sign at both ends.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if xa >= xb:
        raise ValueError(f"Requires xa < xb, got xa={xa}, xb={xb}")

    fa = f(xa)
    fb = f(xb)
    if fa * fb >= 0:
        raise ValueError("f(xa) and f(xb) must have opposite signs")

    for i_iter in range(max_iter):
        xc = (xa + xb) / 2
        fc = f(xc)
        if abs(fc) < ftol:
            if comm.rank == 0:
                logger.info(f"At iter #{i_iter}, ftol achieved. Root={xc}")
            return xc
        if (xb - xa) / 2 < xtol:
            if comm.rank == 0:
                logger.info(f"At iter #{i_iter}, xtol achieved. Root={xc}")
            return xc
        if fc * fa < 0:
            xb, fb = xc, fc
        else:
            xa, fa = xc, fc

    if comm.rank == 0:
        logger.warning(f"Bisection hit max_iter={max_iter} without converging. Root~={(xa + xb) / 2}")
    return (xa + xb) / 2
