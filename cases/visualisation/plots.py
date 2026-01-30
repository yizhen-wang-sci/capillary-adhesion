"""
High-level plotting functions (SimulationIO-dependent).

These functions load data from SimulationIO and use primitives to draw.
"""

import numpy as np
import matplotlib.pyplot as plt

from a_package.domain import Grid
from a_package.model import Term
from a_package.simulation import SimulationIO

from .primitives import (
    get_length_unit,
    compute_extent,
    draw_field_2d,
    draw_masked_field_2d,
    draw_cross_section,
    draw_evolution_curve,
    cmap_height,
    cmap_gap,
    cmap_contact,
    cmap_phase_field,
    eps,
)


# =============================================================================
# Cross-section plots
# =============================================================================

def plot_cross_section_sketch(ax: plt.Axes, io: SimulationIO, idx_step: int, idx_row: int, value_cutoff=eps):
    """
    Plot cross-section with surfaces, contact, and phase regions.

    Wrapper that loads data and calls draw_cross_section primitive.
    """
    grid = io.grid
    unit = get_length_unit(grid)
    data = io.load_step(
        idx_step,
        field_names=[Term.upper_solid, Term.lower_solid, Term.phase],
        single_value_names=[Term.separation],
    )

    # Extract and nondimensionalize
    h_upper = data[Term.upper_solid][0, 0, idx_row]
    sep = data[Term.separation]
    h_upper = (h_upper + sep) / unit
    h_lower = data[Term.lower_solid][0, 0, idx_row] / unit
    x = grid.form_nodal_axis(0) / unit
    phi = data[Term.phase][0, 0, idx_row, :]

    # Add border values for periodic boundary condition
    h_upper = np.append(h_upper, h_upper[0])
    h_lower = np.append(h_lower, h_lower[0])
    x = np.append(x, grid.lengths[0] / unit)
    phi = np.append(phi, phi[0])

    draw_cross_section(ax, x, h_upper, h_lower, phi, cutoff=value_cutoff)


def plot_cross_section_phase_field(ax: plt.Axes, io: SimulationIO, idx_step: int, idx_row: int):
    """Plot phase field along a cross-section row."""
    data = io.load_step(idx_step, field_names=[Term.phase])
    phi = data[Term.phase][0, 0, idx_row, :]
    x_dimensionless = io.grid.form_nodal_axis(0) / min(io.grid.element_sizes)
    ax.plot(x_dimensionless, phi, color='C0')


# =============================================================================
# Topography plots (2D field views)
# =============================================================================

def plot_height_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot upper surface height field."""
    data = io.load_step(idx_step, field_names=[Term.upper_solid])
    unit = get_length_unit(io.grid)
    h = data[Term.upper_solid].squeeze() / unit
    extent = compute_extent(io.grid, unit)
    return draw_field_2d(ax, h, extent, cmap=cmap_height)


def plot_gap_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot gap field between surfaces."""
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = get_length_unit(io.grid)
    g = data[Term.gap].squeeze() / unit
    extent = compute_extent(io.grid, unit)
    return draw_field_2d(ax, g, extent, cmap=cmap_gap, vmin=0, vmax=g.max())


def plot_contact_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot contact regions (where gap <= 0)."""
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = get_length_unit(io.grid)
    g = data[Term.gap].squeeze() / unit
    extent = compute_extent(io.grid, unit)
    # Mask non-contact regions (gap > 0)
    return draw_masked_field_2d(ax, g, extent, mask=(g > 0),
                                 cmap=cmap_contact, vmin=-1, vmax=1, alpha=0.4)


def plot_droplet_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot liquid droplet with liquid and transition phases overlaid."""
    data = io.load_step(idx_step, field_names=[Term.phase])
    unit = get_length_unit(io.grid)
    extent = compute_extent(io.grid, unit)
    phi = data[Term.phase].squeeze()

    # Use partial colormap range (bluest blue is too dark)
    vmin, vmax = 0, 1.5

    # Liquid phase (phi >= 1 - eps)
    draw_masked_field_2d(ax, phi, extent, mask=(phi <= 1 - eps),
                         cmap=cmap_phase_field, vmin=vmin, vmax=vmax, alpha=0.85)

    # Transition phase (eps < phi < 1 - eps)
    im = draw_masked_field_2d(ax, phi, extent, mask=((phi <= eps) | (phi > 1 - eps)),
                              cmap=cmap_phase_field, vmin=vmin, vmax=vmax, alpha=0.7)
    return im


def plot_phase_field_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot raw phase field values."""
    data = io.load_step(idx_step, field_names=[Term.phase])
    unit = get_length_unit(io.grid)
    extent = compute_extent(io.grid, unit)
    phi = data[Term.phase].squeeze()
    return draw_field_2d(ax, phi, extent, cmap="Blues", vmin=0, vmax=2, interpolation="nearest")


def plot_combined_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot gap, contact, and droplet overlaid."""
    im1 = plot_gap_topography(ax, io, idx_step)
    im2 = plot_contact_topography(ax, io, idx_step)
    im4 = plot_droplet_topography(ax, io, idx_step)
    return im1, im2, im4


# =============================================================================
# Evolution curve plots
# =============================================================================

def plot_gibbs_free_energy(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot Gibbs free energy evolution."""
    data = io.load_trajectory(single_value_names=[Term.energy])
    energy = data[Term.energy][:nb_steps]

    # Non-dimensionalize
    unit = get_length_unit(io.grid)
    energy = energy / (unit**2)

    steps = np.arange(len(energy))
    draw_evolution_curve(ax, steps, energy, color="C1", marker="x", ms=5, label=r"$G$")


def plot_normal_force(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot normal force (numerical derivative of energy) evolution."""
    data = io.load_trajectory(single_value_names=[Term.energy, Term.separation])
    energy = data[Term.energy][:nb_steps]
    displ_z = data[Term.separation][:nb_steps]

    # Numerical difference to get force
    force = -(energy[1:] - energy[:-1]) / (displ_z[1:] - displ_z[:-1])

    # Non-dimensionalize
    unit = get_length_unit(io.grid)
    force = force / unit

    # Midpoint steps for derivative
    steps = (np.arange(len(energy))[1:] + np.arange(len(energy))[:-1]) / 2
    draw_evolution_curve(ax, steps, force, color="b", marker="o", ms=3, label=r"$F_z$")


def plot_pressure(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot pressure (Lagrange multiplier) evolution."""
    data = io.load_trajectory(single_value_names=[Term.pressure])
    pressure = data[Term.pressure][:nb_steps]

    # Non-dimensionalize
    unit = get_length_unit(io.grid)
    pressure = pressure * unit

    steps = np.arange(len(pressure))
    draw_evolution_curve(ax, steps, pressure, color="r", marker="o", ms=5, label=r"$P/\gamma a^{-1}$")
