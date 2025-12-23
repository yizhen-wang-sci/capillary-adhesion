import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

from a_package.domain import Grid
from a_package.problem import SelfAffineRoughness

from .io import SimulationIO, Term


# Setup for colours
cmap_height = "plasma"
cmap_gap = "hot"
cmap_contact = "Greys"
cmap_phase_field = "Blues"

color_gas_phase = "w"
color_solid_phase = "C7"
# color_liquid_phase = "steelblue"
# color_transition_phase = "lightskyblue"
# color_vapour_phase = "aliceblue"
color_liquid_phase = "steelblue"
color_transition_phase = "lightblue"


eps = 1e-2  # cut off value to decide one phase


# =============================================================================
# Shared helpers
# =============================================================================

def _nondimensionalize(grid):
    """Get nondimensionalization unit from grid."""
    return min(grid.element_sizes)


def _compute_extent(grid, unit):
    """Compute imshow extent from grid dimensions."""
    return (0, grid.lengths[0] / unit, 0, grid.lengths[1] / unit)


# =============================================================================
# Low-level primitives (array-based, no io)
# =============================================================================

def draw_field_2d(
    ax: plt.Axes,
    field: np.ndarray,
    extent: tuple,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    interpolation: str = "none",
):
    """
    Draw a 2D field as an image.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    field : np.ndarray
        2D array of values to display.
    extent : tuple
        (xmin, xmax, ymin, ymax) for axis limits.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    alpha : float
        Transparency (0-1).
    interpolation : str
        Interpolation method.

    Returns
    -------
    AxesImage
        The image object (can be used for colorbars).
    """
    return ax.imshow(
        field,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation=interpolation,
    )


def draw_masked_field_2d(
    ax: plt.Axes,
    field: np.ndarray,
    extent: tuple,
    mask: np.ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    interpolation: str = "nearest",
):
    """
    Draw a 2D field with masking.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    field : np.ndarray
        2D array of values to display.
    extent : tuple
        (xmin, xmax, ymin, ymax) for axis limits.
    mask : np.ndarray
        Boolean array where True means masked (hidden).
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    alpha : float
        Transparency (0-1).
    interpolation : str
        Interpolation method.

    Returns
    -------
    AxesImage
        The image object.
    """
    masked_field = np.ma.masked_where(mask, field)
    return ax.imshow(
        masked_field,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation=interpolation,
    )


def draw_cross_section(
    ax: plt.Axes,
    x: np.ndarray,
    h_upper: np.ndarray,
    h_lower: np.ndarray,
    phi: np.ndarray | None = None,
    *,
    cutoff: float = eps,
    color_solid: str | None = None,
    color_liquid: str | None = None,
    color_transition: str | None = None,
    show_legend: bool = True,
):
    """
    Draw a filled cross-section with surfaces, contact, and phase regions.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    x : np.ndarray
        1D array of x coordinates.
    h_upper : np.ndarray
        1D array of upper surface heights.
    h_lower : np.ndarray
        1D array of lower surface heights.
    phi : np.ndarray, optional
        1D array of phase field values. If None, only surfaces and contact shown.
    cutoff : float
        Phase threshold for liquid/transition classification.
    color_solid, color_liquid, color_transition : str, optional
        Colors for regions. Uses module defaults if None.
    show_legend : bool
        Whether to show legend.
    """
    # Use module defaults if not specified
    _color_solid = color_solid or color_solid_phase
    _color_liquid = color_liquid or color_liquid_phase
    _color_transition = color_transition or color_transition_phase

    # Draw surface lines
    ax.plot(x, h_upper, "k-")
    ax.plot(x, h_lower, "k-")

    # Helper to fill contiguous regions
    def fill_regions(indices, color):
        if np.size(indices):
            i_diff = np.diff(indices, prepend=indices[0] - 1)
            i_break = np.hstack((i_diff > 1).nonzero())
            for section in np.split(indices, i_break):
                ax.fill_between(x[section], h_lower[section], h_upper[section], color=color)

    # Fill contact regions (where upper < lower, i.e. overlap)
    at_contact = np.asanyarray(h_upper < h_lower).nonzero()[0]
    fill_regions(at_contact, _color_solid)

    # Fill phase regions if phase field provided
    if phi is not None:
        liquid_phase = np.asarray(phi >= 1 - cutoff).nonzero()[0]
        fill_regions(liquid_phase, _color_liquid)

        transition_phase = np.asarray((phi < 1 - cutoff) & (phi > 0 + cutoff)).nonzero()[0]
        fill_regions(transition_phase, _color_transition)

    # Legend with dummy patches
    if show_legend:
        handles = []
        if phi is not None:
            [p_interface] = ax.fill(np.nan, np.nan, _color_transition, label="Interface")
            [p_liquid] = ax.fill(np.nan, np.nan, _color_liquid, label="Liquid")
            handles.extend([p_interface, p_liquid])
        [p_solid] = ax.fill(np.nan, np.nan, _color_solid, label="Solid")
        handles.append(p_solid)
        ax.legend(handles=handles, loc="upper center", ncol=len(handles))

    # No view margin along x-axis
    ax.set_xlim(x[0], x[-1])


def draw_evolution_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str = "C0",
    linestyle: str = "-",
    marker: str = "o",
    ms: float = 3,
    label: str | None = None,
    show_grid: bool = True,
    show_legend: bool = True,
):
    """
    Draw an evolution curve with markers.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    x : np.ndarray
        1D array of x values (e.g., step numbers).
    y : np.ndarray
        1D array of y values.
    color : str
        Line and marker color.
    linestyle : str
        Line style.
    marker : str
        Marker style.
    ms : float
        Marker size.
    label : str, optional
        Legend label.
    show_grid : bool
        Whether to show grid.
    show_legend : bool
        Whether to show legend (only if label provided).
    """
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, ms=ms, mfc="none", label=label)

    if show_grid:
        ax.grid()

    if show_legend and label:
        ax.legend(loc="upper right")


# =============================================================================
# High-level wrappers (SimulationIO-dependent)
# =============================================================================

def plot_cross_section_sketch(ax: plt.Axes, io: SimulationIO, idx_step: int, idx_row: int, value_cutoff=eps):
    """
    Plot cross-section with surfaces, contact, and phase regions.

    Wrapper that loads data and calls draw_cross_section primitive.
    """
    # FIXME: only shift in x-axis is considered.
    grid = io.grid
    unit = _nondimensionalize(grid)
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
    data = io.load_step(idx_step, field_names=[Term.phase])
    phi = data[Term.phase][0, 0, idx_row,:]
    x_dimensionless = io.grid.form_nodal_axis(0) / min(io.grid.element_sizes)
    ax.plot(x_dimensionless, phi, color='C0')


def plot_height_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot upper surface height field."""
    data = io.load_step(idx_step, field_names=[Term.upper_solid])
    unit = _nondimensionalize(io.grid)
    h = data[Term.upper_solid].squeeze() / unit
    extent = _compute_extent(io.grid, unit)
    return draw_field_2d(ax, h, extent, cmap=cmap_height)


def plot_gap_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot gap field between surfaces."""
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = _nondimensionalize(io.grid)
    g = data[Term.gap].squeeze() / unit
    extent = _compute_extent(io.grid, unit)
    return draw_field_2d(ax, g, extent, cmap=cmap_gap, vmin=0, vmax=g.max())


def plot_contact_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot contact regions (where gap <= 0)."""
    data = io.load_step(idx_step, field_names=[Term.gap])
    unit = _nondimensionalize(io.grid)
    g = data[Term.gap].squeeze() / unit
    extent = _compute_extent(io.grid, unit)
    # Mask non-contact regions (gap > 0)
    return draw_masked_field_2d(ax, g, extent, mask=(g > 0),
                                 cmap=cmap_contact, vmin=-1, vmax=1, alpha=0.4)


def plot_droplet_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """Plot liquid droplet with liquid and transition phases overlaid."""
    data = io.load_step(idx_step, field_names=[Term.phase])
    unit = _nondimensionalize(io.grid)
    extent = _compute_extent(io.grid, unit)
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
    unit = _nondimensionalize(io.grid)
    extent = _compute_extent(io.grid, unit)
    phi = data[Term.phase].squeeze()
    return draw_field_2d(ax, phi, extent, cmap="Blues", vmin=0, vmax=2, interpolation="nearest")


# NOT UPDATED: Has hardcoded pixel size, needs refactoring
def plot_interface_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    """[NOT UPDATED] Plot phase interface using gradient-based edge detection."""
    data = io.load_step(idx_step, field_names=[Term.phase])
    dphi = np.gradient(data[Term.phase].squeeze())
    edge = sum(dphi_i**2 for dphi_i in dphi) > 1e-6
    interface = np.where(edge, data[Term.phase].squeeze(), 0)
    # NOTE: hard-coded pixel size 0.1
    border = np.array([0, io.grid.lengths[0], 0, io.grid.lengths[1]]) / 1e-1
    im = ax.imshow(interface, cmap="binary", vmin=0.0, vmax=None, interpolation='nearest', extent=border)
    return im


def plot_combined_topography(ax: plt.Axes, io: SimulationIO, idx_step: int):
    im1 = plot_gap_topography(ax, io, idx_step)
    im2 = plot_contact_topography(ax, io, idx_step)
    # im3 = plot_interface_topography(ax, io, idx_step)
    im4 = plot_droplet_topography(ax, io, idx_step)
    # im4 = plot_phase_field_topography(ax, io, idx_step)
    return im1, im2, im4


# NOT UPDATED: Legacy function, may not follow new conventions
def demonstrate_dynamics(ax: plt.Axes, io: SimulationIO):
    """[NOT UPDATED] Plot separation vs step index."""
    data = io.load_trajectory(single_value_names=[Term.separation])
    unit = min(io.grid.element_sizes)
    ax.plot(data[Term.separation] / unit)


def plot_gibbs_free_energy(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot Gibbs free energy evolution."""
    data = io.load_trajectory(single_value_names=[Term.energy])
    energy = data[Term.energy][:nb_steps]

    # Non-dimensionalize
    # NOTE: actually needs to be divided again by 'gamma' (surface tension), but 'gamma' is symbolic so far.
    unit = _nondimensionalize(io.grid)
    energy = energy / (unit**2)

    steps = np.arange(len(energy))
    draw_evolution_curve(ax, steps, energy, color="C1", marker="x", ms=5, label=r"$G$")


# NOT UPDATED: Standalone demo function, not used in production
def plot_PSD(ax: plt.Axes):
    """[NOT UPDATED] Plot power spectral density (standalone demo)."""
    # TODO: change to sample the PSD from the height profile of a rough surface
    L = 10           # spatial dimension
    n_grid = 200     # samples in spatial domain
    grid = Grid([L, L], [n_grid, n_grid])

    qR = 2e0  # roll-off
    qS = 2e1  # cut-off
    C0 = 1e7  # prefactor
    H = 0.95  # Hurst exponent
    roughness = SelfAffineRoughness(C0, qR, qS, H)

    # isotropic PSD
    q_iso = grid.form_spectral_axis(0)
    ax.loglog(fft.fftshift(q_iso), fft.fftshift(roughness.mapto_isotropic_psd(q_iso)))
    ax.axvline(abs(q_iso[q_iso.nonzero()]).min(), color="r", linestyle="--")
    ax.axvline(q_iso.max(), color="r", linestyle="--")

    ax.grid()


def plot_normal_force(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot normal force (numerical derivative of energy) evolution."""
    data = io.load_trajectory(single_value_names=[Term.energy, Term.separation])
    energy = data[Term.energy][:nb_steps]
    displ_z = data[Term.separation][:nb_steps]

    # Numerical difference to get force
    force = -(energy[1:] - energy[:-1]) / (displ_z[1:] - displ_z[:-1])

    # Non-dimensionalize
    # NOTE: actually needs to be divided by 'eta gamma', but 'gamma' is symbolic so far.
    unit = _nondimensionalize(io.grid)
    force = force / unit

    # Midpoint steps for derivative
    steps = (np.arange(len(energy))[1:] + np.arange(len(energy))[:-1]) / 2
    draw_evolution_curve(ax, steps, force, color="b", marker="o", ms=3, label=r"$F_z$")


def plot_pressure(ax: plt.Axes, io: SimulationIO, nb_steps: int | None = None):
    """Plot pressure (Lagrange multiplier) evolution."""
    data = io.load_trajectory(single_value_names=[Term.pressure])
    pressure = data[Term.pressure][:nb_steps]

    # Non-dimensionalize
    unit = _nondimensionalize(io.grid)
    pressure = pressure * unit

    steps = np.arange(len(pressure))
    draw_evolution_curve(ax, steps, pressure, color="r", marker="o", ms=5, label=r"$P/\gamma a^{-1}$")


def hide_border(ax: plt.Axes):
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)


def hide_ticks(ax: plt.Axes):
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)


# TO ORGANIZE: The following functions do not have an `ax` parameter, should be organized differently?
def latexify_plot(font_size: int):
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['helvetica'],
        # 'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
    }
    matplotlib.rcParams.update(params)
