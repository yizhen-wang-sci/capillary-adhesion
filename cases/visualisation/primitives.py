"""
Low-level drawing primitives (array-based, no IO dependency).

These functions take raw arrays and draw on matplotlib axes.
They have no knowledge of SimulationIO or data loading.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Color constants
# =============================================================================

cmap_height = "plasma"
cmap_gap = "hot"
cmap_contact = "Greys"
cmap_phase_field = "Blues"

color_gas_phase = "w"
color_solid_phase = "C7"
color_liquid_phase = "steelblue"
color_transition_phase = "lightblue"

eps = 1e-2  # cut off value to decide one phase


# =============================================================================
# Drawing primitives
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
# Utility functions
# =============================================================================

def hide_border(ax: plt.Axes):
    """Hide all borders of an axes."""
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)


def hide_ticks(ax: plt.Axes):
    """Hide all tick marks and labels."""
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)


def latexify_plot(font_size: int):
    """Configure matplotlib for LaTeX-style plots."""
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['helvetica'],
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
    }
    matplotlib.rcParams.update(params)
