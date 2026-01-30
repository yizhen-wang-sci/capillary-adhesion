"""
Low-level drawing primitives (array-based, no IO dependency).

These functions take raw arrays and draw on matplotlib axes.
They have no knowledge of SimulationIO or data loading.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from a_package.domain import Grid


# =============================================================================
# Grid helpers
# =============================================================================

def get_length_unit(grid: Grid):
    """Get characteristic length unit from grid (smallest element size)."""
    return min(grid.element_sizes)


def compute_extent(grid: Grid, unit: float):
    """Compute imshow extent from grid dimensions."""
    return (0, grid.lengths[0] / unit, 0, grid.lengths[1] / unit)


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


# =============================================================================
# Triangularized surface visualization
# =============================================================================

def build_triangulation(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
    """
    Build triangulation for a 2D grid.

    Each pixel is split into two triangles following the FEM convention:
        (i,j) ---- (i+1,j)
          |     /   |
          | 0  /  1 |
          |   /     |
        (i,j+1) -- (i+1,j+1)

    Triangle 0: (i,j), (i+1,j), (i,j+1)
    Triangle 1: (i+1,j), (i+1,j+1), (i,j+1)

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions.
    dx, dy : float
        Element sizes in x and y directions.

    Returns
    -------
    x, y : np.ndarray
        1D arrays of vertex x and y coordinates (length (nx+1)*(ny+1)).
    triangles : np.ndarray
        Triangle connectivity array of shape (2*nx*ny, 3).
    """
    # Vertex coordinates (including boundary nodes for non-periodic viz)
    x_nodes = np.arange(nx + 1) * dx
    y_nodes = np.arange(ny + 1) * dy
    xv, yv = np.meshgrid(x_nodes, y_nodes)
    x = xv.ravel()
    y = yv.ravel()

    # Build triangle connectivity
    triangles = []
    for j in range(ny):
        for i in range(nx):
            # Vertex indices in the flattened array
            v00 = j * (nx + 1) + i          # (i, j)
            v10 = j * (nx + 1) + (i + 1)    # (i+1, j)
            v01 = (j + 1) * (nx + 1) + i    # (i, j+1)
            v11 = (j + 1) * (nx + 1) + (i + 1)  # (i+1, j+1)

            # Triangle 0: (i,j), (i+1,j), (i,j+1)
            triangles.append([v00, v10, v01])
            # Triangle 1: (i+1,j), (i+1,j+1), (i,j+1)
            triangles.append([v10, v11, v01])

    return x, y, np.array(triangles)


def expand_field_to_vertices(field: np.ndarray, periodic: bool = True):
    """
    Expand a nodal field to include boundary vertices for triangulation.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (ny, nx) with nodal values.
    periodic : bool
        If True, wrap values for periodic boundary conditions.

    Returns
    -------
    np.ndarray
        Expanded array of shape (ny+1, nx+1).
    """
    ny, nx = field.shape
    expanded = np.zeros((ny + 1, nx + 1))
    expanded[:ny, :nx] = field

    if periodic:
        # Wrap periodic boundaries
        expanded[:ny, nx] = field[:, 0]
        expanded[ny, :nx] = field[0, :]
        expanded[ny, nx] = field[0, 0]
    else:
        # Extrapolate (simple copy of edge values)
        expanded[:ny, nx] = field[:, -1]
        expanded[ny, :nx] = field[-1, :]
        expanded[ny, nx] = field[-1, -1]

    return expanded


def draw_triangulated_surface_2d(
    ax: plt.Axes,
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolors: str = "k",
    linewidth: float = 0.5,
    show_mesh: bool = True,
):
    """
    Draw a 2D triangulated color plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    field : np.ndarray
        2D array of shape (ny, nx) with nodal values.
    dx, dy : float
        Element sizes in x and y directions.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    edgecolors : str
        Edge color for triangles. Use 'none' to hide edges.
    linewidth : float
        Edge line width.
    show_mesh : bool
        Whether to show triangle edges.

    Returns
    -------
    TriContourSet or PolyCollection
        The plot object (can be used for colorbars).
    """
    ny, nx = field.shape

    # Build triangulation
    x, y, triangles = build_triangulation(nx, ny, dx, dy)

    # Expand field to include boundary vertices
    expanded = expand_field_to_vertices(field, periodic=True)
    z = expanded.ravel()

    # Create triangulation object
    triang = matplotlib.tri.Triangulation(x, y, triangles)

    # Draw with tripcolor
    if show_mesh:
        tpc = ax.tripcolor(
            triang, z, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors=edgecolors, linewidth=linewidth
        )
    else:
        tpc = ax.tripcolor(
            triang, z, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors='none'
        )

    ax.set_aspect('equal')
    return tpc


def draw_triangulated_surface_3d(
    ax,
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolor: str = "k",
    linewidth: float = 0.3,
    alpha: float = 1.0,
    show_mesh: bool = True,
):
    """
    Draw a 3D triangulated surface plot.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes to draw on.
    field : np.ndarray
        2D array of shape (ny, nx) with nodal values (heights).
    dx, dy : float
        Element sizes in x and y directions.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    edgecolor : str
        Edge color for triangles. Use 'none' to hide edges.
    linewidth : float
        Edge line width.
    alpha : float
        Surface transparency (0-1).
    show_mesh : bool
        Whether to show triangle edges.

    Returns
    -------
    Poly3DCollection
        The surface object (can be used for colorbars).
    """
    ny, nx = field.shape

    # Build triangulation
    x, y, triangles = build_triangulation(nx, ny, dx, dy)

    # Expand field to include boundary vertices
    expanded = expand_field_to_vertices(field, periodic=True)
    z = expanded.ravel()

    # Draw with plot_trisurf
    if show_mesh:
        surf = ax.plot_trisurf(
            x, y, z, triangles=triangles,
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor=edgecolor, linewidth=linewidth,
            alpha=alpha
        )
    else:
        surf = ax.plot_trisurf(
            x, y, z, triangles=triangles,
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor='none',
            alpha=alpha
        )

    return surf


def draw_triangulated_surface_combined(
    field: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    *,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    show_mesh: bool = True,
    figsize: tuple = (12, 5),
    title: str | None = None,
):
    """
    Draw both 2D and 3D triangulated surface plots side by side.

    Parameters
    ----------
    field : np.ndarray
        2D array of shape (ny, nx) with nodal values.
    dx, dy : float
        Element sizes in x and y directions.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits. If None, auto-scaled to field range.
    show_mesh : bool
        Whether to show triangle edges.
    figsize : tuple
        Figure size (width, height).
    title : str, optional
        Figure title.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    axes : tuple
        Tuple of (ax_2d, ax_3d) axes.
    """
    # Auto-scale if not provided
    if vmin is None:
        vmin = field.min()
    if vmax is None:
        vmax = field.max()

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    # 2D plot
    tpc = draw_triangulated_surface_2d(
        ax_2d, field, dx, dy,
        cmap=cmap, vmin=vmin, vmax=vmax, show_mesh=show_mesh
    )
    ax_2d.set_xlabel("x")
    ax_2d.set_ylabel("y")
    ax_2d.set_title("2D Triangulated View")
    fig.colorbar(tpc, ax=ax_2d, shrink=0.8)

    # 3D plot
    surf = draw_triangulated_surface_3d(
        ax_3d, field, dx, dy,
        cmap=cmap, vmin=vmin, vmax=vmax, show_mesh=show_mesh
    )
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title("3D Triangulated View")
    fig.colorbar(surf, ax=ax_3d, shrink=0.6, pad=0.1)

    if title:
        fig.suptitle(title)

    return fig, (ax_2d, ax_3d)
