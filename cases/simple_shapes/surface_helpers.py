"""
Surface geometry generators for simple shapes.

Each generator creates a height field on a given grid based on surface parameters.
"""

from typing import Any

import numpy as np

from a_package.domain import Grid


def get_surface_shape(config: dict, which: str) -> str:
    """
    Get the shape name for a surface.

    Parameters
    ----------
    config : dict
        The configuration dict.
    which : str
        Either "upper" or "lower".

    Returns
    -------
    str
        The surface shape name.
    """
    return config["problem"][which]["shape"]


def generate_surface_from_config(grid: Grid, surface_cfg: dict[str, Any]) -> np.ndarray:
    """
    Generate a surface from configuration dict.

    Extracts shape and passes remaining params to generate_surface.
    """
    cfg = dict(surface_cfg)  # copy to avoid mutation
    shape = cfg.pop("shape")
    return generate_surface(grid, shape, **cfg)


# Registry of surface generators
_generators: dict[str, callable] = {}


def _register(shape: str):
    """Decorator to register a surface generator."""
    def decorator(func):
        _generators[shape] = func
        return func
    return decorator


def generate_surface(grid: Grid, shape: str, **params) -> np.ndarray:
    """
    Generate a surface height field based on shape and parameters.

    Parameters
    ----------
    grid : Grid
        The computational grid.
    shape : str
        Surface type identifier ("flat", "tip", "sinusoid").
    **params
        Shape-specific parameters.

    Returns
    -------
    np.ndarray
        Height field array with shape matching grid.nb_elements.

    Examples
    --------
    >>> generate_surface(grid, "flat", constant=0.0)
    >>> generate_surface(grid, "tip", radius=10.0)
    >>> generate_surface(grid, "sinusoid", wavenumber=2.0, amplitude=0.1)
    """
    if shape not in _generators:
        available = list(_generators.keys())
        raise ValueError(f"Unknown surface shape: {shape}. Available: {available}")

    return _generators[shape](grid, **params)


@_register("flat")
def _generate_flat(grid: Grid, constant: float = 0.0) -> np.ndarray:
    """Generate a flat surface at constant height."""
    return constant * np.ones(grid.nb_elements)


@_register("tip")
def _generate_tip(grid: Grid, radius: float) -> np.ndarray:
    """Generate a spherical tip (paraboloid approximation)."""
    R = radius
    [lx, ly] = grid.lengths
    x_center = 0.5 * lx
    y_center = 0.5 * ly
    [x, y] = grid.form_nodal_mesh()
    height = -np.sqrt(np.clip(R**2 - (x - x_center)**2 - (y - y_center)**2, 0, None))
    # Set lowest point to zero
    height += np.amax(abs(height))
    return height


@_register("sinusoid")
def _generate_sinusoid(grid: Grid, wavenumber: float, amplitude: float) -> np.ndarray:
    """Generate a sinusoidal surface."""
    [x, y] = grid.form_nodal_mesh()
    height = amplitude * np.cos(wavenumber * x) * np.cos(wavenumber * y)
    return height
