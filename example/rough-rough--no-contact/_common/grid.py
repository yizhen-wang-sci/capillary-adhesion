"""The computational grid, and the unit that config lengths are written in."""

from a_package.domain import Grid
from a_package.simulation import UnitConversion


def build_grid(config: dict) -> Grid:
    """Create a square grid.

    Recognised keys under ``[grid]``:
    - ``nb_pixels`` (int, required)       -> points per side
    - ``lateral_size`` (float, required)  -> domain length per side
    """
    section = config["grid"]
    N = section["nb_pixels"]
    L = section["lateral_size"]
    return Grid([N, N], [L, L])


def build_length_unit(grid: Grid, unit: str) -> UnitConversion:
    """The unit that a ``length_unit = "..."`` key selects.

    Args:
        grid: Supplies the reference lengths.
        unit: ``"lateral"`` (the domain side) or ``"pixel"`` (one element).

    Returns:
        UnitConversion: Converts a config number into a physical length.

    Raises:
        ValueError: If the unit is neither.
    """
    match unit:
        case "lateral":
            return UnitConversion(grid.domain_lengths[0])
        case "pixel":
            return UnitConversion(grid.element_sizes[0])
        case _:
            raise ValueError(f"Unknown length unit: {unit}")
