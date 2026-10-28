"""The parameters a run holds fixed: whichever axis `[trajectory]` does not describe."""

from a_package.model import CapillaryBridge
from a_package.simulation import UnitConversion

from _common.grid import build_grid, build_length_unit


def liquid_volume_from_percent(capillary: CapillaryBridge, percent: float) -> float:
    """Convert a percentage of the gap's capacity into an absolute liquid volume.

    Args:
        capillary: Supplies the capacity, so its gap must already be set.
        percent: Percentage of that capacity.

    Returns:
        float: The absolute volume.
    """
    return capillary.get_max_volume() * (percent / 100.0)


def build_liquid_volume(capillary: CapillaryBridge, config: dict) -> float:
    """The fixed liquid volume.

    Recognised keys under ``[constraint]``:
    - ``liquid_volume_percent`` (float, required)  -> percentage of the gap's capacity
    """
    return liquid_volume_from_percent(capillary, config["constraint"]["liquid_volume_percent"])


def build_separation(config: dict) -> float:
    """The fixed mean separation.

    Recognised keys under ``[constraint]``:
    - ``separation`` (float, required)
    - ``length_unit`` (str) -> a grid-relative unit, as in `grid.build_length_unit`, or
    - ``length_scale`` (float) -> an absolute length scale carried into the grid's own
      (``[grid].length_scale``), the convention `surface_window` uses

    Which convention applies follows from the key the config writes.
    """
    section = config["constraint"]
    if "length_scale" in section:
        scale_grid = UnitConversion(config["grid"]["length_scale"])
        scale_separation = UnitConversion(section["length_scale"])
        return scale_grid.to_dimensionless(scale_separation.to_physical(section["separation"]))

    length_scale = build_length_unit(build_grid(config), section["length_unit"])
    return length_scale.to_physical(section["separation"])
