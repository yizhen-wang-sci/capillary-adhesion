"""The capillary phase mixture."""

import numpy as np

from a_package.model import PhaseMixture

from _common.grid import build_grid, build_length_unit


def build_phase_mixture(config: dict) -> PhaseMixture:
    """Build the capillary phase mixture.

    Recognised keys under ``[capillary]``:
    - ``length_unit`` (str, required)             -> unit of ``interface_thickness``
    - ``contact_angle_degree`` (float, required)  -> ``theta``
    - ``interface_thickness`` (float, required)   -> ``eta``
    - ``perimeter_weight`` (float, optional)      -> ``epsilon``; omit for the model's
      default of 1.0
    """
    section = config["capillary"]
    length_scale = build_length_unit(build_grid(config), section["length_unit"])
    args = dict(
        theta=(np.pi / 180) * section["contact_angle_degree"],
        eta=length_scale.to_physical(section["interface_thickness"]),
    )
    if section.get("perimeter_weight"):
        args["epsilon"] = section["perimeter_weight"]
    return PhaseMixture(**args)
