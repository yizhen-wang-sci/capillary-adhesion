"""Liquid volume is the swept quantity: the bridge fills at a fixed separation."""

import numpy as np


def build_trajectory(config: dict) -> np.ndarray:
    """Build the sequence of liquid-volume percentages to solve at.

    Recognised keys under ``[trajectory]``:
    - ``min_volume_percent`` (float, required)         -> where the filling starts
    - ``max_volume_percent`` (float, required)         -> where it stops
    - ``nb_increments`` (int, required)                -> steps from min to max, so the filling
      has ``nb_increments + 1`` points
    - ``round_trip`` (bool, optional, default false)  -> append the draining, the filling
      reversed without repeating the turning point

    Returns:
        np.ndarray: Percentages of the gap's capacity, as in `constraint` -- turn a step into a
            volume with `constraint.liquid_volume_from_percent`.
    """
    section = config["trajectory"]
    percents = np.linspace(section["min_volume_percent"], section["max_volume_percent"], section["nb_increments"] + 1)

    if section.get("round_trip", False):
        percents = np.concatenate([percents, np.flip(percents)[1:]])

    return percents
