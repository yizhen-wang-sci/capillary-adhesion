"""Lateral offset is the swept quantity: the upper surface slides past the lower one."""

import numpy as np


def build_trajectory(config: dict) -> np.ndarray:
    """Build the sequence of lateral offsets to solve at.

    Recognised keys under ``[trajectory]``:
    - ``nb_pixels_to_slide`` (int, required)  -> offsets ``0 .. n-1``

    Returns:
        np.ndarray: Offsets in whole pixels, so a shift is a roll of the sampled surface.
    """
    return np.arange(config["trajectory"]["nb_pixels_to_slide"])
