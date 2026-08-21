"""Utilities extracted from visual scripts."""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb, is_color_like


logger = logging.getLogger(__name__)


def split_continuous_indices(i_arr):
    """Split a 1-D array of indices into sub-arrays of contiguous indices.

    Args:
        i_arr: 1-D array of integer indices, sorted in ascending order.

    Returns:
        The sub-arrays, each holding one run of consecutive indices.
    """
    # Because numpy.diff will reduce the size of an array by 1, plus 1 to
    # compensate. The indexing is then correct for the original array.
    i_break = (np.diff(i_arr) != 1).nonzero()[0] + 1
    return np.split(i_arr, i_break)


def slice_colormap(cmap, low: float, high: float, bitwidth=8, name=None):
    """Extract a portion of a colormap, sampled into discrete colors.

    Args:
        cmap: Name of a matplotlib colormap, or a `Colormap` instance.
        low: Lower bound of the range to extract, as a fraction of the colormap domain.
        high: Upper bound of the range to extract, as a fraction of the colormap domain.
        bitwidth: Exponent setting the number of samples, 2**bitwidth. Defaults to 8.
        name: Name of the resulting colormap. Defaults to the input name followed by the two
            bounds.

    Returns:
        A `LinearSegmentedColormap` holding the extracted segment.

    Raises:
        TypeError: If `cmap` is neither a colormap name nor a `Colormap`.
        ValueError: If the bounds are not ordered as 0 <= low < high <= 1.
    """
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    if not isinstance(base, Colormap):
        raise TypeError(f"cmap must be a name or Colormap, got {type(base)}")
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"need 0 <= low < high <= 1, got low={low}, high={high}")

    if name is None:
        name = f"{base.name}[{low:.2f},{high:.2f}]"
    nb_samples = 2**bitwidth
    return LinearSegmentedColormap.from_list(name, base(np.linspace(low, high, nb_samples)), N=nb_samples)


def create_segment_colors(source, nb_steps, *,
                          low=0.0, high=1.0,
                          alpha_begin=1.0, alpha_end=1.0, gamma=1.0):
    """Build a sequence of RGBA colors, one per segment.

    Args:
        source: A matplotlib color, a registered colormap name, or a `Colormap`. A color
            holds the hue constant; a colormap sweeps the hue from `low` to `high`.
        nb_steps: Number of colors to generate. Must be >= 1.
        low: Lower bound of the colormap range to sample. Unused for a plain color.
        high: Upper bound of the colormap range to sample. Unused for a plain color.
        alpha_begin: Alpha of the first color. Defaults to 1.0.
        alpha_end: Alpha of the last color, and of a single one. Defaults to 1.0.
        gamma: Exponent shaping the alpha ramp, t**gamma. Defaults to 1.0, a linear ramp.

    Returns:
        Shape (nb_steps, 4), one RGBA row per step.

    Raises:
        ValueError: If `nb_steps` is below 1, the bounds are not ordered as
            0 <= low < high <= 1, an alpha lies outside [0, 1], or `source` is neither a
            color nor a colormap.
    """
    if nb_steps < 1:
        raise ValueError(f"nb_steps must be >= 1, got {nb_steps}")
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"need 0 <= low < high <= 1, got low={low}, high={high}")
    for name, val in (("alpha_begin", alpha_begin), ("alpha_end", alpha_end)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    if is_color_like(source):
        # If it is a color, create a sequence of the same color
        colors = np.empty((nb_steps, 4))
        colors[:, :3] = to_rgb(source)
    else:
        # If not a color, it should be a colormap
        if isinstance(source, str):
            try:
                cmap = plt.get_cmap(source)
            except ValueError:
                raise ValueError(
                    f"{source!r} is not a valid color or a registered "
                    f"colormap name")
        elif isinstance(source, Colormap):
            cmap = source
        else:
            raise ValueError(
                f"source must be a color, colormap name, or Colormap "
                f"instance; got {source!r}")
        # Create a color gradient from a slice of the colormap
        colors = cmap(np.linspace(low, high, nb_steps))   # (nb_steps, 4)

    if nb_steps == 1:
        colors[:, 3] = alpha_end
    else:
        t = np.linspace(0.0, 1.0, nb_steps) ** gamma
        colors[:, 3] = alpha_begin + (alpha_end - alpha_begin) * t
    return colors


def divide_into_segments(x, y, *, nb_segments=None):
    """Group a series of data points into segments, consecutive ones sharing a boundary point.

    Args:
        x: 1-D sequence of coordinates, holding at least 2 points.
        y: 1-D sequence of coordinates, of the same length as `x`.
        nb_segments: Number of segments. Defaults to nb_points - 1, and is capped at that
            value with a warning.

    Returns:
        `nb_segments` arrays of shape (nb_points_in_segment, 2). The point counts differ by
        one where the points do not divide evenly.

    Raises:
        ValueError: If `x` and `y` differ in length, fewer than 2 points are given, or
            `nb_segments` is below 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x and y must have the same length; got {x.size} and {y.size}")
    if x.size < 2:
        raise ValueError(
            f"need at least 2 points to form a segment; got {x.size}")

    max_segments = x.size - 1
    if nb_segments is None:
        nb_segments = max_segments
    if nb_segments < 1:
        raise ValueError(f"nb_segments must be >= 1, got {nb_segments}")

    # Don't create points to satisfiy the nb_segments
    if nb_segments > max_segments:
        logger.warning(f"nb_segments {nb_segments} exceeds max_segments {max_segments}; reducing to max_segments")
    nb_segments = min(nb_segments, max_segments)

    # Round up so boundary indices are integers
    boundary_idxs = np.round(np.linspace(0, x.size - 1, nb_segments + 1)).astype(int)

    # Plus 1 at stop indices so it includes the ending point, which results in continuous line segments
    return [np.column_stack([x[boundary_idxs[i_segm]:boundary_idxs[i_segm + 1] + 1],
                             y[boundary_idxs[i_segm]:boundary_idxs[i_segm + 1] + 1]])
            for i_segm in range(nb_segments)]
