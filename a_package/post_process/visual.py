import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb, is_color_like


def split_continuous_indices(i_arr):
    """
    Splits a 1-dimensional array of indices into contiguous sub-arrays.

    This function takes a 1-dimensional array of integer indices and divides it
    into multiple sub-arrays wherever the difference between successive elements
    is not equal to 1 (non-contiguous indices). Useful for segmenting continuous
    parts of indices.

    Parameters
    ----------
    i_arr : numpy.ndarray
        1-dimensional array containing integer indices. Assumes input is sorted
        in ascending order.

    Returns
    -------
    list of numpy.ndarray
        A list of 1-dimensional sub-arrays, each containing contiguous indices
        from the input array.
    """
    # Because numpy.diff will reduce the size of an array by 1, plus 1 to
    # compensate. The indexing is then correct for the original array.
    i_break = (np.diff(i_arr) != 1).nonzero()[0] + 1
    return np.split(i_arr, i_break)


def slice_colormap(cmap, low: float, high: float, bitwidth=8, name=None):
    """
    Extracts a portion of a given colormap and returns it as a new colormap. The function
    enables slicing a colormap between specified low and high values, optionally setting
    the number of samples and providing a name for the resulting colormap.

    Parameters
    ----------
    cmap : Union[str, Colormap]
        The input colormap, which can either be a string that specifies the name of a
        matplotlib colormap or a Colormap instance.
    low : float
        The lower bound of the colormap range to extract, specified as a fraction of the
        colormap domain. Must be between 0 and 1.
    high : float
        The upper bound of the colormap range to extract, specified as a fraction of the
        colormap domain. Must be between 0 and 1 and greater than or equal to `low`.
    bitwidth : int, optional
        The bitwidth used to determine the number of discrete samples in the colormap. The
        number of samples will be 2 raised to the power of `bitwidth`. Defaults to 8.
    name : str, optional
        The name for the resulting colormap. If not provided, a name will be automatically
        generated based on the input colormap and the specified `low` and `high` bounds.

    Returns
    -------
    Colormap
        A LinearSegmentedColormap instance corresponding to the extracted segment of the
        input colormap.
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
    """Return an (nb_steps, 4) RGBA array for LineCollection(colors=...).

    `source` decides the coloring mode:
      - a single color (name, hex, RGB/RGBA tuple)  -> constant hue,
        alpha ramps from `alpha_begin` to `alpha_end` (fading effect).
      - a colormap (name or Colormap object)         -> hue sweeps across
        the map from `low` to `high` (spectrum effect); alpha still ramps
        from `alpha_begin` to `alpha_end` on top of it (defaults to fully
        opaque, i.e. no fade, unless you set them).

    Parameters
    ----------
    source : color or str or Colormap
        A matplotlib color spec, or a registered colormap name / Colormap
        instance. See above for how each is interpreted.
    nb_steps : int
        Number of colors to generate (one per segment). Must be >= 1.
    low, high : float in [0, 1], keyword-only
        Sub-range of the colormap to sample. Ignored if `source` is a
        plain color.
    alpha_begin, alpha_end : float in [0, 1], keyword-only
        Alpha of the first and last entry. Equal (default 1.0, 1.0) means
        no fade -- fully opaque throughout. Use 0.0 -> 1.0 for fade-in,
        1.0 -> 0.0 for fade-out, or any pair in between.
    gamma : float, keyword-only
        Shapes the alpha ramp via t**gamma: 1.0 linear, >1 stays close to
        alpha_begin longer, <1 approaches alpha_end sooner.

    Returns
    -------
    numpy.ndarray, shape (nb_steps, 4)
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
