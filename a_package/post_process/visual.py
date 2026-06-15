import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap


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
