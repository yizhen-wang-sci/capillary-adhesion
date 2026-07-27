# import numpy as np
# import numpy.linalg as la
# import matplotlib.pyplot as plt

# import SurfaceTopography.Uniform.GeometryAnalysis as ga


# def _borders(n: int):
#     """Indices to split an array of size 'n' into 2, 4, 8, ..., k (2^{k+1} >= n) parts."""
#     n_sect = 2  # number of sections
#     indices = [0]  # indices of the border elements
#     while n_sect < n:
#         div = n // n_sect
#         indices.extend([index + div for index in indices])
#         indices.sort()
#         yield indices
#         n_sect *= 2


# def _count(blocks: list[np.ndarray]):
#     """Count the number of blocks which are marked."""
#     return sum(np.logical_or.reduce(elem, axis=None) for elem in blocks)


# def box_count_dimension(mark: np.ndarray):
#     """
#     mark: all values are boolean
#     """
#     nx, ny = mark.shape
#     assert nx == ny, "Only implemented for square-shape data"

#     box_side_len = []
#     fractal_area = []

#     for ix in _borders(nx):
#         rows = np.split(mark, ix, axis=0)
#         num_pixels = nx / len(ix)
#         box_side_len.append(num_pixels)
#         fractal_area.append(sum(_count(np.split(row, ix, axis=1)) for row in rows))

#     # NOTE: don't use side length 1 because it cause problem in the weighting of least squares
#     # box_side_len.append(1)
#     # fractal_area.append(mark.sum())

#     return np.array(box_side_len), np.array(fractal_area)


# def fit_linear_least_squares(y: np.ndarray, Phi: np.ndarray):
#     """Parameter fitting with y = Phi theta formulation."""
#     return la.pinv(Phi) @ y  # theta = (Phi)^(-1) y with pseudo-inverse


# def fit_power_curve(l_box: np.ndarray, n_box: np.ndarray, is_area: bool):
#     # Use 'log' to convert power into linear
#     # N = a L^b <=> log(N) = log(a) + b log(L) <=> log(N) = [1 log(L)] [log(a) b]^T
#     y = np.log2(n_box)
#     Phi = np.column_stack([np.ones_like(l_box), np.log2(l_box)])
#     # Add weights. Because the larger the box is, the less accurate the count is.
#     w = np.log2(l_box)**(-1)
#     if is_area:
#         w = w**2
#     print(f"weighting={w}")
#     # Least-squares fitting
#     log_a, b = fit_linear_least_squares(w*y, w[:,np.newaxis]*Phi)
#     a = np.exp2(log_a)
#     return a, b


# def gridwise_area_dimension(patch: np.ndarray):
#     l_box, n_box = box_count_dimension(patch)
#     pref, slope = fit_power_curve(l_box, n_box, is_area=True)
#     dimension = -slope
#     return dimension


# def gridwise_perimeter_dimension(patch: np.ndarray, outer: bool=True):
#     if outer:
#         peri = ga.outer_perimeter_area(patch, True)
#     else:
#         peri = ga.inner_perimeter_area(patch, True)

#     l_box, n_box = box_count_dimension(peri)
#     pref, slope = fit_power_curve(l_box, n_box, is_area=False)
#     dimension = -slope
#     return dimension


# def plot_patches_dimension_fitting(ax: plt.Axes, l_box: np.ndarray, n_box: np.ndarray,
#                                    pref: float, slope: float):
#     ax.plot(l_box, n_box, 'rx', label="Box counting")
#     l_axis = np.linspace(l_box.min(), l_box.max(), num=200)
#     ax.plot(l_axis, pref*(l_axis**slope), 'b--', label="Fitting")

#     ax.set_xlabel(r"Side length of the box $L$ (#pixel)")
#     ax.set_ylabel(r"Dimension $N$")
#     ax.loglog()


# # TODO: instead use area / area plot.
# # def patchwise_area_dimension(patch: np.ndarray):
# #     n_patch, patch = ga.assign_patch_numbers_area(patch, True)
# #     dims = np.empty((n_patch, 2))
# #     for i_patch in np.arange(n_patch):
# #         mark = patch == i_patch + 1  # the non-patch area is marked with 0; first index for patches is 1
# #         dims[i_patch] = gridwise_area_dimension(mark)
# #     return dims


# # def patchwise_perimeter_dimension(patch: np.ndarray):
# #     n_patch, patch = ga.assign_patch_numbers_area(patch, True)
# #     dims = np.empty((n_patch, 2))
# #     for i_patch in np.arange(n_patch):
# #         mark = patch == i_patch + 1  # the non-patch area is marked with 0; first index for patches is 1
# #         dims[i_patch] = gridwise_perimeter_dimension(mark)
# #     return dims
