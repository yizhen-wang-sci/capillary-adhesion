"""
Reference implementations in serial. Can be used to compare with the parallel implementations.
"""

import numpy as np
import scipy.sparse as sparse

from a_package.domain.grid import Grid
from a_package.domain.fem import LinearFiniteElementPixel


class RefFirstOrderElement:
    """
    Reference implementations in serial. Can be used to compare against the parallel implementations.

    It creates sparse matrices from the coefficients of interpolating values and gradients.
    """

    def __init__(self, grid: Grid, sub_pt_coords: np.ndarray):
        self.grid_shape = grid.nb_domain_grid_pts
        self.nb_sub_pts = np.shape(sub_pt_coords)[0]

        fe_pixel = LinearFiniteElementPixel()
        M, N = grid.nb_domain_grid_pts
        MN = M * N

        # mapping nodal value to the value at target points
        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(sub_pt_coords)
        blocks = []
        for sub_pt_coeffs in val_interp_coeffs:
            sub_pt_matrix = sparse.lil_matrix((MN, MN), dtype=float)
            for node_idxs, coeff in sub_pt_coeffs.items():
                fill_cyclic_diagonal_pseudo_2d(sub_pt_matrix, node_idxs, (M, N), coeff)
            blocks.append(sub_pt_matrix)
        self.matrix_val = sparse.vstack(blocks, format="csr")

        # mapping nodal value to the gradient at target points
        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(sub_pt_coords)
        blocks = []
        for compon_idx, compon_name in enumerate(["x1", "x2"]):
            for sub_pt_coeffs in grad_interp_coeffs:
                sub_pt_matrix = sparse.lil_matrix((MN, MN), dtype=float)
                for node_idxs, coeff in sub_pt_coeffs[compon_name].items():
                    fill_cyclic_diagonal_pseudo_2d(sub_pt_matrix, node_idxs, (M, N), coeff)
                blocks.append(sub_pt_matrix / grid.element_sizes[compon_idx])
        self.matrix_grad = sparse.vstack(blocks, format="csr")

    def interpolate_value(self, data: np.ndarray):
        """Map nodal values to the interpolated values at centroid."""
        return (self.matrix_val @ data.ravel()).reshape(-1, self.nb_sub_pts, *self.grid_shape)

    def propag_sens_value(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        return (data.ravel() @ self.matrix_val).reshape(-1, 1, *self.grid_shape)

    def interpolate_gradient(self, data: np.ndarray):
        """Map nodal values to the interpolated gradient values, component in x."""
        return (self.matrix_grad @ data.ravel()).reshape(-1, self.nb_sub_pts, *self.grid_shape)

    def propag_sens_gradient(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        return (data.ravel() @ self.matrix_grad).reshape(-1, 1, *self.grid_shape)


def fill_cyclic_diagonal_1d(mat: sparse.spmatrix, j: int, N: int, val: float):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 1D data to 1D data.
    """
    assert mat.ndim == 2
    i = np.arange(N)
    mat[i, (i + j) % N] = val


def fill_cyclic_diagonal_pseudo_2d(
    mat: sparse.spmatrix, j: tuple[int, int], N: tuple[int, int], val: float, row_maj: bool = True
):
    """Fill cyclically, element-wise in the j-th diagonal of a matrix.
    The matrix represents a mapping from 2D data to 2D data.
    However, the 2D data is ravelled and represented as a 1D array.
    """
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    N1, N2 = N
    i1, i2 = np.mgrid[:N1, :N2]
    i1 = i1.ravel()
    i2 = i2.ravel()

    j1, j2 = j
    if row_maj:
        # the 2D data is flattened with row-major (contiguous in 1st axis)
        mat[i1 * N2 + i2, (i1 + j1) % N1 * N2 + (i2 + j2) % N2] = val
    else:
        # the 2D data is flattened with column-major (contiguous in 2nd axis)
        mat[i1 + i2 * N1, (i1 + j1) % N1 + (i2 + j2) % N2 * N1] = val


def fill_vertical_block_diagonal(mat: sparse.spmatrix, N: int, val: list[float]):
    """Fill cyclically, block-wise in the diagonal of a matrix."""
    assert mat.ndim == 2

    # cartesian product of range(N1) and range(N2)
    m = len(val)
    for i in range(N):
        mat[m * i : m * (i + 1), i] = val
