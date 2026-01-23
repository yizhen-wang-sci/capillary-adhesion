"""
Reference implementations for testing.

Contains serial/non-parallel implementations of Grid and FirstOrderElement
for comparison against parallel implementations.
"""

import typing
import functools
import operator
import dataclasses as dc

import numpy as np
import numpy.fft as fft
import scipy.sparse as sparse

from a_package.domain.fem import LinearFiniteElementPixel


# =============================================================================
# Reference Grid (serial, no parallelization)
# =============================================================================

@dc.dataclass
class Grid:
    """A discrete space in 2D (serial reference implementation)."""

    lengths: typing.Sequence[float]
    nb_elements: typing.Sequence[int]

    def __post_init__(self):
        self.element_sizes = [l / n for [l, n] in zip(self.lengths, self.nb_elements)]
        self.element_area = functools.reduce(operator.mul, self.element_sizes, 1.)

    @property
    def is_in_parallel(self) -> bool:
        return False

    def form_index_axis(self, ax_index: int):
        return np.arange(self.nb_elements[ax_index])

    def form_index_mesh(self):
        return np.meshgrid(self.form_index_axis(0), self.form_index_axis(1))

    def form_nodal_axis(self, ax_index: int, with_endpoint: bool = False):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        if with_endpoint:
            n += 1
        return np.arange(n) * d

    def form_nodal_mesh(self, with_endpoint: bool = False):
        return np.meshgrid(self.form_nodal_axis(0, with_endpoint), self.form_nodal_axis(1, with_endpoint))

    def form_spectral_axis(self, ax_index: int):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))


# =============================================================================
# Reference FirstOrderElement (sparse matrix based, serial)
# =============================================================================

class FirstOrderElement:
    """
    Reference implementation using sparse matrices (serial).

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.
    """

    def __init__(self, grid: Grid, sub_pt_coords: np.ndarray):
        self.grid_shape = grid.nb_elements
        self.nb_sub_pts = sub_pt_coords.shape[0]

        fe_pixel = LinearFiniteElementPixel()
        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(sub_pt_coords)
        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(sub_pt_coords)

        [M, N] = grid.nb_elements
        MN = M * N

        # mapping nodal value to the value at target points
        blocks = []
        for sub_pt_coeffs in val_interp_coeffs:
            sub_pt_matrix = sparse.lil_matrix((MN, MN), dtype=float)
            for [node_idxs, coeff] in sub_pt_coeffs.items():
                fill_cyclic_diagonal_pseudo_2d(sub_pt_matrix, node_idxs, (M, N), coeff)
            blocks.append(sub_pt_matrix)
        self.matrix_val = sparse.vstack(blocks, format="csr")

        # mapping nodal value to the gradient at target points
        blocks = []
        for [compon_idx, compon_name] in enumerate(['x1', 'x2']):
            for sub_pt_coeffs in grad_interp_coeffs:
                sub_pt_matrix = sparse.lil_matrix((MN, MN), dtype=float)
                for [node_idxs, coeff] in sub_pt_coeffs[compon_name].items():
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


# =============================================================================
# Helper functions
# =============================================================================

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
