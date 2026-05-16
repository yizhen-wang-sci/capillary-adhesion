import numpy as np
import scipy.sparse as sparse

from .grid import Grid


class FirstOrderElement:
    """
    Create matrices that maps a vector of nodal values into a vector of values of interest in finite elements.

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.

    It combines interpolation and quadrature.
    """

    def __init__(self, grid: Grid, sub_pt_coords: np.ndarray):
        self.grid_shape = grid.nb_domain_grid_pts
        self.nb_sub_pts = sub_pt_coords.shape[0]

        fe_pixel = LinearFiniteElementPixel()
        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(sub_pt_coords)
        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(sub_pt_coords)

        [M, N] = grid.nb_domain_grid_pts
        MN = M * N

        # mapping nodal value to the value at target points
        # matrix_val_0 = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in val_interp_coeffs[0].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_val_0, i_pt, (M, N), coeff)
        # matrix_val_1 = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in val_interp_coeffs[1].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_val_0, i_pt, (M, N), coeff)
        # self.matrix_val = sparse.vstack([matrix_val_0, matrix_val_1], format="csr")
        blocks = []
        for sub_pt_coeffs in val_interp_coeffs:
            sub_pt_matrix = sparse.lil_matrix((MN, MN), dtype=float)
            for [node_idxs, coeff] in sub_pt_coeffs.items():
                fill_cyclic_diagonal_pseudo_2d(sub_pt_matrix, node_idxs, (M, N), coeff)
            blocks.append(sub_pt_matrix)
        self.matrix_val = sparse.vstack(blocks, format="csr")

        # # mapping nodal value to the gradient x component at target points
        # matrix_grad_0_x = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in grad_interp_coeffs[0]["x"].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_grad_0_x, i_pt, (M, N), coeff)
        # matrix_grad_1_x = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in grad_interp_coeffs[1]["x"].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_grad_1_x, i_pt, (M, N), coeff)
        # self.matrix_Dx = sparse.vstack([matrix_grad_0_x, matrix_grad_1_x], format="csr") / grid.element_sizes[0]

        # # mapping nodal value to the gradient y component at target points
        # matrix_grad_0_y = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in grad_interp_coeffs[0]["y"].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_grad_0_y, i_pt, (M, N), coeff)
        # matrix_grad_1_y = sparse.lil_matrix((MN, MN), dtype=float)
        # for [i_pt, coeff] in grad_interp_coeffs[1]["y"].items():
        #     fill_cyclic_diagonal_pseudo_2d(matrix_grad_1_y, i_pt, (M, N), coeff)
        # self.matrix_Dy = sparse.vstack([matrix_grad_0_y, matrix_grad_1_y], format="csr") / grid.element_sizes[1]

        # mapping nodal value to the gradient at target points
        blocks = []
        for [compon_idx, compon_name] in enumerate(['x', 'y']):
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

    # def interpolate_gradient_x(self, data: np.ndarray):
    #     """Map nodal values to the interpolated gradient values, component in x."""
    #     return (self.matrix_Dx @ data.ravel()).reshape(-1, self.nb_sub_pts, *self.grid_shape)

    # def propag_sens_gradient_x(self, data: np.ndarray):
    #     """Propogate the sensitivity of corresponding interpolation backward."""
    #     return (data.ravel() @ self.matrix_Dx).reshape(-1, *self.grid_shape)

    # def interpolate_gradient_y(self, data: np.ndarray):
    #     """Map nodal values to the interpolated gradient values, component in y."""
    #     return (self.matrix_Dy @ data.ravel()).reshape(-1, self.nb_sub_pts, *self.grid_shape)

    # def propag_sens_gradient_y(self, data: np.ndarray):
    #     """Propogate the sensitivity of corresponding interpolation backward."""
    #     return (data.ravel() @ self.matrix_Dy).reshape(-1, *self.grid_shape)

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


class LinearFiniteElementPixel:
    """A unit pixel discretized with linear (first order) finite element basis. It provides discrete operators
    for interpolation and gradient on pre-specified locations.
    (0,0) ---- (1,0)
      |     /   |
      | 0  /  1 |
      |   /     |
    (0,1) ---- (1,1)
    The vertices of the pixel are (0,0), (1,0), (0,1), (1,1). It is divided into two triangles by
    the line connecting vertices (1,0) and (0,1), x_1 + x_2 = 1. The triangle with (0,0) vertice is
    the "triangle0", the other is the "triangle1".
    """

    def compute_value_interpolation_coefficients(self, target_pts):
        # enforce range
        assert np.all(target_pts >= 0) and np.all(target_pts <= 1)

        # nb_sub_pts = np.size(target_pts, axis=0)
        # res = np.empty([1, nb_sub_pts, 2, 2])
        # for i, (x1, x2) in enumerate(target_pts):
        #     if x1 + x2 < 1:
        #         # Lower triangle
        #         res[:, i] = self.triangle0_shape_function(x1, x2)
        #     else:
        #         # Upper triangle
        #         res[:, i] = self.triangle1_shape_function(x1, x2)
        # return res
        res = []
        for [x1, x2] in target_pts:
            if x1 + x2 < 1:
                res.append(self.triangle0_shape_function(x1, x2))
            else:
                res.append(self.triangle1_shape_function(x1, x2))
        return res

    @staticmethod
    def triangle0_shape_function(x1, x2):
        # return [
        #     [1 - x1 - x2, x2],
        #     [x1, 0],
        # ]
        return {(0, 0): 1 - x1 - x2, (1, 0): x1, (0, 1): x2}

    @staticmethod
    def triangle1_shape_function(x1, x2):
        # return [
        #     [0, 1 - x1],
        #     [1 - x2, x1 + x2 - 1],
        # ]
        return {(1, 1): x1 + x2 - 1, (1, 0): 1 - x2, (0, 1): 1 - x1}

    def compute_gradient_interpolation_coefficients(self, target_pts):
        # check points are inside a unit pixel
        assert np.all(target_pts >= 0) and np.all(target_pts <= 1)

        res = []
        for [x1, x2] in target_pts:
            if x1 + x2 < 1:
                res.append(self.triangle0_shape_function_gradient(x1, x2))
            else:
                res.append(self.triangle1_shape_function_gradient(x1, x2))
        return res

    @staticmethod
    def triangle0_shape_function_gradient(x1, x2):
        # return [
        #     [
        #         [-1, 0],
        #         [1, 0],
        #     ],
        #     [
        #         [-1, 1],
        #         [0, 0],
        #     ],
        # ]
        return {"x": {(0, 0): -1.0, (1, 0): 1.0}, "y": {(0, 0): -1.0, (0, 1): 1.0}}

    @staticmethod
    def triangle1_shape_function_gradient(x1, x2):
        # return [
        #     [
        #         [0, -1],
        #         [0, 1],
        #     ],
        #     [
        #         [0, 0],
        #         [-1, 1],
        #     ],
        # ]
        return {"x": {(0, 1): -1.0, (1, 1): 1.0}, "y": {(1, 0): -1.0, (1, 1): 1.0}}
