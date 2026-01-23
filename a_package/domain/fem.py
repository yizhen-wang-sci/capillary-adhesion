import numpy as np
import muGrid

from .grid import Grid
from .field import adapt_shape


class FirstOrderElement:
    """
    Create matrices that maps a vector of nodal values into a vector of values of interest in finite elements.

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.

    It combines interpolation and quadrature.
    """

    def __init__(self, grid: Grid, sub_pt_coords: np.ndarray):
        self.grid = grid
        # get the field manager
        field_collec: muGrid.GlobalFieldCollection = grid.decomposition.collection
        # create the field to hold values
        nb_components_value = 1
        self.nodal_value_field = field_collec.real_field("nodal_value", nb_components_value)
        tag = "FEM_quadrature"
        field_collec.set_nb_sub_pts(tag, sub_pt_coords.shape[0])

        # create value interpolation operator and necessary fields
        self.value_interp_out_field = field_collec.real_field("value_interp_out", nb_components_value, tag)
        self.value_propag_in_field = field_collec.real_field("value_propag_in", nb_components_value, tag)
        self.value_propag_out_field = field_collec.real_field("value_propag_out", nb_components_value)

        # create gradient interpolation operator and necessary fields
        nb_components_gradient = 2
        self.gradient_interp_out_field = field_collec.real_field("gradient_interp_out", nb_components_gradient, tag)
        self.gradient_propag_in_field = field_collec.real_field("gradient_propag_in", nb_components_gradient, tag)
        self.gradient_propag_out_field = field_collec.real_field("gradient_propag_out", nb_components_value)

        fe_pixel = LinearFiniteElementPixel()
        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(sub_pt_coords)
        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(sub_pt_coords)

        nb_sub_pts = sub_pt_coords.shape[0]
        pixel_nodal_shape = [2, 2]
        # the target pixel is aligned towards the (0, 0) element of the kernel
        offset = [0, 0]
        # construct pixel operator for value interpolation
        convol_value = np.zeros([nb_components_value, nb_sub_pts, *pixel_nodal_shape])
        for [subpt_idx, subpt_coeffs] in enumerate(val_interp_coeffs):
            for [node_idxs, coeff] in subpt_coeffs.items():
                convol_value[0, subpt_idx, *node_idxs] = coeff
        self.op_value = muGrid.GenericLinearOperator(offset, convol_value)
        # construct pixel operator for gradient interpolation
        convol_gradient = np.zeros([nb_components_gradient, nb_sub_pts, *pixel_nodal_shape])
        for [compon_idx, compon_name] in enumerate(['x1', 'x2']):
            for [subpt_idx, subpt_coeffs] in enumerate(grad_interp_coeffs):
                for [node_idxs, coeff] in subpt_coeffs[compon_name].items():
                    convol_gradient[compon_idx, subpt_idx, *node_idxs] = coeff / grid.element_sizes[compon_idx]
        self.op_gradient = muGrid.GenericLinearOperator(offset, convol_gradient)

    def interpolate_value(self, data: np.ndarray):
        """Map nodal values to the interpolated values at centroid."""
        self.nodal_value_field.s[...] = adapt_shape(data)
        self.grid.sync_field(self.nodal_value_field)
        self.op_value.apply(self.nodal_value_field, self.value_interp_out_field)
        return self.value_interp_out_field.s.copy()

    def propag_sens_value(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        self.value_propag_in_field.s[...] = adapt_shape(data)
        self.grid.sync_field(self.value_propag_in_field)
        self.op_value.transpose(self.value_propag_in_field, self.value_propag_out_field)
        return self.value_propag_out_field.s.copy()

    def interpolate_gradient(self, data: np.ndarray):
        """Map nodal values to the interpolated gradient values, component in x."""
        self.nodal_value_field.s[...] = adapt_shape(data)
        self.grid.sync_field(self.nodal_value_field)
        self.op_gradient.apply(self.nodal_value_field, self.gradient_interp_out_field)
        return self.gradient_interp_out_field.s.copy()

    def propag_sens_gradient(self, data: np.ndarray):
        """Propogate the sensitivity of corresponding interpolation backward."""
        self.gradient_propag_in_field.s[...] = adapt_shape(data)
        self.grid.sync_field(self.gradient_propag_in_field)
        self.op_gradient.transpose(self.gradient_propag_in_field, self.gradient_propag_out_field)
        return self.gradient_propag_out_field.s.copy()


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
    the "lower triangle", the other is the "upper triangle".
    """

    def compute_value_interpolation_coefficients(self, target_pts):
        # enforce range
        assert np.all(target_pts >= 0) and np.all(target_pts <= 1)

        res = []
        for [x1, x2] in target_pts:
            if x1 + x2 < 1:
                res.append(self.triangle0_shape_function(x1, x2))
            else:
                res.append(self.triangle1_shape_function(x1, x2))
        return res

    @staticmethod
    def triangle0_shape_function(x1, x2):
        return {(0, 0): 1 - x1 - x2, (1, 0): x1, (0, 1): x2}

    @staticmethod
    def triangle1_shape_function(x1, x2):
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
        return {"x1": {(0, 0): -1.0, (1, 0): 1.0}, "x2": {(0, 0): -1.0, (0, 1): 1.0}}

    @staticmethod
    def triangle1_shape_function_gradient(x1, x2):
        return {"x1": {(0, 1): -1.0, (1, 1): 1.0}, "x2": {(1, 0): -1.0, (1, 1): 1.0}}
