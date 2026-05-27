from typing import Sequence

import numpy as np
import muGrid


class FirstOrderElement:
    """
    Create matrices that maps a vector of nodal values into a vector of values of interest in finite elements.

    Field is interpolated by triangular linear finite elements: 'a + b*xi + c*eta', with 'a' located at the
    centroid. Gradient values are thus constant: '(b / dx, c / dy)'.

    It combines interpolation and quadrature.
    """

    def __init__(self, sub_pt_coords: Sequence[Sequence[float]], element_sizes: Sequence[float] | None = None):
        """

        :param sub_pt_coords: coordinates of sub-points, shape (nb_sub_pts, 2)
        :param element_sizes: size of the element in each direction
        """
        sub_pt_coords = np.asarray(sub_pt_coords)
        nb_sub_pts, nb_spatial_dim = sub_pt_coords.shape
        if nb_spatial_dim != 2:
            raise ValueError(f"Expected 2D sub-point coordinates, got {nb_spatial_dim}D")

        if element_sizes is None:
            element_sizes = (1.0,) * 2
        if len(element_sizes) != nb_spatial_dim:
            raise ValueError(f"Expected {nb_spatial_dim}D element sizes, got {len(element_sizes)}")

        nodal_pixel_shape = (2, 2)
        # the target pixel is aligned towards the (0, 0) element of the kernel
        offset = (0, 0)

        fe_pixel = LinearFiniteElementPixel()

        # construct pixel operator for value interpolation
        val_interp_coeffs = fe_pixel.compute_value_interpolation_coefficients(sub_pt_coords)
        pixel_op_value = np.zeros([1, nb_sub_pts, *nodal_pixel_shape])
        for i_subpt, subpt_coeffs in enumerate(val_interp_coeffs):
            for coords, coeff in subpt_coeffs.items():
                pixel_op_value[(0, i_subpt, *coords)] = coeff
        self._op_value = muGrid.GenericLinearOperator(offset, pixel_op_value)

        # construct pixel operator for gradient interpolation
        grad_interp_coeffs = fe_pixel.compute_gradient_interpolation_coefficients(sub_pt_coords)
        pixel_op_gradient = np.zeros([nb_spatial_dim, nb_sub_pts, *nodal_pixel_shape])
        for i_subpt, subpt_coeffs in enumerate(grad_interp_coeffs):
            for i_component, compon_name in enumerate(["x1", "x2"]):
                for coords, coeff in subpt_coeffs[compon_name].items():
                    pixel_op_gradient[(i_component, i_subpt, *coords)] = coeff / element_sizes[i_component]
        self._op_gradient = muGrid.GenericLinearOperator(offset, pixel_op_gradient)

        # FIXME: the below shall be faster, but it seems using a different convention, which makes them not the same
        # self._op_gradient = muGrid.FEMGradientOperator(spatial_dim=2, grid_spacing=element_sizes)

    def interpolate_value(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Map nodal values to the interpolated values at centroid."""
        self._op_value.apply(field_in, field_out)

    def propag_sens_value(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Propogate the sensitivity of corresponding interpolation backward."""
        self._op_value.transpose(field_in, field_out)

    def interpolate_gradient(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Map nodal values to the interpolated gradient values, component in x."""
        self._op_gradient.apply(field_in, field_out)

    def propag_sens_gradient(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Propogate the sensitivity of corresponding interpolation backward."""
        self._op_gradient.transpose(field_in, field_out)


class LinearFiniteElementPixel:
    """A unit pixel discretized with linear (first order) finite element basis. It provides discrete operators
    for interpolation and gradient on pre-specified locations.
    (0,0) ---- (1,0) --> x_2
      |     /   |
      | 0  /  1 |
      |   /     |
    (0,1) ---- (1,1)
      |
      v
     x_1
    The vertices of the pixel are (0,0), (1,0), (0,1), (1,1). It is divided into two triangles by
    the line connecting vertices (1,0) and (0,1), x_1 + x_2 = 1. The triangle with (0,0) vertice is
    the "triangle0", the other is the "triangle1".
    """

    def compute_value_interpolation_coefficients(self, target_pts):
        """ Interpolation coefficients for a given set of target points.
        :param target_pts: 2D array of target points, shape (nb_target_pts, 2)
        :return: list of dicts, one for each target point
        """
        # enforce range
        assert np.all(target_pts >= 0) and np.all(target_pts <= 1)

        res = []
        for x1, x2 in target_pts:
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
        """ Interpolation coefficients for a given set of target points.

        :param target_pts: 2D array of target points, shape (nb_target_pts, 2)
        :return: list of dicts, one for each target point
        """
        # check points are inside a unit pixel
        assert np.all(target_pts >= 0) and np.all(target_pts <= 1)

        res = []
        for x1, x2 in target_pts:
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
