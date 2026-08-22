"""Finite element interpolation on a regular grid."""

from collections.abc import Sequence

import muGrid
import numpy as np


class FirstOrderElement:
    """Linear (first-order) finite element interpolation of a field, and its adjoint."""

    def __init__(self, sub_pt_coords: Sequence[Sequence[float]], element_sizes: Sequence[float] | None = None):
        """Assemble the pixel stencils for value and gradient interpolation.

        Args:
            sub_pt_coords: Coordinates of the sub-points within a unit pixel, shape
                (nb_sub_pts, 2).
            element_sizes: Size of an element along each dimension, 1.0 by default.

        Raises:
            ValueError: If the sub-point coordinates are not 2D, or if `element_sizes` has a
                different number of dimensions than they do.
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

        # FIXME: the below shall be faster, but it seems using a different convention,
        # which makes them not the same
        # self._op_gradient = muGrid.FEMGradientOperator(spatial_dim=2, grid_spacing=element_sizes)

    def interpolate_value(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Map nodal values to the interpolated values at the sub-points.

        Args:
            field_in: Nodal values.
            field_out: Written in place with the interpolated values.
        """
        self._op_value.apply(field_in, field_out)

    def propag_sens_value(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Propagate the sensitivity of `interpolate_value` backward.

        Args:
            field_in: Derivative with respect to the interpolated values.
            field_out: Written in place with the derivative with respect to the nodal values.
        """
        self._op_value.transpose(field_in, field_out)

    def interpolate_gradient(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Map nodal values to the interpolated gradient at the sub-points.

        Args:
            field_in: Nodal values.
            field_out: Written in place with the gradient, one component per dimension.
        """
        self._op_gradient.apply(field_in, field_out)

    def propag_sens_gradient(self, field_in: muGrid.Field, field_out: muGrid.Field):
        """Propagate the sensitivity of `interpolate_gradient` backward.

        Args:
            field_in: Derivative with respect to the interpolated gradient.
            field_out: Written in place with the derivative with respect to the nodal values.
        """
        self._op_gradient.transpose(field_in, field_out)


class LinearFiniteElementPixel:
    """A unit pixel with a linear finite element basis, split by the line x_1 + x_2 = 1."""

    def compute_value_interpolation_coefficients(self, target_pts):
        """Value interpolation coefficients for a given set of target points.

        Args:
            target_pts: Target points, shape (nb_target_pts, 2), each coordinate within [0,1].

        Returns:
            One dict per target point, holding the coefficient of each corner node keyed by that
            node's (x_1, x_2) index.
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
        """Shape functions of triangle0, the one holding the (0,0) vertice.

        Args:
            x1: First coordinate within the unit pixel.
            x2: Second coordinate within the unit pixel.

        Returns:
            The coefficient of each corner node, keyed by that node's (x_1, x_2) index.
        """
        return {(0, 0): 1 - x1 - x2, (1, 0): x1, (0, 1): x2}

    @staticmethod
    def triangle1_shape_function(x1, x2):
        """Shape functions of triangle1, the one opposite the (0,0) vertice.

        Args:
            x1: First coordinate within the unit pixel.
            x2: Second coordinate within the unit pixel.

        Returns:
            The coefficient of each corner node, keyed by that node's (x_1, x_2) index.
        """
        return {(1, 1): x1 + x2 - 1, (1, 0): 1 - x2, (0, 1): 1 - x1}

    def compute_gradient_interpolation_coefficients(self, target_pts):
        """Gradient interpolation coefficients for a given set of target points.

        Args:
            target_pts: Target points, shape (nb_target_pts, 2), each coordinate within [0,1].

        Returns:
            One dict per target point, keyed by the component names "x1" and "x2", each holding
            the coefficient of every corner node.
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
        """Shape-function gradients of triangle0.

        Args:
            x1: Ignored.
            x2: Ignored.

        Returns:
            Keyed by the component names "x1" and "x2", each holding the coefficient of every
            corner node.
        """
        return {"x1": {(0, 0): -1.0, (1, 0): 1.0}, "x2": {(0, 0): -1.0, (0, 1): 1.0}}

    @staticmethod
    def triangle1_shape_function_gradient(x1, x2):
        """Shape-function gradients of triangle1.

        Args:
            x1: Ignored.
            x2: Ignored.

        Returns:
            Keyed by the component names "x1" and "x2", each holding the coefficient of every
            corner node.
        """
        return {"x1": {(0, 1): -1.0, (1, 1): 1.0}, "x2": {(1, 0): -1.0, (1, 1): 1.0}}
