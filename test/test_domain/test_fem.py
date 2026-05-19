import pytest
import numpy as np

from a_package.domain.fem import LinearFiniteElementPixel


@pytest.fixture
def test_pts():
    """Two points in the interior, two points on the borderline."""
    return np.array([[0.25, 0.25],
                     [0.75, 0.75],
                     [0.5 - 1e-9, 0.5 - 1e-9],
                     [0.5 + 1e-9, 0.5 + 1e-9]])


def test_value_interpolation_coefficients(test_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeff_map = fe_pixel.compute_value_interpolation_coefficients(test_pts)

    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((test_pts.shape[0], *pixel_shape))
    for i_sub_pt, sub_pt_coeffs in enumerate(coeff_map):
        for i_node, coeff in sub_pt_coeffs.items():
            pixel_mapping[i_sub_pt, *i_node] = coeff

    expected_mapping = np.array([[[0.5, 0.25],
                                  [0.25, 0]],
                                 [[0, 0.25],
                                  [0.25, 0.5]],
                                 [[0, 0.5],
                                  [0.5, 0]],
                                 [[0, 0.5],
                                  [0.5, 0]]])

    assert np.allclose(pixel_mapping, expected_mapping)


def test_gradient_interpolation_coefficients(test_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeff_map = fe_pixel.compute_gradient_interpolation_coefficients(test_pts)

    nb_components_gradient = 2
    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((nb_components_gradient, test_pts.shape[0], *pixel_shape))
    for i_sub_pt, sub_pt_coeffs in enumerate(coeff_map):
        for i_node, coeff in sub_pt_coeffs['x'].items():
            pixel_mapping[0, i_sub_pt, *i_node] = coeff
        for i_node, coeff in sub_pt_coeffs['y'].items():
            pixel_mapping[1, i_sub_pt, *i_node] = coeff

    expected_mapping_x = np.array([[[-1, 0],
                                    [1, 0]],
                                   [[0, -1],
                                    [0, 1]],
                                   [[-1, 0],
                                    [1, 0]],
                                   [[0, -1],
                                    [0, 1]]])
    expected_mapping_y = np.array([[[-1, 1],
                                    [0, 0]],
                                   [[0, 0],
                                    [-1, 1]],
                                   [[-1, 1],
                                    [0, 0]],
                                   [[0, 0],
                                    [-1, 1]]])

    assert np.allclose(pixel_mapping[0], expected_mapping_x)
    assert np.allclose(pixel_mapping[1], expected_mapping_y)
