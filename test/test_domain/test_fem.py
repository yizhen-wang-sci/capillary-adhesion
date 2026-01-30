import numpy as np
import pytest

from a_package.domain.fem import LinearFiniteElementPixel


def test_value_interpolation_coefficients():
    fe_pixel = LinearFiniteElementPixel()
    test_pts = np.array([[1/3, 1/3], [2/3, 2/3]])
    coeffs = fe_pixel.compute_value_interpolation_coefficients(test_pts)

    pixel_mapping = np.zeros((test_pts.shape[0], 2, 2))
    for [i_sub_pt, sub_pt_coeffs] in enumerate(coeffs):
        for [i_node, coeff] in sub_pt_coeffs.items():
            pixel_mapping[(i_sub_pt, *i_node)] = coeff

    expected_mapping = np.array([
        [[1/3, 1/3],
         [1/3, 0]],
        [[0, 1/3],
         [1/3, 1/3]]
    ])

    assert np.allclose(pixel_mapping, expected_mapping)


@pytest.fixture
def boundary_pts():
    return np.array([[0.5-1e-10, 0.5-1e-10],
                     [0.5+1e-10, 0.5+1e-10]])


def test_value_interpolation_at_boundary(boundary_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeffs = fe_pixel.compute_value_interpolation_coefficients(boundary_pts)

    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((boundary_pts.shape[0], *pixel_shape))
    for [i_sub_pt, sub_pt_coeffs] in enumerate(coeffs):
        for [i_node, coeff] in sub_pt_coeffs.items():
            pixel_mapping[i_sub_pt, *i_node] = coeff

    expected_mapping = np.array([
        [[0, 0.5],
         [0.5, 0]],
        [[0, 0.5],
         [0.5, 0]]])

    assert np.allclose(pixel_mapping, expected_mapping)


def test_gradient_interpolation_at_boundary(boundary_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeffs = fe_pixel.compute_gradient_interpolation_coefficients(boundary_pts)

    nb_components_gradient = 2
    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((boundary_pts.shape[0], nb_components_gradient, *pixel_shape))
    for [i_sub_pt, sub_pt_coeffs] in enumerate(coeffs):
        for [i_node, coeff] in sub_pt_coeffs['x1'].items():
            pixel_mapping[i_sub_pt, 0, *i_node] = coeff
        for [i_node, coeff] in sub_pt_coeffs['x2'].items():
            pixel_mapping[i_sub_pt, 1, *i_node] = coeff

    expected_mapping_triangle0 = np.array([
        [[-1, 0],
         [1, 0]],
        [[-1, 1],
         [0, 0]]])
    expected_mapping_triangle1 = np.array([
        [[0, -1],
         [0, 1]],
        [[0, 0],
         [-1, 1]]])

    assert np.allclose(pixel_mapping[0], expected_mapping_triangle0)
    assert np.allclose(pixel_mapping[1], expected_mapping_triangle1)
