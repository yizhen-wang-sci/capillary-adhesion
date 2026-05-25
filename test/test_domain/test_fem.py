import pytest
import numpy as np

from a_package.domain.fem import LinearFiniteElementPixel, FirstOrderElement

from test.test_domain.reference import RefFirstOrderElement
from test.test_domain.utils import generate_global_random_field


@pytest.fixture
def mock_sub_pts():
    """Two points in the interior, two points on the borderline."""
    return np.array([[0.25, 0.25],
                     [0.75, 0.75],
                     [0.5 - 1e-9, 0.5 - 1e-9],
                     [0.5 + 1e-9, 0.5 + 1e-9]])


def test_value_interpolation_coefficients(mock_sub_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeff_map = fe_pixel.compute_value_interpolation_coefficients(mock_sub_pts)

    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((mock_sub_pts.shape[0], *pixel_shape))
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


def test_gradient_interpolation_coefficients(mock_sub_pts):
    fe_pixel = LinearFiniteElementPixel()
    coeff_map = fe_pixel.compute_gradient_interpolation_coefficients(mock_sub_pts)

    nb_components_gradient = 2
    pixel_shape = [2, 2]
    pixel_mapping = np.zeros((nb_components_gradient, mock_sub_pts.shape[0], *pixel_shape))
    for i_sub_pt, sub_pt_coeffs in enumerate(coeff_map):
        for i_node, coeff in sub_pt_coeffs['x1'].items():
            pixel_mapping[0, i_sub_pt, *i_node] = coeff
        for i_node, coeff in sub_pt_coeffs['x2'].items():
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


def test_first_order_element(decomposed_grid, mock_sub_pts, comm_world):
    mock_field = generate_global_random_field(decomposed_grid.nb_domain_grid_pts, comm_world)

    # Set up the fields
    decomposition = decomposed_grid.decomposition
    collection = decomposition.collection
    nb_sub_pts, nb_spatial_dims = np.shape(mock_sub_pts)
    collection.set_nb_sub_pts("sub_pt", nb_sub_pts)
    field_origin = collection.real_field("origin", 1)
    field_mapped_value = collection.real_field("value", 1, "sub_pt")
    field_mapped_gradient = collection.real_field("gradient", nb_spatial_dims, "sub_pt")
    field_mapped_value_back_sens = collection.real_field("value_back_sens", 1)
    field_mapped_gradient_back_sens = collection.real_field("gradient_back_sens", 1)

    # Parallel FE to test
    fe = FirstOrderElement(mock_sub_pts, decomposed_grid.element_sizes)

    # Serial implementation as reference
    ref_fe = RefFirstOrderElement(decomposed_grid, mock_sub_pts)

    field_origin.s[0,0,...] = mock_field[*decomposition.icoords]
    decomposition.communicate_ghosts(field_origin)

    fe.interpolate_value(field_origin, field_mapped_value)
    expected_field_value = ref_fe.interpolate_value(mock_field)
    assert np.allclose(field_mapped_value.s, expected_field_value[..., *decomposition.icoords])

    fe.interpolate_gradient(field_origin, field_mapped_gradient)
    expected_field_gradient = ref_fe.interpolate_gradient(mock_field)
    assert np.allclose(field_mapped_gradient.s, expected_field_gradient[..., *decomposition.icoords])

    # Modify values so that `communicate_ghosts` is a must-have step
    field_mapped_value.s[...] += 1.
    decomposition.communicate_ghosts(field_mapped_value)
    fe.propag_sens_value(field_mapped_value, field_mapped_value_back_sens)
    expected_field_value_back_sens = ref_fe.propag_sens_value(expected_field_value + 1.)
    assert np.allclose(field_mapped_value_back_sens.s, expected_field_value_back_sens[..., *decomposition.icoords])

    # Modify values so that `communicate_ghosts` is a must-have step
    field_mapped_gradient.s[...] += 1.
    decomposition.communicate_ghosts(field_mapped_gradient)
    fe.propag_sens_gradient(field_mapped_gradient, field_mapped_gradient_back_sens)
    expected_field_gradient_back_sens = ref_fe.propag_sens_gradient(expected_field_gradient + 1.)
    assert np.allclose(field_mapped_gradient_back_sens.s, expected_field_gradient_back_sens[..., *decomposition.icoords])
