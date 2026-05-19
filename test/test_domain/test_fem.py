import pytest
import numpy as np

from a_package.domain.fem import LinearFiniteElementPixel, FirstOrderElement
from a_package.domain import Grid
from a_package.domain.grid import factorize_closest

from test.test_domain.reference import RefFirstOrderElement


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


def test_first_order_element(test_pts, ref_field, comm_world):
    nb_test_pts, nb_spatial_dims = np.shape(test_pts)
    grid = Grid(ref_field.shape)

    # Grid decomposition
    nb_subdomains = factorize_closest(comm_world.Get_size(), 2)
    decomposition = grid.decompose(nb_subdomains, nb_ghost_layers=(1, 1), communicator=comm_world)
    collection = decomposition.collection
    collection.set_nb_sub_pts("test_pt", nb_test_pts)
    field_in_parallel = collection.real_field("origin", 1)
    field_value = collection.real_field("value", 1, "test_pt")
    field_gradient = collection.real_field("gradient", nb_spatial_dims, "test_pt")
    field_value_back_sens = collection.real_field("value_back_sens", 1)
    field_gradient_back_sens = collection.real_field("gradient_back_sens", 1)

    # FE parallel implementation
    field_in_parallel.s[0,0,...] = ref_field[*decomposition.icoords]
    decomposition.communicate_ghosts(field_in_parallel)

    fe = FirstOrderElement(test_pts, grid.element_sizes)
    fe.interpolate_value(field_in_parallel, field_value)
    fe.interpolate_gradient(field_in_parallel, field_gradient)
    fe.propag_sens_value(field_value, field_value_back_sens)
    fe.propag_sens_gradient(field_gradient, field_gradient_back_sens)

    # reference serial implementation
    ref_fe = RefFirstOrderElement(grid, test_pts)
    expected_field_value = ref_fe.interpolate_value(ref_field)
    expected_field_gradient = ref_fe.interpolate_gradient(ref_field)
    expected_field_value_back_sens = ref_fe.propag_sens_value(expected_field_value)
    expected_field_gradient_back_sens = ref_fe.propag_sens_gradient(expected_field_gradient)

    # assertions
    assert np.allclose(field_value.s, expected_field_value[..., *decomposition.icoords])
    assert np.allclose(field_gradient.s, expected_field_gradient[..., *decomposition.icoords])
    assert np.allclose(field_value_back_sens.s, expected_field_value_back_sens[..., *decomposition.icoords])
    assert np.allclose(field_gradient_back_sens.s, expected_field_gradient_back_sens[..., *decomposition.icoords])
