import numpy as np

from a_package.domain import Grid
from a_package.domain.fem import FirstOrderElement

from test.reference import Grid as ReferenceGrid
from test.reference import FirstOrderElement as ReferenceFirstOrderElement


def test_first_order_element():
    # set up
    test_pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
    nb_element = 4
    field_data = np.diagflat(np.arange(nb_element) + 1)

    # implementation (parallel) to test
    grid = Grid([1., 1.], [nb_element, nb_element])
    fe = FirstOrderElement(grid, test_pts)
    field_in_parallel = np.asarray(field_data[grid.subdomain_slice], dtype=float, order="F")
    field_value = fe.interpolate_value(field_in_parallel)
    field_gradient = fe.interpolate_gradient(field_in_parallel)
    field_value_sens = fe.propag_sens_value(field_value)
    field_gradient_sens = fe.propag_sens_gradient(field_gradient)

    # reference implementation (serial)
    ref_grid = ReferenceGrid([1., 1.], [nb_element, nb_element])
    ref_fe = ReferenceFirstOrderElement(ref_grid, test_pts)
    field_in_serial = np.asarray(field_data, dtype=float, order="C")
    expected_field_value = ref_fe.interpolate_value(field_in_serial)
    expected_field_gradient = ref_fe.interpolate_gradient(field_in_serial)
    expected_field_value_sens = ref_fe.propag_sens_value(expected_field_value)
    expected_field_gradient_sens = ref_fe.propag_sens_gradient(expected_field_gradient)

    # assertions
    assert np.allclose(field_value, expected_field_value[grid.subdomain_slice])
    assert np.allclose(field_gradient, expected_field_gradient[grid.subdomain_slice])
    assert np.allclose(field_value_sens, expected_field_value_sens[grid.subdomain_slice])
    assert np.allclose(field_gradient_sens, expected_field_gradient_sens[grid.subdomain_slice])
