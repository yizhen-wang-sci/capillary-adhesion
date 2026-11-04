"""Tests of the smooth switches."""

import numpy as np
import pytest

from a_package.domain.switch import AlgebraicSwitch, ArctangentSwitch, CubicSwitch, ExponentialSwitch


@pytest.mark.parametrize("activation_length", [1e-3, 0.7, 50.0])
@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-6])
@pytest.mark.parametrize("switch_class", [CubicSwitch, ExponentialSwitch, AlgebraicSwitch, ArctangentSwitch])
def test_switch_is_within_tolerance_of_off_at_zero_and_on_at_activation_length(
    switch_class, tolerance, activation_length
):
    switch = switch_class(activation_length, tolerance)
    ends = switch.value(np.array([0.0, activation_length]))
    np.testing.assert_allclose(ends, [tolerance, 1.0 - tolerance], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("activation_length", [1e-3, 0.7, 50.0])
@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-6])
@pytest.mark.parametrize("switch_class", [CubicSwitch, ExponentialSwitch, AlgebraicSwitch, ArctangentSwitch])
def test_switch_derivative_is_derivative_of_value(switch_class, tolerance, activation_length):
    switch = switch_class(activation_length, tolerance)
    x = np.linspace(-1.0, 3.0, 401) * activation_length
    peak_slope = np.max(switch.derivative(x))
    h = 1e-4 / peak_slope
    x_plus, x_minus = x + h, x - h
    finite_difference = (switch.value(x_plus) - switch.value(x_minus)) / (x_plus - x_minus)
    np.testing.assert_allclose(switch.derivative(x), finite_difference, rtol=0, atol=1e-6 * peak_slope)
