"""Tests of the smooth switches."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from a_package.domain.switch import AlgebraicSwitch, ArctangentSwitch, CubicSwitch, ExponentialSwitch

show_me_plot = False


def sharp_indicator(x):
    return (x > 0).astype(float)


def gaps_up_to(activation_length, nb_lengths):
    inside = np.linspace(0.0, 1.0, 1001)
    beyond = np.logspace(0.0, np.log10(nb_lengths), 1001)
    return activation_length * np.concatenate((inside, beyond))


@pytest.mark.parametrize("activation_length", [1e-3, 1e-1, 1e1])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("switch_class", [CubicSwitch, ExponentialSwitch, AlgebraicSwitch, ArctangentSwitch])
def test_switch_lies_between_zero_and_sharp_indicator(switch_class, tolerance, activation_length):
    switch = switch_class(activation_length, tolerance)
    x = gaps_up_to(activation_length, 1e4)
    value = switch.value(x)
    assert np.all(value >= 0.0)
    assert np.all(value <= sharp_indicator(x))


@pytest.mark.parametrize("activation_length", [1e-3, 1e-1, 1e1])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("switch_class", [CubicSwitch, ExponentialSwitch, AlgebraicSwitch, ArctangentSwitch])
def test_switch_falls_short_of_sharp_indicator_by_tolerance_at_activation_length_and_at_most_that_beyond(
    switch_class, tolerance, activation_length
):
    switch = switch_class(activation_length, tolerance)
    x = activation_length * np.logspace(0.0, 4.0, 1001)
    deficit = sharp_indicator(x) - switch.value(x)
    assert deficit[0] == pytest.approx(tolerance, rel=1e-6, abs=1e-15)
    assert np.all(deficit <= tolerance + 1e-15)


def finite_difference_deviation(switch, x, step_sizes):
    """Largest deviation of `derivative` from a central difference of `value`, per step size,
    relative to the peak slope."""
    derivative = switch.derivative(x)
    peak_slope = np.max(derivative)
    deviations = np.full(np.size(step_sizes), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i_step, step_size in enumerate(step_sizes):
            x_plus, x_minus = x + step_size / peak_slope, x - step_size / peak_slope
            finite_difference = (switch.value(x_plus) - switch.value(x_minus)) / (x_plus - x_minus)
            deviation = np.max(np.abs(derivative - finite_difference)) / peak_slope
            if np.isfinite(deviation):
                deviations[i_step] = deviation
    return deviations


@pytest.mark.parametrize("activation_length", [1e-3, 1e-1, 1e1])
@pytest.mark.parametrize("switch_class", [CubicSwitch, ExponentialSwitch, AlgebraicSwitch, ArctangentSwitch])
def test_switch_derivative_is_derivative_of_value(switch_class, activation_length):
    tolerances = np.logspace(-1, -10, 10)
    step_sizes = np.logspace(-8, -1, 29)
    x = np.linspace(-1.0, 3.0, 401) * activation_length
    best_deviations = np.array(
        [np.min(finite_difference_deviation(switch_class(activation_length, tol), x, step_sizes)) for tol in tolerances]
    )

    if show_me_plot:
        plt.plot(tolerances, best_deviations, "x-", label=switch_class.__name__)
        plt.loglog()
        plt.xlabel("tolerance")
        plt.ylabel("finite difference deviation / peak slope")
        plt.legend()
        plt.show()

    assert np.all(best_deviations < 1e-6), dict(zip(tolerances, best_deviations))
