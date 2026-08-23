"""Tests for the conversion between dimensionless and physical quantities."""

import pytest

from a_package.simulation import UnitConversion


@pytest.mark.parametrize("exponent", [1, 2, 3, -1])
def test_conversion_round_trips(exponent):
    unit = UnitConversion(scale=1e-6, base_unit="m")
    value = 3.7
    assert unit.to_dimensionless(unit.to_physical(value, exponent), exponent) == pytest.approx(value)
    assert unit.to_physical(unit.to_dimensionless(value, exponent), exponent) == pytest.approx(value)


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [(1, "m"), (2, "m^2"), (-1, "/m"), (-3, "/m^3")],
)
def test_unit_string_carries_the_exponent(exponent, expected):
    assert UnitConversion(base_unit="m", exponent=exponent).unit == expected
