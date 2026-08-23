import numpy as np
import pytest

from a_package.domain import adapt_shape


def test_adapt_shape_2d():
    """A 2D array is the element axes alone, so a component and a sub-point are prepended."""
    array = np.arange(12.0).reshape(3, 4)

    field = adapt_shape(array)

    assert field.shape == (1, 1, 3, 4)
    # a view, not a copy
    assert np.shares_memory(field, array)
    assert np.array_equal(field[0, 0], array)


def test_adapt_shape_3d_is_rejected():
    """A 3D array is ambiguous between a missing component and a missing sub-point."""
    array = np.zeros((2, 3, 4))

    with pytest.raises(ValueError, match="2D or 4D"):
        adapt_shape(array)


def test_adapt_shape_4d():
    """A 4D array already conforms, so it comes back untouched."""
    array = np.arange(24.0).reshape(1, 2, 3, 4)

    field = adapt_shape(array)

    assert field is array


def test_adapt_shape_is_idempotent():
    """Adapting an already-adapted array leaves it alone."""
    array = np.zeros((3, 4))

    once = adapt_shape(array)
    twice = adapt_shape(once)

    assert twice is once
