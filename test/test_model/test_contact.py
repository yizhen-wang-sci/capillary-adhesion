"""Tests of the rigid contact between two surfaces."""

import numpy as np
import pytest

from a_package.domain import field_component_ax, field_sub_pt_ax
from a_package.model import RigidContact


@pytest.fixture
def flat_surfaces():
    return np.zeros((4, 4)), np.zeros((4, 4))


def test_gap_of_flat_surfaces_is_the_separation(flat_surfaces):
    contact = RigidContact(*flat_surfaces)
    contact.set_mean_separation(0.5)
    np.testing.assert_allclose(contact.get_gap(), 0.5)


def test_gap_follows_the_height_difference():
    upper = np.array([[0.0, 0.2]])
    lower = np.array([[0.0, 0.1]])
    contact = RigidContact(upper, lower)
    contact.set_mean_separation(1.0)
    gap = contact.get_gap().squeeze(axis=(field_component_ax, field_sub_pt_ax))
    np.testing.assert_allclose(gap, [[1.0, 1.1]])


def test_gap_is_zeroed_where_the_surfaces_interpenetrate():
    upper = np.array([[0.0, -2.0]])
    lower = np.array([[0.0, 0.0]])
    contact = RigidContact(upper, lower)
    contact.set_mean_separation(1.0)
    gap = contact.get_gap().squeeze(axis=(field_component_ax, field_sub_pt_ax))
    np.testing.assert_allclose(gap, [[1.0, 0.0]])
