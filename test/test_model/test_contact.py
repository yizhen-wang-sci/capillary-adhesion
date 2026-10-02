"""Tests of the contact between two surfaces."""

import numpy as np
import pytest
from NuMPI import MPI

from a_package.domain import Grid, field_component_ax, field_sub_pt_ax
from a_package.model.contact import ElasticContact, RigidContact


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


SEPARATION = 1.0
DOMAIN_LENGTH = 8.0
ELASTIC_PARAMS = {"youngs_modulus": 1.0, "poisson_ratio": 0.0}


def build_elastic_contact(nb_grid_pts):
    """An elastic contact between two flat surfaces, over a square grid of the given size."""
    grid = Grid([nb_grid_pts, nb_grid_pts], [DOMAIN_LENGTH, DOMAIN_LENGTH])
    grid.decompose([1, MPI.COMM_WORLD.Get_size()], (1, 1), communicator=MPI.COMM_WORLD)
    flat = grid.get_local(np.zeros(grid.nb_domain_grid_pts))
    contact = ElasticContact(grid, ELASTIC_PARAMS, flat, flat, communicator=MPI.COMM_WORLD)
    contact.set_mean_separation(SEPARATION)
    return grid, contact


@pytest.fixture
def elastic_contact():
    return build_elastic_contact(8)


@pytest.fixture
def small_steps():
    return np.pow(10.0, np.arange(-8.0, 1.0))


def compute_numerical_jacobian(x, func, step):
    """Jacobian of a scalar function by central differences of one step size."""
    jacobian = np.empty(x.shape)
    for indices in np.ndindex(x.shape):
        original = x[indices]
        x[indices] = original + step
        plus = func(x)
        x[indices] = original - step
        minus = func(x)
        x[indices] = original
        jacobian[indices] = 0.5 * (plus - minus) / step
    return jacobian


def test_uniform_gap_stores_no_energy(elastic_contact):
    grid, contact = elastic_contact
    contact.set_gap(grid.get_local(np.full(grid.nb_domain_grid_pts, 1.)))
    assert contact.get_energy() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("get_value", "get_jacobian"),
    [
        ("get_energy", "get_energy_jacobian"),
        ("get_gap_volume", "get_gap_volume_jacobian"),
    ],
)
def test_jacobian_is_the_gradient_of_its_quantity(elastic_contact, small_steps, get_value, get_jacobian):
    grid, contact = elastic_contact
    rng = np.random.default_rng(0)
    gap = SEPARATION - 0.2 * rng.random(grid.nb_domain_grid_pts)

    def value_of(value):
        contact.set_gap(grid.get_local(value))
        return getattr(contact, get_value)()

    contact.set_gap(grid.get_local(gap))
    impl_jacobian = getattr(contact, get_jacobian)()

    differences = [
        np.max(np.abs(impl_jacobian - grid.get_local(compute_numerical_jacobian(gap, value_of, step))))
        for step in small_steps
    ]
    assert np.amin(differences) < 1e-6
