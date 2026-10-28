"""Tests of the contact between two surfaces."""

import numpy as np
import pytest
from NuMPI import MPI

from a_package.domain import Grid, ProjectedLbfgs, field_component_ax, field_sub_pt_ax
from a_package.model.contact import ElasticContact, RigidContact
from a_package.model.equilibrium import formulate_constant_volume_gap_problem

show_me_plot = False


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
    contact.set_gap(grid.get_local(np.full(grid.nb_domain_grid_pts, 1.0)))
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


def test_pressure_under_a_sphere_is_hertzian():
    nb_grid_pts = 256
    radius = 8.0
    grid = Grid([nb_grid_pts, nb_grid_pts], [DOMAIN_LENGTH, DOMAIN_LENGTH])
    grid.decompose([1, MPI.COMM_WORLD.Get_size()], (1, 1), communicator=MPI.COMM_WORLD)
    x, y = grid.form_spatial_mesh()
    r_squared = grid.get_local((x - 0.5 * DOMAIN_LENGTH) ** 2 + (y - 0.5 * DOMAIN_LENGTH) ** 2)
    sphere = r_squared / (2 * radius)
    flat = np.zeros_like(sphere)
    contact = ElasticContact(grid, ELASTIC_PARAMS, sphere, flat, communicator=MPI.COMM_WORLD)
    contact.set_mean_separation(-0.08)

    problem = formulate_constant_volume_gap_problem(contact)
    x0 = np.full(sphere.shape, contact.get_gap_origin_volume() / DOMAIN_LENGTH**2)
    solver = ProjectedLbfgs(max_inner_iter=1000, tol_gradient=1e-8)
    result = solver.solve_minimisation(problem, x0)
    assert result["success"]

    # Get pressure
    contact.set_gap(result["x"])
    pressure = contact.get_normal_stress() + result["dual"]

    force = result["dual"] * np.multiply.reduce(grid.domain_lengths)
    contact_modulus = ELASTIC_PARAMS["youngs_modulus"] / (1 - ELASTIC_PARAMS["poisson_ratio"] ** 2)
    contact_radius = np.cbrt(3 * force * radius / (4 * contact_modulus))
    peak_pressure = 2 * contact_modulus * contact_radius / (np.pi * radius)
    hertz_pressure = peak_pressure * np.sqrt(np.clip(1 - r_squared / contact_radius**2, 0, None))

    if show_me_plot:
        import matplotlib.pyplot as plt

        # ruff: noqa: RUF059
        disp = contact.get_gap_origin() - contact.get_gap()
        fig, ax = plt.subplots()
        ax.plot(disp[nb_grid_pts // 2])
        ax.set_title("Displacement")
        fig, ax = plt.subplots()
        ax.imshow(disp.T, cmap=plt.cm.Blues)
        ax.set_title("Displacement 2D")
        fig, ax = plt.subplots()
        ax.plot(pressure[nb_grid_pts // 2], linestyle="-")
        ax.plot(hertz_pressure[nb_grid_pts // 2], linestyle="--")
        ax.set_title("Pressure")
        plt.show()

    np.testing.assert_allclose(pressure, hertz_pressure, atol=3e-3)
