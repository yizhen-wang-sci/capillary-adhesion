import numpy as np

from a_package.domain import CentroidQuadrature, FirstOrderElement, Grid
from a_package.domain.optimizer import Problem, ProjectedLbfgs


def test_problem(decompose_stitch, comm_world):
    """A testing problem where the objective is to minimize the gradient, while keeping the
    sum quantity constant, with a sinusoidal field as the initial guess."""
    nb_pts = (8, 8)
    grid = Grid(nb_pts)

    decompose, _stitch = decompose_stitch
    decomposition = decompose(grid)

    mean_field = np.ones(nb_pts)
    xm, ym = grid.form_spatial_mesh()
    Lx, Ly = grid.domain_lengths
    sinusoidal_field = np.cos(2 * np.pi * xm / Lx) * np.sin(2 * np.pi * ym / Ly)

    quadr = CentroidQuadrature(communicator=comm_world)
    fem = FirstOrderElement(quadr.quad_pt_coords, grid.element_sizes)

    collection = decomposition.collection
    collection.set_nb_sub_pts("nodal", 1)
    collection.set_nb_sub_pts("quadr", quadr.nb_quad_pts)

    field_nodal = decomposition.collection.real_field("nodal", 1, "nodal")
    field_quadr_1 = decomposition.collection.real_field("quadr_1", 1, "quadr")
    field_quadr_1_gradient = decomposition.collection.real_field("quadr_1_gradient", 2, "quadr")

    field_quadr_2_gradient = decomposition.collection.real_field("quadr_2_gradient", 2, "quadr")
    field_quadr_2_gradient_back_sens = decomposition.collection.real_field("quadr_2_gradient_back_sens", 1, "nodal")

    field_quadr_3 = decomposition.collection.real_field("quadr_3", 1, "quadr")
    field_quadr_3_back_sens = decomposition.collection.real_field("quadr_3_back_sens", 1, "nodal")

    def set_field(x: np.ndarray):
        field_nodal.s[0, 0, ...] = np.reshape(x, decomposition.nb_subdomain_grid_pts)
        decomposition.communicate_ghosts(field_nodal)
        fem.interpolate_value(field_nodal, field_quadr_1)
        fem.interpolate_gradient(field_nodal, field_quadr_1_gradient)

    def get_field():
        return field_nodal.s[0, 0, ...]

    def objective():
        return quadr.integrate(0.5 * np.sum(field_quadr_1_gradient.s**2, axis=0, keepdims=True), grid.element_area)

    def objective_gradient():
        integrand_derivative = field_quadr_1_gradient.s
        field_quadr_2_gradient.s[...] = quadr.propag_integral_weight(integrand_derivative, grid.element_area)
        decomposition.communicate_ghosts(field_quadr_2_gradient)
        fem.propag_sens_gradient(field_quadr_2_gradient, field_quadr_2_gradient_back_sens)
        return field_quadr_2_gradient_back_sens.s[0, 0, ...]

    def constraint_jacobian():
        integrand_derivative = np.ones_like(field_quadr_1.s)
        field_quadr_3.s[...] = quadr.propag_integral_weight(integrand_derivative, grid.element_area)
        decomposition.communicate_ghosts(field_quadr_3)
        fem.propag_sens_value(field_quadr_3, field_quadr_3_back_sens)
        return field_quadr_3_back_sens.s[0, 0, ...]

    problem = Problem(
        set_x=set_field,
        get_x=get_field,
        get_f=objective,
        get_f_Dx=objective_gradient,
        A=constraint_jacobian().ravel(),
        b=grid.element_area * np.sum(mean_field),
        communicator=comm_world,
    )
    optimizer = ProjectedLbfgs(max_inner_iter=10)

    result = optimizer.solve_minimisation(problem, x0=grid.get_local(sinusoidal_field))
    solved_field = result["x"].reshape(decomposition.nb_subdomain_grid_pts)
    print(result)
    assert result["success"]
    assert result["nit"] < optimizer.max_inner_iter
    np.testing.assert_allclose(solved_field, grid.get_local(mean_field))
