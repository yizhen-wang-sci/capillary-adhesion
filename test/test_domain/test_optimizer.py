import types

import numpy as np

from a_package.domain.optimizer import Optimizer


def test_unconstrained_numopt():

    saved_x = np.zeros(0)

    def get_x():
        return saved_x

    def set_x(x):
        nonlocal saved_x
        saved_x = x

    def get_f():
        return np.sum(saved_x**2)

    def get_f_Dx():
        return 2 * saved_x

    num_opt = types.SimpleNamespace(get_x=get_x, set_x=set_x, get_f=get_f, get_f_Dx=get_f_Dx)

    optimizer = Optimizer(max_loop=1)
    result = optimizer.solve_minimisation(num_opt, x0=[10, 10])

    assert np.all(np.isclose(result['primal'], 0.))
    assert result['is_converged']


def test_eq_constrained_numopt():

    saved_x = np.zeros(0)

    def get_x():
        return saved_x

    def set_x(x):
        nonlocal saved_x
        saved_x = x

    def get_f():
        return np.sum(saved_x**2)

    def get_f_Dx():
        return 2 * saved_x

    def get_g():
        x1, x2 = saved_x
        return (x1 - 2)**2 + x2**2 - 1

    def get_g_Dx():
        x1, x2 = saved_x
        return np.array([2 * (x1 - 2), 2 * x2])

    num_opt = types.SimpleNamespace(get_x=get_x, set_x=set_x, get_f=get_f, get_f_Dx=get_f_Dx, get_g=get_g, get_g_Dx=get_g_Dx)

    optimizer = Optimizer(max_loop=10)
    result = optimizer.solve_minimisation(num_opt, x0=[10., 10.], lam0=0., alpha0=1e1)

    assert np.all(np.isclose(result['primal'], [1., 0.]))
    assert result['is_converged']


def test_bound_constrained_numopt():

    saved_x = np.zeros(0)

    def get_x():
        return saved_x

    def set_x(x):
        nonlocal saved_x
        saved_x = x

    def get_f():
        return np.sum(saved_x**2)

    def get_f_Dx():
        return 2 * saved_x

    num_opt = types.SimpleNamespace(get_x=get_x, set_x=set_x, get_f=get_f, get_f_Dx=get_f_Dx, x_lb=np.array([2., -1]), x_ub=10.)

    optimizer = Optimizer(max_loop=5)
    result = optimizer.solve_minimisation(num_opt, x0=[10., 10.], beta0=1e-2)

    assert np.all(np.isclose(result['primal'], [2., 0.]))
    assert result['is_converged']


def test_eq_and_bound_constrained_numopt():

    saved_x = np.zeros(0)

    def get_x():
        return saved_x

    def set_x(x):
        nonlocal saved_x
        saved_x = x

    def get_f():
        return np.sum(saved_x**2)

    def get_f_Dx():
        return 2 * saved_x

    def get_g():
        x1, x2 = saved_x
        return (x1 - 2)**2 + x2**2 - 1

    def get_g_Dx():
        x1, x2 = saved_x
        return np.array([2 * (x1 - 2), 2 * x2])

    num_opt = types.SimpleNamespace(get_x=get_x, set_x=set_x, get_f=get_f, get_f_Dx=get_f_Dx, get_g=get_g, get_g_Dx=get_g_Dx, x_lb=np.array([2., -1.]), x_ub=10.)

    optimizer = Optimizer(max_loop=10)
    result = optimizer.solve_minimisation(num_opt, x0=np.array([10., 10.]), lam0=0., alpha0=1e1, beta0=1e-2)

    assert np.all(np.isclose(result['primal'], [2., 1.]))
    assert result['is_converged']
