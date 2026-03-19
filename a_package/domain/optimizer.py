"""
Solving the numerical optimization problem. No physics meaning in this file.
"""

import types
import logging
import timeit
import typing

import numpy as np
import scipy.optimize


logger = logging.getLogger(__name__)


class NumOpt(typing.Protocol):
    """Numerical optimization problem, unconstrained.

    x* = arg min f(x)
    """

    def get_x(self) -> np.ndarray: ...

    def set_x(self, x: np.ndarray): ...

    def get_f(self) -> float: ...

    def get_f_Dx(self) -> np.ndarray: ...


class NumOptEq(NumOpt, typing.Protocol):
    """Numerical optimization problem with equality constraints.

    x* = arg min f(x)

    s.t. g(x) == 0
    """

    def get_g(self) -> float: ...

    def get_g_Dx(self) -> np.ndarray: ...


class NumOptB(NumOpt, typing.Protocol):
    """Numerical optimization problem with simple bounds.

    x* = arg min f(x)

    s.t. x_lb <= x <= x_ub
    """

    @property
    def x_lb(self) -> float: ...

    @property
    def x_ub(self) -> float: ...


class NumOptEqB(NumOptEq, NumOptB, typing.Protocol):
    """Numerical optimization problem with equality constraints and simple bounds.

    x* = arg min f(x)

    s.t. g(x) == 0, x_lb <= x <= x_ub
    """


class OptimizerResult(typing.TypedDict, total=False):
    primal: typing.Required[np.ndarray]
    is_converged: typing.Required[bool]
    dual: float
    slack: np.ndarray
    time: float
    nit: int
    reached_iter_limit: bool
    had_abnormal_stop: bool
    message: str
    final_penalty: float


class Optimizer:

    def __init__(self, max_loop: int=20, max_iter: int = 1000, tol_convergence: float = 1e-6,
                 tol_eq_constraint: float = 1e-9, tol_bound_constraint=1e-6, tol_creeping: float = 1e-12):
        if max_loop < 0:
            raise ValueError("Maximum number of loop must be non-negative.")
        self.max_loop = max_loop
        self.max_iter = max_iter
        self.tol_gradient = tol_convergence
        self.tol_creeping = tol_creeping
        self.tol_eq_constraint = tol_eq_constraint
        self.tol_bound_constraint = tol_bound_constraint
        self.sufficient_eq_decrease = 1e-2
        self.eq_weight_multiplier = 3e0

    def solve_minimisation(self, num_opt: typing.Union[NumOpt, NumOptEq, NumOptB, NumOptEqB], x0: typing.Sequence[float],
                           lam0: float=None, alpha0: float=None, s0: np.ndarray=None, beta0: float=None):
        """
        :param num_opt: NumOpt, NumOptEq, NumOptB, NumOptEqB
        :param x0: Initial guess.
        :param lam0: Initial dual variable.
        :param alpha0: Initial penalty weight for equality constraint.
        :param s0: Initial value for unbound x.
        :param beta0: Initial squashing parameter for bound constraint.
        :return: OptimizerResult
        """
        # Check problem type
        if hasattr(num_opt, "get_g") ^ hasattr(num_opt, "get_g_Dx"):
            raise ValueError("When EQ constraint is provided, one must specify both value and gradient function.")
        has_eq_constraint = hasattr(num_opt, "get_g") and hasattr(num_opt, "get_g_Dx")

        if hasattr(num_opt, "x_lb") ^ hasattr(num_opt, "x_ub"):
            raise ValueError("One must specify a two-side bound. One-side bound is not supported yet.")
        has_bound_constraint = hasattr(num_opt, "x_lb") and hasattr(num_opt, "x_ub")

        # Check values
        x0 = np.asarray(x0)
        if has_bound_constraint:
            if np.any(x0 < num_opt.x_lb) or np.any(x0 > num_opt.x_ub):
                if s0 is None or beta0 is None:
                    x0 = np.clip(x0, num_opt.x_lb, num_opt.x_ub)
                else:
                    x0 = squashing(s0, num_opt.x_lb, num_opt.x_ub, beta0)

        # Initial values (primal, dual, parameters)
        x_plus = x0
        lam_plus = lam0
        alpha_plus = alpha0
        beta_plus = beta0

        # State flags
        is_converged = False
        reached_limit = False

        for loop_count in range(self.max_loop + 1):
            # Update
            x = x_plus
            lam = lam_plus
            alpha = alpha_plus
            beta = beta_plus

            # Check gradients
            num_opt.set_x(x)
            l_Dx = num_opt.get_f_Dx()
            if has_eq_constraint:
                g_Dx = num_opt.get_g_Dx()
                # NOTE: here the plus is aligned with augmented Lagrangian
                l_Dx += lam * g_Dx
            criteria_l_Dx = np.abs(l_Dx) < self.tol_gradient
            if has_bound_constraint:
                criteria_x_lb = (x - num_opt.x_lb) < self.tol_bound_constraint
                criteria_x_ub = (num_opt.x_ub - x) < self.tol_bound_constraint
                criteria_l_Dx |= (criteria_x_lb | criteria_x_ub)

            # Check equality constraint
            criteria_g = True
            if has_eq_constraint:
                g = num_opt.get_g()
                criteria_g = np.abs(g) < self.tol_eq_constraint

            # When both are satisfied, the solver is converged
            if np.all(criteria_l_Dx) and criteria_g:
                is_converged = True
                break

            # For last iter, no more trial
            if loop_count == self.max_loop:
                reached_limit = True
                break

            # reform and solve
            reformed = num_opt
            if has_eq_constraint:
                reformed = approx_eq_by_augmented_lagrangian(reformed, lam, alpha)
            if has_bound_constraint:
                reformed = approx_bound_by_squashing(reformed, beta)
            result = solve_unconstrained(reformed, x0)

            # Prepare for the next loop
            x_plus = result['primal']
            if has_bound_constraint:
                x_plus = squashing(x_plus, num_opt.x_lb, num_opt.x_ub, beta)

            if has_eq_constraint and not criteria_g:
                num_opt.set_x(x_plus)
                g_plus = num_opt.get_g()
                lam_plus = lam + alpha * g_plus
                # Increase penalty weight of equality constraint if not much improved
                if not np.abs(g_plus) < self.sufficient_eq_decrease * np.abs(g):
                    alpha_plus = alpha * self.eq_weight_multiplier

        # Print based on flags
        if reached_limit:
            logger.warning(f"WARNING: reached loop limit.")
        if is_converged:
            logger.info(f"INFO: achieving required tolerance at trial #{loop_count}")

        # Prepare return value
        result = OptimizerResult(primal=x_plus, is_converged=is_converged)
        if has_eq_constraint:
            result['dual'] = lam_plus
        return result


def approx_eq_by_augmented_lagrangian(num_opt: typing.Union[NumOptEq, NumOptEqB], lam: float, alpha: float):

    def get_augmented_lagrangian():
        g = num_opt.get_g()
        return num_opt.get_f() + lam * g + (0.5 * alpha) * g**2

    def get_augmented_lagrangian_Dx():
        return num_opt.get_f_Dx() + (lam + alpha * num_opt.get_g()) * num_opt.get_g_Dx()

    reformed = {"get_x": num_opt.get_x, "set_x": num_opt.set_x,
                "get_f": get_augmented_lagrangian, "get_f_Dx": get_augmented_lagrangian_Dx}

    # If it has bound constraints, pass it
    try:
        reformed.update({"x_lb": num_opt.x_lb, "x_ub": num_opt.x_ub})
    except AttributeError:
        pass

    return types.SimpleNamespace(**reformed)


def approx_bound_by_squashing(num_opt: NumOptB, beta: float):

    # store the "free x" as extra states
    free_x = num_opt.get_x()

    def get_free_x():
        return free_x

    def set_x_with_squashing(x: np.ndarray):
        nonlocal free_x
        free_x = x
        num_opt.set_x(squashing(free_x, num_opt.x_lb, num_opt.x_ub, beta))

    def get_f_Dx_with_squashing():
        return num_opt.get_f_Dx() * squashing_Dx(free_x, num_opt.x_lb, num_opt.x_ub, beta)
        # return num_opt.get_f_Dx() * sb.squashing_Dx(s)

    return types.SimpleNamespace(
        get_x=get_free_x,
        set_x=set_x_with_squashing,
        get_f=num_opt.get_f,
        get_f_Dx=get_f_Dx_with_squashing)


def squashing(x: np.ndarray, x_lb: float, x_ub: float, beta: float):
    x_c = (x_ub + x_lb) / 2
    return (x_ub - x_lb) / 2 * np.tanh(beta * (x - x_c)) + x_c


def squashing_Dx(x: np.ndarray, x_lb: float, x_ub: float, beta: float):
    x_c = (x_ub + x_lb) / 2
    return (x_ub - x_lb) / 2 * beta * (1 - np.tanh(beta * (x - x_c))**2)


def solve_unconstrained(numopt: NumOpt, x0: np.ndarray, max_iter: int = 10000, tol_convergence=1e-6, tol_creeping=1e2):
    """
    Solve unconstrained minimization using L-BFGS.

    Parameters
    ----------
    numopt : NumOpt
        Problem with get_x, set_x, get_f, get_f_Dx.
    x0 : np.ndarray
        Initial guess.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    SolverResult
    """
    x_shape = x0.shape
    t_exec = -timeit.default_timer()

    def compute_f(x):
        numopt.set_x(x.reshape(x_shape))
        return numopt.get_f()

    def compute_f_Dx(x):
        numopt.set_x(x.reshape(x_shape))
        return numopt.get_f_Dx()

    # Serial implementation using scipy
    [x_plus, f_plus, info] = scipy.optimize.fmin_l_bfgs_b(
        compute_f,
        x0,
        fprime=compute_f_Dx,
        maxiter=max_iter,
        # relative decrease of 'f', in units of 'eps'
        factr=tol_creeping,
        # this 'pg' should be zero at exactly a local minimizer
        pgtol=tol_convergence,
    )

    t_exec += timeit.default_timer()
    numopt.set_x(np.reshape(x_plus, x_shape))
    is_converged = info["warnflag"] == 0
    return OptimizerResult(
        primal=numopt.get_x(),
        time=t_exec,
        nit=info["nit"],
        is_converged=is_converged,
        reached_iter_limit=info["nit"] >= max_iter,
        had_abnormal_stop=not is_converged and info["nit"] < max_iter,
        message=info["task"].decode() if isinstance(info["task"], bytes) else info["task"],
    )
