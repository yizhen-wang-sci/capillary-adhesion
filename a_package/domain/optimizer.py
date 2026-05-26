"""
Solving the numerical optimization problem. No physics meaning in this file.
"""

import logging
import timeit
import typing
from typing import Callable, Protocol

import numpy as np
import NuMPI.Optimization
import NuMPI.Tools
import scipy.optimize


logger = logging.getLogger(__name__)


class Problem:
    """Numerical optimization problem with optional constraints.

    x* = arg min f(x)
    s.t. A x - b == 0 (linear equality constraints)
    s.t. g(x) == 0 (equality constraints)
    s.t. x_lb <= x <= x_ub (simple bounds)

    Accessing constraints not provided at instantiation raises AttributeError.
    Return values are reshaped to the conventions optimizers expect.
    So far, b, g, x_lb, and x_ub are assumed to be scalars.
    """

    def __init__(self,
                 get_x: Callable[[], np.ndarray],
                 set_x: Callable[[np.ndarray], None],
                 get_f: Callable[[], float],
                 get_f_Dx: Callable[[], np.ndarray],
                 A: np.ndarray | None = None,
                 b: float | None = None,
                 get_g: Callable[[], float] | None = None,
                 get_g_Dx: Callable[[], np.ndarray] | None = None,
                 x_lb: float | None = None,
                 x_ub: float | None = None,
                 is_zero: np.ndarray | None = None):
        self._get_x = get_x
        self._set_x = set_x
        self._get_f = get_f
        self._get_f_Dx = get_f_Dx
        if A is not None:
            self._A = A
        if b is not None:
            self._b = b
        if get_g is not None:
            self._get_g = get_g
        if get_g_Dx is not None:
            self._get_g_Dx = get_g_Dx
        if x_lb is not None:
            self._x_lb = x_lb
        if x_ub is not None:
            self._x_ub = x_ub
        if is_zero is not None:
            self._is_zero = is_zero

    @property
    def has_linear_constraints(self):
        return hasattr(self, "_A") and hasattr(self, "_b")

    @property
    def has_equality_constraints(self):
        return hasattr(self, "_get_g") and hasattr(self, "_get_g_Dx")

    @property
    def has_bounds(self):
        return hasattr(self, "_x_lb") and hasattr(self, "_x_ub")

    @property
    def has_zeros(self):
        return hasattr(self, "_is_zero")

    def get_x(self):
        return np.asarray(self._get_x()).ravel()

    def set_x(self, x):
        """Set x, skipping the underlying call when x is unchanged.

        Caching matters because optimizers typically re-query f and its
        gradient at the same point during backtracking.
        """
        if np.any(np.asarray(x).ravel() != self.get_x()):
            self._set_x(x)

    def get_f(self):
        return np.asarray(self._get_f()).item()

    def get_f_Dx(self):
        return np.asarray(self._get_f_Dx()).ravel()

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    def get_g(self):
        return np.asarray(self._get_g()).item()

    def get_g_Dx(self):
        return np.asarray(self._get_g_Dx()).ravel()

    @property
    def x_lb(self):
        return self._x_lb

    @property
    def x_ub(self):
        return self._x_ub

    @property
    def is_zero(self):
        return np.asarray(self._is_zero).ravel()


class OptimizerResult(typing.TypedDict, total=False):
    """Result of an optimizer."""

    x: typing.Required[np.ndarray]
    dual: float
    success: typing.Required[bool]
    reached_iter_limit: bool
    had_abnormal_stop: bool
    message: str
    fun: float
    jac: float
    nit: int
    time: float


class Optimizer(Protocol):

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, *args, callback=None, **kwargs) -> OptimizerResult:
        pass


class AugmentedLagrangian(Optimizer):

    def __init__(self, max_outer_loop: int=20, max_inner_iter: int = 1000, tol_gradient: float = 1e-6,
                 tol_eq_constraint: float = 1e-6, tol_creeping: float = 1e-12):
        if max_outer_loop < 0:
            raise ValueError("Maximum number of loop must be non-negative.")
        self.max_outer_loop = max_outer_loop
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient
        self.tol_eq_constraint = tol_eq_constraint
        self.tol_creeping = tol_creeping

        self.sufficient_eq_decrease = 1e-2
        self.eq_weight_multiplier = 3e0
        self.eq_weight_maximum = 1e10

        self.bound_weight_minimum = 5e-5
        self.bound_weight_multiplier = 2e-1

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, callback=None, **kwargs) -> OptimizerResult:
        """
        :param problem: Problem instance
        :param x0: Initial guess.
        :param kwargs: Additional parameters like lam0, alpha0, beta0, etc.
        :return: OptimizerResult
        """
        lam0 = kwargs.pop("lam0", None)
        alpha0 = kwargs.pop("alpha0", None)
        beta0 = kwargs.pop("beta0", None)

        # Check problem type
        has_eq_constraint = problem.has_equality_constraints
        has_bound_constraint = problem.has_bounds

        # Check values
        x0 = np.asarray(x0)
        # Initialize dual variables and parameters with defaults
        if has_eq_constraint:
            if lam0 is None:
                lam0 = 0.0
            if alpha0 is None:
                alpha0 = 1e0

        if has_bound_constraint:
            if np.any(x0 < problem.x_lb) or np.any(x0 > problem.x_ub):
                x0 = np.clip(x0, problem.x_lb, problem.x_ub)
            # FIXME: replace clipping?
            # if beta0 is None:
            #     beta0 = 1e0

        # Initial values (primal, dual, parameters)
        x_plus = x0
        lam_plus = lam0
        alpha_plus = alpha0
        beta_plus = beta0
        # FIXME: using clipping as a work-around
        # if has_bound_constraint:
        #     s0 = inverse_squashing(x0, problem.x_lb, problem.x_ub)
        #     s_plus = s0

        # Print headers
        symb_nabla = "\u2207"
        symb_alpha = "\u03B1"
        symb_lam = "\u03BB"
        tabel_headers_line1 = ["Loop", "f", f"|{symb_nabla}f+{symb_lam}{symb_nabla}g|", "|g|"]
        if has_eq_constraint:
            tabel_headers_line1 += [f"{symb_lam}", f"{symb_alpha}"]
        tabel_headers_line2 = ["Iter", "L", f"|{symb_nabla}L|", "Message"]
        separator = "  "
        line_width = 80
        logger.info(separator.join("{:<4}".format(col_name) if col_name in [
                    "Loop"] else "{:<8}".format(col_name)for col_name in tabel_headers_line1))
        logger.info(separator.join("{:<4}".format(col_name) if col_name in [
                    "Iter"] else "{:<8}".format(col_name)for col_name in tabel_headers_line2))
        logger.info("=" * line_width)

        # State flags
        is_converged = False
        reached_limit = False
        had_abnormal_stop = False

        loop_count = 0
        for loop_count in range(self.max_outer_loop + 1):
            # Update
            x = x_plus
            # FIXME: replace the clipping?
            # if has_bound_constraint:
            #     s = s_plus
            lam = lam_plus
            alpha = alpha_plus
            beta = beta_plus

            problem.set_x(x)
            f = problem.get_f()

            # Compute gradients
            l_Dx = problem.get_f_Dx()
            if has_eq_constraint:
                # Add the contribution of equality constraint gradient
                l_Dx += lam * problem.get_g_Dx()
            # FIXME: replace the clipping?
            if has_bound_constraint:
                # Project the gradient to the bounds
                l_Dx[(x <= problem.x_lb) & (l_Dx > 0)] = 0
                l_Dx[(x >= problem.x_ub) & (l_Dx < 0)] = 0
            l_Dx_norm = np.amax(abs(l_Dx))

            # Compute equality constraint
            g_norm = 0
            if has_eq_constraint:
                # FIXME: g is considered a scalar function as of now
                g_norm = abs(problem.get_g())

            # Print states
            padded_literals = [f"{loop_count:>4d}", f"{f:>8.1e}", f"{l_Dx_norm:>8.1e}", f"{g_norm:>8.1e}"]
            if has_eq_constraint:
                padded_literals += [f"{lam:>8.1e}", f"{alpha:>8.1e}"]
            logger.info(separator.join(padded_literals))

            # Check convergence
            criteria_l_Dx = l_Dx_norm < self.tol_gradient
            criteria_g = g_norm < self.tol_eq_constraint
            if criteria_l_Dx and criteria_g:
                is_converged = True
                break

            # For last iter, no more trial
            if loop_count == self.max_outer_loop:
                reached_limit = True
                break

            # reform and solve
            reformed = problem
            if has_eq_constraint:
                reformed = approx_eq_by_augmented_lagrangian(reformed, lam, alpha)
            if has_bound_constraint:
                # FIXME: replace the clipping?
                # reformed = approx_bound_by_squashing(reformed, beta)
                reformed = approx_bound_by_clipping(reformed)
            result = solve_unconstrained(reformed, x, max_iter=self.max_inner_iter, tol_gradient=self.tol_gradient,
                                         tol_creeping=self.tol_creeping, callback=callback)
            x_plus = result['x']

            # FIXME: replace the clipping?
            # if has_bound_constraint:
            #     result = solve_unconstrained(reformed, s, max_iter=self.max_iter, tol_gradient=1e-8, tol_creeping=1e-12)
            #     s_plus = result['x']
            #     x_plus = squashing(s_plus, num_opt.x_lb, num_opt.x_ub)
            # else:
            #     result = solve_unconstrained(reformed, x, max_iter=self.max_iter, tol_gradient=1e-8, tol_creeping=1e-12)
            #     x_plus = result['x']

            # Print progress
            augm_lagr = reformed.get_f()
            augm_lagr_Dx_norm = np.amax(abs(reformed.get_f_Dx()))
            padded_literals = [f"{result['nit']:>4d}", f"{augm_lagr:>8.1e}", f"{augm_lagr_Dx_norm:>8.1e}", result['message']]
            logger.info(separator.join(padded_literals))
            logger.info("-" * line_width)

            if result['had_abnormal_stop']:
                had_abnormal_stop = True
                break

            # FIXME: replace the clipping?
            # if has_bound_constraint:
            #     if not np.all(criteria_l_Dx) and beta > self.bound_weight_minimum:
            #         beta_plus = beta * self.bound_weight_multiplier

            # Prepare for the next loop
            if has_eq_constraint:
                problem.set_x(x_plus)
                g_plus = problem.get_g()
                lam_plus = lam + alpha * g_plus

                # Increase penalty weight of equality constraint if not much improved
                if not criteria_g and not abs(g_plus) < self.sufficient_eq_decrease * g_norm and alpha < self.eq_weight_maximum:
                    alpha_plus = alpha * self.eq_weight_multiplier

        # Print based on flags
        if is_converged:
            logger.info(f"INFO: achieving required tolerance at trial #{loop_count}")
        if reached_limit:
            logger.warning(f"WARNING: reached loop limit.")
        if had_abnormal_stop:
            logger.warning(f"WARNING: abnormal stop.")

        # Prepare return value
        result = OptimizerResult(x=x_plus, success=is_converged, nit=loop_count,
                                 reached_iter_limit=reached_limit, had_abnormal_stop=had_abnormal_stop)
        if has_eq_constraint:
            result['dual'] = lam_plus
            result['final_penalty'] = alpha_plus
        return result

def solve_unconstrained(self, problem: Problem, x0: np.ndarray, max_iter: int, tol_gradient: float,
                        tol_creeping: float, callback=None):
    """
    Solve unconstrained minimization using L-BFGS.

    Parameters
    ----------
    problem : Problem
        Problem instance.
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
        problem.set_x(x.reshape(x_shape))
        return problem.get_f()

    def compute_f_Dx(x):
        problem.set_x(x.reshape(x_shape))
        return problem.get_f_Dx()

    bounds = None
    # if problem.has_bounds:
    #     bounds = [(problem.x_lb, problem.x_ub)] * np.size(x0)

    # Serial implementation using scipy
    [x_plus, f_plus, info] = scipy.optimize.fmin_l_bfgs_b(
        compute_f,
        x0,
        fprime=compute_f_Dx,
        maxiter=max_iter,
        bounds=bounds,
        # relative decrease of 'f', in units of 'eps'
        factr=tol_creeping / np.finfo(np.float64).resolution,
        # this 'pg' should be zero at exactly a local minimizer
        pgtol=tol_gradient,
        callback=callback,
    )

    t_exec += timeit.default_timer()
    problem.set_x(np.reshape(x_plus, x_shape))
    is_converged = info["warnflag"] == 0
    return OptimizerResult(
        x=problem.get_x(),
        success=is_converged,
        reached_iter_limit=info["nit"] >= max_iter,
        had_abnormal_stop=not is_converged and info["nit"] < max_iter,
        message=info["task"].decode() if isinstance(info["task"], bytes) else info["task"],
        fun=f_plus,
        jac=info["grad"],
        nit=info["nit"],
        time=t_exec,
    )


def approx_eq_by_augmented_lagrangian(problem: Problem, lam: float, alpha: float):

    def get_augmented_lagrangian():
        g = problem.get_g()
        return problem.get_f() + lam * g + (0.5 * alpha) * g**2

    def get_augmented_lagrangian_Dx():
        return problem.get_f_Dx() + (lam + alpha * problem.get_g()) * problem.get_g_Dx()

    reformed = {"get_x": problem.get_x, "set_x": problem.set_x,
                "get_f": get_augmented_lagrangian, "get_f_Dx": get_augmented_lagrangian_Dx}

    # If it has bound constraints, pass it
    if problem.has_bounds:
        reformed.update({"x_lb": problem.x_lb, "x_ub": problem.x_ub, "has_bounds": True})

    return Problem(**reformed)


def approx_bound_by_clipping(problem: Problem):

    def set_x_clipped(x: np.ndarray):
        problem.set_x(np.clip(x, problem.x_lb, problem.x_ub))

    def get_f_Dx_projected():
        f_Dx = problem.get_f_Dx()
        x = problem.get_x()
        f_Dx[(x <= problem.x_lb) & (f_Dx > 0)] = 0
        f_Dx[(x >= problem.x_ub) & (f_Dx < 0)] = 0
        return f_Dx

    return Problem(get_x=problem.get_x, set_x=set_x_clipped, get_f=problem.get_f, get_f_Dx=get_f_Dx_projected)


def approx_bound_by_squashing(problem: Problem, beta: float=1e-2):
    # store the "free x" as extra states
    free_x = problem.get_x()

    def get_free_x():
        return free_x

    def set_x_with_squashing(x: np.ndarray):
        nonlocal free_x
        free_x = x
        problem.set_x(squashing(free_x, problem.x_lb, problem.x_ub))

    def get_f_with_barrier():
        f = problem.get_f()
        return f + beta * barrier_squashed(free_x, problem.x_lb, problem.x_ub)

    def get_f_Dx_with_squashing_and_barrier():
        return problem.get_f_Dx() * squashing_Dx(free_x, problem.x_lb, problem.x_ub) + beta * barrier_squashed_Dx(
            free_x, problem.x_lb, problem.x_ub)

    return Problem(
        get_x=get_free_x,
        set_x=set_x_with_squashing,
        get_f=get_f_with_barrier,
        get_f_Dx=get_f_Dx_with_squashing_and_barrier)


def squashing(x: np.ndarray, x_lb: float, x_ub: float):
    x_c = (x_ub + x_lb) / 2
    return (x_ub - x_lb) / 2 * np.tanh(x - x_c) + x_c


def squashing_Dx(x: np.ndarray, x_lb: float, x_ub: float):
    x_c = (x_ub + x_lb) / 2
    return (x_ub - x_lb) / 2 * (1 - np.tanh(x - x_c)**2)


def inverse_squashing(x: np.ndarray, x_lb: float, x_ub: float):
    x_c = (x_ub + x_lb) / 2
    clipped = np.clip(2 / (x_ub - x_lb) * (x - x_c), -0.999999, 0.999999)
    return np.arctanh(clipped) + x_c


def barrier_squashed(x: np.ndarray, x_lb: float, x_ub: float):
    # barrier = (inverse_squashing(x, x_lb, x_ub) - x_c)**2
    x_c = (x_ub + x_lb) / 2
    return 0.5 * np.sum((x - x_c)**2)


def barrier_squashed_Dx(x: np.ndarray, x_lb: float, x_ub: float):
    x_c = (x_ub + x_lb) / 2
    return x - x_c


class ProjectedLbfgs(Optimizer):
    """Can handle linear equality constraint and box inequality constraint."""

    def __init__(self, max_inner_iter: int = 1000, tol_gradient: float = 1e-6):
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, communicator=None, callback=None, **kwargs) -> OptimizerResult:
        linear_constraint = NuMPI.Optimization.LinearConstraint(problem.A, problem.b, NuMPI.Tools.Reduction(communicator))

        def compute_f(x):
            problem.set_x(x)
            return problem.get_f()

        def compute_f_Dx(x):
            problem.set_x(x)
            return problem.get_f_Dx()

        bounds_lo = None
        bounds_hi = None
        if problem.has_bounds:
            bounds_lo = problem.x_lb
            bounds_hi = problem.x_ub

        zero_mask = None
        if problem.has_zeros:
            zero_mask = problem.is_zero

        init_shape = x0.shape
        result = NuMPI.Optimization.l_bfgs_projected(
            compute_f,
            x0.ravel(),
            linear_constraint,
            jac=compute_f_Dx,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            zero_mask=zero_mask,
            maxiter=self.max_inner_iter,
            gtol=self.tol_gradient,
            comm=communicator,
            callback=callback,
        )
        return OptimizerResult(x=result['x'].reshape(init_shape), dual=result['multiplier'], success=result['success'],
                               message=result['message'], nit=result['nit'])


class BoundedLbfgs(Optimizer):
    """Can handle linear equality constraint and box inequality constraint."""

    def __init__(self, max_inner_iter: int = 1000, tol_gradient: float = 1e-6):
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, communicator=None, callback=None, **kwargs) -> OptimizerResult:

        def compute_f(x):
            problem.set_x(x)
            return problem.get_f()

        def compute_f_Dx(x):
            problem.set_x(x)
            return problem.get_f_Dx()

        bounds_lo = None
        bounds_hi = None
        if problem.has_bounds:
            bounds_lo = problem.x_lb
            bounds_hi = problem.x_ub

        zero_mask = None
        if problem.has_zeros:
            zero_mask = problem.is_zero

        init_shape = x0.shape
        result = NuMPI.Optimization.l_bfgs_bounded(
            compute_f,
            x0.ravel(),
            jac=compute_f_Dx,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            zero_mask=zero_mask,
            maxiter=self.max_inner_iter,
            gtol=self.tol_gradient,
            comm=communicator,
            callback=callback,
        )
        return OptimizerResult(x=result['x'].reshape(init_shape), success=result['success'],
                               message=result['message'], nit=result['nit'])
