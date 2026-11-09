"""Solving the numerical optimization problem. No physics meaning in this file."""

import logging
import sys
from collections.abc import Callable
from typing import Protocol, TypedDict

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

import NuMPI.Optimization
import NuMPI.Tools
import numpy as np
from NuMPI import MPI

logger = logging.getLogger(__name__)


class Problem:
    """A minimisation of f(x), optionally subject to g(x) == 0 and x_lb <= x <= x_ub."""

    def __init__(
        self,
        get_x: Callable[[], np.ndarray],
        set_x: Callable[[np.ndarray], None],
        get_f: Callable[[], float],
        get_f_Dx: Callable[[], np.ndarray],
        get_g: Callable[[], np.ndarray] | None = None,
        get_g_Dx: Callable[[], np.ndarray] | None = None,
        get_g_D2x_dot: Callable[[np.ndarray], np.ndarray] | None = None,
        g_degree: int | None = None,
        x_lb: float | np.ndarray | None = None,
        x_ub: float | np.ndarray | None = None,
        is_zeroed: np.ndarray | None = None,
        communicator=MPI.COMM_SELF,
    ):
        """Store the model's callables, and whichever constraints were given.

        Args:
            get_x: Read the model's current x.
            set_x: Write the model's x.
            get_f: The objective at the current x.
            get_f_Dx: The objective's gradient at the current x.
            get_g: The equality constraints g(x) at the current x, one value per constraint.
            get_g_Dx: The Jacobian ∇g(x) at the current x, one gradient per constraint.
            get_g_D2x_dot: The Hessian products ∇²g_j(x) u at the current x, one per constraint.
            g_degree: 1 if g is linear, 2 if g is quadratic, None if neither is declared.
            x_lb: Lower bound, either one for every entry of x or one per entry.
            x_ub: Upper bound, either one for every entry of x or one per entry.
            is_zeroed: Mask of entries held at zero.
            communicator: Communicator spanning the ranks the unknowns are spread across.

        Raises:
            ValueError: If `get_g` and `get_g_Dx` are not given together, if `get_g_D2x_dot` or
                `g_degree` is given without them, if `g_degree` is neither None, 1 nor 2, if
                `g_degree` is 2 without `get_g_D2x_dot`, or if `g_degree` is 1 with `get_g_D2x_dot`.
        """
        if (get_g is None) != (get_g_Dx is None):
            raise ValueError("get_g and get_g_Dx must be given together")
        if get_g is None and (get_g_D2x_dot is not None or g_degree is not None):
            raise ValueError("get_g_D2x_dot and g_degree require get_g")
        if g_degree not in (None, 1, 2):
            raise ValueError(f"g_degree must be None, 1 or 2, got {g_degree}")
        if g_degree == 2 and get_g_D2x_dot is None:
            raise ValueError("g_degree 2 requires get_g_D2x_dot")
        if g_degree == 1 and get_g_D2x_dot is not None:
            raise ValueError("g_degree 1 excludes get_g_D2x_dot")

        self._get_x = get_x
        self._set_x = set_x
        self._get_f = get_f
        self._get_f_Dx = get_f_Dx
        if get_g is not None and get_g_Dx is not None:
            self._get_g = get_g
            self._get_g_Dx = get_g_Dx
        if get_g_D2x_dot is not None:
            self._get_g_D2x_dot = get_g_D2x_dot
        self.g_degree = g_degree
        if x_lb is not None:
            self._x_lb = x_lb
        if x_ub is not None:
            self._x_ub = x_ub
        if is_zeroed is not None:
            self._is_zero = is_zeroed
        self.communicator = communicator

    @property
    def has_equality_constraints(self):
        """Whether an equality constraint was given."""
        return hasattr(self, "_get_g")

    @property
    def has_constraint_hessian_products(self):
        """Whether the Hessian products of the equality constraints were given."""
        return hasattr(self, "_get_g_D2x_dot")

    @property
    def has_bounds(self):
        """Whether simple bounds were given."""
        return hasattr(self, "_x_lb") and hasattr(self, "_x_ub")

    @property
    def has_zeros(self):
        """Whether a zero mask was given."""
        return hasattr(self, "_is_zero")

    def get_x(self):
        """The model's current unknowns, ravelled."""
        return np.asarray(self._get_x()).ravel()

    def set_x(self, x):
        """Set x, skipping the underlying call when x is unchanged.

        Args:
            x: The new unknowns, in the model's own shape.
        """
        is_changed = np.any(np.asarray(x).ravel() != self.get_x())
        is_changed = self.communicator.allreduce(is_changed, op=MPI.LOR)
        if is_changed:
            self._set_x(x)

    def get_f(self):
        """The objective at the current x, as a scalar."""
        return np.asarray(self._get_f()).item()

    def get_f_Dx(self):
        """The objective's gradient at the current x, ravelled."""
        return np.asarray(self._get_f_Dx()).ravel()

    def get_g(self):
        """The equality constraints g(x) at the current x, as an (m,) array.

        Raises:
            AttributeError: If no equality constraint was given.
        """
        return np.atleast_1d(np.asarray(self._get_g(), dtype=float)).ravel()

    def get_g_Dx(self):
        """The Jacobian ∇g(x) at the current x, as an (m, n) array.

        Raises:
            AttributeError: If no equality constraint was given.
        """
        return np.reshape(self._get_g_Dx(), (-1, np.size(self._get_x())))

    def get_g_D2x_dot(self, u):
        """The Hessian products ∇²g_j(x) u at the current x, as an (m, n) array.

        Args:
            u: The vector the Hessians are applied to, ravelled.

        Raises:
            AttributeError: If no Hessian product was given.
        """
        return np.reshape(self._get_g_D2x_dot(u), (-1, np.size(self._get_x())))

    @property
    def x_lb(self):
        """Lower bound on every entry of x.

        Raises:
            AttributeError: If no lower bound was given.
        """
        return self._x_lb

    @property
    def x_ub(self):
        """Upper bound on every entry of x.

        Raises:
            AttributeError: If no upper bound was given.
        """
        return self._x_ub

    @property
    def is_zero(self):
        """Mask of entries held at zero, ravelled.

        Raises:
            AttributeError: If no zero mask was given.
        """
        return np.asarray(self._is_zero).ravel()


def _measure_constraint_violation(problem: Problem, x: np.ndarray):
    """The absolute residual of each equality constraint at x.

    Args:
        problem: A problem with an equality constraint.
        x: The unknowns to evaluate at.

    Returns:
        An (m,) array.
    """
    problem.set_x(x)
    return np.abs(problem.get_g())


class OptimizerResult(TypedDict, total=False):
    """Result of an optimizer."""

    x: Required[np.ndarray]
    dual: np.ndarray
    constr_violation: np.ndarray
    success: Required[bool]
    reached_iter_limit: bool
    had_abnormal_stop: bool
    message: str
    fun: float
    jac: float
    nit: int
    time: float


class Optimizer(Protocol):
    """The interface a solving strategy presents to a caller."""

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, *args, callback=None, **kwargs) -> OptimizerResult:
        """Minimise the objective of `problem`, starting from `x0`.

        Args:
            problem: The problem to solve.
            x0: Initial guess.
            *args: Positional extras an implementation may take.
            callback: Called once per iteration by the underlying solver.
            **kwargs: Strategy-specific options.

        Returns:
            The result of the minimisation.
        """
        raise NotImplementedError


class ProjectedLbfgs(Optimizer):
    """Hands a linear equality constraint and box bounds to NuMPI to enforce natively."""

    def __init__(self, max_inner_iter: int = 1000, tol_gradient: float = 1e-6, tol_step: float = 0.0):
        """Set the iteration limit and the two convergence tolerances.

        Args:
            max_inner_iter: Iteration limit handed to NuMPI.
            tol_gradient: Tolerance on the infinity norm of the KKT-masked tangent gradient.
            tol_step: Tolerance on the infinity norm of the iterate step. Zero disables it.
        """
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient
        self.tol_step = tol_step

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, callback=None, **kwargs) -> OptimizerResult:
        """Minimise the objective subject to one linear equality constraint and bounds.

        Args:
            problem: The problem to solve.
            x0: Initial guess. Its shape is restored on the returned x.
            callback: Called once per iteration by NuMPI.
            **kwargs: To match the `Optimizer` interface.

        Returns:
            The result of the minimisation, carrying `dual`, the multiplier of the constraint,
            and `constr_violation`, the absolute residual of g at the returned x.

        Raises:
            TypeError: If the problem has no equality constraint, or if g is not declared linear.
            ValueError: If g has more than one constraint.
        """
        if not problem.has_equality_constraints or problem.g_degree != 1:
            raise TypeError(f"ProjectedLbfgs requires a linear g, got g_degree {problem.g_degree}")
        gradient = problem.get_g_Dx()
        if gradient.shape[0] != 1:
            raise ValueError(f"ProjectedLbfgs requires one constraint in g, got {gradient.shape[0]}")
        a = np.array(gradient[0])
        target = problem.communicator.allreduce(a @ problem.get_x(), op=MPI.SUM) - problem.get_g()[0]
        linear_constraint = NuMPI.Optimization.LinearConstraint(a, target, NuMPI.Tools.Reduction(problem.communicator))

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
            xtol=self.tol_step,
            comm=problem.communicator,
            callback=callback,
        )
        return OptimizerResult(
            x=result["x"].reshape(init_shape),
            dual=np.atleast_1d(np.asarray(result["multiplier"], dtype=float)),
            constr_violation=_measure_constraint_violation(problem, result["x"]),
            success=result["success"],
            message=result["message"],
            nit=result["nit"],
        )


class BoundedLbfgs(Optimizer):
    """Hands box bounds to NuMPI to enforce natively, without an equality constraint."""

    def __init__(self, max_inner_iter: int = 1000, tol_gradient: float = 1e-6):
        """Set the iteration limit and the gradient tolerance.

        Args:
            max_inner_iter: Iteration limit handed to NuMPI.
            tol_gradient: Tolerance on the projected gradient.
        """
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient

    def solve_minimisation(self, problem: Problem, x0: np.ndarray, callback=None, **kwargs) -> OptimizerResult:
        """Minimise the objective subject to box bounds only.

        Args:
            problem: The problem to solve.
            x0: Initial guess. Its shape is restored on the returned x.
            callback: Called once per iteration by NuMPI.
            **kwargs: To match the `Optimizer` interface.

        Returns:
            The result of the minimisation.

        Raises:
            TypeError: If the problem carries an equality constraint.
        """
        if problem.has_equality_constraints:
            raise TypeError("BoundedLbfgs does not support an equality constraint")

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
            comm=problem.communicator,
            callback=callback,
        )
        return OptimizerResult(
            x=result["x"].reshape(init_shape), success=result["success"], message=result["message"], nit=result["nit"]
        )
