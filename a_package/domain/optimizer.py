"""Solving the numerical optimization problem. No physics meaning in this file."""

import logging
from typing import Callable, Protocol, TypedDict
import sys
if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

import numpy as np
import NuMPI.Optimization
import NuMPI.Tools
from NuMPI import MPI


logger = logging.getLogger(__name__)


class Problem:
    """A minimisation of f(x), optionally subject to A x - b == 0, g(x) == 0, x_lb <= x <= x_ub."""

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
                 is_zeroed: np.ndarray | None = None,
                 communicator=MPI.COMM_SELF):
        """Store the model's callables, and whichever constraints were given.

        Args:
            get_x: Read the model's current x.
            set_x: Write the model's x.
            get_f: The objective at the current x.
            get_f_Dx: The objective's gradient at the current x.
            A: Jacobian of the linear equality constraint.
            b: Right-hand side of the linear equality constraint.
            get_g: The general equality constraint at the current x.
            get_g_Dx: That constraint's gradient at the current x.
            x_lb: Lower bound, applied to every entry of x.
            x_ub: Upper bound, applied to every entry of x.
            is_zeroed: Mask of entries held at zero.
            communicator: Communicator spanning the ranks the unknowns are spread across.
        """
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
        if is_zeroed is not None:
            self._is_zero = is_zeroed
        self.communicator = communicator

    @property
    def has_linear_constraints(self):
        """Whether a linear equality constraint was given."""
        return hasattr(self, "_A") and hasattr(self, "_b")

    @property
    def has_equality_constraints(self):
        """Whether a general equality constraint was given."""
        return hasattr(self, "_get_g") and hasattr(self, "_get_g_Dx")

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

    @property
    def A(self):
        """Jacobian of the linear equality constraint.

        Raises:
            AttributeError: If no linear equality constraint was given.
        """
        return self._A

    @property
    def b(self):
        """Right-hand side of the linear equality constraint.

        Raises:
            AttributeError: If no linear equality constraint was given.
        """
        return self._b

    def get_g(self):
        """The general equality constraint at the current x, as a scalar.

        Raises:
            AttributeError: If no general equality constraint was given.
        """
        return np.asarray(self._get_g()).item()

    def get_g_Dx(self):
        """That constraint's gradient at the current x, ravelled.

        Raises:
            AttributeError: If no general equality constraint was given.
        """
        return np.asarray(self._get_g_Dx()).ravel()

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


class OptimizerResult(TypedDict, total=False):
    """Result of an optimizer."""

    x: Required[np.ndarray]
    dual: float
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

    def __init__(self, max_inner_iter: int = 1000, tol_gradient: float = 1e-6,
                 tol_step: float = 0.0):
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
        """Minimise the objective subject to a linear equality constraint and bounds.

        Args:
            problem: The problem to solve.
            x0: Initial guess. Its shape is restored on the returned x.
            callback: Called once per iteration by NuMPI.
            **kwargs: To match the `Optimizer` interface.

        Returns:
            The result of the minimisation, carrying `dual`, the multiplier of the linear
            constraint.
        """
        linear_constraint = NuMPI.Optimization.LinearConstraint(problem.A, problem.b, NuMPI.Tools.Reduction(problem.communicator))

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
        return OptimizerResult(x=result['x'].reshape(init_shape), dual=result['multiplier'], success=result['success'],
                               message=result['message'], nit=result['nit'])


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
            TypeError: If the problem carries a linear or an equality constraint.
        """
        if problem.has_linear_constraints or problem.has_equality_constraints:
            raise TypeError("BoundedLbfgs does not support linear or equality constraints")

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
        return OptimizerResult(x=result['x'].reshape(init_shape), success=result['success'],
                               message=result['message'], nit=result['nit'])
