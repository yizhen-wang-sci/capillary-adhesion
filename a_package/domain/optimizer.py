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

    def __init__(self, max_loop: int=20, max_iter: int = 1000, tol_gradient: float = 1e-6,
                 tol_eq_constraint: float = 1e-9, tol_bound_constraint=1e-9, tol_creeping: float = 1e-12):
        if max_loop < 0:
            raise ValueError("Maximum number of loop must be non-negative.")
        self.max_loop = max_loop
        self.max_iter = max_iter
        self.tol_gradient = tol_gradient
        self.tol_creeping = tol_creeping
        self.tol_eq_constraint = tol_eq_constraint
        self.sufficient_eq_decrease = 1e-2
        self.eq_weight_multiplier = 3e0
        self.tol_bound_constraint = tol_bound_constraint
        self.active_bound_threshold = 1e-3
        self.squashing_parameter_multiplier = 2e0

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
        s0 = None  # Initialize s0
        
        # Initialize dual variables and parameters with defaults
        if has_eq_constraint:
            if lam0 is None:
                lam0 = 0.0
            if alpha0 is None:
                alpha0 = 10.0  # Default penalty weight
        
        if has_bound_constraint:
            # Compute default beta if not provided
            if beta0 is None:
                beta0 = default_beta(num_opt.x_lb, num_opt.x_ub)
                x0 = np.clip(x0, num_opt.x_lb, num_opt.x_ub)
                # Compute s0 from clipped x0 using inverse squashing
                s0 = inverse_squashing(x0, num_opt.x_lb, num_opt.x_ub, beta0)
            elif s0 is None:
                # User provided x0 (desired bounded position) and beta0
                # Clip x0 to be strictly interior to avoid gradient vanishing at bounds
                x0 = _clip_x0_interior(x0, num_opt.x_lb, num_opt.x_ub)
                # Compute corresponding free variable s0 using inverse squashing
                s0 = inverse_squashing(x0, num_opt.x_lb, num_opt.x_ub, beta0)
            # else: user provided both s0 and beta0, use them directly
            
            # Compute initial bounded x from s0
            x0 = squashing(s0, num_opt.x_lb, num_opt.x_ub, beta0)

        # Initial values (primal, dual, parameters)
        x_plus = x0
        s_plus = s0 if has_bound_constraint else None
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
                # NOTE: here the plus is aligned with augmented Lagrangian
                l_Dx += lam * num_opt.get_g_Dx()
            criteria_l_Dx = np.abs(l_Dx) < self.tol_gradient
            
            # Compute bound constraint metrics
            max_bound_violation = None
            n_active_lb = None
            n_active_ub = None
            if has_bound_constraint:
                # lower bound
                x_diff_lb = x - num_opt.x_lb
                i_active_lb = np.argwhere((x_diff_lb < self.active_bound_threshold) & (l_Dx > 0))
                n_active_lb = len(i_active_lb)
                criteria_x_lb = x_diff_lb[i_active_lb] < self.tol_bound_constraint
                criteria_l_Dx[i_active_lb] = criteria_x_lb
                # upper bound
                x_diff_ub = num_opt.x_ub - x
                i_active_ub = np.argwhere((x_diff_ub < self.active_bound_threshold) & (l_Dx < 0))
                n_active_ub = len(i_active_ub)
                criteria_x_ub = x_diff_ub[i_active_ub] < self.tol_bound_constraint
                criteria_l_Dx[i_active_ub] = criteria_x_ub
                
                # Compute max bound violation for active bounds
                violations = []
                if len(i_active_lb) > 0:
                    violations.extend(x_diff_lb[i_active_lb])
                if len(i_active_ub) > 0:
                    violations.extend(x_diff_ub[i_active_ub])
                if len(violations) > 0:
                    max_bound_violation = max(violations)
                else:
                    max_bound_violation = 0.0

            # Check equality constraint
            g = None
            criteria_g = True
            if has_eq_constraint:
                g = num_opt.get_g()
                criteria_g = np.abs(g) < self.tol_eq_constraint

            # Log at beginning of loop (after computing metrics, before convergence check)
            lagrangian_grad_norm = float(np.linalg.norm(l_Dx))
            g_violation_str = f"{float(np.abs(g)):.6e}" if g is not None else "None"
            if max_bound_violation is None:
                bound_viol_str = "None"
            elif np.ndim(max_bound_violation) > 0:
                bound_viol_str = f"[{', '.join(f'{v:.6e}' for v in max_bound_violation)}]"
            else:
                bound_viol_str = f"{float(max_bound_violation):.6e}"
            logger.info(
                f"Loop #{loop_count}: |grad L|={lagrangian_grad_norm:.6e}, "
                f"g_viol={g_violation_str}, "
                f"bound_viol={bound_viol_str}, "
                f"active_lb={n_active_lb}, active_ub={n_active_ub}"
            )

            # When both are satisfied, the solver is converged
            # For bound-constrained problems, also verify we're not prematurely converged
            # due to gradient vanishing from small beta
            if np.all(criteria_l_Dx) and criteria_g:
                # Additional check for bound constraints: ensure we're not stuck at wrong bound
                if has_bound_constraint:
                    # Check if any bound is active but not satisfied to tolerance
                    x_diff_lb = x - num_opt.x_lb
                    x_diff_ub = num_opt.x_ub - x
                    i_near_lb = np.argwhere(x_diff_lb < self.active_bound_threshold)
                    i_near_ub = np.argwhere(x_diff_ub < self.active_bound_threshold)

                    # If near a bound but gradient doesn't point outward, beta might be too small
                    # or we might be at the wrong location
                    if len(i_near_lb) > 0 or len(i_near_ub) > 0:
                        # Check if bound constraint is actually satisfied
                        bound_satisfied = True
                        if len(i_near_lb) > 0:
                            bound_satisfied = bound_satisfied and np.all(x_diff_lb[i_near_lb] < self.tol_bound_constraint)
                        if len(i_near_ub) > 0:
                            bound_satisfied = bound_satisfied and np.all(x_diff_ub[i_near_ub] < self.tol_bound_constraint)

                        if not bound_satisfied:
                            # Not truly converged, need to continue
                            is_converged = False
                        else:
                            is_converged = True
                    else:
                        is_converged = True
                else:
                    is_converged = True

                if is_converged:
                    break

            # For last iter, no more trial
            if loop_count == self.max_loop:
                reached_limit = True
                break

            # Store old values for logging changes
            lam_old = lam_plus
            alpha_old = alpha_plus
            beta_old = beta_plus

            # reform and solve
            reformed = num_opt
            if has_eq_constraint:
                reformed = approx_eq_by_augmented_lagrangian(reformed, lam, alpha)
            if has_bound_constraint:
                reformed = approx_bound_by_squashing(reformed, beta)

            # Use s_plus as initial guess for free variable optimization
            initial_guess = s_plus if has_bound_constraint else x0
            result = solve_unconstrained(reformed, initial_guess)

            # Prepare for the next loop
            x_plus = result['primal']

            if has_bound_constraint:
                s_plus = result['primal']
                x_plus = squashing(s_plus, num_opt.x_lb, num_opt.x_ub, beta)

            num_opt.set_x(x_plus)
            l_Dx_plus = num_opt.get_f_Dx()
            if has_eq_constraint:
                l_Dx_plus += lam * num_opt.get_g_Dx()

            if has_bound_constraint:
                # lower bound
                x_diff_lb = x_plus - num_opt.x_lb
                i_active_lb = np.argwhere((x_diff_lb < self.active_bound_threshold) & (l_Dx_plus > 0))
                criteria_x_lb = x_diff_lb[i_active_lb] < self.tol_bound_constraint

                # upper bound
                x_diff_ub = num_opt.x_ub - x_plus
                i_active_ub = np.argwhere((x_diff_ub < self.active_bound_threshold) & (l_Dx_plus < 0))
                criteria_x_ub = x_diff_ub[i_active_ub] < self.tol_bound_constraint

                # Tighten the squashing if any active bound does not satisfy the criteria
                if not np.all(criteria_x_lb) or not np.all(criteria_x_ub):
                    # Check if gradient points INTO the feasible region (beta too large)
                    # vs OUT of the feasible region (beta needs to increase)
                    gradient_points_inward = False

                    # At lower bound: if gradient < 0, it points inward (toward larger x)
                    if len(i_active_lb) > 0:
                        if np.any(l_Dx_plus[i_active_lb] < 0):
                            gradient_points_inward = True

                    # At upper bound: if gradient > 0, it points inward (toward smaller x)
                    if len(i_active_ub) > 0:
                        if np.any(l_Dx_plus[i_active_ub] > 0):
                            gradient_points_inward = True

                    if gradient_points_inward:
                        # Beta is too large, causing gradient vanishing at bounds
                        # Decrease beta to allow movement away from bound
                        beta_plus = beta / self.squashing_parameter_multiplier
                        logger.debug(f"Decreasing beta: {beta} -> {beta_plus} (gradient points inward)")
                    else:
                        # Gradient points outward, bound is truly active
                        # Increase beta to enforce bound more strictly
                        beta_plus = beta * self.squashing_parameter_multiplier
                        logger.debug(f"Increasing beta: {beta} -> {beta_plus} (bound active)")

            if has_eq_constraint and not criteria_g:
                num_opt.set_x(x_plus)
                g_plus = num_opt.get_g()
                lam_plus = lam + alpha * g_plus
                # Increase penalty weight of equality constraint if not much improved
                if not np.abs(g_plus) < self.sufficient_eq_decrease * np.abs(g):
                    alpha_plus = alpha * self.eq_weight_multiplier

            # Log changes after reform and solve
            # Format values, handling arrays and scalars
            def format_value(val):
                if val is None:
                    return "None"
                elif np.ndim(val) > 0:
                    return f"[{', '.join(f'{v:.6e}' for v in val)}]"
                else:
                    return f"{float(val):.6e}"
            
            lam_str = f"{format_value(lam_old)} -> {format_value(lam_plus)}" if has_eq_constraint else "None"
            alpha_str = f"{format_value(alpha_old)} -> {format_value(alpha_plus)}" if has_eq_constraint else "None"
            beta_str = f"{format_value(beta_old)} -> {format_value(beta_plus)}" if has_bound_constraint else "None"
            
            logger.info(
                f"  After solve: lam={lam_str}, alpha={alpha_str}, beta={beta_str}"
            )

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


def inverse_squashing(x: np.ndarray, x_lb: float, x_ub: float, beta: float):
    """
    Inverse of the squashing function.

    Given a desired position x in the bounded space, compute the corresponding
    free variable s such that squashing(s, beta) = x.

    This is essential for proper initialization: when user provides x0 as a
    desired starting point in bounded space, we must compute the corresponding
    s0 for the unconstrained optimizer.

    Parameters
    ----------
    x : np.ndarray
        Desired position in bounded space (x_lb <= x <= x_ub).
    x_lb : float
        Lower bound.
    x_ub : float
        Upper bound.
    beta : float
        Squashing parameter.

    Returns
    -------
    s : np.ndarray
        Free variable such that squashing(s, beta) ≈ x.
    """
    x_c = (x_ub + x_lb) / 2
    half_range = (x_ub - x_lb) / 2
    normalized = (x - x_c) / half_range
    # Clip to avoid numerical issues at bounds (arctanh diverges at ±1)
    normalized = np.clip(normalized, -0.9999, 0.9999)
    return x_c + np.arctanh(normalized) / beta


def default_beta(x_lb: np.ndarray, x_ub: np.ndarray, safety_factor: float = 0.3) -> np.ndarray:
    """
    Compute a default beta that gives O(1) gradient scaling at the center.

    Analysis:
    - squashing_Dx at center = (x_ub - x_lb) / 2 * beta
    - For squashing_Dx ≈ 1: beta = 2 / (x_ub - x_lb) = 1 / half_range

    Using a safety factor < 1 ensures we start with beta on the smaller side,
    which provides more uniform gradient scaling. Beta will be increased
    adaptively if bound constraints are not satisfied.

    Parameters
    ----------
    x_lb : np.ndarray
        Lower bounds.
    x_ub : np.ndarray
        Upper bounds.
    safety_factor : float
        Multiplier to reduce beta for safety (default 0.3).

    Returns
    -------
    beta : np.ndarray
        Recommended beta value per dimension.
    """
    half_range = (x_ub - x_lb) / 2
    # Avoid division by zero for very small ranges
    half_range = np.maximum(half_range, 1e-10)
    return safety_factor / half_range


def _clip_x0_interior(x0: np.ndarray, x_lb: np.ndarray, x_ub: np.ndarray, 
                       interior_factor: float = 0.99) -> np.ndarray:
    """
    Clip x0 to be strictly inside the bounds to avoid gradient vanishing.

    When x0 is exactly at a bound, squashing_Dx ≈ 0, causing the optimizer
    to see zero gradient and prematurely converge. This function ensures x0
    is slightly interior to the feasible region.

    Parameters
    ----------
    x0 : np.ndarray
        Original starting point.
    x_lb : np.ndarray
        Lower bounds.
    x_ub : np.ndarray
        Upper bounds.
    interior_factor : float
        Factor to keep x0 away from bounds (default 0.99 means 99% of the way).

    Returns
    -------
    x0_clipped : np.ndarray
        Starting point clipped to be strictly interior.
    """
    x_c = (x_lb + x_ub) / 2
    half_range = (x_ub - x_lb) / 2
    # Clip to [x_lb + ε, x_ub - ε] where ε is 1% of the range
    x0_clipped = np.clip(x0, 
                          x_lb + (1 - interior_factor) * half_range,
                          x_ub - (1 - interior_factor) * half_range)
    return x0_clipped


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
