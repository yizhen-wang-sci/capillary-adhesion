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
    dual: float
    is_converged: typing.Required[bool]
    reached_iter_limit: typing.Required[bool]
    had_abnormal_stop: typing.Required[bool]
    nit: int
    final_penalty: float
    time: float
    message: str


class Optimizer:

    def __init__(self, max_outer_loop: int=20, max_inner_iter: int = 1000, tol_gradient: float = 1e-6,
                 tol_eq_constraint: float = 1e-6, tol_creeping: float = 1e-9):
        if max_outer_loop < 0:
            raise ValueError("Maximum number of loop must be non-negative.")
        self.max_outer_loop = max_outer_loop
        self.max_inner_iter = max_inner_iter
        self.tol_gradient = tol_gradient
        self.tol_eq_constraint = tol_eq_constraint
        self.tol_creeping = tol_creeping

        self.sufficient_eq_decrease = 5e-2
        self.eq_weight_maximum = 2e16
        self.eq_weight_multiplier = 5e0

        self.bound_weight_minimum = 5e-5
        self.bound_weight_multiplier = 2e-1

    def solve_minimisation(self, num_opt: typing.Union[NumOpt, NumOptEq, NumOptB, NumOptEqB], x0: typing.Sequence[float],
                           lam0: float=None, alpha0: float=None, beta0: float=None):
        """
        :param num_opt: NumOpt, NumOptEq, NumOptB, NumOptEqB
        :param x0: Initial guess.
        :param lam0: Initial dual variable.
        :param alpha0: Initial penalty weight for equality constraint.
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
        # Initialize dual variables and parameters with defaults
        if has_eq_constraint:
            if lam0 is None:
                lam0 = 0.0
            if alpha0 is None:
                alpha0 = 1e0

        # FIXME: using clipping as a work-around
        # if has_bound_constraint:
        #     if np.any(x0 > num_opt.x_ub) or np.any(x0 < num_opt.x_lb):
        #         x0 = np.clip(x0, num_opt.x_lb, num_opt.x_ub)
        #     if beta0 is None:
        #         beta0 = 1e0

        # Initial values (primal, dual, parameters)
        x_plus = x0
        lam_plus = lam0
        alpha_plus = alpha0
        beta_plus = beta0
        # FIXME: using clipping as a work-around
        # if has_bound_constraint:
        #     s0 = inverse_squashing(x0, num_opt.x_lb, num_opt.x_ub)
        #     s_plus = s0

        # Print headers
        symb_nabla = "\u2207"
        symb_alpha = "\u03B1"
        symb_lam = "\u03BB"
        tabel_headers_line1 = ["Loop", "f", f"|{symb_nabla}f+{symb_lam}{symb_nabla}g|", "|g|"]
        if has_eq_constraint:
            tabel_headers_line1 += [f"|{symb_lam}|", f"{symb_alpha}"]
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

            num_opt.set_x(x)
            f = num_opt.get_f()

            # Compute gradients
            l_Dx = num_opt.get_f_Dx()
            if has_eq_constraint:
                # Add the contribution of equality constraint gradient
                l_Dx += lam * num_opt.get_g_Dx()
            # FIXME: replace the clipping?
            if has_bound_constraint:
                # Project the gradient to the bounds
                l_Dx[(x <= num_opt.x_lb) & (l_Dx > 0)] = 0
                l_Dx[(x >= num_opt.x_ub) & (l_Dx < 0)] = 0
            l_Dx_norm = np.amax(abs(l_Dx))

            # Compute equality constraint
            g_norm = 0
            if has_eq_constraint:
                # FIXME: g is considered a scalar function as of now
                g_norm = abs(num_opt.get_g())

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
            reformed = num_opt
            if has_eq_constraint:
                reformed = approx_eq_by_augmented_lagrangian(reformed, lam, alpha)
            if has_bound_constraint:
                # FIXME: replace the clipping?
                # reformed = approx_bound_by_squashing(reformed, beta)
                reformed = approx_bound_by_clipping(reformed)
            result = solve_unconstrained(reformed, x, max_iter=self.max_inner_iter, tol_gradient=self.tol_gradient,
                                         tol_creeping=self.tol_creeping)
            x_plus = result['primal']

            # FIXME: replace the clipping?
            # if has_bound_constraint:
            #     result = solve_unconstrained(reformed, s, max_iter=self.max_iter, tol_gradient=1e-8, tol_creeping=1e-12)
            #     s_plus = result['primal']
            #     x_plus = squashing(s_plus, num_opt.x_lb, num_opt.x_ub)
            # else:
            #     result = solve_unconstrained(reformed, x, max_iter=self.max_iter, tol_gradient=1e-8, tol_creeping=1e-12)
            #     x_plus = result['primal']

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
                num_opt.set_x(x_plus)
                g_plus = num_opt.get_g()
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
        result = OptimizerResult(primal=x_plus, is_converged=is_converged, reached_iter_limit=reached_limit,
                                 had_abnormal_stop=had_abnormal_stop, nit=loop_count)
        if has_eq_constraint:
            result['dual'] = lam_plus
            result['final_penalty'] = alpha_plus
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


def approx_bound_by_clipping(num_opt: NumOptB):

    def set_x_clipped(x: np.ndarray):
        nonlocal num_opt
        num_opt.set_x(np.clip(x, num_opt.x_lb, num_opt.x_ub))

    def get_f_Dx_projected():
        f_Dx = num_opt.get_f_Dx()
        x = num_opt.get_x()
        f_Dx[(x <= num_opt.x_lb) & (f_Dx > 0)] = 0
        f_Dx[(x >= num_opt.x_ub) & (f_Dx < 0)] = 0
        return f_Dx

    return types.SimpleNamespace(get_x=num_opt.get_x, set_x=set_x_clipped, get_f=num_opt.get_f, get_f_Dx=get_f_Dx_projected)


def approx_bound_by_squashing(num_opt: NumOptB, beta: float=1e-2):
    # store the "free x" as extra states
    free_x = num_opt.get_x()

    def get_free_x():
        return free_x

    def set_x_with_squashing(x: np.ndarray):
        nonlocal free_x
        free_x = x
        num_opt.set_x(squashing(free_x, num_opt.x_lb, num_opt.x_ub))

    def get_f_with_barrier():
        f = num_opt.get_f()
        return f + beta * barrier_squashed(free_x, num_opt.x_lb, num_opt.x_ub)

    def get_f_Dx_with_squashing_and_barrier():
        return num_opt.get_f_Dx() * squashing_Dx(free_x, num_opt.x_lb, num_opt.x_ub) + beta * barrier_squashed_Dx(
            free_x, num_opt.x_lb, num_opt.x_ub)

    return types.SimpleNamespace(
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


def solve_unconstrained(numopt: NumOpt, x0: np.ndarray, max_iter: int, tol_gradient: float, tol_creeping: float):
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
        factr=tol_creeping / np.finfo(np.float64).resolution,
        # this 'pg' should be zero at exactly a local minimizer
        pgtol=tol_gradient,
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
