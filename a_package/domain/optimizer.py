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


class OptimizerResult(typing.NamedTuple):
    primal: np.ndarray
    dual: float
    time: float
    nit: int
    is_converged: bool
    reached_iter_limit: bool
    had_abnormal_stop: bool
    message: str


def solve_unconstrained(numopt: NumOpt, x0: np.ndarray, max_iter: int = 10000):
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
        # relative decreasing of 'f', in units of 'eps'
        factr=1e2,
        # this 'pg' should be zero at exactly a local minimizer
        pgtol=1e-6,
    )

    t_exec += timeit.default_timer()
    numopt.set_x(np.reshape(x_plus, x_shape))
    is_converged = info["warnflag"] == 0
    return OptimizerResult(
        primal=numopt.get_x(),
        dual=0.0,
        time=t_exec,
        nit=info["nit"],
        is_converged=is_converged,
        reached_iter_limit=info["nit"] >= max_iter,
        had_abnormal_stop=not is_converged and info["nit"] < max_iter,
        message=info["task"].decode() if isinstance(info["task"], bytes) else info["task"],
    )


class BoundConstrainedSolver:
    """Solver for bound-constrained optimization (no equality constraints)."""

    max_iter: int
    tol_convergence: float
    tol_creeping: float

    def __init__(self, max_iter=1000, tol_convergence=1e-6, tol_creeping=1e2):
        self.max_iter = max_iter
        self.tol_convergence = tol_convergence
        self.tol_creeping = tol_creeping

    def solve_minimisation(self, numopt: NumOptB, x0: np.ndarray):
        """
        Solve bound-constrained minimization.

        Parameters
        ----------
        numopt : NumOptB
            Problem with get_x, set_x, get_f, get_f_Dx, x_lb, x_ub.
        x0 : np.ndarray
            Initial guess.

        Returns
        -------
        SolverResult
        """
        logger.info("BoundConstrainedSolver: starting")

        # reform bounds with clipping
        reformed = self._reform_simple_bounds_with_clipping(numopt)

        # solve using unconstrained solver
        result = solve_unconstrained(reformed, x0, max_iter=self.max_iter)

        logger.info(f"BoundConstrainedSolver: nit={result.nit}, {result.message}")
        logger.info(f"BoundConstrainedSolver: time={result.time:.1e}s")

        return result

    @staticmethod
    def _reform_simple_bounds_with_clipping(num_opt: NumOptB):
        def set_x_clipped(x: np.ndarray):
            num_opt.set_x(np.clip(x, num_opt.x_lb, num_opt.x_ub))

        def get_f_Dx_masked():
            f_Dx = num_opt.get_f_Dx().ravel()
            x = num_opt.get_x().ravel()
            f_Dx[(x <= num_opt.x_lb) & (f_Dx > 0)] = 0
            f_Dx[(x >= num_opt.x_ub) & (f_Dx < 0)] = 0
            return f_Dx

        return types.SimpleNamespace(
            get_x=num_opt.get_x,
            set_x=set_x_clipped,
            get_f=num_opt.get_f,
            get_f_Dx=get_f_Dx_masked,
        )


class AugmentedLagrangian:

    max_inner_iter: int
    max_outer_loop: int
    init_penalty_weight: float
    sufficient_constr_dec: float
    penalty_weight_growth: float
    tol_convergence: float
    tol_creeping: float
    tol_constraint: float

    def __init__(self, max_inner_iter=1000, max_outer_loop=50, init_penalty_weight=1e0, sufficient_constr_dec=1e-2,
                 penalty_weight_growth=3e0, tol_convergence=1e-6, tol_creeping=1e2, tol_constraint=1e-8):
        self.max_inner_iter = max_inner_iter
        self.max_outer_loop = max_outer_loop
        self.init_penalty_weight = init_penalty_weight
        self.sufficient_constr_dec = sufficient_constr_dec
        self.penalty_weight_growth = penalty_weight_growth
        self.tol_convergence = tol_convergence
        self.tol_creeping = tol_creeping
        self.tol_constraint = tol_constraint

    def solve_minimisation(self, numopt: typing.Union[NumOptEq, NumOptEqB], x0: np.ndarray, lam0: float):
        # print headers
        nabla = "\u2207"
        delta = "\u0394"
        tabel_headers_line1 = ["Loop", "f", f"|Pr({nabla}L)|", "|g|", f"|{delta} lam|", "c"]
        tabel_headers_line2 = ["Iter", f"|res {nabla}|", "Message"]
        separator = "  "
        line_width = 80
        logger.info(separator.join("{:<4}".format(col_name) if col_name in [
                    "Loop"] else "{:<8}".format(col_name)for col_name in tabel_headers_line1))
        logger.info(separator.join("{:<4}".format(col_name) if col_name in [
                    "Iter"] else "{:<8}".format(col_name)for col_name in tabel_headers_line2))
        logger.info("=" * line_width)

        # select inner solver based on whether problem has bounds
        if hasattr(numopt, "x_lb") and hasattr(numopt, "x_ub"):
            inner_solver = BoundConstrainedSolver(
                max_iter=self.max_inner_iter,
                tol_convergence=self.tol_convergence,
                tol_creeping=self.tol_creeping,
            )
            solve_inner = inner_solver.solve_minimisation
        else:
            def solve_inner(numopt, x0):
                return solve_unconstrained(numopt, x0, max_iter=self.max_inner_iter)

        # initial values
        total_nit = 0
        t_exec = 0
        x_plus = x0
        lam_plus = lam0
        c_plus = self.init_penalty_weight
        is_converged = False
        reached_last_loop = False
        had_abnormal_stop = False
        lam = lam0  # only for the purpose of printing delta-lam at count=0

        for count in range(self.max_outer_loop + 1):
            # compute values that must be evaluated before update
            norm_delta_lam = abs(lam_plus - lam)

            # update primal, dual and penalty parameter
            x = x_plus
            lam = lam_plus
            c = c_plus

            # get the reformulated problem (equality constraint -> augmented lagrangian)
            reformed = self.approximate_equality_constraint_with_augmented_lagrangian(numopt, lam, c)

            # print status before calling inner solver
            numopt.set_x(x)
            obj_value = numopt.get_f()
            constr_violation = abs(numopt.get_g())
            reformed.set_x(x)
            norm_lagr_gradient = np.max(abs(reformed.get_f_Dx()))
            padded_literals = [f"{count:>4d}", f"{obj_value:>8.1e}", f"{norm_lagr_gradient:>8.1e}",
                               f"{constr_violation:>8.1e}", f"{norm_delta_lam:>8.1e}", f"{c:>8.1e}"]
            logger.info(separator.join(padded_literals))

            # convergence criteria
            if norm_lagr_gradient < self.tol_convergence and constr_violation < self.tol_constraint:
                is_converged = True
                logger.info(f"Notice: achieving required tolerance at iter #{count}")
                break

            # quit due to reaching maximal loop count
            if count == self.max_outer_loop:
                reached_last_loop = True
                break

            # solve minimisation of augmented lagrangian
            result = solve_inner(reformed, x_plus)
            total_nit += result.nit
            t_exec += result.time

            # print inner solver progress
            reformed.set_x(result.primal)
            res_norm_grad = np.max(abs(reformed.get_f_Dx()))
            padded_literals = [f"{result.nit:>4d}", f"{res_norm_grad:>8.1e}", result.message]
            logger.info(separator.join(padded_literals))
            logger.info("-" * line_width)

            # value for new iteration
            x_plus = result.primal

            # measures when it fails to achieve convergence
            if not result.is_converged:
                if result.reached_iter_limit:
                    # if reaches maximal inner iterations, as 'x' will be updated, simply do another loop to continue
                    continue
                else:
                    # otherwise, it is most likely an error elsewhere, more loops are not helpful
                    had_abnormal_stop = True
                    break
            # else, it achieves convergence
            # update Lagrangian multiplier following the formula
            numopt.set_x(x_plus)
            constr_violation_plus = numopt.get_g()
            lam_plus = lam + c * constr_violation_plus
            # increase penalty weight if the constraint didn't improve enough
            if not (
                    abs(constr_violation_plus) < self.tol_constraint) and not (
                    abs(constr_violation_plus) < self.sufficient_constr_dec * abs(constr_violation)):
                c_plus = c * self.penalty_weight_growth

        # show a warning under necessary conditions
        if is_converged:
            message = "converged"
        elif reached_last_loop:
            message = "reached outer loop limit"
            logger.warning(f"WARNING: {message}.")
        elif had_abnormal_stop:
            message = "had abnormal stop"
            logger.warning(f"WARNING: {message}.")
        else:
            message = "not converged"
            logger.warning(f"WARNING: {message}.")

        # more prints
        logger.info(f"Total time for inner solver: {t_exec:.1e} seconds.")
        logger.info(f"Ends with dual variable lambda={lam:.6f}")

        return OptimizerResult(reformed.get_x(), lam, t_exec, total_nit, is_converged, reached_last_loop, had_abnormal_stop, message)

    @staticmethod
    def approximate_equality_constraint_with_augmented_lagrangian(num_opt: NumOptEq, lam: float, c: float):

        def get_augmented_lagrangian():
            g = num_opt.get_g()
            return num_opt.get_f() + lam * g + (0.5 * c) * g**2

        def get_augmented_lagrangian_Dx():
            return num_opt.get_f_Dx() + (lam + c * num_opt.get_g()) * num_opt.get_g_Dx()

        reformed = {"get_x": num_opt.get_x, "set_x": num_opt.set_x,
                    "get_f": get_augmented_lagrangian, "get_f_Dx": get_augmented_lagrangian_Dx}
        try:
            reformed.update({"x_lb": num_opt.x_lb, "x_ub": num_opt.x_ub})
        except AttributeError:
            pass

        return types.SimpleNamespace(**reformed)
