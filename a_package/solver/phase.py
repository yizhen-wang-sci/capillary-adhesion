"""
Phase field solvers.

Solves for equilibrium phase field under various constraints.
"""

import logging
import types

import numpy as np

from a_package.problem import NodalFormCapillary

from .optimizer import AugmentedLagrangian


logger = logging.getLogger(__name__)


class PhaseSolver:
    """
    Solves for the equilibrium phase field under various constraints.

    Receives a formulation (problem) and optimizer, provides methods for
    different constraint types.
    """

    def __init__(self, formulation: NodalFormCapillary, optimizer_args: dict):
        """
        Parameters
        ----------
        formulation : NodalFormCapillary
            Problem formulation providing energy/volume evaluation.
        optimizer_args : dict
            Arguments for the AugmentedLagrangian optimizer.
        """
        self.formulation = formulation
        self.optimizer = AugmentedLagrangian(**optimizer_args)

    def solve_with_constant_volume(self, gap, phase, pressure, volume):
        """
        Solve for equilibrium phase field with constant volume constraint.

        Parameters
        ----------
        gap : np.ndarray
            Gap field between surfaces.
        phase : np.ndarray
            Initial guess for phase field.
        pressure : float
            Initial guess for pressure (Lagrange multiplier).
        volume : float
            Target liquid volume.

        Returns
        -------
        tuple[np.ndarray, float]
            (phase, pressure) - optimized phase field and pressure.
        """
        # gap is simply a parameter of the optimization problem
        self.formulation.set_gap(gap)

        # save the original shape
        original_shape = phase.shape

        # wrap volume constraint into a function
        def volume_constraint():
            return self.formulation.get_volume() - volume

        # pack into an object matching the "NumOptEqB" protocol typing
        problem = types.SimpleNamespace(
            get_x=self.formulation.get_phase,
            set_x=self.formulation.set_phase,
            get_f=self.formulation.get_energy,
            get_f_Dx=self.formulation.get_energy_jacobian,
            get_g=volume_constraint,
            get_g_Dx=self.formulation.get_volume_jacobian,
            x_lb=self.formulation.phase_lb,
            x_ub=self.formulation.phase_ub,
        )

        # call the minimizer
        res = self.optimizer.solve_minimisation(problem, x0=phase, lam0=pressure)
        phase = np.reshape(res.primal, original_shape)
        pressure = res.dual
        return phase, pressure

    def check_phase_feasibility(self, phase: np.ndarray):
        """Check and warn if phase field exceeds bounds."""
        # check lower bound
        if np.any(phase < self.formulation.phase_lb):
            outlier = np.where(phase < self.formulation.phase_lb, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding lower bound, "
                f"min at {self.formulation.phase_lb:.1f}-{-extreme:.2e}."
            )
        # check upper bound
        if np.any(phase > self.formulation.phase_ub):
            outlier = np.where(phase > self.formulation.phase_ub, phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(
                f"Notice: phase field has {count} values exceeding upper bound, "
                f"max at {self.formulation.phase_ub:.1f}+{extreme - 1:.2e}."
            )
