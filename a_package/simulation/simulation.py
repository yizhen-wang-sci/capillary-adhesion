"""
Simulation orchestration.

Provides the main Simulation class that coordinates contact solving,
phase field solving, and result storage.
"""

import logging
import pathlib

import numpy as np

from a_package.domain import Grid
from a_package.problem import NodalFormCapillary
from a_package.solver import RigidContactSolver, PhaseSolver

from .io import SimulationIO, Term


logger = logging.getLogger(__name__)


class Simulation:
    """
    Main simulation orchestrator.

    Coordinates contact solving, phase solving, and IO for different
    simulation types.
    """

    def __init__(
        self,
        grid: Grid,
        capillary_args: dict,
        optimizer_args: dict,
    ):
        self.grid = grid
        self.capillary_args = capillary_args
        self.optimizer_args = optimizer_args

    def run_with_constant_volume(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        trajectory: np.ndarray,
        volume: float,
        store_dir: pathlib.Path | str,
        phase_init: np.ndarray | None = None,
        pressure_init: float = 0,
    ) -> SimulationIO:
        """
        Run simulation with constant liquid volume constraint.

        Parameters
        ----------
        upper : np.ndarray
            Upper surface height field.
        lower : np.ndarray
            Lower surface height field.
        trajectory : np.ndarray
            Array of separation values to simulate.
        volume : float
            Target liquid volume to maintain.
        store_dir : Path | str
            Directory to store simulation results.
        phase_init : np.ndarray, optional
            Initial guess for phase field. Zeros if not provided.
        pressure_init : float, optional
            Initial guess for pressure (Lagrange multiplier).

        Returns
        -------
        SimulationIO
            IO object for accessing saved results.
        """
        assert upper.shape[-2:] == tuple(self.grid.nb_elements)
        assert lower.shape[-2:] == tuple(self.grid.nb_elements)
        assert volume > 0, "Liquid volume must be positive."

        # Truncate to remove trailing digits due to rounding errors
        trajectory = np.round(trajectory, 6)

        # Default initial phase field
        if phase_init is None:
            phase_init = np.zeros(self.grid.nb_elements)

        # Create solvers
        contact_solver = RigidContactSolver(upper, lower)
        formulation = NodalFormCapillary(self.grid, self.capillary_args)
        phase_solver = PhaseSolver(formulation, self.optimizer_args)

        # IO
        io = SimulationIO(self.grid, store_dir)
        io.save_constant(
            fields={Term.phase_init: phase_init},
            single_values={Term.pressure_init: pressure_init},
        )

        # Log info
        logger.info(
            f"Problem size: {'x'.join(str(dim) for dim in phase_init.shape)}. "
            f"Simulating for {len(trajectory)} separation values..."
        )

        # Initial guess
        phase = np.asarray(phase_init)
        pressure = pressure_init

        # Simulation loop
        for index, separation in enumerate(trajectory):
            logger.info(f"Step {index}: separation={separation}")
            gap = contact_solver.solve_gap(separation)
            phase, pressure = phase_solver.solve_with_constant_volume(
                gap, phase, pressure, volume
            )

            # Check feasibility
            phase_solver.check_phase_feasibility(phase)

            # Save step
            io.save_step(
                index,
                fields={
                    Term.upper_solid: contact_solver.upper,
                    Term.lower_solid: contact_solver.lower,
                    Term.gap: gap,
                    Term.phase: phase,
                },
                single_values={
                    Term.separation: separation,
                    Term.pressure: pressure,
                    Term.energy: phase_solver.formulation.get_energy(),
                },
            )

        return io
