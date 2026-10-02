"""Equilibrium formulations for capillary contact problems."""

import numpy as np

from a_package.domain import OptimizerResult, Problem

from .capillary import CapillaryBridge
from .contact import ElasticContact


def formulate_constant_volume_phase_problem(
    capillary: CapillaryBridge, volume: float, explicit_phase_bounds: bool = True
):
    """Minimise energy(phase) subject to volume(phase) == volume.

    Args:
        capillary: The physics model providing the energy and its Jacobian w.r.t. phase field.
        volume: The liquid volume to hold constant.
        explicit_phase_bounds: Whether to pass the phase bounds to the optimizer.

    Returns:
        An adapted problem the optimizer can handle, whose dual variable is the pressure.
    """
    # Exploit the linearity in the volume Jacobian
    args = {
        "get_x": capillary.get_phase,
        "set_x": capillary.set_phase,
        "get_f": capillary.get_energy,
        "get_f_Dx": capillary.get_energy_jacobian,
        "A": capillary.get_volume_jacobian().ravel(),
        "b": volume,
        "is_zeroed": capillary.gap_is_closed,
        "communicator": capillary.communicator,
    }

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update({"x_lb": capillary.phase_lb, "x_ub": capillary.phase_ub})
    return Problem(**args)


def extract_pressure_in_constant_volume_solution(result: OptimizerResult):
    """Read the pressure out of a solved constant-volume problem.

    Args:
        result: From solving a problem built by `formulate_constant_volume_phase_problem`.

    Returns:
        The pressure, divided by the surface tension.
    """
    # NOTE: in NuMPI LinearConstraint, it defines lagrangian multiplier with "-lambda ...",
    # hence lambda and pressure have the same sign. For this problem, precisely,
    # lambda = pressure / surface tension
    pressure_per_surface_tension = result["dual"]
    return pressure_per_surface_tension


def formulate_constant_pressure_phase_problem(
    capillary: CapillaryBridge, pressure: float, explicit_phase_bounds: bool = True
):
    """Minimise energy(phase) - pressure * volume(phase).

    Args:
        capillary: The physics model providing the energy and its Jacobian w.r.t. phase field.
        pressure: The pressure to hold constant, in units of the surface tension.
        explicit_phase_bounds: Whether to pass the phase bounds to the optimizer.

    Returns:
        An adapted problem the optimizer can handle.
    """

    def helmholtz_potential():
        """Free energy of the capillary minus the work done against the constant pressure."""
        return capillary.get_energy() - pressure * capillary.get_volume()

    def helmholtz_potential_jacobian():
        """Derivative of `helmholtz_potential` with respect to the phase."""
        return capillary.get_energy_jacobian() - pressure * capillary.get_volume_jacobian()

    # Exploit the linearity in the volume Jacobian
    args = {
        "get_x": capillary.get_phase,
        "set_x": capillary.set_phase,
        "get_f": helmholtz_potential,
        "get_f_Dx": helmholtz_potential_jacobian,
        "is_zeroed": capillary.gap_is_closed,
        "communicator": capillary.communicator,
    }

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update({"x_lb": capillary.phase_lb, "x_ub": capillary.phase_ub})
    return Problem(**args)


def formulate_constant_separation_gap_problem(elastic: ElasticContact):
    """Minimise energy(gap) subject to displaced_volume(gap) == 0.

    Args:
        elastic: The physics model providing the energy and its Jacobian w.r.t. gap,
            with its mean separation already set.

    Returns:
        An adapted problem the optimizer can handle, whose dual variable is the mean pressure.
    """
    # Exploit the linearity in the displaced volume Jacobian
    return Problem(
        get_x=elastic.get_gap,
        set_x=elastic.set_gap,
        get_f=elastic.get_energy,
        get_f_Dx=elastic.get_energy_jacobian,
        x_lb=elastic.gap_lb,
        x_ub=elastic.gap_ub,
        communicator=elastic.communicator,
    )


def formulate_constant_volume_gap_problem(elastic: ElasticContact):
    """Minimise energy(gap) subject to displaced_volume(gap) == 0.

    Args:
        elastic: The physics model providing the energy and its Jacobian w.r.t. gap,
            with its mean separation already set.

    Returns:
        An adapted problem the optimizer can handle, whose dual variable is the mean pressure.
    """
    # Exploit the linearity in the displaced volume Jacobian
    return Problem(
        get_x=elastic.get_gap,
        set_x=elastic.set_gap,
        get_f=elastic.get_energy,
        get_f_Dx=elastic.get_energy_jacobian,
        A=elastic.get_gap_volume_jacobian().ravel(),
        b=elastic.get_gap_origin_volume(),
        x_lb=elastic.gap_lb,
        x_ub=elastic.gap_ub,
        communicator=elastic.communicator,
    )


def formulate_constant_load_gap_coupling_constant_pressure_phase_problem(
    capillary: CapillaryBridge, elastic: ElasticContact, surface_tension: float, pressure: float, load: float
):
    """Minimise elastic energy(gap) + load * gap_volume(gap) + helmholtz potential(phase, gap).

    Args:
        capillary: The physics model providing the energy, the liquid volume and their Jacobians
            w.r.t. both phase field and gap.
        elastic: The physics model providing the energy, the gap volume and their Jacobians w.r.t.
            gap, with its mean separation already set.
        surface_tension: The surface tension, scaling the capillary energy into elastic units.
        pressure: The capillary pressure to hold constant, in units of the surface tension.
        load: The mean pressure pushing the surfaces together, compressive positive.

    Returns:
        An adapted problem the optimizer can handle, whose x is the phase followed by the gap.
    """
    nb_nodes = np.size(capillary.get_phase())

    def get_phase_and_gap():
        """Phase and gap, ravelled and concatenated."""
        return np.concatenate((np.ravel(capillary.get_phase()), np.ravel(elastic.get_gap())))

    def set_phase_and_gap(value: np.ndarray):
        """Set the gap in both models, then the phase, so the phase is masked by the new gap."""
        phase, gap = np.split(np.ravel(value), [nb_nodes])
        capillary.set_gap(gap)
        elastic.set_gap(gap)
        capillary.set_phase(phase)

    def gibbs_potential():
        """Elastic energy, plus work against the constant load and capillary helmholtz potential."""
        capillary_potential = capillary.get_energy() - pressure * capillary.get_volume()
        return elastic.get_energy() + load * elastic.get_gap_volume() + surface_tension * capillary_potential

    def gibbs_potential_jacobian():
        """Derivative of `gibbs_potential` with respect to the phase and the gap."""
        phase_jacobian = surface_tension * (
            capillary.get_energy_phase_jacobian() - pressure * capillary.get_volume_phase_jacobian()
        )
        gap_jacobian = (
            elastic.get_energy_jacobian()
            + load * elastic.get_gap_volume_jacobian()
            + surface_tension * (capillary.get_energy_gap_jacobian() - pressure * capillary.get_volume_gap_jacobian())
        )
        return np.concatenate((np.ravel(phase_jacobian), np.ravel(gap_jacobian)))

    return Problem(
        get_x=get_phase_and_gap,
        set_x=set_phase_and_gap,
        get_f=gibbs_potential,
        get_f_Dx=gibbs_potential_jacobian,
        x_lb=np.concatenate((np.full(nb_nodes, capillary.phase_lb), np.full(nb_nodes, elastic.gap_lb))),
        x_ub=np.concatenate((np.full(nb_nodes, capillary.phase_ub), np.full(nb_nodes, elastic.gap_ub))),
        communicator=capillary.communicator,
    )
