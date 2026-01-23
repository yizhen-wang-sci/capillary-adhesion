"""
Capillary bridge physics: model and formulation.

Structure:
- CapillaryBridge: Pure physics model (numerics-free), Field → Field relations
- NodalFormCapillary: Formulation that discretizes the physics for optimization

Convention:
- CapillaryBridge is private (implementation detail)
- NodalFormCapillary is public (optimization-ready interface)
"""

import dataclasses as dc
import logging

import numpy as np

from a_package.domain import Grid, Field, adapt_shape, field_component_ax, FirstOrderElement, Quadrature, centroid_quadrature


logger = logging.getLogger(__name__)


# =============================================================================
# Pure Physics Model (numerics-free, private)
# =============================================================================

@dc.dataclass(init=True)
class CapillaryBridge:
    """Pure physics model for capillary bridge.

    Numerics-free: only Field → Field relations.
    This class is private; use NodalFormCapillary for optimization.
    """

    eta: float
    """interface thickness"""
    theta: float
    """contact angle"""

    def __post_init__(self):
        # convention
        self.phase_vapour = 0.
        self.phase_liquid = 1.
        # According to Modica-Mortola's theorem, the perimeter of liquid-vapour interface is propotional to its energy.
        # That propotion equals to the integral of the square root of the double-well penalty, on the interval connected
        # by two phases. Therefore, we have to set a prefactor equal to the inverse of that propotion, then that value
        # would exactly be the perimeter
        self.perimeter_prefactor = 3.

        # more parameters
        self.curv = 0.5 * (abs(np.sin(self.theta)) + np.asin(np.cos(self.theta)) / np.cos(self.theta))
        self.gamma = -np.cos(self.theta)

        # To save gap heights in quadrature points
        self.gap: Field = None
        """gap between two rigid bodies"""

    def compute_perimeter(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        return self.perimeter_prefactor * (
            (1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad)
        )

    def compute_energy(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        liquid_vapour = (
            self.perimeter_prefactor
            * ((1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad))
            * self.gap
            * self.curv
        )
        # upper and lower surface, so the 2. (height gradient squareis one order higher and omitted)
        liquid_solid = 2.0 * phase
        return liquid_vapour + self.gamma * liquid_solid

    @staticmethod
    def double_well_penalty(x):
        return np.sum(x**2 * (1 - x) ** 2, axis=field_component_ax, keepdims=True)

    @staticmethod
    def square_penalty(x):
        return np.sum(x**2, axis=field_component_ax, keepdims=True)

    # def compute_force(self, phase: Field, phase_grad: Field):
    #     # the common part of 3 components
    #     liquid_vapour_D_gap = (
    #         self.perimeter_prefactor
    #         * ((1 / self.eta) * self.double_well_penalty(phase) + self.eta * self.square_penalty(phase_grad))
    #         * self.curv
    #     )

    #     # the different part of 3 components
    #     f_x = (liquid_vapour_D_gap * self.dg_dx).sum()
    #     f_y = (liquid_vapour_D_gap * self.dg_dy).sum()
    #     f_z = liquid_vapour_D_gap.sum()  # dg_dz = 1

    #     return f_x, f_y, f_z

    def compute_energy_jacobian(self, phase: Field, phase_grad: Field):
        """
        - phase: phase-field
        - phase_grad: gradient of phase-field
        """
        liquid_vapour_D_phase = (
            self.perimeter_prefactor
            * ((1 / self.eta) * self.double_well_penalty_derivative(phase))
            * self.gap
            * self.curv
        )
        liquid_vapour_D_phase_grad = (
            self.perimeter_prefactor * (self.eta * self.square_penalty_derivatie(phase_grad)) * self.gap * self.curv
        )

        liquid_solid_D_phase = 2.0
        # liquid_vapour_D_phase_grad = 0.

        return liquid_vapour_D_phase + self.gamma * liquid_solid_D_phase, liquid_vapour_D_phase_grad

    @staticmethod
    def double_well_penalty_derivative(x):
        return 2 * x * (1 - x) * (1 - 2 * x)

    @staticmethod
    def square_penalty_derivatie(x):
        return 2 * x

    def compute_volume(self, phase: Field):
        return phase * self.gap

    def compute_volume_jacobian(self, phase: Field):
        return (self.gap,)


# =============================================================================
# Formulation (uses numerics toolbox, public)
# =============================================================================

class NodalFormCapillary:
    """Formulates capillary bridge physics into an optimisation problem.

    Combines:
    - Physics: CapillaryBridge (energy, volume computations)
    - Numerics: FEM (interpolation), Quadrature (integration)

    Provides evaluation interface for optimizers:
    - get_energy(), get_energy_jacobian()
    - get_volume(), get_volume_jacobian()
    """

    nodal_shape: list[int]
    nodal_gap: Field
    nodal_phase: Field
    quadr_phase: Field
    quadr_phase_gradient: Field
    bridge: CapillaryBridge
    fem: FirstOrderElement
    quadrature: Quadrature
    phase_lb: float
    phase_ub: float

    def __init__(self, grid: Grid, capillary_args: dict, quadrature: Quadrature = centroid_quadrature):
        self.grid = grid
        self.nodal_shape = [1, 1, *grid.nb_elements]

        self.nodal_gap = np.zeros(self.nodal_shape)
        self.nodal_phase = np.zeros(self.nodal_shape)
        self.quadr_phase = np.zeros([])
        self.quadr_phase_gradient = np.zeros([])

        self.bridge = CapillaryBridge(**capillary_args)
        self.fem = FirstOrderElement(grid, quadrature.quad_pt_coords)
        self.quadrature = quadrature

    def get_gap(self):
        return self.nodal_gap

    def set_gap(self, value):
        self.nodal_gap = adapt_shape(value)
        # map to quadrature points
        self.bridge.gap = self.fem.interpolate_value(self.nodal_gap)

    @property
    def quadr_gap(self):
        return self.bridge.gap

    @property
    def gap_is_closed(self):
        return self.nodal_gap == 0

    def get_phase(self):
        return self.nodal_phase

    @property
    def phase_lb(self):
        return self.bridge.phase_vapour

    @property
    def phase_ub(self):
        return self.bridge.phase_liquid

    def set_phase(self, value):
        # Clean the phase-field where the solid bodies contact
        value[self.gap_is_closed] = 0.
        self.nodal_phase = value
        # map to quadrature points
        self.quadr_phase = self.fem.interpolate_value(self.nodal_phase)
        # self._quad_phase_grad = np.stack(
        #     [self.fem.interpolate_gradient_x(nodal_phase), self.fem.interpolate_gradient_y(nodal_phase)],
        #     axis=0
        # )
        self.quadr_phase_gradient = self.fem.interpolate_gradient(self.nodal_phase)

    def validate_phase_field(self, nodal_phase: Field):
        # check phase field < 0
        if np.any(nodal_phase < 0):
            outlier = np.where(nodal_phase < 0, nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")
        # check phase field > 1
        if np.any(nodal_phase > 1):
            outlier = np.where(nodal_phase > 1, nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")

    def get_energy(self):
        integrand = self.bridge.compute_energy(self.quadr_phase, self.quadr_phase_gradient)
        return self.quadrature.integrate(self.grid, integrand).item()

    def get_energy_jacobian(self):
        [energy_D_phase, energy_D_phase_grad] = self.bridge.compute_energy_jacobian(
            self.quadr_phase, self.quadr_phase_gradient)
        jacobian = self.fem.propag_sens_value(self.quadrature.propag_integral_weight(
            self.grid, energy_D_phase)) + self.fem.propag_sens_gradient(self.quadrature.propag_integral_weight(
                self.grid, energy_D_phase_grad))
        # clean the contact part because there won't be either liquid or vapour
        jacobian[self.gap_is_closed] = 0
        return jacobian

    def get_volume(self):
        integrand = self.bridge.compute_volume(self.quadr_phase)
        return self.quadrature.integrate(self.grid, integrand).item()

    def get_volume_jacobian(self):
        [volume_D_phase] = self.bridge.compute_volume_jacobian(self.quadr_phase)
        jacobian = self.fem.propag_sens_value(self.quadrature.propag_integral_weight(self.grid, volume_D_phase))
        # clean the contact part because there won't be either liquid or vapour
        jacobian[self.gap_is_closed] = 0
        return jacobian

    def get_perimeter(self):
        integrand = self.bridge.compute_perimeter(self.quadr_phase, self.quadr_phase_gradient)
        return self.quadrature.integrate(self.grid, integrand).item()
