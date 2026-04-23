"""
Capillary bridge physics: model and formulation.

Structure:
- CapillaryBridge: Pure physics model (numerics-free), Field → Field relations
- NodalFormCapillary: Formulation that discretizes the physics for optimization

Convention:
- CapillaryBridge is private (implementation detail)
- NodalFormCapillary is public (optimization-ready interface)
"""

import logging

import numpy as np

from a_package.domain import Grid, Field, adapt_shape, field_component_ax, FirstOrderElement, Quadrature, centroid_quadrature


logger = logging.getLogger(__name__)


# =============================================================================
# Pure Physics Model (numerics-free, private)
# =============================================================================

class CapillaryBridge:
    """Pure physics model for capillary bridge.

    Numerics-free: only Field → Field relations.
    This class is private; use NodalFormCapillary for optimization.
    """

    def __init__(self, eta: float, theta: float, epsilon: float):
        """
        Parameters
        ----------
        eta : float
            Interface thickness of the diffuse interface model.
        theta : float
            Contact angle at the liquid-solid interface, in radians.
        epsilon: float
            An extra prefactor on the perimeter term.
        """
        self._eta = eta
        self._theta = theta
        self._curv = 0.5 * (abs(np.sin(theta)) + np.asin(np.cos(theta)) / np.cos(theta))
        self._gamma = -np.cos(theta)
        self._epsilon = epsilon

    @property
    def phase_vapour(self):
        """Phase-field value representing vapour phase."""
        return 0.

    @property
    def phase_liquid(self):
        """Phase-field value representing liquid phase."""
        return 1.

    @property
    def perimeter_prefactor(self):
        """Prefactor for perimeter computation.

        According to Modica-Mortola's theorem, the perimeter of liquid-vapour
        interface is proportional to its energy. That proportion equals to the
        integral of the square root of the double-well penalty, on the interval
        connected by two phases. Therefore, we set a prefactor equal to the
        inverse of that proportion, then that value would exactly be the perimeter.
        """
        return 3

    def compute_local_perimeter(self, phase: Field, phase_grad: Field):
        """Compute local perimeter density of liquid-vapour interface."""
        return self.perimeter_prefactor * (
            (1 / self._eta) * self.double_well_penalty(phase) + self._eta * self.square_penalty(phase_grad))

    def compute_local_energy(self, gap: Field, phase: Field, phase_grad: Field):
        """Compute local energy density (liquid-vapour + liquid-solid contributions)."""
        liquid_vapour = self.compute_local_perimeter(phase, phase_grad) * gap * self._curv * self._epsilon
        # FIXME: switch on whether to use small-slope-approx.?
        # upper and lower surface, hence the 2. (height gradient square is one order higher and omitted)
        liquid_solid = 2.0 * phase
        return liquid_vapour + self._gamma * liquid_solid

    @staticmethod
    def double_well_penalty(x):
        """Double-well potential W(x) = x^2(1-x)^2, summed over components."""
        return np.sum(x**2 * (1 - x) ** 2, axis=field_component_ax, keepdims=True)

    @staticmethod
    def square_penalty(x):
        """Square norm |x|^2, summed over components."""
        return np.sum(x**2, axis=field_component_ax, keepdims=True)

    def compute_local_energy_jacobian(self, gap: Field, phase: Field, phase_grad: Field):
        """Compute derivatives of local energy w.r.t. phase and phase gradient."""
        liquid_vapour_D_phase = (self.perimeter_prefactor * (1 / self._eta)
                                 * self.double_well_penalty_derivative(phase) * gap * self._curv * self._epsilon)
        liquid_vapour_D_phase_grad = (self.perimeter_prefactor * self._eta
                                      * self.square_penalty_derivative(phase_grad) * gap * self._curv * self._epsilon)

        liquid_solid_D_phase = 2.0

        return liquid_vapour_D_phase + self._gamma * liquid_solid_D_phase, liquid_vapour_D_phase_grad

    @staticmethod
    def double_well_penalty_derivative(x):
        """Derivative of double-well potential: dW/dx = 2x(1-x)(1-2x)."""
        return 2 * x * (1 - x) * (1 - 2 * x)

    @staticmethod
    def square_penalty_derivative(x):
        """Derivative of square norm: d|x|^2/dx = 2x."""
        return 2 * x

    def compute_local_volume(self, gap: Field, phase: Field):
        """Compute local liquid volume density."""
        return phase * gap

    def compute_local_volume_jacobian(self, gap: Field, phase: Field):
        """Compute derivative of local volume w.r.t. phase."""
        return (gap,)


# =============================================================================
# Formulation (uses numerics toolbox, public)
# =============================================================================

class NodalFormCapillary:
    """Nodal-value interface for capillary bridge evaluation.

    Combines CapillaryBridge physics with FEM interpolation and quadrature
    integration. Accepts nodal values via set_gap/set_phase and provides
    integrated quantities via get_energy/get_volume and their jacobians.
    """

    def __init__(self, grid: Grid, capillary_args: dict, quadrature: Quadrature = centroid_quadrature):
        self._grid = grid
        self._bridge = CapillaryBridge(**capillary_args)
        self._quadrature = quadrature
        self._fem = FirstOrderElement(grid, quadrature.quad_pt_coords)

        # Initialise to None so it raises error if user forgets to set them
        self._nodal_gap: Field = None
        self._quadr_gap: Field = None
        self._nodal_phase: Field = None
        self._quadr_phase: Field = None
        self._quadr_phase_gradient: Field = None

    def get_gap(self):
        """Return nodal gap field."""
        return self._nodal_gap

    def set_gap(self, value):
        """Set nodal gap and update quadrature-point values."""
        self._nodal_gap = adapt_shape(value)
        self._quadr_gap = self._fem.interpolate_value(self._nodal_gap)

    @property
    def gap_is_closed(self):
        """Boolean mask where gap equals zero (solid contact)."""
        return self._nodal_gap == 0

    def get_phase(self):
        """Return nodal phase field."""
        return self._nodal_phase

    def set_phase(self, value):
        """Set nodal phase and update quadrature-point values and gradients."""
        value[self.gap_is_closed] = 0.
        self._nodal_phase = value
        self._quadr_phase = self._fem.interpolate_value(self._nodal_phase)
        self._quadr_phase_gradient = self._fem.interpolate_gradient(self._nodal_phase)

    @property
    def phase_lb(self):
        """Lower bound for phase field (vapour)."""
        return self._bridge.phase_vapour

    @property
    def phase_ub(self):
        """Upper bound for phase field (liquid)."""
        return self._bridge.phase_liquid

    def validate_phase(self):
        """Log warnings if phase field exceeds [0, 1] bounds."""
        if np.any(self._nodal_phase < 0):
            outlier = np.where(self._nodal_phase < 0, self._nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")
        if np.any(self._nodal_phase > 1):
            outlier = np.where(self._nodal_phase > 1, self._nodal_phase, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")

    def get_energy(self):
        """Compute total capillary energy."""
        integrand = self._bridge.compute_local_energy(self._quadr_gap, self._quadr_phase, self._quadr_phase_gradient)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_energy_jacobian(self):
        """Compute gradient of energy w.r.t. nodal phase."""
        [energy_D_phase, energy_D_phase_grad] = self._bridge.compute_local_energy_jacobian(
            self._quadr_gap, self._quadr_phase, self._quadr_phase_gradient)
        jacobian = self._fem.propag_sens_value(self._quadrature.propag_integral_weight(
            self._grid, energy_D_phase)) + self._fem.propag_sens_gradient(self._quadrature.propag_integral_weight(
                self._grid, energy_D_phase_grad))
        jacobian[self.gap_is_closed] = 0
        return jacobian

    def get_volume(self):
        """Compute total liquid volume."""
        integrand = self._bridge.compute_local_volume(self._quadr_gap, self._quadr_phase)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_volume_jacobian(self):
        """Compute gradient of volume w.r.t. nodal phase."""
        [volume_D_phase] = self._bridge.compute_local_volume_jacobian(self._quadr_gap, self._quadr_phase)
        jacobian = self._fem.propag_sens_value(self._quadrature.propag_integral_weight(self._grid, volume_D_phase))
        jacobian[self.gap_is_closed] = 0
        return jacobian

    def get_perimeter(self):
        """Compute total perimeter of liquid-vapour interface."""
        integrand = self._bridge.compute_local_perimeter(self._quadr_phase, self._quadr_phase_gradient)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_max_volume(self):
        """Compute maximum available volume."""
        return self._quadrature.integrate(self._grid, self._quadr_gap).item()

    def get_liquid_area(self):
        """Compute area of liquid-solid interface."""
        return self._quadrature.integrate(self._grid, self._quadr_phase).item()
