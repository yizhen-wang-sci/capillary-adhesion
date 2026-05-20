"""
Capillary bridge model and formulation.
"""

import logging

import numpy as np
import muGrid

from a_package.domain import Grid, FirstOrderElement, Quadrature, centroid_quadrature, factorize_closest
from a_package.domain import Field, field_component_ax


logger = logging.getLogger(__name__)


class PhaseMixture:
    """Physics of the liquid-vapour interface.

    Minimal numerics info: it only touches the component axis and keeps the axis not squeezed after reduction.
    """

    def __init__(self, eta: float, theta: float, epsilon: float=1.0, component_axis: int=field_component_ax):
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
        self._component_axis = component_axis

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
        return self.perimeter_prefactor * ((1 / self._eta) * self.double_well_penalty(
            phase, self._component_axis) + self._eta * self.square_penalty(phase_grad, self._component_axis))

    def compute_local_energy(self, gap: Field, phase: Field, phase_grad: Field):
        """Compute local energy density (liquid-vapour + liquid-solid contributions)."""
        liquid_vapour = self.compute_local_perimeter(phase, phase_grad) * gap * self._curv * self._epsilon
        # FIXME: switch on whether to use small-slope-approx.?
        # upper and lower surface, hence the 2. (height gradient square is one order higher and omitted)
        liquid_solid = 2.0 * phase
        return liquid_vapour + self._gamma * liquid_solid

    @staticmethod
    def double_well_penalty(x, axis):
        """Double-well potential W(x) = x^2(1-x)^2, summed over components."""
        return np.sum(x**2 * (1 - x) ** 2, axis=axis, keepdims=True)

    @staticmethod
    def square_penalty(x, axis):
        """Square norm |x|^2, summed over components."""
        return np.sum(x**2, axis=axis, keepdims=True)

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


class CapillaryBridge:
    """Nodal-value interface for capillary bridge evaluation.

    This class combines physics and numerics, and functions as the interface to use in simulation.
    All its methods accept and return nodal values, and it handles the interpolation and integral.
    """

    def __init__(self, grid: Grid, phase_mixture, quadrature: Quadrature = centroid_quadrature, communicator=None):
        self._grid = grid
        self._mixture = phase_mixture

        # numeric helpers
        self._quadrature = quadrature
        self._fem = FirstOrderElement(self._quadrature.quad_pt_coords, grid.element_sizes)

        # wrap communicator in muGrid.Communicator. The constructor has a mechanism to avoid
        # overhead if the communicator is already a muGrid.Communicator object.
        communicator = muGrid.Communicator(communicator)

        # decomposition and field collection setup
        nb_subdomains = factorize_closest(communicator.size, 2)
        self._decomposition = grid.decompose(nb_subdomains, nb_ghost_layers=(1, 1), communicator=communicator)
        self._collection = self._decomposition.collection
        self._collection.set_nb_sub_pts("nodal", 1)
        self._collection.set_nb_sub_pts("quadr", self._quadrature.nb_quad_pts)

        # fields
        self._nodal_gap = muGrid.Field(self._collection.real_field("nodal_gap", 1, "nodal"))
        self._quadr_gap = muGrid.Field(self._collection.real_field("quadr_gap", 1, "quadr"))
        self._nodal_phase = muGrid.Field(self._collection.real_field("nodal_phase", 1, "nodal"))
        self._quadr_phase = muGrid.Field(self._collection.real_field("quadr_phase", 1, "quadr"))
        self._quadr_phase_gradient = muGrid.Field(self._collection.real_field("quadr_phase_gradient", 2, "quadr"))

        # more fields for backward propagation
        self._quadr_value_1 = muGrid.Field(self._collection.real_field("quadr_value_1", 1, "quadr"))
        self._quadr_value_1_back_sens = muGrid.Field(self._collection.real_field("quadr_value_1_back_sens", 1, "nodal"))
        self._quadr_value_2 = muGrid.Field(self._collection.real_field("quadr_value_2", 1, "quadr"))
        self._quadr_value_2_back_sens = muGrid.Field(self._collection.real_field("quadr_value_2_back_sens", 1, "nodal"))
        self._quadr_gradient = muGrid.Field(self._collection.real_field("quadr_gradient", 2, "quadr"))
        self._quadr_gradient_back_sens = muGrid.Field(self._collection.real_field("quadr_gradient_back_sens", 1, "nodal"))

    @property
    def grid_shape(self):
        return self._grid.nb_domain_grid_pts

    def get_gap(self):
        """Return nodal gap field."""
        return self._nodal_gap.s

    def set_gap(self, value: np.ndarray):
        """Set nodal gap and update quadrature-point values."""
        self._nodal_gap.s[...] = np.reshape(value, (1, 1, *self._grid.nb_domain_grid_pts))
        self._decomposition.communicate_ghosts(self._nodal_gap)
        self._fem.interpolate_value(self._nodal_gap, self._quadr_gap)

    @property
    def gap_is_closed(self):
        """Boolean mask where gap equals zero (solid contact)."""
        return self._nodal_gap.s == 0

    def get_phase(self):
        """Return nodal phase field."""
        return self._nodal_phase.s

    def set_phase(self, value: np.ndarray):
        """Set nodal phase and update quadrature-point values and gradients."""
        self._nodal_phase.s[...] = np.reshape(value, (1, 1, *self._grid.nb_domain_grid_pts))
        self._nodal_phase.s[self.gap_is_closed] = 0.
        self._decomposition.communicate_ghosts(self._nodal_phase)
        self._fem.interpolate_value(self._nodal_phase, self._quadr_phase)
        self._fem.interpolate_gradient(self._nodal_phase, self._quadr_phase_gradient)

    @property
    def phase_lb(self):
        """Lower bound for phase field (vapour)."""
        return self._mixture.phase_vapour

    @property
    def phase_ub(self):
        """Upper bound for phase field (liquid)."""
        return self._mixture.phase_liquid

    def validate_phase(self):
        """Log warnings if phase field exceeds [0, 1] bounds."""
        if np.any(self._nodal_phase.s < 0):
            outlier = np.where(self._nodal_phase.s < 0, self._nodal_phase.s, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmin(outlier)
            logger.warning(f"Notice: phase field has {count} values < 0, min at {extreme:.2e}")
        if np.any(self._nodal_phase.s > 1):
            outlier = np.where(self._nodal_phase.s > 1, self._nodal_phase.s, np.nan)
            count = np.count_nonzero(~np.isnan(outlier))
            extreme = np.nanmax(outlier)
            logger.warning(f"Notice: phase field has {count} values > 1, max at 1.0+{extreme - 1:.2e}.")

    def get_energy(self):
        """Compute total capillary energy."""
        integrand = self._mixture.compute_local_energy(self._quadr_gap.s, self._quadr_phase.s,
                                                       self._quadr_phase_gradient.s)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_energy_jacobian(self):
        """Compute gradient of energy w.r.t. nodal phase."""
        [energy_D_phase, energy_D_phase_gradient] = self._mixture.compute_local_energy_jacobian(
            self._quadr_gap.s, self._quadr_phase.s, self._quadr_phase_gradient.s)

        self._quadr_value_1.s[...] = self._quadrature.propag_integral_weight(self._grid, energy_D_phase)
        self._decomposition.communicate_ghosts(self._quadr_value_1)
        self._fem.propag_sens_value(self._quadr_value_1, self._quadr_value_1_back_sens)

        self._quadr_gradient.s[...] = self._quadrature.propag_integral_weight(self._grid, energy_D_phase_gradient)
        self._decomposition.communicate_ghosts(self._quadr_gradient)
        self._fem.propag_sens_gradient(self._quadr_gradient, self._quadr_gradient_back_sens)

        jacobian = self._quadr_value_1_back_sens.s + self._quadr_gradient_back_sens.s
        jacobian[self.gap_is_closed] = 0
        return jacobian.squeeze()

    def get_volume(self):
        """Compute total liquid volume."""
        integrand = self._mixture.compute_local_volume(self._quadr_gap.s, self._quadr_phase.s)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_volume_jacobian(self):
        """Compute gradient of volume w.r.t. nodal phase."""
        [volume_D_phase] = self._mixture.compute_local_volume_jacobian(self._quadr_gap.s, self._quadr_phase.s)

        self._quadr_value_2.s[...] = self._quadrature.propag_integral_weight(self._grid, volume_D_phase)
        self._decomposition.communicate_ghosts(self._quadr_value_2)
        self._fem.propag_sens_value(self._quadr_value_2, self._quadr_value_2_back_sens)

        jacobian = self._quadr_value_2_back_sens.s.copy()
        jacobian[self.gap_is_closed] = 0
        return jacobian.squeeze()

    def get_perimeter(self):
        """Compute total perimeter of liquid-vapour interface."""
        integrand = self._mixture.compute_local_perimeter(self._quadr_phase, self._quadr_phase_gradient)
        return self._quadrature.integrate(self._grid, integrand).item()

    def get_max_volume(self):
        """Compute maximum available volume."""
        return self._quadrature.integrate(self._grid, self._quadr_gap.s).item()

    def get_liquid_area(self):
        """Compute area of liquid-solid interface."""
        return self._quadrature.integrate(self._grid, self._quadr_phase.s).item()
