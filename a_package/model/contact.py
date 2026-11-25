"""Contact between surfaces."""

from typing import Protocol, runtime_checkable

import muGrid
import numpy as np
from NuMPI import MPI

from a_package.domain import (
    Grid,
    NodalQuadrature,
    adapt_shape,
    field_component_ax,
    field_element_axs,
    field_sub_pt_ax,
)


class RigidContact:
    """Computes the gap field between two rigid surfaces at a given separation."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        """Store the two surface profiles, shaped to the field convention.

        Args:
            upper: Height profile of the upper surface. Passed through `adapt_shape`, so either
                the bare grid shape or the full field shape will do.
            lower: Height profile of the lower surface, same shapes accepted.
        """
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def set_mean_separation(self, value: float):
        """Set the mean separation the gap is measured at.

        Args:
            value: The mean separation.
        """
        self.separation = value

    def get_gap(self):
        """Gap between the two surfaces at the separation set by `set_mean_separation`.

        Returns:
            The gap, zeroed wherever the surfaces would collide.
        """
        return np.clip(self.separation + self.upper - self.lower, 0, None)


class SpectralElasticPlane:
    """Response of a semi-infinite elastic solid to a surface stress, per wavevector."""

    def __init__(self, grid: Grid, youngs_modulus: float, poisson_ratio: float = 0.0):
        """Derive the contact modulus from the elastic constants, and the wavevectors from the grid.

        Args:
            grid: The discrete space, decomposed for a transform, providing this rank's share
                of the spectrum.
            youngs_modulus: Young's modulus of the solid.
            poisson_ratio: Poisson's ratio of the solid, 0.0 by default.
        """
        # Wavevector
        wavevector = grid.form_local_spectral_mesh()
        self._q_norm = np.linalg.norm(wavevector, ord=2, axis=0, keepdims=True)
        # Effective contact modulus
        self._E_star = youngs_modulus / (1 - poisson_ratio**2)

    def to_stress_spectrum(self, spectral_displacement):
        """Surface stress holding a given surface displacement.

        Args:
            spectral_displacement: Surface displacement, one value per wavevector.

        Returns:
            The stress, one value per wavevector.
        """
        return 0.5 * self._E_star * self._q_norm * spectral_displacement


@runtime_checkable
class ContactVariable(Protocol):
    """The field a contact energy is minimised over, one-to-one and monotonic in the nodal gap."""

    def to_gap(self, x: np.ndarray, gap_origin: np.ndarray) -> np.ndarray:
        """Gap at given values of the variable.

        Args:
            x: Nodal values of the variable.
            gap_origin: Gap of the undeformed surfaces, shaped like `x`.

        Returns:
            The nodal gap.
        """
        raise NotImplementedError

    def from_gap(self, gap: np.ndarray, gap_origin: np.ndarray) -> np.ndarray:
        """Values of the variable at a given gap.

        Args:
            gap: Nodal gap.
            gap_origin: Gap of the undeformed surfaces, broadcastable against `gap`.

        Returns:
            The nodal values of the variable.
        """
        raise NotImplementedError

    def gap_D_x(self, x: np.ndarray, gap_origin: np.ndarray) -> np.ndarray | float:
        """Derivative of the nodal gap w.r.t. the variable, per node.

        Args:
            x: Nodal values of the variable.
            gap_origin: Gap of the undeformed surfaces, shaped like `x`.

        Returns:
            The derivative, shaped like `x` or a single value for all nodes.
        """
        raise NotImplementedError


class GapVariable(ContactVariable):
    """The gap itself."""

    def to_gap(self, x, gap_origin):
        """Gap at given values of the gap."""
        return x

    def from_gap(self, gap, gap_origin):
        """Gap at a given gap."""
        return gap

    def gap_D_x(self, x, gap_origin):
        """Derivative of the gap w.r.t. itself, one."""
        return 1.0


class DisplacementVariable(ContactVariable):
    """The displacement of the elastic solid, `gap_origin - gap`."""

    def to_gap(self, x, gap_origin):
        """Gap at given values of the displacement."""
        return gap_origin - x

    def from_gap(self, gap, gap_origin):
        """Displacement at a given gap."""
        return gap_origin - gap

    def gap_D_x(self, x, gap_origin):
        """Derivative of the gap w.r.t. the displacement, minus one."""
        return -1.0


class ElasticContact:
    """Computes the gap field between a rigid upper surface and an elastic lower one.

    The state is `x`, the `ContactVariable` chosen at construction, which the energy is minimised
    over. The gap follows from `x` and the gap of the undeformed surfaces, the displacement of the
    elastic solid is `gap_origin - gap`, and the stress follows from the displacement.
    """

    def __init__(
        self,
        grid: Grid,
        elastic_params: dict | None,
        upper: np.ndarray,
        lower: np.ndarray,
        communicator=MPI.COMM_SELF,
        variable: ContactVariable | None = None,
        solid: SpectralElasticPlane | None = None,
    ):
        """Set up the numerics and the fields over a grid, and store the two surface profiles.

        Args:
            grid: The discrete space, decomposed for a transform, providing the convolution
                with the stiffness and the element sizes.
            elastic_params: Elastic constants of a `SpectralElasticPlane` built as the solid, or
                None when `solid` is given.
            upper: Height profile of the rigid upper surface over this rank's subdomain. Passed
                through `adapt_shape`, so either the bare element shape or the full field shape
                will do.
            lower: Height profile of the undeformed lower surface, same shapes accepted.
            communicator: Communicator spanning the ranks the fields are spread across,
                `MPI.COMM_SELF` by default.
            variable: The field `x` stands for, the gap by default.
            solid: The elastic solid, built from `elastic_params` by default.

        Raises:
            TypeError: If `variable` is not a `ContactVariable`.
            ValueError: If other than exactly one of `elastic_params` and `solid` is given, or a
                surface profile spans other than this rank's subdomain.
        """
        variable = GapVariable() if variable is None else variable
        if not isinstance(variable, ContactVariable):
            raise TypeError(f"The variable must be a ContactVariable, got {type(variable).__name__}.")
        if (elastic_params is None) == (solid is None):
            given = "neither" if solid is None else "both"
            raise ValueError(f"Exactly one of elastic_params and solid must be given, got {given}.")
        self.variable = variable

        self._grid = grid
        self._solid = SpectralElasticPlane(grid, **elastic_params) if solid is None else solid
        self._upper = adapt_shape(upper)
        self._lower = adapt_shape(lower)

        subdomain_shape = tuple(grid.decomposition.nb_subdomain_grid_pts)
        for name, profile in (("upper", self._upper), ("lower", self._lower)):
            element_shape = profile.shape[field_element_axs[0] :]
            if element_shape != subdomain_shape:
                raise ValueError(
                    f"The {name} surface profile must span this rank's subdomain "
                    f"{subdomain_shape}, got {element_shape}."
                )

        # numeric setup
        self._quadrature = NodalQuadrature(communicator)

        grid.collection_real.set_nb_sub_pts("nodal", 1)
        grid.collection_spectral.set_nb_sub_pts("nodal", 1)

        self.communicator = communicator

        # fields
        self._gap_origin_nodal = muGrid.Field(grid.collection_real.real_field("nodal_gap_origin", 1, "nodal"))
        self._gap_nodal = muGrid.Field(grid.collection_real.real_field("nodal_gap", 1, "nodal"))
        self._x_nodal = muGrid.Field(grid.collection_real.real_field("nodal_contact_x", 1, "nodal"))

        self._displacement_nodal = muGrid.Field(grid.collection_real.real_field("nodal_displacement", 1, "nodal"))
        self._displacement_nodal_spectrum = muGrid.Field(
            grid.collection_spectral.complex_field("nodal_displacement_spectrum", 1, "nodal")
        )
        self._stress_nodal_spectrum = muGrid.Field(
            grid.collection_spectral.complex_field("nodal_stress_spectrum", 1, "nodal")
        )
        self._stress_nodal = muGrid.Field(grid.collection_real.real_field("nodal_stress", 1, "nodal"))

    def set_mean_separation(self, separation: float):
        """Set the mean separation the gap is measured at, and update the state at the current `x`.

        Args:
            separation: The mean separation.
        """
        self._gap_origin_nodal.s[...] = self._upper + separation - self._lower
        self._update_from_x()

    def get_gap_origin(self):
        """Gap of the undeformed surfaces at the mean separation."""
        return self._gap_origin_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def get_gap(self):
        """Gap of the deformed surfaces."""
        return self._gap_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def set_gap(self, value: np.ndarray):
        """Set the gap through `x`, and update the displacement and the stress holding it.

        Args:
            value: Nodal gap values, reshaped to this rank's subdomain.
        """
        gap = value.reshape(self._grid.decomposition.nb_subdomain_grid_pts)
        self._x_nodal.s[...] = self.variable.from_gap(gap, self.get_gap_origin())
        self._update_from_x()

    def get_displacement(self):
        """Displacement of the elastic solid."""
        return self._displacement_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def set_displacement(self, value: np.ndarray):
        """Set the displacement through `x`, and update the gap and the stress holding it.

        Args:
            value: Nodal displacement values, reshaped to this rank's subdomain.
        """
        displacement = value.reshape(self._grid.decomposition.nb_subdomain_grid_pts)
        gap_origin = self.get_gap_origin()
        self._x_nodal.s[...] = self.variable.from_gap(gap_origin - displacement, gap_origin)
        self._update_from_x()

    def get_x(self):
        """The field the energy is minimised over, per `variable`, as last set."""
        return self._x_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def set_x(self, value: np.ndarray):
        """Set the field the energy is minimised over, and update the gap, displacement and stress.

        Args:
            value: Nodal values of `variable`, reshaped to this rank's subdomain.
        """
        self._x_nodal.s[...] = value.reshape(self._grid.decomposition.nb_subdomain_grid_pts)
        self._update_from_x()

    def _update_from_x(self):
        """Compute the gap, the displacement and the stress holding it, at the current `x`."""
        self._gap_nodal.s[...] = self.variable.to_gap(self._x_nodal.s, self._gap_origin_nodal.s)
        self._displacement_nodal.s[...] = self._gap_origin_nodal.s - self._gap_nodal.s
        self._grid.decomposition.fft(self._displacement_nodal, self._displacement_nodal_spectrum)
        self._stress_nodal_spectrum.s[...] = self._solid.to_stress_spectrum(self._displacement_nodal_spectrum.s)
        self._grid.decomposition.ifft(self._stress_nodal_spectrum, self._stress_nodal)
        self._stress_nodal.s[...] *= self._grid.decomposition.normalisation

    def get_normal_stress(self):
        """Get the load distribution."""
        return self._stress_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    @property
    def gap_lb(self):
        """Lower bound for the gap, where the surfaces touch."""
        return 0

    @property
    def gap_ub(self):
        """Upper bound for the gap, which it does not have."""
        return np.inf

    @property
    def x_lb(self):
        """Lower bound for `x`, the lesser of the two gap bounds mapped to `variable`."""
        return np.minimum(*self._map_gap_bounds())

    @property
    def x_ub(self):
        """Upper bound for `x`, the greater of the two gap bounds mapped to `variable`."""
        return np.maximum(*self._map_gap_bounds())

    def _map_gap_bounds(self):
        """The lower and the upper gap bound, mapped to `variable`."""
        gap_origin = self.get_gap_origin()
        return self.variable.from_gap(self.gap_lb, gap_origin), self.variable.from_gap(self.gap_ub, gap_origin)

    @property
    def gap_is_closed(self):
        """Boolean mask where the displacement reaches its upper bound (solid contact)."""
        return self._gap_origin_nodal.s == 0

    def get_energy(self):
        """Compute total elastic energy."""
        integrand = 0.5 * self._stress_nodal.s * self._displacement_nodal.s
        return self._quadrature.integrate(integrand, self._grid.element_area).item()

    def get_energy_jacobian(self):
        """Compute gradient of energy w.r.t. `x`."""
        energy_D_gap = self._quadrature.propag_integral_weight(-self._stress_nodal.s, self._grid.element_area)
        energy_D_x = energy_D_gap * self._get_gap_D_x()
        return energy_D_x.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def get_gap_volume(self):
        """Compute total volume of the gap."""
        return self._quadrature.integrate(self._gap_nodal.s, self._grid.element_area).item()

    def get_gap_volume_jacobian(self):
        """Compute gradient of the gap volume w.r.t. `x`."""
        volume_D_gap = self._quadrature.propag_integral_weight(np.ones_like(self._gap_nodal.s), self._grid.element_area)
        volume_D_x = volume_D_gap * self._get_gap_D_x()
        return volume_D_x.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def _get_gap_D_x(self):
        """Derivative of the nodal gap w.r.t. `x`, at the current `x`."""
        return self.variable.gap_D_x(self._x_nodal.s, self._gap_origin_nodal.s)

    def get_gap_volume_intercept(self):
        """Gap volume at `x` equal to zero, the constant term of the gap volume as affine in `x`."""
        gap_origin = self._gap_origin_nodal.s
        gap_at_zero = self.variable.to_gap(np.zeros_like(gap_origin), gap_origin)
        return self._quadrature.integrate(gap_at_zero, self._grid.element_area).item()

    def get_gap_origin_volume(self):
        """Compute total volume of the gap of the undeformed surfaces."""
        return self._quadrature.integrate(self._gap_origin_nodal.s, self._grid.element_area).item()
