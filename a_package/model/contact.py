"""Contact between surfaces."""
import muGrid
import numpy as np
from NuMPI import MPI

from a_package.domain import (
    CentroidQuadrature,
    FirstOrderElement,
    Grid,
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
    """Response of a semi-infinite elastic solid to a surface stress over its surface, per wavevector."""

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
        self._E_star = youngs_modulus / (1 - poisson_ratio ** 2)

    def to_stress_spectrum(self, spectral_displacement):
        """Surface stress holding a given surface displacement.

        Args:
            spectral_displacement: Surface displacement, one value per wavevector.

        Returns:
            The stress, one value per wavevector.
        """
        return 0.5 * self._E_star * self._q_norm * spectral_displacement


class ElasticContact:
    """Computes the gap field between a rigid upper surface and an elastic lower one.

    The gap follows the surfaces, the separation and the displacement of the elastic solid,
    which is the field this holds and the one the energy is minimised over.
    """

    def __init__(
        self,
        grid: Grid,
        elastic_params: dict,
        upper: np.ndarray,
        lower: np.ndarray,
        communicator=MPI.COMM_SELF,
    ):
        """Set up the numerics and the fields over a grid, and store the two surface profiles.

        Args:
            grid: The discrete space, decomposed for a transform, providing the convolution
                with the stiffness and the element sizes.
            elastic_params: Elastic constants of the solid, passed to `SpectralElasticPlane`.
            upper: Height profile of the rigid upper surface over this rank's subdomain. Passed
                through `adapt_shape`, so either the bare element shape or the full field shape
                will do.
            lower: Height profile of the undeformed lower surface, same shapes accepted.
            communicator: Communicator spanning the ranks the fields are spread across,
                `MPI.COMM_SELF` by default.

        Raises:
            ValueError: If a surface profile spans other than this rank's subdomain.
        """
        self._grid = grid
        self._solid = SpectralElasticPlane(grid, **elastic_params)
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
        self._quadrature = CentroidQuadrature(communicator)
        self._fem = FirstOrderElement(self._quadrature.quad_pt_coords, grid.element_sizes)

        grid.collection_real.set_nb_sub_pts("nodal", 1)
        grid.collection_real.set_nb_sub_pts("quadr", self._quadrature.nb_quad_pts)
        grid.collection_spectral.set_nb_sub_pts("quadr", self._quadrature.nb_quad_pts)

        self.communicator = communicator

        # fields
        self._gap_origin_nodal = muGrid.Field(grid.collection_real.real_field("nodal_gap_origin", 1, "nodal"))
        self._gap_origin_quadr = muGrid.Field(grid.collection_real.real_field("quadr_gap_origin", 1, "quadr"))

        self._gap_nodal = muGrid.Field(grid.collection_real.real_field("nodal_gap", 1, "nodal"))
        self._gap_quadr = muGrid.Field(grid.collection_real.real_field("quadr_gap", 1, "quadr"))

        self._displacement_quadr = muGrid.Field(grid.collection_real.real_field("quadr_displacement", 1, "quadr"))
        self._displacement_quadr_spectrum = muGrid.Field(grid.collection_spectral.complex_field("quadr_displacement_spectrum", 1, "quadr"))
        self._stress_quadr_spectrum = muGrid.Field(grid.collection_spectral.complex_field("quadr_stress_spectrum", 1, "quadr"))
        self._stress_quadr = muGrid.Field(grid.collection_real.real_field("quadr_stress", 1, "quadr"))
        self._stress_derivative_quadr_spectrum = muGrid.Field(grid.collection_spectral.complex_field("quadr_stress_derivative_spectrum", 1, "quadr"))
        self._stress_derivative_quadr = muGrid.Field(grid.collection_real.real_field("quadr_stress_derivative", 1, "quadr"))

        self._energy_D_gap_quadr = muGrid.Field(grid.collection_real.real_field("energy_D_gap_quadr", 1, "quadr"))
        self._energy_D_gap_nodal = muGrid.Field(grid.collection_real.real_field("energy_D_gap_nodal", 1, "nodal"))

        self._volume_D_gap_quadr = muGrid.Field(grid.collection_real.real_field("volume_D_gap_quadr", 1, "quadr"))
        self._volume_D_gap_nodal = muGrid.Field(grid.collection_real.real_field("volume_D_gap_nodal", 1, "nodal"))

    def set_mean_separation(self, separation: float):
        """Set the mean separation the gap is measured at.

        Args:
            separation: The mean separation.
        """
        self._gap_origin_nodal.s[...] = self._upper + separation - self._lower
        self._grid.decomposition.communicate_ghosts(self._gap_origin_nodal)
        self._fem.interpolate_value(self._gap_origin_nodal, self._gap_origin_quadr)

    def get_gap(self):
        return self._gap_nodal.s

    def set_gap(self, value: np.ndarray):
        """Set nodal displacement and update the traction holding it.

        Args:
            value: Nodal displacement values, reshaped to this rank's subdomain.
        """
        self._gap_nodal.s[...] = value.reshape(self._grid.decomposition.nb_subdomain_grid_pts)
        self._grid.decomposition.communicate_ghosts(self._gap_nodal)
        self._fem.interpolate_value(self._gap_nodal, self._gap_quadr)
        self._displacement_quadr.s[...] = self._gap_origin_quadr.s - self._gap_quadr.s
        self._grid.decomposition.fft(self._displacement_quadr, self._displacement_quadr_spectrum)
        # precompute the stress because setting the gap also sets the displacement
        self._stress_quadr_spectrum.s[...] = self._solid.to_stress_spectrum(
            self._displacement_quadr_spectrum.s)
        self._grid.decomposition.ifft(self._stress_quadr_spectrum, self._stress_quadr)
        self._stress_quadr.s[...] *= self._grid.decomposition.normalisation

    @property
    def gap_lb(self):
        """Lower bound for the displacement, which the elastic solid does not have."""
        return 0

    @property
    def gap_ub(self):
        """Upper bound for the displacement, the gap of the undeformed surfaces."""
        return np.inf

    @property
    def gap_is_closed(self):
        """Boolean mask where the displacement reaches its upper bound (solid contact)."""
        return self._gap_origin_nodal.s == 0

    def get_energy(self):
        """Compute total elastic energy."""
        integrand = 0.5 * self._stress_quadr.s * self._displacement_quadr.s
        return self._quadrature.integrate(integrand, self._grid.element_area).item()

    def get_energy_jacobian(self):
        """Compute gradient of energy w.r.t. nodal gap."""
        self._energy_D_gap_quadr.s[...] = self._quadrature.propag_integral_weight(
            -self._stress_quadr.s, self._grid.element_area)
        self._grid.decomposition.communicate_ghosts(self._energy_D_gap_quadr)
        self._fem.propag_sens_value(self._energy_D_gap_quadr, self._energy_D_gap_nodal)
        return self._energy_D_gap_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def get_gap_volume(self):
        """Compute total volume swept by the displacement."""
        return self._quadrature.integrate(self._gap_quadr.s, self._grid.element_area).item()

    def get_gap_volume_jacobian(self):
        """Compute gradient of displaced volume w.r.t. nodal displacement."""
        self._volume_D_gap_quadr.s[...] = self._quadrature.propag_integral_weight(
            np.ones_like(self._gap_quadr.s), self._grid.element_area)
        self._grid.decomposition.communicate_ghosts(self._volume_D_gap_quadr)
        self._fem.propag_sens_value(self._volume_D_gap_quadr, self._volume_D_gap_nodal)
        return self._volume_D_gap_nodal.s.squeeze(axis=(field_component_ax, field_sub_pt_ax))

    def get_gap_origin_volume(self):
        return self._quadrature.integrate(self._gap_origin_quadr.s, self._grid.element_area).item()
