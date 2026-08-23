"""The two solid surfaces, described in absolute physical units.

Same model and same per-face layout as `_common.surface`, different unit algebra: each face
gives its PSD an absolute length scale, and every quantity is carried into physical units and
back out into the grid's. Sweeping `grid.length_scale` varies how much of a fixed PSD a fixed
pixel count covers.

Note:
    `grid.length_scale` leaves `grid.nb_pixels` alone, so the phasor -- see the note in
    `_common.surface` -- is the same at every window size. One seed across the batch gives one
    phase field with the amplitude envelope rescaled: every mode index is reused at a different
    physical q. It is not a crop out of one fixed surface, which would keep the low-q content
    and add new high-q content.
"""

import numpy as np

from a_package.model import SelfAffineRoughness
from a_package.simulation import UnitConversion

from _common.grid import build_grid
from _common.surface import FACES, face_params, face_shape, report_surface_stats  # noqa: F401


def build_roughness(config: dict, face: str) -> SelfAffineRoughness:
    """Build one face's self-affine roughness, carried from its length scale into the grid's.

    Recognised keys:
    - ``[grid].length_scale`` (float, required)                     -> unit the grid works in
    - ``[surface.<face>.rough].length_scale`` (float, required)     -> unit the PSD is in
    - ``[surface.<face>.rough].rolloff_wavelength`` (required)      -> ``qR``
    - ``[surface.<face>.rough].cutoff_wavelength`` (required)       -> ``qS``
    - ``[surface.<face>.rough].termination_wavelength`` (required)  -> ``qT``
    - ``[surface.<face>.rough].hurst_exponent`` (required)          -> ``H``
    - ``[surface.<face>.rough].prefactor`` (required)               -> ``C0``, rescaled by
      ``2 - 2H``
    """
    section = face_params(config, face)
    scale_psd = UnitConversion(section["length_scale"])
    scale_grid = UnitConversion(config["grid"]["length_scale"])

    def regrid(value, exponent=1):
        return scale_grid.to_dimensionless(scale_psd.to_physical(value, exponent=exponent), exponent=exponent)

    H = section["hurst_exponent"]
    return SelfAffineRoughness(
        C0=regrid(section["prefactor"], exponent=2 - 2 * H),
        H=H,
        qR=regrid((2 * np.pi) / section["rolloff_wavelength"], exponent=-1),
        qS=regrid((2 * np.pi) / section["cutoff_wavelength"], exponent=-1),
        qT=regrid((2 * np.pi) / section["termination_wavelength"], exponent=-1),
    )


def build_surface(config: dict) -> list[np.ndarray]:
    """Build the upper and lower height profiles.

    Recognised keys, per face:
    - ``[surface.<face>].shape`` (str, required)      -> ``"rough"`` or ``"flat"``
    - ``[surface.<face>.rough].seed`` (int, optional) -> the realisation, with `grid.nb_pixels`
    - plus `build_roughness`'s keys when the shape is ``"rough"``

    Returns:
        list[np.ndarray]: Upper then lower, whole rather than decomposed.

    Raises:
        ValueError: If a face declares a shape that is neither.
    """
    grid = build_grid(config)
    profiles = []
    for face in FACES:
        shape = face_shape(config, face)
        match shape:
            case "flat":
                profiles.append(np.zeros(grid.nb_domain_grid_pts))
            case "rough":
                profiles.append(
                    build_roughness(config, face).generate_height_profile(grid, face_params(config, face).get("seed"))
                )
            case _:
                raise ValueError(f"Unknown surface shape for {face}: {shape}")
    return profiles
