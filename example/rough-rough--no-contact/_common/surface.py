"""The two solid surfaces bounding the liquid bridge.

`[surface.upper]` and `[surface.lower]` always both exist, every key lives inside one of them,
and each states its own `shape`. `surface_window` is the variant that works in absolute physical
units.

Note:
    A seed is not a realisation. `psd_to_height` builds `ifft2(sqrt(psd * A) * phasor)`, and
    `generate_phasor_2D_random(shape, seed)` randomises phases alone. The realisation is
    therefore `(seed, grid.nb_pixels)` and the PSD is a filter over it: at one pixel count, one
    seed gives one surface however the PSD is changed. `random_amplitude=True` breaks this, and
    nothing passes it.
"""

import numpy as np

from a_package.domain import Grid
from a_package.model import SelfAffineRoughness

from _common.grid import build_grid, build_length_unit


FACES = ("upper", "lower")


def face_shape(config: dict, face: str) -> str:
    """Which shape a face declares, from ``[surface.<face>].shape``.

    Raises:
        KeyError: If the face states no shape. There is no default.
    """
    return config["surface"][face]["shape"]


def face_params(config: dict, face: str) -> dict:
    """A face's shape parameters, from its ``[surface.<face>.<shape>]`` subsection.

    Note:
        `shape` is the tag and the subsection it names is the payload. A shape taking no
        parameters -- "flat" -- has no subsection, hence the empty default.
    """
    return config["surface"][face].get(face_shape(config, face), {})


def build_roughness(config: dict, face: str) -> SelfAffineRoughness:
    """Build one face's self-affine roughness, with wavevectors in grid units.

    Recognised keys under ``[surface.<face>.rough]``:
    - ``length_unit`` (str, required)               -> unit of the wavelengths below
    - ``rolloff_wavelength`` (float, required)      -> ``qR``
    - ``cutoff_wavelength`` (float, required)       -> ``qS``
    - ``hurst_exponent`` (float, required)          -> ``H``
    - ``prefactor`` (float, required)               -> ``C0``, in grid units
    - ``termination_wavelength`` (float, optional)  -> ``qT``; omit for the model's own default
    """
    section = face_params(config, face)
    scale = build_length_unit(build_grid(config), section["length_unit"])

    args = dict(
        C0=section["prefactor"],
        H=section["hurst_exponent"],
        qR=scale.to_physical((2 * np.pi) / section["rolloff_wavelength"], exponent=-1),
        qS=scale.to_physical((2 * np.pi) / section["cutoff_wavelength"], exponent=-1),
    )
    if "termination_wavelength" in section:
        args["qT"] = scale.to_physical((2 * np.pi) / section["termination_wavelength"], exponent=-1)
    return SelfAffineRoughness(**args)


def build_surface(config: dict) -> list[np.ndarray]:
    """Build the upper and lower height profiles.

    Recognised keys, per face:
    - ``[surface.<face>].shape`` (str, required)      -> ``"rough"`` or ``"flat"``
    - ``[surface.<face>.rough].seed`` (int, optional) -> omit for a non-reproducible surface
    - plus `build_roughness`'s keys when the shape is ``"rough"``

    Returns:
        list[np.ndarray]: Upper then lower, whole rather than decomposed. Generation is serial,
            so distributing them is the caller's job.

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


def report_surface_stats(grid: Grid, upper: np.ndarray, lower: np.ndarray, separations: np.ndarray) -> None:
    """Print the roughness amplitude and the gap range in pixels, warning where they touch.

    Args:
        grid: Supplies the pixel size.
        upper: Whole height profile, as `build_surface` returns it.
        lower: Whole height profile, as `build_surface` returns it.
        separations: The separations to bracket the gap over; one element is fine.
    """
    pixel_size = grid.element_sizes[0]
    print(
        f"RMS amplitude, upper={np.std(upper - upper.mean()) / pixel_size}px, "
        f"lower={np.std(lower - lower.mean()) / pixel_size}px"
    )

    d_min, d_max = np.min(separations), np.max(separations)
    gap_at_closest = (upper + d_min - lower).min()
    gap_at_farthest = (upper + d_max - lower).max()
    print(
        f"Gap range: [{d_min / pixel_size} - {(d_min - gap_at_closest) / pixel_size}px, "
        f"{d_max / pixel_size} + {(gap_at_farthest - d_max) / pixel_size}px]"
    )

    if gap_at_closest < 0:
        print("WARNING: the two surfaces will contact.")
