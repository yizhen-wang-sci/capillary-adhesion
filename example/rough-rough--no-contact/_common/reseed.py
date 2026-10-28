"""Per-realisation surfaces, for cases measuring spread over realisations."""

from typing import Callable

import numpy as np

from _common.config import value_at_path
from _common.surface import FACES, face_shape


def derive_face_seeds(config: dict, root: int, index: int) -> dict[str, int]:
    """Install the seed of every rough face, derived from a root and a realisation index.

    Args:
        config: The configuration, modified in place under
            ``[surface.<face>.<shape>].seed``.
        root: The root entropy, one per recipe.
        index: The realisation index.

    Returns:
        dict: The seeds, keyed by face.
    """
    child = np.random.SeedSequence(root, spawn_key=(index,))
    states = child.generate_state(len(FACES), dtype=np.uint32)

    seeds = {}
    for face, state in zip(FACES, states):
        shape = face_shape(config, face)
        if shape == "flat":
            continue
        config["surface"][face].setdefault(shape, {})["seed"] = int(state)
        seeds[face] = int(state)
    return seeds


def seeds_from_index(root_path: str, index_path: str) -> Callable[[dict], None]:
    """Build the `resolve_config` of a case whose seeds follow a realisation index.

    Args:
        root_path: Dotted config path holding the root entropy.
        index_path: Dotted config path holding the realisation index.

    Returns:
        Callable: Takes a config and installs the derived seeds in place.
    """

    def resolve(config: dict) -> None:
        derive_face_seeds(config, value_at_path(config, root_path), value_at_path(config, index_path))

    return resolve
