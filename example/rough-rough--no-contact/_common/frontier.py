"""How far each record of a run got. Serial."""

from collections.abc import Sequence

import numpy as np
from a_package.dataset import NpyBack, QuantityFront, RecordDir
from a_package.dataset.quantity import QuantityError


def seed_point_scalars(quantities: QuantityFront, names: Sequence[str], shape: tuple[int, ...]) -> None:
    """Fill every per-point scalar with NaN, so an unwritten point reads as unwritten.

    Note:
        Only for a fresh record. `NpyBack.save_value` at one address creates the whole array
        with `np.zeros`, so an unwritten point would otherwise read as zero; called on a record
        being resumed, this erases the frontier `count_complete_points` reads.
    """
    for name in names:
        quantities.save_value(name, np.full(shape, np.nan))


def count_complete_points(record: RecordDir, names: Sequence[str]) -> int:
    """How many leading points of this record have all of `names` on disk. Serial.

    A point is one place in the trailing frame, counted in the order the solver walks it. The
    count is the minimum across `names`.
    """
    try:
        quantities = QuantityFront(NpyBack(record.data))
    except (FileNotFoundError, OSError):
        return 0

    counts = []
    for name in names:
        if name not in quantities:
            return 0
        try:
            values = np.asarray(quantities.load_value(name)).reshape(-1)
        except (QuantityError, FileNotFoundError, OSError):
            return 0
        unwritten = np.flatnonzero(np.isnan(values))
        counts.append(int(unwritten[0]) if len(unwritten) else len(values))
    return min(counts) if counts else 0
