import muGrid


try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
except ModuleNotFoundError:
    _comm = None
communicator = muGrid.Communicator(_comm)


def serial_exec(action, store=None):
    """Execute action on root, optionally store result, sync all ranks.

    If store is None: broadcast result to all ranks.
    If store provided: call store(result) on root, barrier, return None.
    """
    result = None
    if communicator.rank == 0:
        result = action()

    if store is None:
        communicator.bcast(result, 0)
        return result
    else:
        if communicator.rank == 0:
            store(result)
        communicator.barrier()
        return None


def factorize_closest(value: int, nb_ints: int):
    """Find the maximal combination of nb_ints integers whose product is less or equal to value."""
    nb_divisions = []
    for root_degree in range(nb_ints, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        nb_divisions.append(max_divisor)
        value //= max_divisor
    return nb_divisions
