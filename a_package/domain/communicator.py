import muGrid
try:
    from mpi4py import MPI
    communicator = muGrid.Communicator(MPI.COMM_WORLD)
except ModuleNotFoundError:
    MPI = None
    communicator = muGrid.Communicator()


def factorize_closest(value: int, nb_ints: int):
    """Find the maximal combination of nb_ints integers whose product is less or equal to value."""
    nb_divisions = []
    for root_degree in range(nb_ints, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        nb_divisions.append(max_divisor)
        value //= max_divisor
    return nb_divisions
