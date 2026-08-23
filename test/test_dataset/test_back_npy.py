"""Tests for numeric arrays and quantities kept as `.npy` files."""

import numpy as np
import pytest
from NuMPI.Testing.Assertions import assert_all_array_equal

from a_package.dataset.back_npy import NpyBack, NpyBackError, NpyIO
from a_package.dataset.quantity import QuantityError, QuantityFront
from test.utils import generate_global_random_field


@pytest.fixture
def mock_field(mock_grid, comm_world):
    return generate_global_random_field(mock_grid.nb_domain_grid_pts, comm_world)


@pytest.fixture
def mock_array(comm_world):
    array = np.empty(10, dtype=float)
    if comm_world.rank == 0:
        rng = np.random.default_rng()
        array[...] = rng.random(array.shape)
    comm_world.Bcast(array, root=0)
    return array


@pytest.fixture
def io(decomposed_grid, comm_world):
    return NpyIO(**decomposed_grid.owned_layout(), communicator=comm_world)


# =============================================================================
# NpyIO


def test_a_decomposed_array_round_trips_through_one_file(mpi_tmp_path, decomposed_grid, mock_field, io, comm_world):
    local = decomposed_grid.get_local(mock_field)
    io.save_distributed(mpi_tmp_path / "field.npy", local)
    assert_all_array_equal(comm_world, io.load_distributed(mpi_tmp_path / "field.npy"), local)


def test_a_singular_array_round_trips_on_rank_zero_alone(mpi_tmp_path, mock_array, io, comm_world):
    io.save_singular(mpi_tmp_path / "array.npy", mock_array)
    loaded = io.load_singular(mpi_tmp_path / "array.npy")
    if comm_world.rank == 0:
        np.testing.assert_equal(loaded, mock_array)
    else:
        assert loaded is None


def test_what_one_rank_wrote_is_replicated_to_every_rank(mpi_tmp_path, mock_array, io):
    io.save_singular(mpi_tmp_path / "array.npy", mock_array)
    np.testing.assert_equal(io.load_replicated(mpi_tmp_path / "array.npy"), mock_array)


def test_an_unset_layout_reads_as_undecomposed_and_round_trips(mpi_tmp_path, mock_array, comm_world):
    io = NpyIO()
    assert not io.is_decomposed()
    path = mpi_tmp_path / f"whole-{comm_world.rank}.npy"
    io.save_distributed(path, mock_array)
    np.testing.assert_equal(io.load_distributed(path), mock_array)


@pytest.mark.parametrize(
    "layout",
    [
        {"domain_shape": (4, 4)},
        {"domain_shape": (4, 4), "owned_shape": (4,), "owned_offset": (0, 0)},
        {"domain_shape": (4, 4), "owned_shape": (0, 4), "owned_offset": (0, 0)},
        {"domain_shape": (4, 4), "owned_shape": (4, 4), "owned_offset": (-1, 0)},
        {"domain_shape": (4, 4), "owned_shape": (4, 4), "owned_offset": (1, 6)},
    ],
    ids=["incomplete", "ndim_mismatch", "owns_nothing", "starts_outside", "ends_outside"],
)
def test_a_layout_that_is_no_decomposition_is_refused(layout, comm_world):
    with pytest.raises(ValueError):
        NpyIO(**layout, communicator=comm_world)


@pytest.mark.parametrize("verb", ["load_singular", "load_replicated", "load_distributed"])
def test_a_missing_file_is_refused(mpi_tmp_path, verb, io):
    with pytest.raises(FileNotFoundError):
        getattr(io, verb)(mpi_tmp_path / "absent.npy")


@pytest.mark.parametrize(
    ("verb", "broken_on_root"),
    [
        ("load_distributed", True),
        ("load_distributed", False),
        ("save_distributed", True),
        ("save_distributed", False),
        ("load_singular", True),
        ("load_replicated", True),
        ("load_replicated", False),
    ],
)
def test_a_path_one_rank_cannot_reach_raises_on_every_rank(
    verb, broken_on_root, mpi_tmp_path, decomposed_grid, mock_field, mock_array, io, comm_world
):
    local = decomposed_grid.get_local(mock_field)
    io.save_distributed(mpi_tmp_path / "field.npy", local)
    io.save_singular(mpi_tmp_path / "array.npy", mock_array)
    io.barrier()

    absent = mpi_tmp_path / "absent"
    reachable = mpi_tmp_path / ("field.npy" if "distributed" in verb else "array.npy")
    is_broken = (comm_world.rank == 0) == broken_on_root
    path = (absent / reachable.name) if is_broken else reachable

    call = getattr(io, verb)
    args = (path, local) if verb == "save_distributed" else (path,)
    if broken_on_root or comm_world.Get_size() > 1:
        with pytest.raises(FileNotFoundError):
            call(*args)
    else:
        call(*args)


# =============================================================================
# NpyBack


@pytest.fixture
def bare_quantities(mpi_tmp_path, io):
    return QuantityFront(NpyBack(mpi_tmp_path / "data", io, decomposed=("x", "y")))


@pytest.fixture
def quantities(bare_quantities, decomposed_grid):
    nb_x, nb_y = decomposed_grid.nb_domain_grid_pts
    length = bare_quantities.define("L")
    bare_quantities.save_value("L", 1.0)
    bare_quantities.define("x", unit=length, is_basis=True)
    bare_quantities.define("y", unit=length, is_basis=True)
    bare_quantities.save_value("x", np.arange(nb_x, dtype=float))
    bare_quantities.save_value("y", np.arange(nb_y, dtype=float))
    bare_quantities.define("step", is_basis=True)
    bare_quantities.save_value("step", np.arange(3))
    return bare_quantities


def test_a_value_round_trips_at_the_point_it_was_saved_at(quantities, decomposed_grid, mock_field, comm_world):
    quantities.define("gap", frame=("step", "x", "y"))
    local = decomposed_grid.get_local(mock_field)
    quantities.save_value("gap", local, at={"step": 1})
    assert_all_array_equal(comm_world, quantities.load_value("gap", at={"step": 1}), local)


def test_a_step_left_unwritten_reads_as_nan(quantities):
    quantities.define("pressure", frame=("step",))
    quantities.save_value("pressure", 0.5, at={"step": 2})
    np.testing.assert_equal(quantities.load_value("pressure"), [np.nan, np.nan, 0.5])


def test_a_reopened_directory_gives_back_the_quantities_it_was_given(mpi_tmp_path, quantities, io):
    quantities.define("gap", unit=quantities["L"] ** 2, frame=("step", "x", "y"))
    reopened = QuantityFront(NpyBack(mpi_tmp_path / "data", io, decomposed=("x", "y")))
    assert sorted(reopened) == sorted(quantities)
    assert all(reopened[name] == quantities[name] for name in quantities)


@pytest.mark.parametrize("exponent", [2, -1, 0.5])
def test_a_unit_comes_back_as_the_kind_of_number_it_was_defined_with(mpi_tmp_path, quantities, io, exponent):
    unit = quantities["L"] ** exponent
    quantities.define("derived", unit=unit)
    reopened = QuantityFront(NpyBack(mpi_tmp_path / "data", io, decomposed=("x", "y")))
    assert reopened["derived"].unit == unit
    assert type(reopened["derived"].unit["L"]) is type(exponent)


def test_an_exponent_the_back_cannot_hold_is_refused(quantities):
    from fractions import Fraction

    with pytest.raises(NpyBackError):
        quantities.define("derived", unit=quantities["L"] ** Fraction(1, 3))


def test_a_decomposed_basis_takes_no_point(quantities, decomposed_grid, mock_field, comm_world):
    if comm_world.Get_size() == 1:
        pytest.skip("nothing is decomposed on one rank")
    quantities.define("gap", frame=("step", "x", "y"))
    quantities.save_value("gap", decomposed_grid.get_local(mock_field), at={"step": 0})
    with pytest.raises(NpyBackError):
        quantities.load_value("gap", at={"step": 0, "x": 0.0})


def test_a_frame_putting_a_decomposed_basis_before_another_is_refused(quantities):
    with pytest.raises(NpyBackError):
        quantities.define("gap", frame=("x", "step"))


def test_a_value_of_the_wrong_number_of_dimensions_is_refused(quantities):
    quantities.define("gap", frame=("step", "x", "y"))
    with pytest.raises(QuantityError):
        quantities.save_value("gap", 1.0, at={"step": 0})


def test_a_value_never_saved_is_refused(quantities):
    quantities.define("gap", frame=("step", "x", "y"))
    with pytest.raises(QuantityError):
        quantities.load_value("gap", at={"step": 0})


@pytest.mark.parametrize("decomposed", [("y", "x"), ()])
def test_a_back_over_a_decomposing_io_needs_its_bases_named_in_order(mpi_tmp_path, io, decomposed, comm_world):
    if not decomposed and comm_world.Get_size() == 1:
        pytest.skip("nothing is decomposed on one rank")
    with pytest.raises(NpyBackError):
        NpyBack(mpi_tmp_path / "other", io, decomposed=decomposed)
