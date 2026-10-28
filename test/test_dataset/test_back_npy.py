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


def test_a_decomposed_field_round_trips_through_one_file(mpi_tmp_path, decomposed_grid, mock_field, io, comm_world):
    local = decomposed_grid.get_local(mock_field)
    io.write_data(mpi_tmp_path / "field.npy", local, decomposed=True)
    assert_all_array_equal(comm_world, io.read_data(mpi_tmp_path / "field.npy", decomposed=True), local)


def test_a_field_written_decomposed_reads_back_whole(mpi_tmp_path, decomposed_grid, mock_field, io):
    io.write_data(mpi_tmp_path / "field.npy", decomposed_grid.get_local(mock_field), decomposed=True)
    np.testing.assert_equal(NpyIO().read_data(mpi_tmp_path / "field.npy"), mock_field)


def test_an_undecomposed_array_round_trips_to_every_rank(mpi_tmp_path, mock_array, io, comm_world):
    io.write_data(mpi_tmp_path / "array.npy", mock_array)
    assert_all_array_equal(comm_world, io.read_data(mpi_tmp_path / "array.npy"), mock_array)


def test_a_text_file_round_trips_to_every_rank(mpi_tmp_path, io):
    io.write_text(mpi_tmp_path / "note.txt", "the whole content")
    assert io.read_text(mpi_tmp_path / "note.txt") == "the whole content"


def test_an_unset_layout_gives_each_rank_a_file_of_its_own(mpi_tmp_path, comm_world):
    io = NpyIO()
    assert not io.is_decomposed()
    own = np.full(4, comm_world.rank, dtype=float)
    path = mpi_tmp_path / f"own-{comm_world.rank}.npy"
    io.write_data(path, own)
    np.testing.assert_equal(io.read_data(path), own)


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


def test_an_error_seen_in_one_rank_is_raised_everywhere(io, comm_world):
    if comm_world.size == 1:
        pytest.skip("nothing diverges on one rank")

    with pytest.raises(AssertionError, match="1"), io.agreeing_on_error():
        if comm_world.rank == 1:
            raise AssertionError("1")


@pytest.mark.parametrize("decomposed", [True, False])
def test_a_missing_file_is_refused(mpi_tmp_path, decomposed, io):
    with pytest.raises(FileNotFoundError):
        io.read_data(mpi_tmp_path / "absent.npy", decomposed=decomposed)


@pytest.mark.parametrize("verb", ["read_data", "write_data"])
def test_a_path_one_rank_cannot_reach_is_refused_on_every_rank(
    verb, mpi_tmp_path, decomposed_grid, mock_field, io, comm_world
):
    if comm_world.size == 1:
        pytest.skip("nothing diverges on one rank")

    reachable = mpi_tmp_path / "reachable"
    io.make_dir(reachable)
    data = decomposed_grid.get_local(mock_field)
    io.write_data(reachable / "field.py", data, decomposed=True)

    method = getattr(io, verb)
    path = mpi_tmp_path / "absent" / "field.py" if comm_world.rank == 1 else reachable / "field.py"
    args = (path, data) if verb == "write_data" else (path,)
    with pytest.raises(FileNotFoundError):
        method(*args, decomposed=True)


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
    if comm_world.size == 1:
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
    if not decomposed and comm_world.size == 1:
        pytest.skip("nothing is decomposed on one rank")
    with pytest.raises(NpyBackError):
        NpyBack(mpi_tmp_path / "other", io, decomposed=decomposed)
