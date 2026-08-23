"""Tests for the run directory layout."""

from pathlib import Path

import pytest

from a_package.dataset.dirs import RunDir, SourceDir

# =============================================================================
# Directories


def test_dir_refuses_an_existing_path_when_not_allowed(tmp_path):
    RunDir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        RunDir(tmp_path / "run", exist_ok=False)


# =============================================================================
# SourceDir


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "simulate.py").write_text("print('hi')\n")
    (path / "config.toml").write_text("a = 1\n")
    (path / "notes.md").write_text("skip me\n")
    (path / "sub").mkdir()
    (path / "sub" / "nested.py").write_text("pass\n")
    return SourceDir(path)


@pytest.mark.parametrize(
    ("suffixes", "copied"),
    [
        ({}, ["simulate.py"]),
        ({"include_suffixes": (".py", ".toml")}, ["config.toml", "simulate.py"]),
    ],
    ids=["default", "given"],
)
def test_snapshot_copies_the_top_level_files_of_the_given_suffixes(source, suffixes, copied):
    snapshot = Path(source.snapshot(tag="baseline", **suffixes))
    assert sorted(p.name for p in snapshot.iterdir()) == copied


def test_snapshot_is_a_sibling_of_the_sources_named_by_tag_and_index(source, tmp_path):
    first = source.snapshot(tag="baseline")
    second = source.snapshot(tag="baseline")
    third = source.snapshot(tag="other")
    assert [Path(p).parent for p in (first, second, third)] == [tmp_path] * 3
    assert [p.name for p in (first, second, third)] == ["baseline--01", "baseline--02", "other--01"]


# =============================================================================
# RunDir


def test_metadata_read_back_is_what_was_added(tmp_path):
    run = RunDir(tmp_path / "run")
    assert run.read_metadata() == {}
    run.add_metadata({"git-hash": "abc", "created": "yesterday"})
    run.add_metadata({"git-hash": "def"})
    assert run.read_metadata() == {"git-hash": "def", "created": "yesterday"}


def test_metadata_that_does_not_parse_is_refused(tmp_path):
    run = RunDir(tmp_path / "run")
    (run / "metadata.json").write_text("not json")
    with pytest.raises(ValueError):
        run.read_metadata()


def test_a_new_record_is_found_by_its_parameters(tmp_path):
    run = RunDir(tmp_path / "run")
    first = run.new_record(theta=30, volume=1)
    second = run.new_record(theta=60, volume=1)

    assert first.name == "theta=30--volume=1"
    assert Path(first).is_dir()
    assert sorted(r.name for r in run.find_records()) == sorted([first.name, second.name])
    assert [r.name for r in run.find_records(theta="30")] == [first.name]
    assert run.get_record(theta="30").name == first.name


def test_a_declared_naming_survives_reopening_and_queries_by_the_declared_type(tmp_path):
    run = RunDir(tmp_path / "run")
    run.declare_record_naming({"theta": float, "steps": int})
    record = run.new_record(theta=30, steps=4)

    reopened = RunDir(tmp_path / "run")
    assert reopened.record_naming.parse(record.name) == {"theta": 30.0, "steps": 4}
    assert [r.name for r in reopened.find_records(theta=30.0)] == [record.name]


def test_declare_record_naming_refuses_a_type_it_cannot_write_down(tmp_path):
    run = RunDir(tmp_path / "run")
    with pytest.raises(ValueError):
        run.declare_record_naming({"theta": complex})


def test_new_record_refuses_a_taken_combination(tmp_path):
    run = RunDir(tmp_path / "run")
    run.new_record(theta=30)
    with pytest.raises(FileExistsError):
        run.new_record(theta=30)


def test_find_records_skips_directories_of_another_convention(tmp_path):
    run = RunDir(tmp_path / "run")
    run.new_record(theta=30)
    (tmp_path / "run" / "unrelated").mkdir()
    assert [r.name for r in run.find_records()] == ["theta=30"]


@pytest.mark.parametrize(
    ("records", "error"),
    [
        ([], FileNotFoundError),
        ([{"theta": 30, "volume": 1}, {"theta": 30, "volume": 2}], LookupError),
    ],
)
def test_get_record_raises_unless_exactly_one_matches(records, error, tmp_path):
    run = RunDir(tmp_path / "run")
    for fields in records:
        run.new_record(**fields)
    with pytest.raises(error):
        run.get_record(theta="30")
