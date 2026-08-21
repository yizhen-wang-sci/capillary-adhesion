"""Tests for the run directory layout and its naming conventions."""

import json
from pathlib import Path

import pytest

from a_package.simulation.dirs import ParameterCombo, RunDir, SourceDir, TaggedIndex

# =============================================================================
# TaggedIndex
# =============================================================================


@pytest.mark.parametrize(("tag", "normalized"), [("baseline", "baseline"), (" Base Line ", "base-line")])
def test_tagged_index_round_trips_a_tag_and_an_index(tag, normalized):
    naming = TaggedIndex()
    assert naming.parse(naming.format(tag=tag, index=2)) == {"tag": normalized, "index": 2}


# =============================================================================
# ParameterCombo
# =============================================================================


def test_parameter_combo_round_trips_typed_values():
    naming = ParameterCombo(types={"theta": float, "steps": int})
    fields = {"theta": 30.0, "steps": 4}
    assert naming.parse(naming.format(**fields)) == fields


def test_parameter_combo_round_trips_as_strings_when_untyped():
    typed = ParameterCombo(types={"theta": float, "steps": int})
    assert ParameterCombo().parse(typed.format(theta=30.0, steps=4)) == {"theta": "30.0", "steps": "4"}


@pytest.mark.parametrize("name", ["theta30", "theta=abc"])
def test_parameter_combo_parse_rejects_a_name_it_cannot_decode(name):
    assert ParameterCombo(types={"theta": float}).parse(name) is None


def test_parameter_combo_derive_next_refuses_a_taken_combination_whatever_the_value_type():
    naming = ParameterCombo(types={"theta": float})
    with pytest.raises(FileExistsError):
        naming.derive_next(["theta=30.0"], theta="30")


# =============================================================================
# Directories
# =============================================================================


def test_dir_refuses_an_existing_path_when_not_allowed(tmp_path):
    RunDir(tmp_path / "run")
    with pytest.raises(FileExistsError):
        RunDir(tmp_path / "run", exist_ok=False)


# =============================================================================
# SourceDir
# =============================================================================


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


def test_snapshot_copies_the_top_level_sources_only(source):
    snapshot = Path(source.snapshot(tag="baseline"))
    assert sorted(p.name for p in snapshot.iterdir()) == ["config.toml", "simulate.py"]


def test_snapshot_names_itself_by_tag_and_index(source):
    first = source.snapshot(tag="baseline")
    second = source.snapshot(tag="baseline")
    third = source.snapshot(tag="other")
    assert first.name == "baseline--01"
    assert second.name == "baseline--02"
    assert third.name == "other--01"


# =============================================================================
# RunDir
# =============================================================================


def test_add_metadata_merges_and_overrides(tmp_path):
    run = RunDir(tmp_path / "run")
    run.add_metadata({"git-hash": "abc", "created": "yesterday"})
    run.add_metadata({"git-hash": "def"})
    assert json.loads((run / "metadata.json").read_text()) == {"git-hash": "def", "created": "yesterday"}


def test_add_metadata_replaces_an_unparsable_file(tmp_path):
    run = RunDir(tmp_path / "run")
    (run / "metadata.json").write_text("not json")
    run.add_metadata({"git-hash": "abc"})
    assert json.loads((run / "metadata.json").read_text()) == {"git-hash": "abc"}


def test_a_new_record_is_found_by_its_parameters(tmp_path):
    run = RunDir(tmp_path / "run")
    first = run.new_record(theta=30, volume=1)
    second = run.new_record(theta=60, volume=1)

    assert first.name == "theta=30--volume=1"
    assert Path(first).is_dir()
    assert sorted(r.name for r in run.find_records()) == sorted([first.name, second.name])
    assert [r.name for r in run.find_records(theta="30")] == [first.name]
    assert run.get_record(theta="30").name == first.name


def test_new_record_refuses_a_taken_combination(tmp_path):
    run = RunDir(tmp_path / "run")
    run.new_record(theta=30)
    with pytest.raises(FileExistsError):
        run.new_record(theta=30)


def test_new_record_follows_the_given_naming(tmp_path):
    run = RunDir(tmp_path / "run", record_naming=TaggedIndex())
    assert run.new_record(tag="trial").name == "trial--01"
    assert run.new_record(tag="trial").name == "trial--02"


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
