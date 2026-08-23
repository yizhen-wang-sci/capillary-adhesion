"""Tests for how the directories are named."""

import pytest

from a_package.dataset._naming import ParameterCombo, TaggedIndex

# =============================================================================
# TaggedIndex


@pytest.mark.parametrize(("tag", "normalized"), [("baseline", "baseline"), (" Base Line ", "base-line")])
def test_tagged_index_round_trips_a_tag_and_an_index(tag, normalized):
    naming = TaggedIndex()
    assert naming.parse(naming.format(tag=tag, index=2)) == {"tag": normalized, "index": 2}


# =============================================================================
# ParameterCombo


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
