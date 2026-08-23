"""Tests for the physical meaning given to stored numeric values."""

import pytest

from a_package.dataset.quantity import BASIS, Quantity, QuantityBack, QuantityError, QuantityFront, Scale


class DictBack(QuantityBack):
    """A back that keeps quantities and values in memory."""

    def __init__(self):
        self.quantities = {}
        self.values = {}

    def get_all_quantities(self):
        return dict(self.quantities)

    def new_quantity(self, new):
        self.quantities[new.name] = new

    def save_value(self, quantity, address, value):
        self.values[(quantity.name, address)] = value

    def load_value(self, quantity, address):
        return self.values[(quantity.name, address)]


@pytest.fixture
def front():
    return QuantityFront(DictBack())


@pytest.fixture
def front_with_step(front):
    front.define("step", is_basis=True)
    front.save_value("step", [0, 1, 2])
    return front


# =============================================================================
# Scale


def test_a_scale_is_the_same_whichever_order_it_was_built_in(front):
    length = front.define("L")
    time = front.define("T")
    assert length * time == time * length
    assert hash(length * time) == hash(time * length)


def test_opposite_powers_of_one_quantity_leave_nothing_behind(front):
    length = front.define("L")
    assert len(length * length**-1) == 0
    assert front.define("ratio", unit=length * length**-1).unit == ""


@pytest.mark.parametrize(
    ("exponents", "composition"),
    [
        ((2, -1), {"L": 2, "T": -1}),
        ((0.5, -0.5), {"L": 0.5, "T": -0.5}),
        ((1, -3), {"L": 1, "T": -3}),
    ],
    ids=["whole", "fractional", "mixed"],
)
def test_a_scale_holds_the_exponent_each_quantity_was_raised_to(front, exponents, composition):
    length = front.define("L")
    time = front.define("T")
    scale = length ** exponents[0] * time ** exponents[1]
    assert {name: scale[name] for name in scale} == composition


def test_an_exponent_that_is_not_finite_is_refused(front):
    length = front.define("L")
    with pytest.raises(QuantityError):
        Scale({"L": float("inf")})
    with pytest.raises(QuantityError):
        length ** float("nan")


@pytest.mark.parametrize("against", ["basis", "not_constant", "undefined"])
def test_only_a_defined_constant_quantity_can_be_a_unit(front_with_step, against):
    front_with_step.define("L")
    unit = {
        "basis": BASIS,
        "not_constant": front_with_step.define("gap", frame=("step",)),
        "undefined": Scale({"nowhere": 1}),
    }[against]
    with pytest.raises(QuantityError):
        front_with_step.define("derived", unit=unit)


# =============================================================================
# QuantityFront


def test_a_definition_is_what_comes_back_under_its_name(front):
    defined = front.define("L", unit="m")
    assert front["L"] == defined == Quantity("L", "m", ())
    assert "L" in front
    assert sorted(front) == ["L"]
    assert len(front) == 1


def test_a_value_comes_back_from_the_point_it_was_saved_at(front_with_step):
    front_with_step.define("pressure", frame=("step",))
    front_with_step.save_value("pressure", 0.25, at={"step": 1})
    assert front_with_step.load_value("pressure", at={"step": 1}) == 0.25


def test_defining_the_same_quantity_twice_over_is_accepted(front):
    front.define("L", unit="m")
    assert front.define("L", unit="m").unit == "m"


def test_defining_a_name_again_with_another_meaning_is_refused(front):
    front.define("L", unit="m")
    with pytest.raises(QuantityError):
        front.define("L", unit="mm")


@pytest.mark.parametrize("name", ["has space", "3rd", ""])
def test_a_name_that_is_not_a_valid_identifier_is_refused(front, name):
    with pytest.raises(QuantityError):
        front.define(name)


def test_the_name_reserved_for_the_basis_marker_is_refused(front):
    with pytest.raises(QuantityError):
        front.define(BASIS.name)


def test_a_frame_takes_only_a_basis_quantity(front_with_step):
    front_with_step.define("L")
    with pytest.raises(QuantityError):
        front_with_step.define("gap", frame=("L",))


def test_a_basis_quantity_takes_no_frame(front_with_step):
    with pytest.raises(QuantityError):
        front_with_step.define("other", is_basis=True, frame=("step",))


def test_the_values_of_a_basis_must_increase(front):
    front.define("step", is_basis=True)
    with pytest.raises(QuantityError):
        front.save_value("step", [0, 2, 1])


@pytest.mark.parametrize("verb", ["save_value", "load_value"])
def test_the_values_of_a_basis_itself_take_no_point(front, verb):
    front.define("step", is_basis=True)
    front.save_value("step", [0, 1, 2])
    args = ([0, 1, 2],) if verb == "save_value" else ()
    with pytest.raises(QuantityError):
        getattr(front, verb)("step", *args, at={"step": 1})


def test_a_quantity_never_defined_is_refused(front):
    with pytest.raises(QuantityError):
        front["nowhere"]
    with pytest.raises(QuantityError):
        front.save_value("nowhere", 1.0)
    with pytest.raises(QuantityError):
        front.load_value("nowhere")


def test_a_point_is_given_in_a_basis_that_spans_the_quantity(front_with_step):
    front_with_step.define("pressure", frame=("step",))
    with pytest.raises(QuantityError):
        front_with_step.save_value("pressure", 0.25, at={"nowhere": 0})


def test_a_point_no_basis_value_sits_at_is_refused(front_with_step):
    front_with_step.define("pressure", frame=("step",))
    with pytest.raises(QuantityError):
        front_with_step.load_value("pressure", at={"step": 99})
