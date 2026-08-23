"""Physical quantity represented by the numeric data kept in storage."""

import dataclasses as dc
import itertools
import operator
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class QuantityError(Exception):
    """A unified error name for quantity."""


@dc.dataclass(frozen=True)
class Quantity:
    """Gives numeric values their meaning, unit and frame of bases."""

    name: str
    """A unique name. Should reveal its meaning. Used as the identifier in QuantityFront."""

    unit: "str | Scale | None" = None
    """A unit literal, or a reference scale composed of constant quantities.

    Note:
        Empty str for unitless. None for not specified.
    """

    frame: "tuple[Quantity, ...]" = ()
    """Spatial coordinates, temporal axis, state space, and anything similar. Ordered.

    Note:
       Each basis in frame defines one dimension in value of the quantity. Only quantities marked
       as basis (see QuantityFront.define) can be used in frame.
    """

    def __pow__(self, exponent):
        """Raise it to a power, so it reads as the unit of something else."""
        return Scale(self) ** exponent

    def __mul__(self, other):
        """Multiply it by another, so it reads as the unit of something else."""
        return Scale(self) * other

    __rmul__ = __mul__


BASIS = Quantity("_")
"""A reserved quantity, used internally to mark a quantity as basis.

Note:
    This exists to distinguish between a constant quantity and a basis quantity.
"""


class Scale:
    """A reference scale composed of constant quantities.

    Note:
        Use * and ** operators of a quantity to build a scale.
    """

    def __init__(self, composition: Quantity | Mapping[str, int | float]):
        """Convert a quantity to a scale and verify all quantities involved are constant.

        Args:
            composition: A single quantity, or the combination of powers of several quantities,
                keyed by name.

        Raises:
            QuantityError: If the scale measures against the basis marker, or against a quantity
                that is not constant.
        """
        if isinstance(composition, Quantity):
            # Reject non-constant quantities. Only what has one value can be measured against.
            if len(composition.frame):
                raise QuantityError(f"{composition.name} is not constant, so it cannot be used as a unit.")
            self._composition = {composition.name: 1}
        else:
            self._composition = self._normalise(composition)

        if BASIS.name in self._composition:
            raise QuantityError(f"{BASIS.name} is reserved. It marks a quantity as basis.")

    @staticmethod
    def _normalise(composition: Mapping[str, int | float]) -> dict[str, int | float]:
        """Drop the name when powers cancel out; reduce to integer if exponent is a whole-number.

        Args:
            composition: The power of each quantity, keyed by name.

        Returns:
            A clean mapping from quantity name to exponent.

        Raises:
            QuantityError: If an exponent is not finite.
        """
        normalised = {}
        for name in composition:
            exponent = composition[name]
            if exponent == 0:
                continue
            try:
                integer_part = int(exponent)
            except (OverflowError, ValueError) as err:
                # Rule out Inf and NaN
                raise QuantityError(f"{exponent} cannot be an exponent.") from err
            normalised[name] = integer_part if exponent == integer_part else exponent
        return normalised

    # =========================================================================
    # Turn itself into a mapping.

    def __contains__(self, item):
        """Whether a quantity of that name composes it."""
        return item in self._composition

    def __getitem__(self, item):
        """The exponent a composing quantity is raised to."""
        return self._composition[item]

    def __iter__(self):
        """Iterate the composing quantities by name."""
        return iter(self._composition)

    def __len__(self):
        """Number of composing quantities."""
        return len(self._composition)

    # =========================================================================
    # Keep Quantity a frozen dataclass.

    def __eq__(self, value):
        """Whether two reference scales are equal."""
        if not isinstance(value, (Quantity, Scale)):
            return NotImplemented
        return self._composition == Scale(value)._composition

    def __hash__(self):
        """The uniqueness based on the powers of quantities."""
        return hash(tuple(sorted(self._composition.items())))

    # =========================================================================
    # Derive quantities and scales.

    def __pow__(self, exponent):
        """Power by raise every composing quantity to the power of the given exponent."""
        return Scale({name: each * exponent for name, each in self._composition.items()})

    def __mul__(self, other):
        """Multiply by summing up exponents of quantities."""
        if not isinstance(other, (Quantity, Scale)):
            return NotImplemented

        # A dict only to add up with, never to keep
        summed = dict(self._composition)
        for name, exponent in Scale(other)._composition.items():
            summed[name] = summed.get(name, 0) + exponent
        return Scale(summed)

    __rmul__ = __mul__


class QuantityBack(Protocol):
    """The part interacting with the storage that keeps defined quantities and values."""

    def get_all_quantities(self) -> dict[str, Quantity]:
        """Load all stored quantities.

        Returns:
            Keyed by name.

        Raises:
            QuantityError
        """
        raise NotImplementedError

    def new_quantity(self, new: Quantity):
        """Save the new quantity.

        Args:
            new: The quantity.

        Raises:
            QuantityError
        """
        raise NotImplementedError

    def save_value(self, quantity: Quantity, address: tuple[int | None, ...], value: Any):
        """Save the value of a quantity at an address in its frame.

        Args:
            quantity: The quantity.
            address: One entry per basis in the frame, in frame order. An index takes one
                point of that basis; None takes the whole of it.
            value: The value at that address. In parallel, it may be just a partition of the
                value if certain bases are decomposed.

        Raises:
            QuantityError
        """
        raise NotImplementedError

    def load_value(self, quantity: Quantity, address: tuple[int | None, ...]) -> Any:
        """Load the value of a quantity at an address in its frame.

        Args:
            quantity: The quantity.
            address: One entry per basis in the frame, in frame order. An index takes one
                point of that basis; None takes the whole of it.

        Returns:
            The value at that address. In parallel, it may be just a partition of the value
            if certain bases are decomposed.

        Raises:
            QuantityError
        """
        raise NotImplementedError


class QuantityFront:
    """The part interacting with simulation and visualisation scripts."""

    def __init__(self, back: QuantityBack):
        """Load the quantities saved in the back."""
        self._back = back
        self._saved = back.get_all_quantities()

    # =========================================================================
    # Make it work like a read-only dict.

    def __contains__(self, name: str):
        """Whether a quantity with that name is declared."""
        return name in self._saved

    def __getitem__(self, name: str):
        """Get a declared quantity."""
        try:
            return self._saved[name]
        except KeyError:
            raise QuantityError(f"{name} is not defined.") from None

    def __iter__(self):
        """Iterate all the quantities by its name."""
        return iter(self._saved)

    def __len__(self):
        """Number of quantities saved."""
        return len(self._saved)

    # =========================================================================
    # Define quantities and address their values.

    def define(
        self,
        name: str,
        unit: str | Quantity | Scale | None = None,
        frame: Sequence[str | Quantity] = (),
        is_basis: bool = False,
    ) -> Quantity:
        """Define a new quantity. Also returns it for convenience.

        Args:
            name: A unique name to refer to the quantity by.
            unit: What one unit of it is measured against.
            frame: The quantities spanning it, by name or by value. Each must be a basis.
            is_basis: Whether it is a basis quantity that other quantities are addressed along.
                Basis quantity is assigned BASIS as its single frame entry.

        Returns:
            The definition, which is also what `self[name]` gives back.

        Raises:
            QuantityError: If the name is taken or reserved, the unit is not a scale of
                constant quantities, a member of the frame is not defined or is not a basis,
                or a basis is given a frame.
        """
        if not name.isidentifier():
            raise QuantityError(f"{name} is not a valid identifier.")
        # Protect the name reserved for BASIS
        if name == BASIS.name:
            raise QuantityError(f"{name} is reserved for the marker of basis.")

        unit = self._normalise_unit(unit)

        if is_basis:
            # A basis quantity is marked with "BASIS" in frame
            if len(frame):
                raise QuantityError(f"{name} is a basis quantity, so it takes no frame.")
            frame = (BASIS,)
        else:
            frame = tuple(self._to_quantity(basis) for basis in frame)
            # Only basis quantities can be used in frame
            for basis in frame:
                if basis.frame != (BASIS,):
                    raise QuantityError(f"{basis.name} is not a basis quantity, so cannot be used in frame.")

        new = Quantity(name, unit, frame)

        # Check duplicated name
        duplicated = self._saved.get(name)
        if duplicated is not None and duplicated != new:
            raise QuantityError(f"{name} is a duplicated name. It already has {duplicated}")

        # Create the new quantity
        self._back.new_quantity(new)
        self._saved[name] = new
        return new

    def _normalise_unit(self, unit: str | Quantity | Scale | None):
        """Convert a quantity to scale. Collapse to an empty str if the scale is unitless.

        Raises:
            QuantityError: If the scale involves undefined quantities.
        """
        if unit is None or isinstance(unit, str):
            return unit

        scale = Scale(unit)
        if not len(scale):
            return ""

        unknown = sorted(set(scale) - set(self._saved))
        if len(unknown):
            raise QuantityError(f"{unknown} are not defined, so they cannot be measured against.")

        return scale

    def _to_quantity(self, name: str | Quantity):
        """Turn a name to the corresponding quantity."""
        if isinstance(name, Quantity):
            return name
        return self[name]

    def save_value(self, name: str, value, at: Mapping[str, int | float] | None = None):
        """Save the value of a quantity at specified point(s) in frame."""
        # Check undefine name
        if name not in self._saved:
            raise QuantityError(f"No quantity defined with name {name}. Define it first.")
        quantity = self._saved[name]
        # Specific rules for basis quantities
        if quantity.frame == (BASIS,):
            if not _is_increasing(value):
                raise QuantityError(f"{name} is a basis quantity, so value must increase monotonically.")
            if at is not None:
                raise QuantityError(f"{name} is a basis quantity, so its own values take no point.")
        self._back.save_value(quantity, self._address(quantity, at), value)

    def load_value(self, name: str, at: Mapping[str, int | float] | None = None):
        """Load the value of a quantity at specified point(s) in frame."""
        # Check undefine name
        if name not in self._saved:
            raise QuantityError(f"No quantity defined with name {name}.")
        quantity = self._saved[name]
        # Specific rules for basis quantities
        if quantity.frame == (BASIS,) and at is not None:
            raise QuantityError(f"{name} is a basis quantity, so its own values take no point.")
        return self._back.load_value(quantity, self._address(quantity, at))

    def _address(self, quantity: Quantity, at: Mapping[str, int | float] | None):
        """Get the complete address in quantity frame."""
        at = dict(at or {})

        # Check any invalid name
        basis_names = {q.name for q in quantity.frame}
        unknown_names = set(at) - basis_names
        if unknown_names:
            raise QuantityError(f"{unknown_names} do not span {quantity.name}; {basis_names} do")

        address = []
        for basis in quantity.frame:
            point = at.get(basis.name)
            if point is None:
                address.append(None)
                continue
            # It is guaranteed at `define` that frame entries are basis quantities, and at
            # `save_value` that their values increase, so a point sits at one place or none
            sampled = tuple(self._back.load_value(basis, (None,)))
            try:
                address.append(sampled.index(point))
            except ValueError:
                raise QuantityError(f"{basis.name} is specified at an inaddressible point: {point}.") from None

        return tuple(address)


def _is_increasing(value: Any) -> bool:
    """Whether the value is monotonically increasing.

    Args:
        value: Value to check.

    Returns:
        False as soon as one entry is not greater than its predecessor.
    """
    return all(map(operator.lt, value, itertools.islice(iter(value), 1, None)))
