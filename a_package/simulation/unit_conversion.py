"""Unit conversion between dimensionless and physical quantities."""

import dataclasses as dc


@dc.dataclass
class UnitConversion:
    """Conversion between dimensionless and physical quantities."""

    scale: float = 1.0
    """Multiplier taking a dimensionless value to a physical one."""
    base_unit: str = ""
    """Label of the base physical unit, e.g. 'm', 's', 'kg'."""
    exponent: int = 1
    """Power the base unit is raised to."""

    def __pow__(self, n: int):
        """Return new UnitConversion with scale**n and exponent*n.

        Args:
            n: The power to raise the conversion to.
        """
        return UnitConversion(
            scale=self.scale**n,
            base_unit=self.base_unit,
            exponent=self.exponent * n,
        )

    def to_physical(self, value, exponent: int = 1):
        """Convert dimensionless value to physical value.

        Args:
            value: The dimensionless value.
            exponent: The power the scale is raised to.
        """
        return value * self.scale**exponent

    def to_dimensionless(self, value, exponent: int = 1):
        """Convert physical value to dimensionless value.

        Args:
            value: The physical value.
            exponent: The power the scale is raised to.
        """
        return value / self.scale**exponent

    @property
    def unit(self) -> str:
        """Formatted unit string, empty when there is no base unit."""
        if not self.base_unit:
            return ""
        prefix = "/" if self.exponent < 0 else ""
        suffix = f"^{abs(self.exponent)}" if abs(self.exponent) > 1 else ""
        return f"{prefix}{self.base_unit}{suffix}"
