"""
Unit conversion between dimensionless and physical quantities.
"""

import dataclasses as dc


@dc.dataclass
class UnitConversion:
    """Conversion between dimensionless and physical quantities.

    scale: numeric multiplier (dimensionless -> physical)
    base_unit: string label for the base physical unit (e.g., 'm', 's', 'kg')
    exponent: power of the base unit (default 1)
    """

    scale: float = 1.0
    base_unit: str = ""
    exponent: int = 1

    def __pow__(self, n: int):
        """Return new UnitConversion with scale**n and exponent*n."""
        return UnitConversion(
            scale=self.scale**n,
            base_unit=self.base_unit,
            exponent=self.exponent * n,
        )

    def to_physical(self, value, exponent: int = 1):
        """Convert dimensionless value to physical value."""
        return value * self.scale**exponent

    def to_dimensionless(self, value, exponent: int = 1):
        """Convert physical value to dimensionless value."""
        return value / self.scale**exponent

    @property
    def unit(self) -> str:
        """Formatted unit string."""
        if not self.base_unit:
            return ""
        prefix = "/" if self.exponent < 0 else ""
        suffix = f"^{abs(self.exponent)}" if abs(self.exponent) > 1 else ""
        return f"{prefix}{self.base_unit}{suffix}"
