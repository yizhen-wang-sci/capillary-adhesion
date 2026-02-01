"""
Unit conversion between dimensionless and physical quantities.
"""

import dataclasses as dc


@dc.dataclass
class UnitConversion:
    """Conversion between dimensionless and physical quantities.

    scale: numeric multiplier (dimensionless -> physical)
    unit: string label for the physical unit (e.g., 'm', 's', 'kg')
    """

    scale: float = 1.0
    unit: str = ''

    def to_physical(self, value: float) -> float:
        """Convert dimensionless value to physical value."""
        return value * self.scale

    def to_dimensionless(self, value: float) -> float:
        """Convert physical value to dimensionless value."""
        return value / self.scale

    def format(self, value: float, fmt: str = '.2e') -> str:
        """Format dimensionless value as physical value with unit."""
        physical = self.to_physical(value)
        return f"{physical:{fmt}}{self.unit}"
