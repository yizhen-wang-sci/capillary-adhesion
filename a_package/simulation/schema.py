import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class Term(StrEnum):
    upper_solid = "upper"
    lower_solid = "lower"
    separation = "separation"
    pressure = "pressure"
    volume = "volume"
    gap = "gap"
    phase = "phase"
    energy = "energy"
    perimeter = "perimeter"
    phase_init = "phase_init"
    pressure_init = "pressure_init"
