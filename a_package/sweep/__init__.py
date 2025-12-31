"""
Parameter sweep for parametric exploration.

Provides:
- Sweep expansion: config with sweeps → multiple configs
- Sweep execution: run simulations for each config
"""

from .sweep import expand_sweeps, count_sweep_combinations
