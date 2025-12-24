"""
Configuration module for TOML-based simulation parameters.

Provides:
- Schema dataclasses for typed configuration
- TOML loading and saving
- Parameter sweep expansion for parametric exploration
"""

from .schema import Config

from .loader import load_config, save_config

from .sweep import expand_sweeps, count_sweep_combinations
