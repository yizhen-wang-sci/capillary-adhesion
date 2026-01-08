"""
Configuration module for TOML-based simulation parameters.

Provides:
- Schema dataclasses for typed configuration
- TOML loading and saving
- Sweep expansion utilities
"""

from .schema import Config
from .loader import load_config, save_config
from .sweep import expand_sweep_spec, count_sweep_combinations, expand_configs
