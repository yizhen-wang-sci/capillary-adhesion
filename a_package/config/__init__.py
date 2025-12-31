"""
Configuration module for TOML-based simulation parameters.

Provides:
- Schema dataclasses for typed configuration
- TOML loading and saving
"""

from .schema import Config

from .loader import load_config, save_config
