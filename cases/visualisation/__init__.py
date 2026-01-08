"""
Visualization utilities for simulation results.

Modules:
- primitives: Low-level drawing functions (array-based, no IO dependency)
- plots: High-level plotting functions (SimulationIO-dependent)
- animations: Animation creation
- preview: Config preview before running
- sweep_data: Data extraction from parameter sweeps
"""

from .primitives import (
    draw_field_2d,
    draw_masked_field_2d,
    draw_cross_section,
    draw_evolution_curve,
    latexify_plot,
    hide_border,
    hide_ticks,
)
from .plots import (
    plot_cross_section_sketch,
    plot_height_topography,
    plot_gap_topography,
    plot_contact_topography,
    plot_droplet_topography,
    plot_phase_field_topography,
    plot_combined_topography,
    plot_gibbs_free_energy,
    plot_normal_force,
    plot_pressure,
)
from .animations import (
    create_overview_animation,
    animate_droplet_evolution,
    animate_droplet_evolution_with_curves,
)
from .preview import preview_surface_and_gap
from .sweep_data import (
    extract_from_sweep,
    get_config_value,
    get_trajectory_value,
    collect_sweep_data,
)
