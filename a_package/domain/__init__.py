"""
Domain module: spatial foundation, discretization, and optimization.

Provides:
- Grid: spatial discretization (will support domain decomposition)
- Field: data living on the grid
- NpyIO: parallel-aware persistence (future)
- FEM: finite element interpolation
- Quadrature: integration rules
- Optimizer: numerical optimization (will support parallelization)
"""

from .grid import Grid, factorize_closest
from .field import Field, adapt_shape, field_component_ax, field_sub_pt_ax, field_element_axs
from .io import NpyIO
from .fem import FirstOrderElement
from .quadrature import Quadrature, centroid_quadrature
from .optimizer import Optimizer, Problem, AugmentedLagrangian, ProjectedLbfgs, BoundedLbfgs
