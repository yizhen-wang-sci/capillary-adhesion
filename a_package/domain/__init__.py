"""The discrete space, and the numerical problem posed on it."""

from .grid import Grid, factorize_closest
from .field import Field, adapt_shape, field_component_ax, field_sub_pt_ax, field_element_axs
from .io import NpyIO
from .fem import FirstOrderElement
from .quadrature import Quadrature, NodalQuadrature, CentroidQuadrature, ThreePtQuadrature
from .optimizer import Optimizer, Problem, OptimizerResult, AugmentedLagrangian, ProjectedLbfgs, BoundedLbfgs
