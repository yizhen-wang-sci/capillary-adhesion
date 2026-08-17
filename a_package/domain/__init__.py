"""The discrete space, and the numerical problem posed on it."""

from .fem import FirstOrderElement
from .field import Field, adapt_shape, field_component_ax, field_element_axs, field_sub_pt_ax
from .grid import Grid, factorize_closest
from .optimizer import BoundedLbfgs, Optimizer, OptimizerResult, Problem, ProjectedLbfgs
from .quadrature import CentroidQuadrature, NodalQuadrature, Quadrature, ThreePtQuadrature
