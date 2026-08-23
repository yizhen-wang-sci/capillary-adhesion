"""The minimisation solver."""

from a_package.domain import ProjectedLbfgs, BoundedLbfgs


def build_optimizer(config: dict, linear_eq_constraints: bool = True):
    """Build the solver.

    Recognised keys under ``[optimizer]``:
    - ``max_nb_iters`` (int, required)    -> ``max_inner_iter``
    - ``tol_gradient`` (float, required)  -> ``tol_gradient`` (gtol)
    - ``tol_step`` (float, optional)      -> ``tol_step`` (xtol); omit for the solver's own
      default. Ignored by ``BoundedLbfgs``, which has no step criterion.

    Args:
        config: The configuration.
        linear_eq_constraints: True for ``ProjectedLbfgs``, which has an equality constraint to
            project onto (fixed volume); False for ``BoundedLbfgs``, which has none (fixed
            pressure).
    """
    section = config["optimizer"]
    args = dict(
        max_inner_iter=section["max_nb_iters"],
        tol_gradient=section["tol_gradient"],
    )
    if linear_eq_constraints:
        if "tol_step" in section:
            args["tol_step"] = section["tol_step"]
        return ProjectedLbfgs(**args)
    return BoundedLbfgs(**args)
