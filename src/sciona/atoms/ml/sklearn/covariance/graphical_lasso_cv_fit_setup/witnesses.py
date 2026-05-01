"""Ghost witnesses for GraphicalLassoCV fit-setup helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_graphical_lasso_cv_location(
    X: NDArray[float],
    assume_centered: bool,
) -> NDArray[float]:
    """Describe GraphicalLassoCV's fitted location vector before path solving."""
    del X
    del assume_centered
    raise NotImplementedError


def witness_graphical_lasso_cv_inner_verbose(
    verbose: int,
) -> int:
    """Describe GraphicalLassoCV's inner graphical_lasso_path verbosity level."""
    del verbose
    return 0


def witness_graphical_lasso_cv_use_explicit_alphas(
    alphas: object,
) -> bool:
    """Describe whether GraphicalLassoCV treats alphas as an explicit sequence."""
    del alphas
    return False


def witness_graphical_lasso_cv_explicit_alphas(
    alphas: object,
) -> NDArray[float]:
    """Describe the validated explicit alpha vector used directly by GraphicalLassoCV."""
    del alphas
    raise NotImplementedError


def witness_graphical_lasso_cv_refinement_count(
    use_explicit_alphas: bool,
    n_refinements: int,
) -> int:
    """Describe GraphicalLassoCV's refinement-count selection from explicit or generated alphas."""
    del use_explicit_alphas
    del n_refinements
    return 1
