"""Explicit recipe-stage boundaries for the complete independent Amex pipeline."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete qualified pipeline intermediates required')


@register_atom(concrete_only)
def populations(prepared: object) -> dict:
    """Validate canonical prepared input and construct ordered joint populations."""
    from sciona._runtime.ml.decomposed_amex_pipeline import populations as run
    return run(prepared)


@register_atom(concrete_only)
def features(population: dict) -> dict:
    """Build all nine manual blocks, encoded sequences and raw row-learner inputs."""
    from sciona._runtime.ml.decomposed_amex_pipeline import features as run
    return run(population)


@register_atom(concrete_only)
def row_models(population: dict, feature_values: dict) -> dict:
    """Fit all five customer-grouped row DART models at full source controls."""
    from sciona._runtime.ml.decomposed_amex_pipeline import row_models as run
    return run(population,feature_values)


@register_atom(concrete_only)
def assembly(feature_values: dict, row_result: dict) -> dict:
    """Assemble source-ordered tree features and normalized neural features."""
    from sciona._runtime.ml.decomposed_amex_pipeline import assembly as run
    return run(feature_values,row_result)


@register_atom(concrete_only)
def tree_models(population: dict, assembled: dict) -> dict:
    """Fit all ten downstream DART models with and without row-prediction slots."""
    from sciona._runtime.ml.decomposed_amex_pipeline import tree_models as run
    return run(population,assembled)


@register_atom(concrete_only)
def neural_models(population: dict, feature_values: dict, assembled: dict) -> dict:
    """Fit both five-fold neural variants with full epochs and checkpoint rules."""
    from sciona._runtime.ml.decomposed_amex_pipeline import neural_models as run
    return run(population,feature_values,assembled)


@register_atom(concrete_only)
def combination_inputs(tree_result: dict, neural_result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Validate and order four probability vectors and supply literal source weights."""
    from sciona._runtime.ml.decomposed_amex_pipeline import combination_inputs as run
    return run(tree_result,neural_result)


@register_atom(concrete_only)
def result(scores: np.ndarray) -> dict:
    """Encode finite combined scores and the complete recipe's model counts."""
    from sciona._runtime.ml.decomposed_amex_pipeline import result as run
    return run(scores)
