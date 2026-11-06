"""Complete CHAMPS recipe boundaries around a separately bound generic mean."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete recipe inputs required')


def check_training(training_result):
    if not isinstance(training_result, dict) or set(training_result) != {
            'version', 'execution_version', 'model_predictions', 'coupling_types',
            'models_completed', 'training'}:
        raise ValueError('Complete same-invocation training result required')


@register_atom(concrete_only)
def train_models(prepared: object) -> dict:
    """Run complete chemistry, fitted preprocessing and thirteen full models.

    Preserve caller-declared schedules, source checkpoint selection, complete
    state reload, prediction unscaling and six-decimal per-model boundaries.
    Forward every model prediction before the separate ensemble reduction.
    """
    from sciona.decomposed_champs_runtime import train_models as implementation
    return implementation(prepared)


@register_atom(concrete_only)
def ensemble_inputs(training_result: dict) -> tuple[tuple, np.ndarray, int]:
    """Align identities and select the source nine models for each coupling.

    All thirteen models are required and validated, including values discarded
    by selection. Return opaque record keys, selected values and retained count.
    Molecular categories, model masks and central-five policy stay here.
    """
    from sciona.decomposed_champs_ensemble import ensemble_inputs as implementation
    check_training(training_result)
    return implementation(training_result['model_predictions'], training_result['coupling_types'])


@register_atom(concrete_only)
def result(training_result: dict, keys: tuple, values: np.ndarray) -> dict:
    """Restore the complete source result envelope after the generic reduction.

    Inputs must belong to the same graph invocation; shape and key checks do
    not authenticate training provenance. Private evidence fields are retained.
    """
    from sciona.decomposed_champs_ensemble import result as implementation
    check_training(training_result)
    if keys != tuple(training_result['coupling_types']):
        raise ValueError('Result identities differ from training invocation')
    return dict(version=training_result['version'], execution_version=training_result['execution_version'],
                predictions=implementation(keys, values), models_completed=training_result['models_completed'],
                training=training_result['training'])
