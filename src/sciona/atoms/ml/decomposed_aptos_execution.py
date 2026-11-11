"""Concrete recipe adapters around complete APTOS training and explicit score cuts."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete recipe inputs required')


@register_atom(concrete_only)
def train_scores(plan: object, query_keys: list, images: dict,
        references: dict, first_stage_epochs: dict, seeds: dict, batch_size: int,
        learning_rate: float, lower_deviation: float, upper_deviation: float,
        tie_policy: str, work_root: str) -> dict:
    """Fit all eight models and both stages; leave final class indexing to the graph.

    References map four families to [local safetensors path, reviewed SHA256].
    Budgets and seeds map (family, replica) pairs to integers. All first-stage
    fits precede the ensemble teacher; second-stage model weights continue
    with fresh Adam. Reference augmentation, normalization and single-view
    inference are explicit reconstruction choices. Results include private
    predictions, soft targets and aggregate fit histories. Temporary trained
    checkpoints are verified by replay and removed, never published here.
    """
    from sciona._runtime.ml.aptos_population import Population
    from sciona._runtime.ml.decomposed_aptos_pipeline import train_scores as run_reference
    if not isinstance(plan, Population):
        raise ValueError('APTOS population plan required')
    return run_reference(base=(plan.base_keys, plan.base_labels),
        average=(plan.average_keys, plan.average_labels), grouped=(plan.group_keys, plan.group_labels),
        pseudo_keys=plan.pseudo_keys, query_keys=query_keys, images=images, pretrained=references,
        first_stage_epochs=first_stage_epochs, seeds=seeds, batch_size=batch_size,
        learning_rate=learning_rate, lower_deviation=lower_deviation, upper_deviation=upper_deviation,
        tie_policy=tie_policy, work_root=work_root)


@register_atom(concrete_only)
def cut_inputs(training_result: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Expose scores and source cut policy without applying numeric indexing."""
    from sciona._runtime.ml.decomposed_aptos_pipeline import cut_inputs as implementation
    return implementation(training_result)


@register_atom(concrete_only)
def result(training_result: dict, ordinals: np.ndarray) -> dict:
    """Attach generic class indices to the complete same-invocation training result."""
    from sciona._runtime.ml.decomposed_aptos_pipeline import result as implementation
    return implementation(training_result, ordinals)
