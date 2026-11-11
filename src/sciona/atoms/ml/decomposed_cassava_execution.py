"""Concrete Cassava recipe adapters; reusable numerical sums remain graph nodes."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete recipe inputs required')


@register_atom(concrete_only)
def train_families(plan: object, source_rows: list, efficientnet_rows: list,
        query_rows: list, references: dict, torch_batch_size: int,
        torch_workers: int, efficientnet_batch_size: int, output_directory: str) -> dict:
    """Run all fifteen CV fits, full B4 refit and frozen CropNet inference.

    Retain complete source image preparation, family losses, training budgets,
    normalization and checkpoint selection. All four reviewed external model
    artifacts are mandatory. Runtime identities, predictions and checkpoints
    remain private. This stage forwards family predictions before final sums.
    """
    from sciona._runtime.ml.decomposed_cassava_pipeline import train_families as implementation
    return implementation(plan, source_rows=source_rows, efficientnet_rows=efficientnet_rows,
        query_rows=query_rows, references=references, torch_batch_size=torch_batch_size,
        torch_workers=torch_workers, efficientnet_batch_size=efficientnet_batch_size,
        output_directory=output_directory)


def check_result(training_result):
    if not isinstance(training_result, dict) or set(training_result) != {
            'query_keys', 'family_outputs', 'histories', 'evaluations', 'checkpoints'}:
        raise ValueError('Complete same-invocation training result required')


@register_atom(concrete_only)
def pair_inputs(training_result: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Align all four family outputs and expose the first unscaled pair sum.

    Validate exact query identity coverage and normalized five-class family
    matrices. Source alignment and class count remain recipe policy. The
    returned context, pair values and coefficients belong to this invocation.
    """
    from sciona._runtime.ml.decomposed_cassava_assembly import pair_inputs as implementation
    check_result(training_result)
    return implementation(training_result['query_keys'], **training_result['family_outputs'])


@register_atom(concrete_only)
def final_inputs(context: dict, paired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expose the completed pair sum, B4 and CropNet with weights [.5,1,1].

    The pair must already be summed: distributing .5 into both original
    vectors can change subnormal rounding. No normalization is applied.
    """
    from sciona._runtime.ml.decomposed_cassava_assembly import final_inputs as implementation
    return implementation(context, paired)


@register_atom(concrete_only)
def result(training_result: dict, context: dict, scores: np.ndarray) -> dict:
    """Restore source predictions and preserve all training evidence fields.

    Inputs must originate from the same graph invocation. This encoder checks
    representation and bounds; it does not authenticate training provenance.
    """
    from sciona._runtime.ml.decomposed_cassava_assembly import result as implementation
    check_result(training_result)
    predictions = implementation(context, scores)
    if predictions['scores'].shape[0] != len(training_result['query_keys']):
        raise ValueError('Query count differs from assembly context')
    return dict(predictions=predictions, **{k: training_result[k] for k in
        ('family_outputs', 'histories', 'evaluations', 'checkpoints')})
