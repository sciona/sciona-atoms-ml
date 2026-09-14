"""Draft providers for the corrected Cassava four-family CPU workflow.

Inputs, predictions and checkpoint locations are private runtime values. These
providers do not establish historical leaderboard equivalence or Tier 1 review.
"""
from sciona.ghost.registry import register_atom


def witness_cassava_fold_plan(sample_keys: list, labels: list, assignments: list) -> dict:
    return {'kind': 'Cassava.FoldPlan'}


@register_atom(witness_cassava_fold_plan)
def cassava_fold_plan(sample_keys: list, labels: list, assignments: list) -> object:
    """Validate unique runtime keys and aligned class/fold ordinals 0 through 4.

    All five validation folds must exist and every training complement must
    contain all five classes. The caller resolves aliases of physical images.
    """
    from sciona.cassava_fold_contract import build_plan
    return build_plan(sample_keys, labels, assignments)


def witness_cassava_train_ensemble(plan: object, source_rows: list,
        efficientnet_rows: list, query_rows: list, references: dict,
        torch_batch_size: int, torch_workers: int, efficientnet_batch_size: int,
        output_directory: str) -> dict:
    if plan != {'kind': 'Cassava.FoldPlan'}:
        raise ValueError('Cassava fold-plan witness required')
    return {'kind': 'Cassava.ExecutionResult'}


@register_atom(witness_cassava_train_ensemble)
def cassava_train_ensemble(plan: object, source_rows: list,
        efficientnet_rows: list, query_rows: list, references: dict,
        torch_batch_size: int, torch_workers: int, efficientnet_batch_size: int,
        output_directory: str) -> dict:
    """Train 15 CV models, refit B4 for 14 epochs, and blend with frozen CropNet.

    Source/query rows contain keyed encoded 600x800 RGB images; prepared B4
    rows contain keyed encoded 512x512 RGB images. The caller establishes the
    preparation relationship. References supply local model paths and reviewed
    hashes for all four families; model downloads are not implicit. Output
    directory must not exist. Runtime results include private keys, scores
    (row sum 3, not probabilities), class ordinals, histories and checkpoints.
    CPU batch sizes and worker count are explicit. Source corrections and
    qualified reference-artifact limits are documented in the CDG metadata.
    """
    from sciona.cassava_pipeline import execute
    result, outputs, histories, evaluations, artifacts = execute(plan,
        source_rows=source_rows, efficientnet_rows=efficientnet_rows,
        query_rows=query_rows, references=references,
        torch_batch_size=torch_batch_size, torch_workers=torch_workers,
        efficientnet_batch_size=efficientnet_batch_size,
        output_directory=output_directory)
    return {'predictions': result, 'family_outputs': outputs,
            'histories': histories, 'evaluations': evaluations,
            'checkpoints': artifacts}
