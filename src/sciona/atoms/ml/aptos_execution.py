"""Draft providers for the corrected APTOS two-stage CPU reference.

Inputs and predictions are private runtime values. This implementation does
not establish leaderboard parity or human-reviewed Tier 1 status.
"""
from sciona.ghost.registry import register_atom


def witness_aptos_population(base: list, average: list, grouped: list, pseudo_keys: list) -> dict:
    return {'kind': 'Aptos.Population'}


@register_atom(witness_aptos_population)
def aptos_population(base: list, average: list, grouped: list, pseudo_keys: list) -> object:
    """Validate disjoint first-stage base and three second-stage added roles.

    Labeled roles contain [identities, labels]. Base and averaging labels use
    five ordinals 0..4; group-bounded labels use four ordinals 0..3. Pseudo
    identities have no supplied labels. Source-role provenance is caller-owned.
    """
    from sciona.aptos_population import prepare_population
    return prepare_population(base=base, average=average, grouped=grouped, pseudo_keys=pseudo_keys)


def witness_aptos_train_ensemble(plan: object, query_keys: list, images: dict,
        references: dict, first_stage_epochs: dict, seeds: dict, batch_size: int,
        learning_rate: float, lower_deviation: float, upper_deviation: float,
        tie_policy: str, work_root: str) -> dict:
    if plan != {'kind': 'Aptos.Population'}:
        raise ValueError('APTOS population witness required')
    return {'kind': 'Aptos.ExecutionResult'}


@register_atom(witness_aptos_train_ensemble)
def aptos_train_ensemble(plan: object, query_keys: list, images: dict,
        references: dict, first_stage_epochs: dict, seeds: dict, batch_size: int,
        learning_rate: float, lower_deviation: float, upper_deviation: float,
        tie_policy: str, work_root: str) -> dict:
    """Fit eight pretrained models, refine targets and continue ten epochs.

    References map four families to [local safetensors path, reviewed SHA256].
    Budgets and seeds map (family, replica) pairs to integers. All first-stage
    fits precede the ensemble teacher; second-stage model weights continue
    with fresh Adam. Reference augmentation, normalization and single-view
    inference are explicit reconstruction choices. Results include private
    predictions, soft targets and aggregate fit histories. Temporary trained
    checkpoints are verified by replay and removed, never published here.
    """
    from sciona.aptos_population import Population
    from sciona.aptos_pipeline import run_reference
    if not isinstance(plan, Population):
        raise ValueError('APTOS population plan required')
    return run_reference(base=(plan.base_keys, plan.base_labels),
        average=(plan.average_keys, plan.average_labels), grouped=(plan.group_keys, plan.group_labels),
        pseudo_keys=plan.pseudo_keys, query_keys=query_keys, images=images, pretrained=references,
        first_stage_epochs=first_stage_epochs, seeds=seeds, batch_size=batch_size,
        learning_rate=learning_rate, lower_deviation=lower_deviation, upper_deviation=upper_deviation,
        tie_policy=tie_policy, work_root=work_root)
