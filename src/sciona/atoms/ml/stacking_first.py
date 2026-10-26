"""Draft explicit first-level source adapters; stage-kind witnesses only.

Source-specific nine-class/five-fold constraints remain in runtime validators.
Registration is not publication or a static proof of numeric correctness.
"""
from sciona.ghost.registry import register_atom


def witness_prepare(training: object, labels: object, folds: object, training_ids: list,
                    query: object, query_ids: list, embedding: object, seed: int, controls: dict) -> dict:
    return {'kind': 'Otto.FirstInputs'}


@register_atom(witness_prepare)
def prepare_first(training: object, labels: object, folds: object, training_ids: list,
                  query: object, query_ids: list, embedding: object, seed: int, controls: dict) -> object:
    from sciona.stacking_first_operations import prepare_first_inputs
    return prepare_first_inputs(training, labels, folds, training_ids, query, query_ids, embedding, seed, controls)


def witness_policy() -> tuple:
    return tuple(range(1, 34))


@register_atom(witness_policy)
def source_entry_policy() -> tuple:
    """Explicit fixed source entry indices, independent of private input values."""
    return tuple(range(1, 34))


def witness_entry(inputs: dict, entry: int) -> dict:
    if inputs != {'kind': 'Otto.FirstInputs'} or type(entry) is not int or not 1 <= entry <= 23:
        raise ValueError('Prepared population and source entry required')
    return {'kind': 'Otto.FirstBlock'}


@register_atom(witness_entry)
def fit_first_entry(inputs: object, entry: int) -> object:
    from sciona.stacking_first_operations import fit_entry
    return fit_entry(inputs, entry)


def witness_bank(inputs: dict, representative: int) -> dict:
    if inputs != {'kind': 'Otto.FirstInputs'} or type(representative) is not int or not 24 <= representative <= 33:
        raise ValueError('Prepared population and neighbor representative required')
    return {'kind': 'Otto.NeighborBank'}


@register_atom(witness_bank)
def fit_neighbor_bank(inputs: object, representative: int) -> object:
    from sciona.stacking_first_operations import fit_neighbor_bank as fit
    return fit(inputs, representative)


def witness_column(inputs: dict, bank: dict, entry: int) -> dict:
    witness_bank(inputs, entry)
    if bank != {'kind': 'Otto.NeighborBank'}:
        raise ValueError('Shared neighbor bank required')
    return {'kind': 'Otto.FirstBlock'}


@register_atom(witness_column)
def select_neighbor_entry(inputs: object, bank: object, entry: int) -> object:
    from sciona.stacking_first_operations import select_neighbor_entry as select
    return select(inputs, bank, entry)


def witness_collect(block_1: dict, block_2: dict, block_3: dict, block_4: dict, block_5: dict, block_6: dict, block_7: dict, block_8: dict, block_9: dict, block_10: dict, block_11: dict, block_12: dict, block_13: dict, block_14: dict, block_15: dict, block_16: dict, block_17: dict, block_18: dict, block_19: dict, block_20: dict, block_21: dict, block_22: dict, block_23: dict, block_24: dict, block_25: dict, block_26: dict, block_27: dict, block_28: dict, block_29: dict, block_30: dict, block_31: dict, block_32: dict, block_33: dict) -> dict:
    if any(value != {'kind': 'Otto.FirstBlock'} for value in locals().values()):
        raise ValueError('All source blocks required')
    return {'kind': 'Otto.FirstLevel'}


@register_atom(witness_collect)
def collect_first(block_1: object, block_2: object, block_3: object, block_4: object, block_5: object, block_6: object, block_7: object, block_8: object, block_9: object, block_10: object, block_11: object, block_12: object, block_13: object, block_14: object, block_15: object, block_16: object, block_17: object, block_18: object, block_19: object, block_20: object, block_21: object, block_22: object, block_23: object, block_24: object, block_25: object, block_26: object, block_27: object, block_28: object, block_29: object, block_30: object, block_31: object, block_32: object, block_33: object) -> dict:
    from sciona.stacking_first_operations import collect_first_blocks
    return collect_first_blocks({1: block_1, 2: block_2, 3: block_3, 4: block_4, 5: block_5, 6: block_6, 7: block_7, 8: block_8, 9: block_9, 10: block_10, 11: block_11, 12: block_12, 13: block_13, 14: block_14, 15: block_15, 16: block_16, 17: block_17, 18: block_18, 19: block_19, 20: block_20, 21: block_21, 22: block_22, 23: block_23, 24: block_24, 25: block_25, 26: block_26, 27: block_27, 28: block_28, 29: block_29, 30: block_30, 31: block_31, 32: block_32, 33: block_33})
