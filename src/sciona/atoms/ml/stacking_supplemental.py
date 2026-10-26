"""Draft supplemental source adapters; witnesses mark stage kinds only."""
from sciona.ghost.registry import register_atom


def witness_prepare(training: object, labels: object, folds: object, training_ids: list,
                    query: object, query_ids: list, embedding: object, seed: int,
                    raw_metrics: list, tfidf_metrics: list, embedding_metrics: list,
                    tfidf_controls: dict, cluster_controls: dict) -> dict:
    return {'kind': 'Otto.SupplementalInputs'}


@register_atom(witness_prepare)
def prepare_supplemental(training: object, labels: object, folds: object, training_ids: list,
                         query: object, query_ids: list, embedding: object, seed: int,
                         raw_metrics: list, tfidf_metrics: list, embedding_metrics: list,
                         tfidf_controls: dict, cluster_controls: dict) -> object:
    from sciona.stacking_supplemental_operations import prepare_supplemental as prepare
    return prepare(training, labels, folds, training_ids, query, query_ids, embedding, seed,
                   raw_metrics, tfidf_metrics, embedding_metrics, tfidf_controls, cluster_controls)


def witness_block(inputs: dict) -> dict:
    if inputs != {'kind': 'Otto.SupplementalInputs'}:
        raise ValueError('Prepared supplemental inputs required')
    return {'kind': 'Otto.SupplementalBlock'}


def witness_raw(inputs: dict) -> tuple:
    block = witness_block(inputs)
    return block.copy(), block.copy(), block.copy()


@register_atom(witness_raw)
def raw_class_distances(inputs: object) -> tuple:
    from sciona.stacking_supplemental_operations import raw_class_distances as run
    return run(inputs)


@register_atom(witness_block)
def weighted_class_distances(inputs: object) -> object:
    from sciona.stacking_supplemental_operations import weighted_class_distances as run
    return run(inputs)


@register_atom(witness_block)
def embedded_class_distances(inputs: object) -> object:
    from sciona.stacking_supplemental_operations import embedded_class_distances as run
    return run(inputs)


@register_atom(witness_block)
def nonzero_counts(inputs: object) -> object:
    from sciona.stacking_supplemental_operations import nonzero_counts as run
    return run(inputs)


@register_atom(witness_block)
def raw_neural_features(inputs: object) -> object:
    from sciona.stacking_supplemental_operations import raw_neural_features as run
    return run(inputs)


def witness_fit(inputs: dict) -> dict:
    witness_block(inputs)
    return {'kind': 'Otto.ReferenceClusters'}


@register_atom(witness_fit)
def fit_reference_clusters(inputs: object) -> object:
    from sciona.stacking_supplemental_operations import fit_reference_clusters as fit
    return fit(inputs)


def witness_assign(inputs: dict, model: dict) -> dict:
    if model != {'kind': 'Otto.ReferenceClusters'}:
        raise ValueError('Fitted reference clusters required')
    return witness_block(inputs)


@register_atom(witness_assign)
def cluster_assignments(inputs: object, model: object) -> object:
    from sciona.stacking_supplemental_operations import cluster_assignments as assign
    return assign(inputs, model)


def witness_collect(raw_one: dict, raw_two: dict, raw_four: dict, weighted: dict,
                    embedded: dict, clusters: dict, counts: dict, raw_neural: dict) -> tuple:
    if any(value != {'kind': 'Otto.SupplementalBlock'} for value in locals().values()):
        raise ValueError('All supplemental source blocks required')
    return {'kind': 'Otto.SupplementalBlocks'}, {'kind': 'Otto.SupplementalBlock'}


@register_atom(witness_collect)
def collect_supplemental(raw_one: object, raw_two: object, raw_four: object, weighted: object,
                         embedded: object, clusters: object, counts: object, raw_neural: object) -> tuple:
    from sciona.stacking_supplemental_operations import collect_supplemental as collect
    return collect(raw_one, raw_two, raw_four, weighted, embedded, clusters, counts, raw_neural)
