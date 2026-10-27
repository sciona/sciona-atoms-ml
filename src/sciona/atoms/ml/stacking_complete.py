"""Draft complete-stack boundaries; numerical methods remain separate nodes."""
import json
import numpy as np
from sciona.ghost.registry import register_atom


UNPACK_PORTS = (
    ('training','object'),('labels','object'),('first_folds','object'),('meta_folds','object'),
    ('training_ids','list'),('query','object'),('query_ids','list'),('seed','int'),
    ('first_controls','dict'),('embedding_controls','dict'),('raw_metrics','list'),
    ('tfidf_metrics','list'),('embedding_metrics','list'),('tfidf_controls','dict'),
    ('cluster_controls','dict'),('candidates','dict'),('tsne_interpretation','str'))


def witness_unpack(payload: dict) -> tuple:
    # Boundary validation is runtime-only. These are stage/port markers.
    return tuple({'kind':'Otto.Input.'+name} for name,kind in UNPACK_PORTS)


@register_atom(witness_unpack)
def unpack_payload(payload: dict) -> tuple:
    from sciona.otto_execution import prepare, _first_controls
    data=json.loads(prepare(payload).configuration)
    t,q,c=data['training'],data['query'],data['controls'];s=c['supplemental']
    return (t['values'],t['labels'],t['first_folds'],t['meta_folds'],t['identities'],
            q['values'],q['identities'],c['seed'],_first_controls(c['first_level']),c['embedding'],
            s['raw_metrics'],s['tfidf_metrics'],s['embedding_metrics'],s['tfidf_controls'],
            s['cluster_controls'],c['meta_candidates'],c['tsne_interpretation'])


def witness_embedding(training: object, query: object, training_ids: list, query_ids: list,
                      seed: int, embedding_controls: dict) -> dict:
    return {'kind':'Otto.PopulationEmbedding'}


@register_atom(witness_embedding)
def fit_population_embedding(training: object, query: object, training_ids: list, query_ids: list,
                             seed: int, embedding_controls: dict) -> object:
    from sciona.otto_tsne import fit_population
    return fit_population(np.vstack((training,query)),training_ids+query_ids,seed=seed,**embedding_controls)


def witness_layout(first_level: dict, supplemental: dict, raw_neural: object,
                   training_ids: list, query_ids: list, folds: object, tsne_interpretation: str) -> dict:
    return {'kind':'Otto.MetaLayout'}


@register_atom(witness_layout)
def assemble_layout(first_level: dict, supplemental: dict, raw_neural: object,
                    training_ids: list, query_ids: list, folds: object, tsne_interpretation: str) -> dict:
    from sciona.otto_meta_layout import assemble
    return assemble(first_level,supplemental,raw_neural,training_ids,query_ids,folds,
                    tsne_interpretation=tsne_interpretation)


def witness_sharing(first_controls: dict, representative_24: int, representative_25: int, representative_26: int, representative_27: int, representative_28: int, representative_29: int, representative_30: int, representative_31: int, representative_32: int, representative_33: int) -> dict:
    return {'kind':'Otto.FirstControls'}


@register_atom(witness_sharing)
def validate_neighbor_sharing(first_controls: dict, representative_24: int, representative_25: int, representative_26: int, representative_27: int, representative_28: int, representative_29: int, representative_30: int, representative_31: int, representative_32: int, representative_33: int) -> dict:
    representatives=(representative_24, representative_25, representative_26, representative_27, representative_28, representative_29, representative_30, representative_31, representative_32, representative_33,)
    if type(first_controls) is not dict or any(type(i) is not int for i in first_controls) or set(first_controls)!=set(range(1,34)):
        raise ValueError('Complete first-level controls required')
    for entry,representative in zip(range(24,34),representatives):
        if type(representative) is not int or not 24<=representative<=33:
            raise ValueError('Valid shared neighbor representative required')
        if json.dumps(first_controls[entry],sort_keys=True)!=json.dumps(first_controls[representative],sort_keys=True):
            raise ValueError('Runtime neighbor controls differ from compiled sharing groups')
    return first_controls
