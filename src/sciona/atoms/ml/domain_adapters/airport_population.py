"""Explicit airport partitioning and state pairing; no hidden model execution."""
import math
import pandas as pd

from sciona.ghost.registry import register_atom
from sciona.airport_model_state import validate_adapter_state
from sciona.nasa_population_dispatch import partition_records, assemble_predictions
from sciona.atoms.ml.domain_adapters.airport_features import witness_unmaterialized


@register_atom(witness=witness_unmaterialized)
def training_jobs(training_records: dict, airports: tuple, maximum_errors: dict) -> dict:
    """Require one labeled population and explicit nonnegative cutoff per slot."""
    partitions,_=partition_records(training_records,airports)
    if set(partitions)!=set(airports) or not isinstance(maximum_errors,dict) or set(maximum_errors)!=set(airports):
        raise ValueError('All declared training populations and cutoffs are required')
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0
           for value in maximum_errors.values()):
        raise ValueError('Finite nonnegative residual cutoffs required')
    return {airport:dict(training_records=records,airport=airport,maximum_error=float(maximum_errors[airport]))
            for airport,records in partitions.items()}


@register_atom(witness=witness_unmaterialized)
def training_slot(job: dict) -> tuple:
    if not isinstance(job,dict) or set(job)!={'training_records','airport','maximum_error'}:
        raise ValueError('Explicit training job required')
    return job['training_records'],job['airport'],job['maximum_error']


@register_atom(witness=witness_unmaterialized)
def inference_jobs(prediction_records: dict, model_states: dict) -> tuple:
    """Pair each queried slot with saved state; unqueried models are not executed."""
    if not isinstance(model_states,dict) or not model_states:
        raise ValueError('Explicit nonempty population states required')
    for airport,state in model_states.items():
        adapter,_,_=validate_adapter_state(state)
        if adapter['airport']!=airport:
            raise ValueError('State key and adapter population differ')
    partitions,identities=partition_records(prediction_records,tuple(model_states))
    return ({airport:dict(model_state=model_states[airport],prediction_records=records)
             for airport,records in partitions.items()},identities)


@register_atom(witness=witness_unmaterialized)
def inference_slot(job: dict) -> tuple:
    if not isinstance(job,dict) or set(job)!={'model_state','prediction_records'}:
        raise ValueError('Explicit inference job required')
    return job['model_state'],job['prediction_records']


@register_atom(witness=witness_unmaterialized)
def collect_predictions(identities: pd.DataFrame, slot_predictions: dict) -> pd.DataFrame:
    return assemble_predictions(identities,slot_predictions)
