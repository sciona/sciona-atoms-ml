"""Field conversion and column projection around reusable feature atoms."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.nasa_first_graph_adapters import prepare_arguments, project_tables


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Materialized domain tables and explicit policies required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def prepare(queries: pd.DataFrame, raw_options: dict) -> tuple:
    """Validate the declared snapshot and convert public software fields to explicit generic operands."""
    return prepare_arguments(queries, raw_options)


@register_atom(witness=witness_materialized)
def project(context: dict, lexical: pd.DataFrame, attributes: pd.DataFrame, configuration: dict,
        estimates: dict, calendar: dict, forecasts: dict, counts: np.ndarray,
        delay_count: np.ndarray, delay_value_count: np.ndarray, delay_mean: np.ndarray,
        delay_maximum: np.ndarray) -> tuple:
    """Name generic outputs and supply explicit keyed tables to the reusable join atom."""
    return project_tables(context, lexical, attributes, configuration, estimates, calendar, forecasts,
                          counts, delay_count, delay_value_count, delay_mean, delay_maximum)
