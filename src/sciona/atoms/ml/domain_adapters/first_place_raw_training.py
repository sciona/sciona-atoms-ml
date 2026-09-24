"""Raw population boundaries and identity-aligned target attachment for training."""
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def raw_inputs(raw_populations: dict) -> tuple:
    """Expose exact caller-supplied population records and fixed feature policies."""
    if set(raw_populations)!=set(POPULATIONS):
        raise ValueError('Complete explicit raw population inputs required')
    values=[]
    for label in POPULATIONS:
        entry=raw_populations[label]
        fields=['queries','raw_options','feature_columns','numeric_fills','categorical_fills']
        if not isinstance(entry,dict) or set(entry)!=set(fields):
            raise ValueError('Explicit raw tables and complete fixed model policy required')
        values.extend(entry[field] for field in fields)
        values.append('int16')
    return tuple(values)


@register_atom(witness=witness_materialized)
def attach_target(queries: pd.DataFrame, frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Attach caller targets positionally after fixed feature preparation, retaining row order."""
    if target_name!='minutes_until_pushback' or not queries.columns.is_unique or not frame.columns.is_unique:
        raise ValueError('Unique source columns and explicit training target required')
    if not queries.index.equals(frame.index) or len(queries)!=len(frame):
        raise ValueError('Training features and target row identities differ')
    if set(frame)&{'gufi','timestamp',target_name}:
        raise ValueError('Model features contain training metadata')
    result=pd.concat([queries[['gufi','timestamp']].copy(),frame.copy()],axis=1)
    result[target_name]=queries[target_name].to_numpy(copy=True)
    return result


@register_atom(witness=witness_materialized)
def collect_tables(table0: pd.DataFrame, table1: pd.DataFrame, table2: pd.DataFrame,
        table3: pd.DataFrame, table4: pd.DataFrame, table5: pd.DataFrame, table6: pd.DataFrame,
        table7: pd.DataFrame, table8: pd.DataFrame, table9: pd.DataFrame) -> dict:
    """Bind prepared training tables to the fixed public-software population order."""
    return dict(zip(POPULATIONS,[table0,table1,table2,table3,table4,table5,table6,table7,table8,table9]))
