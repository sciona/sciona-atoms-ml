"""Explicit reusable labeled population concatenation."""
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.population_tables import concatenate_labeled_populations
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def concatenate(tables: list, labels: list, columns: list, label_column: str, order_column: str) -> pd.DataFrame:
    """Select and label each population, concatenate, and sort using explicit columns."""
    return concatenate_labeled_populations(tables,labels,columns,label_column,order_column)
