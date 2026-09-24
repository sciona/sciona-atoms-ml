"""Source validation constants and residual model output naming."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def score_policy(frame: pd.DataFrame) -> tuple:
    """Zero validation offset, integer clip bounds and initial source best-score ceiling."""
    return np.zeros(len(frame)),1,299,1e9


@register_atom(witness=witness_materialized)
def residual_outputs(model: object, importance: pd.DataFrame) -> tuple:
    """Preserve v0/v2 model alias and source feature-importance field names."""
    return model,model,importance.rename(columns={'feature':'features','importance':'imp'}).copy()
