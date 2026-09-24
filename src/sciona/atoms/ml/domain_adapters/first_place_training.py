"""Pinned source eligibility and native parameter policy for regression training."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


def _inputs(table, target_name, end_train, residual):
    names = list(table.columns)
    for name in ['gufi','timestamp',target_name]:
        names.remove(name)
    frame = table[names].copy()
    times = pd.DatetimeIndex(table['timestamp'])
    eligible = ((times > pd.Timestamp('2020-11-02')) & (table[target_name].to_numpy() != 0))
    offsets = frame['etd_time_till_est_dep'].to_numpy() if residual else np.zeros(len(frame))
    categories = [name for name in names if '_cat_' in name]
    parameters = dict(eta=0.01,depth=7,rsm=1,subsample=0.8,max_leaves=21,l2_leaf_reg=3,
        min_data_in_leaf=5000,n_estimators=20000,task_type='CPU',grow_policy='Lossguide',has_time=True,
        random_seed=4,loss_function='MAE',boosting_type='Plain',max_ctr_complexity=12,bootstrap_type='Bernoulli')
    return frame,table[target_name].to_numpy(),times,np.asarray(eligible,dtype=bool),end_train,offsets,categories,parameters,60


@register_atom(witness=witness_materialized)
def residual_inputs(table: pd.DataFrame, target_name: str, end_train: str) -> tuple:
    """Source temporal eligibility, residual targets and fixed training parameters."""
    return _inputs(table,target_name,end_train,True)


@register_atom(witness=witness_materialized)
def direct_inputs(table: pd.DataFrame, target_name: str, end_train: str) -> tuple:
    """Source temporal eligibility, direct targets and fixed training parameters."""
    return _inputs(table,target_name,end_train,False)
