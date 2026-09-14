"""Five-classifier training orchestration for the causal three-system ensemble."""
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_train_causal_classifiers(training_features: AbstractArray, training_labels: AbstractArray,
    n_estimators: int=500, max_depth: int=9, min_samples_split: int=8,
    learning_rate: float=.1, random_state: int=1) -> tuple:
    if len(training_features.shape)!=2 or training_labels.shape!=(training_features.shape[0],):
        raise ValueError('aligned training matrix and label vector required')
    # Model objects are runtime state, not numerical arrays or certified models.
    return tuple({'type':'GradientBoostingClassifier','n_features':training_features.shape[1]} for _ in range(5))


def causal_training_partitions(training_features,training_labels):
    """Construct source-defined roles and weights without fitting a model."""
    X=np.asarray(training_features);y=np.asarray(training_labels)
    if X.dtype.kind not in 'iuf' or X.ndim!=2 or min(X.shape)==0 or not np.all(np.isfinite(X)):
        raise ValueError('finite nonempty real feature matrix required')
    if y.dtype.kind not in 'iu' or y.shape!=(len(X),) or set(y.tolist())!={-1,0,1}:
        raise ValueError('aligned integer targets containing -1,0,1 required')
    if len(y)%2 or not np.array_equal(y[0::2],-y[1::2]):
        raise ValueError('adjacent reversed training pairs require opposite labels')
    dependent=y!=0
    targets=[y,(y!=0).astype(np.int64),y[dependent],(y==1).astype(np.int64),(y==-1).astype(np.int64)]
    matrices=[X,X,X[dependent],X,X]
    weights=[]
    for index,target in enumerate(targets):
        if index==0:
            weights.append(None)
        elif index==2:
            # The source's target==0 assignment affects no direction rows.
            weights.append(np.ones(len(target)))
        else:
            value=np.ones(len(target))
            value[target==0]=np.count_nonzero(target==1)/np.count_nonzero(target==0)
            weights.append(value)
    return list(zip(matrices,targets,weights))


@register_atom(witness_train_causal_classifiers)
def train_causal_classifiers(training_features: NDArray[np.float64], training_labels: NDArray[np.int64],
    n_estimators: int=500, max_depth: int=9, min_samples_split: int=8,
    learning_rate: float=.1, random_state: int=1) -> tuple:
    """Fit one-step, dependence, direction, left and right gradient boosters.

    Features must already be extracted in a consistent column order. Adjacent
    training rows represent reversed observation pairs with opposite labels.
    All three labels -1/0/+1 must be present. Direction training excludes label
    zero; binary indicator tasks use the source's negative-class balancing.
    Defaults reproduce the upstream boosting configuration with modern sklearn
    log_loss replacing the former deviance spelling. Models remain in memory;
    no files are loaded or persisted. No predictive-quality claim is implied.

    Source: https://github.com/jarfo/cause-effect/blob/master/estimator.py
    """
    for name,value,minimum in [('n_estimators',n_estimators,1),('max_depth',max_depth,1),('min_samples_split',min_samples_split,2)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum:
            raise ValueError(name+' outside integer domain')
    if isinstance(random_state,bool) or not isinstance(random_state,(int,np.integer)) or not 0<=random_state<2**32:
        raise ValueError('unsigned 32-bit random seed required')
    if isinstance(learning_rate,bool) or not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('finite positive learning rate required')
    models=[]
    for X,y,weight in causal_training_partitions(training_features,training_labels):
        model=GradientBoostingClassifier(loss='log_loss',n_estimators=int(n_estimators),max_depth=int(max_depth),min_samples_split=int(min_samples_split),learning_rate=float(learning_rate),random_state=int(random_state))
        model.fit(X,y,sample_weight=weight)
        models.append(model)
    return tuple(models)
