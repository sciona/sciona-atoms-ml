"""Reusable sparse binary logistic fitting and independent probability scoring."""
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete CSR features and binary fitted state required')


def _matrix(features):
    if not isinstance(features,csr_matrix) or features.dtype!=np.float64 or not all(features.shape):
        raise ValueError('Nonempty float64 CSR matrix required')
    matrix=features.copy()
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
        raise ValueError('Canonical finite CSR entries required')
    return matrix


@register_atom(concrete_only)
def fit_sparse_binary_logistic(features: object, labels: np.ndarray, inverse_regularization: float,
                               tolerance: float, maximum_iterations: int, seed: int) -> object:
    """Fit L2 binary logistic regression using LBFGS and an intercept.

    Features are a nonempty canonical float64 scipy CSR matrix; aligned int64
    labels (N,) contain both 0 and 1. C (inverse regularization) and tolerance
    are finite positive Python numbers; maximum_iterations is a positive Python
    int and seed is a Python int in [0,2**32). Controls are caller supplied.
    No feature scaling, weighting, class balancing or split policy is imposed.

    Returns a fitted sklearn LogisticRegression. Nonconvergence raises rather
    than returning a partial fit. Native thread pools are limited to one during
    fitting; caller owns aggregate concurrency. A sparse copy protects input
    storage. Fitted coefficients remain dense; source features are not densified.
    """
    matrix=_matrix(features)
    targets=np.asarray(labels)
    if targets.dtype!=np.int64 or targets.shape!=(matrix.shape[0],) or not np.array_equal(np.unique(targets),[0,1]):
        raise ValueError('Aligned int64 binary labels with both classes required')
    for value in (inverse_regularization,tolerance):
        if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
            raise ValueError('Finite positive regularization and tolerance required')
    if type(maximum_iterations) is not int or maximum_iterations<1:
        raise ValueError('Positive iteration count required')
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Valid integer seed required')
    model=LogisticRegression(C=inverse_regularization,tol=tolerance,max_iter=maximum_iterations,
        random_state=seed,solver='lbfgs',fit_intercept=True)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(matrix,targets)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Nonfinite fitted coefficients')
    return model


@register_atom(concrete_only)
def predict_sparse_binary_probability(features: object, model: object) -> np.ndarray:
    """Score class 1 in row order using a fitted binary logistic model.

    Requires canonical finite float64 CSR features and a fitted sklearn
    LogisticRegression with classes [0,1], finite coefficients and matching
    feature width. Returns finite float64 probabilities (N,) in [0,1]. No
    query labels, refitting, calibration or threshold policy. Inputs are not
    mutated; a sparse copy is used and native pools are limited to one.
    """
    matrix=_matrix(features)
    if not isinstance(model,LogisticRegression) or not np.array_equal(getattr(model,'classes_',None),[0,1]):
        raise ValueError('Fitted binary logistic model required')
    if (getattr(model,'n_features_in_',None)!=matrix.shape[1]
            or np.shape(getattr(model,'coef_',None))!=(1,matrix.shape[1])
            or np.shape(getattr(model,'intercept_',None))!=(1,)
            or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()):
        raise ValueError('Aligned finite fitted coefficients required')
    with threadpool_limits(limits=1):
        result=model.predict_proba(matrix)[:,1]
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():
        raise ValueError('Nonfinite probability computation')
    return result
