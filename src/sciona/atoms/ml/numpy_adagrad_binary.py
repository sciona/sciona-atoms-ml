"""Reusable one-hidden-layer ReLU/softmax fitting with explicit AdaGrad controls."""
import numbers
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits
from sciona.ghost.abstract import AbstractArray,AbstractScalar
from sciona.ghost.registry import register_atom
from .model_selection.masked_operations import _selected,_mask_metadata
from .xgboost.model_io import _names


def _configuration(config):
    keys={'hidden_units','epochs','batch_size','seed','learning_rate','epsilon'}
    if not isinstance(config,dict) or set(config)!=keys:
        raise ValueError('Complete explicit AdaGrad configuration required')
    for key in ('hidden_units','epochs','batch_size','seed'):
        value=config[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or value<(0 if key=='seed' else 1):
            raise ValueError('Invalid integer '+key)
    if config['seed']>=2**32:raise ValueError('RandomState seed must fit uint32')
    for key in ('learning_rate','epsilon'):
        value=config[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value) or value<=0:
            raise ValueError('Positive finite '+key+' required')


def _metadata(features,names):
    if features.dtype!='float64' or len(features.shape)!=2 or min(features.shape)<1:
        raise ValueError('Nonempty float64 feature matrix metadata required')
    if features.dim is not None and not features.dim.is_dimensionless:
        raise ValueError('Explicitly standardized dimensionless features required')
    _names(names,features.shape[1])


def _matrix(features,names):
    x=np.asarray(features)
    if x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Nonempty finite float64 feature matrix required')
    _names(names,x.shape[1])
    return x


def _loss_gradients(x,y,parameters):
    first,bias,second,output_bias=parameters
    preactivation=x@first+bias
    hidden=np.maximum(preactivation,0)
    logits=hidden@second+output_bias
    logits-=logits.max(axis=1,keepdims=True)
    exponential=np.exp(logits)
    probabilities=exponential/exponential.sum(axis=1,keepdims=True)
    loss=np.mean(np.log(exponential.sum(axis=1))-logits[np.arange(len(y)),y])
    delta=probabilities.copy();delta[np.arange(len(y)),y]-=1;delta/=len(y)
    back=(delta@second.T)*(preactivation>0)
    return loss,(x.T@back,back.sum(axis=0),hidden.T@delta,delta.sum(axis=0))


def witness_fit_masked_adagrad_binary(features: AbstractArray,targets: AbstractArray,feature_names: list,
        selection: AbstractArray,configuration: dict) -> tuple:
    _metadata(features,feature_names);_mask_metadata(selection,features.shape[0]);_configuration(configuration)
    if targets.shape!=(features.shape[0],) or targets.dtype!='int64':
        raise ValueError('Aligned int64 targets required')
    if targets.dim is not None and not targets.dim.is_dimensionless:raise ValueError('Binary labels are dimensionless')
    return (dict(format='sciona.adagrad_binary.abstract.v1',feature_names=list(feature_names),hidden_units=configuration['hidden_units']),
            int(configuration['epochs']),AbstractScalar(dtype='int64',min_val=configuration['epochs']))


@register_atom(witness_fit_masked_adagrad_binary)
def fit_masked_adagrad_binary(features: NDArray[np.float64],targets: NDArray[np.int64],feature_names: list,
        selection: NDArray[np.bool_],configuration: dict) -> tuple[dict,int,int]:
    """Fit only selected ordered rows; emit numeric model state, epochs, updates.

    Inputs must already be standardized. No scaling, partition inference,
    shuffling, early stopping, validation feedback or query fitting occurs.
    Initialization is RandomState Glorot uniform with zero biases. Optimize
    mean cross entropy in fixed sequential batches using sqrt(sum(g²)+epsilon).
    All configured epochs run with one native thread. Excluded values cannot
    influence fitting. State is numeric JSON data, not executable serialization.
    """
    _configuration(configuration)
    x,y=_selected(features,targets,selection);x=_matrix(x,feature_names)
    if y.dtype!=np.int64 or set(y)!={0,1}:raise ValueError('Both int64 binary classes required in selected rows')
    hidden=int(configuration['hidden_units']);rng=np.random.RandomState(int(configuration['seed']))
    a=np.sqrt(6/(x.shape[1]+hidden));b=np.sqrt(6/(hidden+2))
    parameters=[rng.uniform(-a,a,(x.shape[1],hidden)),np.zeros(hidden),rng.uniform(-b,b,(hidden,2)),np.zeros(2)]
    accumulators=[np.zeros_like(p) for p in parameters];updates=0
    with threadpool_limits(limits=1),np.errstate(over='raise',invalid='raise',divide='raise'):
        for _ in range(int(configuration['epochs'])):
            for start in range(0,len(x),int(configuration['batch_size'])):
                stop=start+int(configuration['batch_size'])
                loss,gradients=_loss_gradients(x[start:stop],y[start:stop],parameters)
                if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
                    raise ValueError('Nonfinite neural optimization state')
                for parameter,accumulator,gradient in zip(parameters,accumulators,gradients):
                    accumulator+=gradient*gradient
                    parameter-=configuration['learning_rate']*gradient/np.sqrt(accumulator+configuration['epsilon'])
                updates+=1
    if any(not np.isfinite(p).all() for p in parameters):raise ValueError('Nonfinite fitted parameters')
    state=dict(format='sciona.adagrad_binary.v1',feature_names=list(feature_names),
               parameters=[p.tolist() for p in parameters])
    return state,int(configuration['epochs']),updates


def _parameters(state,names):
    if (not isinstance(state,dict) or set(state)!={'format','feature_names','parameters'}
            or state['format']!='sciona.adagrad_binary.v1' or state['feature_names']!=names
            or not isinstance(state['parameters'],list) or len(state['parameters'])!=4):
        raise ValueError('Matching numeric neural state and ordered feature names required')
    p=[np.asarray(v,dtype=np.float64) for v in state['parameters']]
    if (p[0].ndim!=2 or p[0].shape[0]!=len(names) or p[0].shape[1]<1
            or p[1].shape!=(p[0].shape[1],) or p[2].shape!=(p[0].shape[1],2)
            or p[3].shape!=(2,) or any(not np.isfinite(v).all() for v in p)):
        raise ValueError('Finite shape-consistent neural parameters required')
    return p


def witness_predict_adagrad_binary(features: AbstractArray,feature_names: list,state: dict) -> AbstractArray:
    _metadata(features,feature_names)
    if not isinstance(state,dict) or state.get('format')!='sciona.adagrad_binary.abstract.v1' or state.get('feature_names')!=feature_names:
        raise ValueError('Matching abstract neural state required')
    return AbstractArray(shape=(features.shape[0],),dtype='float64')


@register_atom(witness_predict_adagrad_binary)
def predict_adagrad_binary(features: NDArray[np.float64],feature_names: list,state: dict) -> NDArray[np.float64]:
    """Predict class-one probabilities from explicitly standardized features.

    Ordered feature names must match fitting. Caller applies the fitted scaler;
    this operation neither infers nor refits preprocessing. One native thread.
    """
    x=_matrix(features,feature_names);a,b,c,d=_parameters(state,feature_names)
    with threadpool_limits(limits=1),np.errstate(over='raise',invalid='raise',divide='raise'):
        logits=np.maximum(x@a+b,0)@c+d
        logits-=logits.max(axis=1,keepdims=True)
        exponential=np.exp(logits)
        result=exponential[:,1]/exponential.sum(axis=1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite neural predictions')
    return result
