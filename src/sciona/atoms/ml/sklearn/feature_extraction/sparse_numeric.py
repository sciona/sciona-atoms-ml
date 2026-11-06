"""Iterable numeric feature mappings to sparse matrices with frozen vocabulary."""
from collections.abc import Mapping
from numbers import Real
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete numeric mappings and fitted vocabulary required')


def _records(records):
    count=0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError('Feature mappings required')
        checked={}
        for key,value in record.items():
            if not isinstance(key,str) or not key:
                raise ValueError('Nonempty string feature names required')
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
                raise ValueError('Real numeric feature values required')
            try: number=float(value)
            except (ValueError,OverflowError) as error:
                raise ValueError('Representable finite feature values required') from error
            if not np.isfinite(number):raise ValueError('Finite feature values required')
            checked[key]=number
        count+=1
        yield checked
    if count==0:raise ValueError('At least one feature mapping required')


def _state(state):
    if (not isinstance(state,DictVectorizer) or not state.sparse or not state.sort
            or np.dtype(state.dtype)!=np.dtype('float64')):
        raise ValueError('Sparse sorted float64 DictVectorizer required')
    names=getattr(state,'feature_names_',None)
    vocabulary=getattr(state,'vocabulary_',None)
    if (not isinstance(names,list) or not names or any(not isinstance(n,str) or not n for n in names)
            or names!=sorted(set(names)) or vocabulary!={name:i for i,name in enumerate(names)}):
        raise ValueError('Consistent nonempty fitted numeric vocabulary required')


@register_atom(concrete_only)
def fit_numeric_vocabulary(records: object) -> tuple[object, object]:
    """Consume numeric mappings once and return fitted vocabulary and CSR rows.

    Accepts a nonempty iterable, including a one-shot generator, of mappings
    from nonempty string keys to finite real values (booleans excluded). Signed
    and zero values are accepted. Empty rows are allowed but the total feature
    vocabulary must be nonempty. Columns use sorted feature names. Returns a
    fitted sklearn DictVectorizer and float64 CSR matrix in input row order.

    Source mappings are validated and copied one at a time, never materialized
    as a full list. The output CSR and vocabulary remain resident. Caller owns
    training/query separation and must not mutate fitted state between calls.
    No categorical encoding, labels, scaling or application policy is imposed.
    """
    state=DictVectorizer(sparse=True,sort=True,dtype=np.float64)
    matrix=state.fit_transform(_records(records))
    _state(state)
    return state,matrix


@register_atom(concrete_only)
def transform_numeric_vocabulary(records: object, state: object) -> object:
    """Transform a nonempty numeric-mapping iterable using frozen vocabulary.

    Uses the fit operation's mapping contract and returns float64 CSR in input
    row order with the original column order. Unknown keys are validated then
    ignored; empty mappings yield all-zero rows. Does not refit or mutate state
    or input mappings. Source mappings are consumed once; output CSR is resident.
    """
    _state(state)
    return state.transform(_records(records))
