"""Explicit physical feature calculation and numerical view assembly."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def _unit(value,expected):
    if value.dim is not None and not value.dim.is_compatible(expected):
        raise ValueError('Incompatible physical units')


def witness_positive_longitudinal_kinematics(momentum: AbstractArray,transverse_momentum: AbstractArray,
        parent_transverse_momentum: AbstractArray,flight_distance: AbstractArray,lifetime: AbstractArray) -> tuple:
    if len(momentum.shape)!=2 or min(momentum.shape)<1 or transverse_momentum.shape!=momentum.shape:
        raise ValueError('Aligned nonempty daughter matrices required')
    rows=momentum.shape[0];momentum_unit=DimensionalSignature(M=1,L=1,T=-1)
    for value in (momentum,transverse_momentum,parent_transverse_momentum):_unit(value,momentum_unit)
    _unit(flight_distance,DimensionalSignature(L=1));_unit(lifetime,DimensionalSignature(T=1))
    if any(v.dtype!='float64' for v in (momentum,transverse_momentum,parent_transverse_momentum,flight_distance,lifetime)):
        raise ValueError('Float64 physical quantities required')
    if any(v.shape!=(rows,) for v in (parent_transverse_momentum,flight_distance,lifetime)):
        raise ValueError('Physical quantity rows must align')
    return tuple(AbstractArray(shape=(rows,),dtype='float64',dim=d) for d in
        (DimensionalSignature(M=1),momentum_unit,DimensionalSignature(L=1,T=-1),momentum_unit))


@register_atom(witness_positive_longitudinal_kinematics)
def positive_longitudinal_kinematics(momentum: NDArray[np.float64],transverse_momentum: NDArray[np.float64],
        parent_transverse_momentum: NDArray[np.float64],flight_distance: NDArray[np.float64],lifetime: NDArray[np.float64]) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    """Return momentum/speed mass proxy, total momentum, speed, longitudinal sum.

    Daughter count is arbitrary and positive. The positive square-root branch
    is explicit: no inferred longitudinal signs, relativistic invariant-mass
    claim or unit conversion. Caller supplies consistent momentum, length and
    time units. Invalid magnitudes fail rather than being clipped.
    """
    p,pt,parent,distance,time=map(np.asarray,(momentum,transverse_momentum,parent_transverse_momentum,flight_distance,lifetime))
    witness_positive_longitudinal_kinematics(*(AbstractArray(shape=v.shape,dtype=str(v.dtype)) for v in (p,pt,parent,distance,time)))
    if any(not np.isfinite(v).all() for v in (p,pt,parent,distance,time)):
        raise ValueError('Finite physical quantities required')
    if np.any(p<0) or np.any(pt<0) or np.any(pt>p) or np.any(parent<0) or np.any(distance<=0) or np.any(time<=0):
        raise ValueError('Invalid physical magnitudes')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        longitudinal=(np.sqrt(p-pt)*np.sqrt(p)*np.sqrt(1+pt/np.where(p>0,p,1))).sum(axis=1)
        total=np.hypot(parent,longitudinal);speed=distance/time;proxy=total*(time/distance)
    if any(not np.isfinite(v).all() for v in (proxy,total,speed,longitudinal)):raise ValueError('Nonfinite derived physical quantities')
    return proxy,total,speed,longitudinal


def witness_assemble_source_feature_views(base: AbstractArray,mass_proxy: AbstractArray,total_momentum: AbstractArray,
        speed: AbstractArray,longitudinal: AbstractArray,excluded_column: int) -> dict:
    if len(base.shape)!=2 or base.shape[0]<1 or base.shape[1]!=46 or base.dtype!='float64':
        raise ValueError('Nonempty float64 base matrix with 46 ordered columns required')
    if isinstance(excluded_column,(bool,np.bool_)) or not isinstance(excluded_column,(int,np.integer)) or not 0<=excluded_column<46:
        raise ValueError('Explicit excluded column position 0..45 required')
    for value in (mass_proxy,total_momentum,speed,longitudinal):
        if value.shape!=(base.shape[0],) or value.dtype!='float64':raise ValueError('Aligned float64 physical vectors required')
    _unit(mass_proxy,DimensionalSignature(M=1));_unit(total_momentum,DimensionalSignature(M=1,L=1,T=-1))
    _unit(longitudinal,DimensionalSignature(M=1,L=1,T=-1));_unit(speed,DimensionalSignature(L=1,T=-1))
    # Heterogeneous feature columns do not have a single physical dimension.
    return {key:AbstractArray(shape=(base.shape[0],width),dtype='float64') for key,width in
        [('base',46),('restricted',45),('proxy',47),('regression',50)]} | {'mass_proxy':mass_proxy}


@register_atom(witness_assemble_source_feature_views)
def assemble_source_feature_views(base: NDArray[np.float64],mass_proxy: NDArray[np.float64],total_momentum: NDArray[np.float64],
        speed: NDArray[np.float64],longitudinal: NDArray[np.float64],excluded_column: int) -> dict:
    """Assemble the source's explicit positional views from separate quantities.

    Exactly 46 caller-ordered base columns; no dataset schema, feature lookup,
    inferred exclusion or unit conversion. Regression appends proxy, total
    momentum, speed and longitudinal sum in that order. Inputs are not mutated.
    """
    values=list(map(np.asarray,(base,mass_proxy,total_momentum,speed,longitudinal)))
    witness_assemble_source_feature_views(*(AbstractArray(shape=v.shape,dtype=str(v.dtype)) for v in values),excluded_column)
    if any(not np.isfinite(v).all() for v in values):raise ValueError('Finite feature values required')
    b,proxy,total,velocity,z=values
    return dict(base=b.copy(),restricted=np.delete(b,excluded_column,axis=1),proxy=np.column_stack([b,proxy]),
        regression=np.column_stack([b,proxy,total,velocity,z]),mass_proxy=proxy.copy())


def witness_enrich_corrected_mass_views(prepared: dict,corrected_mass: AbstractArray) -> dict:
    if not isinstance(prepared,dict) or set(prepared)!={'base','restricted','proxy','regression','mass_proxy'}:
        raise ValueError('Complete prepared source views required')
    rows=corrected_mass.shape[0] if len(corrected_mass.shape)==1 else 0
    if rows<1 or corrected_mass.dtype!='float64':raise ValueError('Float64 corrected mass vector required')
    _unit(corrected_mass,DimensionalSignature(M=1))
    for key,width in [('base',46),('restricted',45),('proxy',47),('regression',50)]:
        if prepared[key].shape!=(rows,width) or prepared[key].dtype!='float64':raise ValueError('Prepared views must align')
    if prepared['mass_proxy'].shape!=(rows,) or prepared['mass_proxy'].dtype!='float64':raise ValueError('Aligned mass proxy required')
    _unit(prepared['mass_proxy'],DimensionalSignature(M=1))
    return {key:prepared[key] for key in ('base','restricted','proxy')} | {'corrected':AbstractArray(shape=(rows,50),dtype='float64')}


@register_atom(witness_enrich_corrected_mass_views)
def enrich_corrected_mass_views(prepared: dict,corrected_mass: NDArray[np.float64]) -> dict:
    """Append proxy, corrected mass, residual and source-stabilized ratio.

    The 1e-10 offset is in the caller's mass units and is a fixed source choice,
    not unit-invariant regularization. Corrected features do not establish
    nested downstream validation. No fitting or evaluation controls enter here.
    """
    if not isinstance(prepared,dict) or set(prepared)!={'base','restricted','proxy','regression','mass_proxy'}:
        raise ValueError('Complete prepared source views required')
    arrays={k:np.asarray(v) for k,v in prepared.items()};mass=np.asarray(corrected_mass)
    witness_enrich_corrected_mass_views({k:AbstractArray(shape=v.shape,dtype=str(v.dtype)) for k,v in arrays.items()},
        AbstractArray(shape=mass.shape,dtype=str(mass.dtype)))
    if any(not np.isfinite(v).all() for v in (*arrays.values(),mass)):raise ValueError('Finite prepared and corrected features required')
    proxy=arrays['mass_proxy']
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        ratio=(proxy+1e-10)/(mass+1e-10)
        enriched=np.column_stack([arrays['base'],proxy,mass,proxy-mass,ratio])
    if not np.isfinite(enriched).all():raise ValueError('Nonfinite corrected features')
    return {key:arrays[key].copy() for key in ('base','restricted','proxy')} | {'corrected':enriched}
