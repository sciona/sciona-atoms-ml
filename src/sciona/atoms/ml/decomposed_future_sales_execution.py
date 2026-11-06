"""Explicit forecasting recipe adapters and reusable numeric training stages."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args,**kwargs):
    raise ValueError('Concrete complete inputs required')


@register_atom(concrete_only)
def unpack(payload: dict) -> tuple[list,list,list,list,list,int,float,dict,float,float,list]:
    """Expose private root populations and explicit recipe controls."""
    from sciona.decomposed_future_sales_contract import unpack as run
    return run(payload)


@register_atom(concrete_only)
def aggregate(transactions: list,entities: list,boundaries: list) -> tuple[np.ndarray,np.ndarray]:
    """Compensated net transaction sums within explicit periods and entities."""
    from sciona.decomposed_future_sales_features import aggregate as run
    return run(transactions,entities,boundaries)


@register_atom(concrete_only)
def bounded_targets(totals: np.ndarray,lower: float,upper: float) -> np.ndarray:
    """Clip numeric observed totals and append one unknown future row."""
    from sciona.decomposed_future_sales_features import bounded_targets as run
    return run(totals,lower,upper)


@register_atom(concrete_only)
def lag_features(targets: np.ndarray,lags: list,rolling_windows: list,rolling_function: object) -> np.ndarray:
    """Prior-only lags and full windows using the externally bound rolling atom."""
    from sciona.decomposed_future_sales_features import lag_features as run
    return run(targets,lags,rolling_windows,rolling_function)


@register_atom(concrete_only)
def encoding_features(targets: np.ndarray,groups: np.ndarray,calendar_cycle: int,cold_start: float) -> np.ndarray:
    """Prior-only numeric group means and explicit cyclic period features."""
    from sciona.decomposed_future_sales_features import encoding_features as run
    return run(targets,groups,calendar_cycle,cold_start)


@register_atom(concrete_only)
def assemble(targets: np.ndarray,lagged: np.ndarray,encoded: np.ndarray,lags: list,rolling_windows: list) -> object:
    """Apply the retained recipe layout to separate numeric feature blocks."""
    from sciona.decomposed_future_sales_features import assemble as run
    return run(targets,lagged,encoded,lags,rolling_windows)


@register_atom(concrete_only)
def search(panel: object,config: dict,categorical_columns: list,lower: float,upper: float) -> dict:
    """Complete isolated TPE search with early stopping and explicit numeric policy."""
    from sciona.decomposed_future_sales_transport import search as run
    return run(panel,config,categorical_columns,lower,upper)


@register_atom(concrete_only)
def refit(panel: object,config: dict,categorical_columns: list,lower: float,upper: float,selection: dict) -> dict:
    """One isolated selected-round refit over observed rows and bounded forecast."""
    from sciona.decomposed_future_sales_transport import refit as run
    return run(panel,config,categorical_columns,lower,upper,selection)
