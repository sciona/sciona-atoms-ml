"""Draft Community boundaries for explicit causal period-demand forecasting."""
from sciona.ghost.registry import register_atom


def witness_future_sales_prepare(payload: dict) -> dict:return {'kind':'FutureSales.Prepared'}


@register_atom(witness_future_sales_prepare)
def future_sales_prepare(payload: dict) -> object:
    """Validate private entity/period populations and serialized controls."""
    from sciona.future_sales_contract import prepare
    return prepare(payload)


def witness_future_sales_execute(prepared: object) -> dict:
    if prepared!={'kind':'FutureSales.Prepared'}:raise ValueError('Prepared Future Sales input required')
    return {'kind':'FutureSales.Result'}


@register_atom(witness_future_sales_execute)
def future_sales_execute(prepared: object) -> dict:
    """Aggregate, build causal features, search, refit and forecast one period."""
    from sciona.future_sales_contract import execute
    return execute(prepared)
