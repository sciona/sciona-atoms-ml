"""Generic finite-resource route planning provider stages, pending Tier 3 review."""
from sciona.ghost.registry import register_atom


def witness_resource_route_encode(payload: dict) -> dict:
    return {'kind':'ResourceRoute.State'}


@register_atom(witness_resource_route_encode)
def resource_route_encode(payload: dict) -> object:
    """Route planning stage: encode_state."""
    from sciona.resource_route_planning import encode_state
    return encode_state(payload)


def witness_resource_route_generate(state: object) -> dict:
    if state!={'kind':'ResourceRoute.State'}:raise ValueError('Required route witness missing')
    return {'kind':'ResourceRoute.Candidates'}


@register_atom(witness_resource_route_generate)
def resource_route_generate(state: object) -> object:
    """Route planning stage: generate_candidates."""
    from sciona.resource_route_planning import generate_candidates
    return generate_candidates(state)


def witness_resource_route_search(candidates: object) -> dict:
    if candidates!={'kind':'ResourceRoute.Candidates'}:raise ValueError('Required route witness missing')
    return {'kind':'ResourceRoute.Searched'}


@register_atom(witness_resource_route_search)
def resource_route_search(candidates: object) -> object:
    """Route planning stage: search_routes."""
    from sciona.resource_route_planning import search_routes
    return search_routes(candidates)


def witness_resource_route_validate(searched: object) -> dict:
    if searched!={'kind':'ResourceRoute.Searched'}:raise ValueError('Required route witness missing')
    return {'kind':'ResourceRoute.Validated'}


@register_atom(witness_resource_route_validate)
def resource_route_validate(searched: object) -> object:
    """Route planning stage: validate_route."""
    from sciona.resource_route_planning import validate_route
    return validate_route(searched)


def witness_resource_route_select(validated: object) -> dict:
    if validated!={'kind':'ResourceRoute.Validated'}:raise ValueError('Required route witness missing')
    return {'kind':'ResourceRoute.Result'}


@register_atom(witness_resource_route_select)
def resource_route_select(validated: object) -> dict:
    """Route planning stage: select_plan."""
    from sciona.resource_route_planning import select_plan
    return select_plan(validated)
