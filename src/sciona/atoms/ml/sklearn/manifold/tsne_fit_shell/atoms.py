"""t-SNE fit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_tsne_fit_return_self


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


@register_atom(witness_tsne_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(
    lambda result, estimator_token: isinstance(result, str) and result == estimator_token,
    "result must return the estimator token unchanged",
)
def tsne_fit_return_self(estimator_token: str) -> str:
    """Model TSNE.fit returning the fitted estimator itself."""
    return estimator_token
