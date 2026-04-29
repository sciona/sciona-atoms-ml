"""One-vs-rest prediction-output helper atoms."""

from .atoms import (
    one_vs_rest_predict_argmaxima_init,
    one_vs_rest_predict_labels_from_argmaxima,
    one_vs_rest_predict_maxima_init,
    one_vs_rest_predict_multiclass_update,
)

__all__ = [
    "one_vs_rest_predict_argmaxima_init",
    "one_vs_rest_predict_labels_from_argmaxima",
    "one_vs_rest_predict_maxima_init",
    "one_vs_rest_predict_multiclass_update",
]
