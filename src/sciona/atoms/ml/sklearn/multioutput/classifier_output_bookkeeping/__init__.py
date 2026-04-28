"""Multioutput classifier output bookkeeping helper atoms."""

from .atoms import (
    multioutput_classifier_probability_blocks,
    multioutput_classifier_score_require_2d_targets,
    multioutput_classifier_score_require_matching_output_count,
    multioutput_predict_require_base_predict_method,
)

__all__ = [
    "multioutput_classifier_probability_blocks",
    "multioutput_classifier_score_require_2d_targets",
    "multioutput_classifier_score_require_matching_output_count",
    "multioutput_predict_require_base_predict_method",
]
