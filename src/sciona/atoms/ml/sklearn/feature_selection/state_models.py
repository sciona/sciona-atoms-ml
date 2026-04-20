"""State containers for sklearn feature-selection atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UnivariateSelectionState:
    """Fitted univariate selector scores and support mask."""

    scores: NDArray[np.float64]
    pvalues: NDArray[np.float64] | None
    support_mask: NDArray[np.bool_]
    n_features_in: int
    score_func: str
    selector: str
    selector_param: int | float | str
