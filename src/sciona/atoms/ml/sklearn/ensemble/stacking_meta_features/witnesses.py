"""Ghost witnesses for stacking meta-feature helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_prediction_entry(entry: object, method: str, is_binary_classification: bool) -> tuple[int, int]:
    if method not in {"predict", "predict_proba", "decision_function"}:
        raise ValueError("stack methods must be predict, predict_proba, or decision_function")

    if isinstance(entry, tuple):
        if len(entry) < 1:
            raise ValueError("tuple prediction entries must be nonempty")
        n_samples: int | None = None
        total_width = 0
        for block in entry:
            if not isinstance(block, AbstractArray) or len(block.shape) != 2:
                raise ValueError("tuple prediction blocks must be 2D arrays")
            rows, cols = int(block.shape[0]), int(block.shape[1])
            if rows < 1 or cols < 2:
                raise ValueError("tuple prediction blocks must be nonempty with at least two columns")
            if n_samples is None:
                n_samples = rows
            elif rows != n_samples:
                raise ValueError("tuple prediction blocks must share a sample count")
            total_width += cols - 1
        return int(n_samples), total_width

    if not isinstance(entry, AbstractArray):
        raise ValueError("prediction entries must be arrays or tuples of arrays")
    if len(entry.shape) == 1:
        length = int(entry.shape[0])
        if length < 1:
            raise ValueError("1D prediction entries must be nonempty")
        return length, 1
    if len(entry.shape) != 2:
        raise ValueError("2D prediction entries must be matrices")
    rows, cols = int(entry.shape[0]), int(entry.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("2D prediction entries must be nonempty")
    if method == "predict_proba" and is_binary_classification:
        if cols < 2:
            raise ValueError("binary predict_proba entries must have at least two columns")
        return rows, cols - 1
    return rows, cols


def _check_prediction_entries(
    predictions: tuple[object, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
) -> tuple[int, tuple[int, ...], int]:
    if len(predictions) < 1 or len(predictions) != len(stack_method_names):
        raise ValueError("predictions and stack_method_names must be nonempty tuples with matching length")
    n_samples: int | None = None
    widths: list[int] = []
    for entry, method in zip(predictions, stack_method_names):
        rows, width = _check_prediction_entry(entry, method, is_binary_classification)
        if n_samples is None:
            n_samples = rows
        elif rows != n_samples:
            raise ValueError("prediction entries must share a sample count")
        widths.append(width)
    return int(n_samples), tuple(widths), int(sum(widths))


def witness_stacking_meta_feature_matrix(
    predictions: tuple[object, ...],
    stack_method_names: tuple[str, ...],
    *,
    is_binary_classification: bool,
    X: AbstractArray | None = None,
    passthrough: bool = False,
) -> AbstractArray:
    """Describe sklearn stacking meta-feature construction from supplied predictions."""
    n_samples, _, total_width = _check_prediction_entries(
        predictions,
        stack_method_names,
        is_binary_classification,
    )
    if passthrough:
        if not isinstance(X, AbstractArray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D matrix when passthrough is enabled")
        rows, cols = int(X.shape[0]), int(X.shape[1])
        if rows != n_samples or cols < 1:
            raise ValueError("X must match the prediction sample count and be nonempty")
        total_width += cols
    return AbstractArray(shape=(n_samples, total_width), dtype="float64")


def witness_stacking_meta_feature_widths(
    predictions: tuple[object, ...],
    stack_method_names: tuple[str, ...],
    *,
    is_binary_classification: bool,
) -> AbstractArray:
    """Describe sklearn stacking meta-feature widths derived from supplied predictions."""
    _, widths, _ = _check_prediction_entries(
        predictions,
        stack_method_names,
        is_binary_classification,
    )
    return AbstractArray(shape=(len(widths),), dtype="int64", min_val=1.0)


def witness_stacking_feature_names_out(
    class_name: str,
    estimator_names: tuple[str, ...],
    feature_widths: AbstractArray,
    *,
    input_features: AbstractArray | None = None,
    passthrough: bool = False,
) -> AbstractArray:
    """Describe stacking output feature names from estimator names and width counts."""
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("class_name must be nonempty")
    if len(estimator_names) < 1:
        raise ValueError("estimator_names must be nonempty")
    if len(feature_widths.shape) != 1:
        raise ValueError("feature_widths must be a vector")
    if int(feature_widths.shape[0]) != len(estimator_names):
        raise ValueError("feature_widths must align with estimator_names")
    total = 0
    for width in feature_widths.values:
        if int(width) < 1:
            raise ValueError("feature widths must be positive")
        total += int(width)
    if passthrough:
        if not isinstance(input_features, AbstractArray) or len(input_features.shape) != 1:
            raise ValueError("input_features must be a vector when passthrough is enabled")
        total += int(input_features.shape[0])
    return AbstractArray(shape=(total,), dtype="object")
