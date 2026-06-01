"""Ghost witnesses for sklearn LARS CV refit callback atoms."""

from __future__ import annotations


def witness_lars_cv_refit_state_payload(best_alpha: float, cv_alphas: object, mse_path: object) -> dict[str, object]:
    """Describe the selected CV state stored before the final refit callback."""
    return {"alpha_": best_alpha, "cv_alphas_": cv_alphas, "mse_path_": mse_path}


def witness_lars_cv_refit_fit_kwargs(max_iter: int, best_alpha: float) -> dict[str, object]:
    """Describe the keyword payload passed to the final LarsCV._fit callback."""
    return {"max_iter": int(max_iter), "alpha": float(best_alpha), "Xy": None, "fit_path": True}


def witness_lars_cv_refit_fit_call(X: object, y: object, kwargs: object) -> dict[str, object]:
    """Describe the positional and keyword payload for the final LarsCV._fit callback."""
    return {"args": (X, y), "kwargs": kwargs}


def witness_lars_cv_fit_return_self(estimator: object) -> object:
    """Describe the identity returned by LarsCV.fit after the refit callback."""
    return estimator
