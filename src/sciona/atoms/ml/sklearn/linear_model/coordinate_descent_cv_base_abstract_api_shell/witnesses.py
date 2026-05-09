"""Ghost witnesses for sklearn LinearModelCV abstract API atoms."""

from __future__ import annotations


def witness_cd_cv_base_abstract_method_names(class_name: object) -> object:
    """Describe abstract method names declared by LinearModelCV."""
    return class_name


def witness_cd_cv_base_abstract_method_roles(method_name: object) -> object:
    """Describe the role attached to each LinearModelCV abstract method."""
    return method_name


def witness_cd_cv_base_path_signature_payload(X: object, y: object, kwargs: object) -> object:
    """Describe the positional and keyword payload accepted by LinearModelCV.path."""
    return X, y, kwargs
