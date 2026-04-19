from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiLabelBinarizer,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.preprocessing import add_dummy_feature as sklearn_add_dummy_feature
from sklearn.preprocessing import binarize as sklearn_binarize
from sklearn.preprocessing import label_binarize as sklearn_label_binarize
from sklearn.preprocessing import maxabs_scale as sklearn_maxabs_scale
from sklearn.preprocessing import minmax_scale as sklearn_minmax_scale
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.preprocessing import power_transform as sklearn_power_transform
from sklearn.preprocessing import quantile_transform as sklearn_quantile_transform
from sklearn.preprocessing import robust_scale as sklearn_robust_scale
from sklearn.preprocessing import scale as sklearn_scale


def test_preprocessing_basic_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        add_dummy_feature,
        binarize,
        binarizer_transform,
        kbins_discretizer_fit,
        kbins_discretizer_fit_transform,
        kbins_discretizer_inverse_transform,
        kbins_discretizer_transform,
        kernel_centerer_fit,
        kernel_centerer_transform,
        label_binarize,
        label_binarizer_fit,
        label_binarizer_fit_transform,
        label_binarizer_inverse_transform,
        label_binarizer_transform,
        label_encoder_fit,
        label_encoder_fit_transform,
        label_encoder_inverse_transform,
        label_encoder_transform,
        maxabs_scale,
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_partial_fit,
        maxabs_scaler_transform,
        minmax_scale,
        minmax_scaler_fit,
        minmax_scaler_inverse_transform,
        minmax_scaler_partial_fit,
        minmax_scaler_transform,
        multi_label_binarizer_fit,
        multi_label_binarizer_fit_transform,
        multi_label_binarizer_inverse_transform,
        multi_label_binarizer_transform,
        normalize,
        normalizer_transform,
        polynomial_features_fit,
        polynomial_features_fit_transform,
        polynomial_features_transform,
        power_transform,
        power_transformer_fit,
        power_transformer_fit_transform,
        power_transformer_inverse_transform,
        power_transformer_transform,
        quantile_transform,
        quantile_transformer_fit,
        quantile_transformer_fit_transform,
        quantile_transformer_inverse_transform,
        quantile_transformer_transform,
        robust_scale,
        robust_scaler_fit,
        robust_scaler_inverse_transform,
        robust_scaler_transform,
        scale,
        spline_transformer_fit,
        spline_transformer_fit_transform,
        spline_transformer_transform,
        standard_scaler_fit,
        standard_scaler_inverse_transform,
        standard_scaler_partial_fit,
        standard_scaler_transform,
    )

    assert callable(add_dummy_feature)
    assert callable(binarize)
    assert callable(binarizer_transform)
    assert callable(kbins_discretizer_fit)
    assert callable(kbins_discretizer_fit_transform)
    assert callable(kbins_discretizer_inverse_transform)
    assert callable(kbins_discretizer_transform)
    assert callable(kernel_centerer_fit)
    assert callable(kernel_centerer_transform)
    assert callable(label_binarize)
    assert callable(label_binarizer_fit)
    assert callable(label_binarizer_fit_transform)
    assert callable(label_binarizer_inverse_transform)
    assert callable(label_binarizer_transform)
    assert callable(label_encoder_fit)
    assert callable(label_encoder_fit_transform)
    assert callable(label_encoder_inverse_transform)
    assert callable(label_encoder_transform)
    assert callable(maxabs_scale)
    assert callable(maxabs_scaler_fit)
    assert callable(maxabs_scaler_inverse_transform)
    assert callable(maxabs_scaler_partial_fit)
    assert callable(maxabs_scaler_transform)
    assert callable(minmax_scale)
    assert callable(minmax_scaler_fit)
    assert callable(minmax_scaler_inverse_transform)
    assert callable(minmax_scaler_partial_fit)
    assert callable(minmax_scaler_transform)
    assert callable(multi_label_binarizer_fit)
    assert callable(multi_label_binarizer_fit_transform)
    assert callable(multi_label_binarizer_inverse_transform)
    assert callable(multi_label_binarizer_transform)
    assert callable(normalize)
    assert callable(normalizer_transform)
    assert callable(polynomial_features_fit)
    assert callable(polynomial_features_fit_transform)
    assert callable(polynomial_features_transform)
    assert callable(power_transform)
    assert callable(power_transformer_fit)
    assert callable(power_transformer_fit_transform)
    assert callable(power_transformer_inverse_transform)
    assert callable(power_transformer_transform)
    assert callable(quantile_transform)
    assert callable(quantile_transformer_fit)
    assert callable(quantile_transformer_fit_transform)
    assert callable(quantile_transformer_inverse_transform)
    assert callable(quantile_transformer_transform)
    assert callable(robust_scale)
    assert callable(robust_scaler_fit)
    assert callable(robust_scaler_inverse_transform)
    assert callable(robust_scaler_transform)
    assert callable(scale)
    assert callable(spline_transformer_fit)
    assert callable(spline_transformer_fit_transform)
    assert callable(spline_transformer_transform)
    assert callable(standard_scaler_fit)
    assert callable(standard_scaler_inverse_transform)
    assert callable(standard_scaler_partial_fit)
    assert callable(standard_scaler_transform)


def test_add_dummy_feature_matches_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import add_dummy_feature

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    assert np.array_equal(add_dummy_feature(X, value=2.5), sklearn_add_dummy_feature(X, value=2.5))


def test_add_dummy_feature_matches_sklearn_sparse_formats() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import add_dummy_feature

    X_dense = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float64)
    for matrix_factory in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        X = matrix_factory(X_dense)
        result = add_dummy_feature(X, value=-1.0)
        expected = sklearn_add_dummy_feature(X, value=-1.0)
        assert result.getformat() == expected.getformat()
        assert np.array_equal(result.toarray(), expected.toarray())


def test_binarize_matches_sklearn_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarize

    X = np.array([[0.2, 0.6, 1.2], [0.5, -1.0, 0.0]], dtype=np.float64)
    assert np.array_equal(binarize(X, threshold=0.5), sklearn_binarize(X, threshold=0.5))

    X_sparse = sp.csr_matrix([[0.0, 0.7, 0.2], [1.4, 0.0, 0.3]], dtype=np.float64)
    result = binarize(X_sparse, threshold=0.25)
    expected = sklearn_binarize(X_sparse, threshold=0.25)
    assert np.array_equal(result.toarray(), expected.toarray())


def test_binarize_sparse_negative_threshold_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarize

    X = sp.csr_matrix([[0.0, 1.0]])
    with pytest.raises(ValueError, match="threshold < 0"):
        binarize(X, threshold=-0.1)
    with pytest.raises(ValueError, match="threshold < 0"):
        sklearn_binarize(X, threshold=-0.1)


def test_binarizer_transform_matches_sklearn_transform() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarizer_transform

    X = np.array([[1.0, -1.0, 2.0], [0.2, 0.4, 0.6]], dtype=np.float64)
    transformer = Binarizer(threshold=0.3).fit(X)
    assert np.array_equal(binarizer_transform(X, threshold=0.3), transformer.transform(X))

    X_sparse = sp.csr_matrix([[0.0, 0.7, 0.2], [1.4, 0.0, 0.3]], dtype=np.float64)
    sparse_result = binarizer_transform(X_sparse, threshold=0.25)
    sparse_expected = Binarizer(threshold=0.25).fit(X_sparse).transform(X_sparse)
    assert np.array_equal(sparse_result.toarray(), sparse_expected.toarray())


def test_normalize_matches_sklearn_dense_norms_and_axes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        assert np.allclose(normalize(X, norm=norm, axis=1), sklearn_normalize(X, norm=norm, axis=1))
        assert np.allclose(normalize(X, norm=norm, axis=0), sklearn_normalize(X, norm=norm, axis=0))

    result, norms = normalize(X, norm="l2", axis=1, return_norm=True)
    expected, expected_norms = sklearn_normalize(X, norm="l2", axis=1, return_norm=True)
    assert np.allclose(result, expected)
    assert np.allclose(norms, expected_norms)


def test_normalize_matches_sklearn_sparse_l1_l2_and_max_norms() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = sp.csr_matrix([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        result = normalize(X, norm=norm, axis=1)
        expected = sklearn_normalize(X, norm=norm, axis=1)
        assert np.allclose(result.toarray(), expected.toarray())

    max_result, max_norms = normalize(X, norm="max", axis=1, return_norm=True)
    expected_result, expected_norms = sklearn_normalize(X, norm="max", axis=1, return_norm=True)
    assert np.allclose(max_result.toarray(), expected_result.toarray())
    assert np.allclose(max_norms, expected_norms)


def test_normalize_sparse_return_norm_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = sp.csr_matrix([[1.0, 2.0]])
    with pytest.raises(ValueError, match="return_norm=True"):
        normalize(X, norm="l2", return_norm=True)
    with pytest.raises(NotImplementedError, match="return_norm=True"):
        sklearn_normalize(X, norm="l2", return_norm=True)


def test_normalizer_transform_matches_sklearn_transform() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalizer_transform

    X = np.array([[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        transformer = Normalizer(norm=norm).fit(X)
        assert np.allclose(normalizer_transform(X, norm=norm), transformer.transform(X))

    X_sparse = sp.csr_matrix(X)
    sparse_result = normalizer_transform(X_sparse, norm="l2")
    sparse_expected = Normalizer(norm="l2").fit(X_sparse).transform(X_sparse)
    assert np.allclose(sparse_result.toarray(), sparse_expected.toarray())


def test_kernel_centerer_fit_matches_sklearn_state() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit

    K = np.array([[9.0, 2.0, -2.0], [2.0, 14.0, -13.0], [-2.0, -13.0, 21.0]], dtype=np.float64)
    state = kernel_centerer_fit(K)
    expected = KernelCenterer().fit(K)

    assert np.allclose(state.k_fit_rows, expected.K_fit_rows_)
    assert np.isclose(state.k_fit_all, expected.K_fit_all_)
    assert state.n_features_in == expected.n_features_in_


def test_kernel_centerer_transform_matches_sklearn_square_and_rectangular() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit, kernel_centerer_transform

    K_train = np.array([[9.0, 2.0, -2.0], [2.0, 14.0, -13.0], [-2.0, -13.0, 21.0]], dtype=np.float64)
    state = kernel_centerer_fit(K_train)
    transformer = KernelCenterer().fit(K_train)
    assert np.allclose(kernel_centerer_transform(K_train, state), transformer.transform(K_train))

    K_test = np.array([[1.0, 4.0, 5.0], [2.0, -1.0, 3.0]], dtype=np.float64)
    assert np.allclose(kernel_centerer_transform(K_test, state), transformer.transform(K_test))


def test_kernel_centerer_errors_match_sklearn_shape_requirements() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit, kernel_centerer_transform

    K_rectangular = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    with pytest.raises(Exception, match="square"):
        kernel_centerer_fit(K_rectangular)
    with pytest.raises(ValueError, match="square"):
        KernelCenterer().fit(K_rectangular)

    state = kernel_centerer_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="columns"):
        kernel_centerer_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        KernelCenterer().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


def test_kbins_discretizer_uniform_ordinal_and_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        kbins_discretizer_fit,
        kbins_discretizer_inverse_transform,
        kbins_discretizer_transform,
    )

    X = np.array([[-2.0, 1.0], [-1.0, 2.0], [0.0, 3.0], [1.0, 4.0]], dtype=np.float64)
    state = kbins_discretizer_fit(X, n_bins=3, encode="ordinal", strategy="uniform")
    expected = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform").fit(X)
    transformed = kbins_discretizer_transform(X, state)
    expected_transformed = expected.transform(X)

    assert np.array_equal(state.n_bins, expected.n_bins_)
    for result_edges, expected_edges in zip(state.bin_edges, expected.bin_edges_):
        assert np.allclose(result_edges, expected_edges)
    assert np.allclose(transformed, expected_transformed)
    assert np.allclose(kbins_discretizer_inverse_transform(transformed, state), expected.inverse_transform(expected_transformed))


def test_kbins_discretizer_quantile_onehot_dense_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kbins_discretizer_fit_transform

    X = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 20.0], [4.0, 21.0]], dtype=np.float64)
    with pytest.warns(FutureWarning, match="quantile_method"):
        state, transformed = kbins_discretizer_fit_transform(
            X,
            n_bins=np.array([3, 2]),
            encode="onehot-dense",
            strategy="quantile",
        )
    with pytest.warns(FutureWarning, match="quantile_method"):
        expected = KBinsDiscretizer(n_bins=np.array([3, 2]), encode="onehot-dense", strategy="quantile").fit(X)

    assert np.array_equal(state.n_bins, expected.n_bins_)
    assert np.allclose(transformed, expected.transform(X))


def test_kbins_discretizer_onehot_sparse_and_kmeans_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kbins_discretizer_fit, kbins_discretizer_transform

    X = np.array([[-3.0, 0.0], [-2.0, 0.5], [-1.0, 1.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float64)
    state = kbins_discretizer_fit(X, n_bins=3, encode="onehot", strategy="kmeans", random_state=0)
    expected = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="kmeans", random_state=0).fit(X)
    result = kbins_discretizer_transform(X, state)
    expected_result = expected.transform(X)

    assert sp.issparse(result)
    assert result.getformat() == expected_result.getformat()
    assert np.allclose(result.toarray(), expected_result.toarray())


def test_kbins_discretizer_constant_feature_and_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kbins_discretizer_fit, kbins_discretizer_transform

    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    with pytest.warns(UserWarning, match="constant"):
        state = kbins_discretizer_fit(X, n_bins=3, encode="ordinal", strategy="uniform")
    with pytest.warns(UserWarning, match="constant"):
        expected = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform").fit(X)
    assert np.array_equal(state.n_bins, expected.n_bins_)
    assert np.allclose(kbins_discretizer_transform(X, state), expected.transform(X))

    with pytest.raises(Exception, match="at least 2"):
        kbins_discretizer_fit(X, n_bins=1)
    with pytest.raises(ValueError, match="range \\[2"):
        KBinsDiscretizer(n_bins=1).fit(X)

    with pytest.raises(Exception, match="feature"):
        kbins_discretizer_transform(np.ones((2, 3), dtype=np.float64), state)
    with pytest.raises(ValueError):
        expected.transform(np.ones((2, 3), dtype=np.float64))


def test_label_encoder_fit_and_transform_match_sklearn_numeric_labels() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_encoder_fit, label_encoder_transform

    y = np.array([2, 2, 6, 1, 6])
    state = label_encoder_fit(y)
    expected = LabelEncoder().fit(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.array_equal(label_encoder_transform([6, 1, 2], state), expected.transform([6, 1, 2]))


def test_label_encoder_fit_transform_matches_sklearn_string_labels() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_encoder_fit_transform

    y = np.array(["paris", "tokyo", "paris", "amsterdam"], dtype=object)
    state, encoded = label_encoder_fit_transform(y)
    expected = LabelEncoder()
    expected_encoded = expected.fit_transform(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.array_equal(encoded, expected_encoded)


def test_label_encoder_inverse_transform_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_encoder_fit, label_encoder_inverse_transform

    y = np.array(["low", "medium", "high", "medium"], dtype=object)
    state = label_encoder_fit(y)
    expected = LabelEncoder().fit(y)
    encoded = np.array([2, 0, 1, 2])

    assert np.array_equal(label_encoder_inverse_transform(encoded, state), expected.inverse_transform(encoded))


def test_label_encoder_empty_and_unseen_label_behaviors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        label_encoder_fit,
        label_encoder_inverse_transform,
        label_encoder_transform,
    )

    state = label_encoder_fit(["a", "b"])
    expected = LabelEncoder().fit(["a", "b"])

    assert np.array_equal(label_encoder_transform([], state), expected.transform([]))
    assert np.array_equal(label_encoder_inverse_transform([], state), expected.inverse_transform([]))

    with pytest.raises(ValueError, match="previously unseen labels"):
        label_encoder_transform(["c"], state)
    with pytest.raises(ValueError, match="previously unseen labels"):
        expected.transform(["c"])

    with pytest.raises(ValueError, match="previously unseen labels"):
        label_encoder_inverse_transform([2], state)
    with pytest.raises(ValueError, match="previously unseen labels"):
        expected.inverse_transform([2])


def test_label_binarize_multiclass_and_ordering_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarize

    y = np.array([1, 6, 2, 6])
    classes = np.array([1, 6, 4, 2])
    assert np.array_equal(label_binarize(y, classes=classes), sklearn_label_binarize(y, classes=classes))


def test_label_binarize_binary_and_custom_labels_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarize

    y = np.array(["yes", "no", "no", "yes"], dtype=object)
    classes = np.array(["no", "yes"], dtype=object)
    result = label_binarize(y, classes=classes, neg_label=-1, pos_label=2)
    expected = sklearn_label_binarize(y, classes=classes, neg_label=-1, pos_label=2)

    assert np.array_equal(result, expected)


def test_label_binarize_sparse_output_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarize

    y = np.array([1, 3, 2, 1])
    classes = np.array([1, 2, 3])
    result = label_binarize(y, classes=classes, sparse_output=True)
    expected = sklearn_label_binarize(y, classes=classes, sparse_output=True)

    assert sp.issparse(result)
    assert np.array_equal(result.toarray(), expected.toarray())


def test_label_binarize_multilabel_indicator_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarize

    y = np.array([[1, 0, 1], [0, 1, 0]])
    classes = np.array([0, 1, 2])

    assert np.array_equal(label_binarize(y, classes=classes), sklearn_label_binarize(y, classes=classes))


def test_label_binarize_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarize

    with pytest.raises(ValueError, match="strictly less"):
        label_binarize([1, 2], classes=[1, 2], neg_label=1, pos_label=1)
    with pytest.raises(ValueError, match="strictly less"):
        sklearn_label_binarize([1, 2], classes=[1, 2], neg_label=1, pos_label=1)

    with pytest.raises(ValueError, match="Sparse binarization"):
        label_binarize([1, 2], classes=[1, 2], neg_label=-1, sparse_output=True)
    with pytest.raises(ValueError, match="Sparse binarization"):
        sklearn_label_binarize([1, 2], classes=[1, 2], neg_label=-1, sparse_output=True)


def test_label_binarizer_fit_and_transform_match_sklearn_binary() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarizer_fit, label_binarizer_transform

    y = np.array(["yes", "no", "no", "yes"], dtype=object)
    state = label_binarizer_fit(y, neg_label=-1, pos_label=2)
    expected = LabelBinarizer(neg_label=-1, pos_label=2).fit(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert state.y_type == expected.y_type_
    assert state.sparse_input == expected.sparse_input_
    assert np.array_equal(label_binarizer_transform(y, state), expected.transform(y))


def test_label_binarizer_fit_transform_matches_sklearn_multiclass() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarizer_fit_transform

    y = np.array([1, 2, 6, 4, 2])
    state, transformed = label_binarizer_fit_transform(y)
    expected = LabelBinarizer()
    expected_transformed = expected.fit_transform(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert state.y_type == expected.y_type_
    assert np.array_equal(transformed, expected_transformed)


def test_label_binarizer_inverse_transform_matches_sklearn_multiclass_scores() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        label_binarizer_fit,
        label_binarizer_inverse_transform,
        label_binarizer_transform,
    )

    y = np.array([1, 2, 6, 4, 2])
    state = label_binarizer_fit(y)
    expected = LabelBinarizer().fit(y)
    scores = np.array([[0.1, 0.2, 0.7, 0.0], [0.8, 0.1, 0.0, 0.1]])

    assert np.array_equal(label_binarizer_transform([1, 6], state), expected.transform([1, 6]))
    assert np.array_equal(label_binarizer_inverse_transform(scores, state), expected.inverse_transform(scores))


def test_label_binarizer_sparse_multilabel_inverse_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarizer_fit, label_binarizer_inverse_transform

    y = sp.csr_matrix([[0, 1, 1], [1, 0, 0]])
    state = label_binarizer_fit(y)
    expected = LabelBinarizer().fit(y)
    Y = np.array([[1, 0, 1], [0, 1, 0]])

    result = label_binarizer_inverse_transform(Y, state)
    expected_result = expected.inverse_transform(Y)

    assert sp.issparse(result)
    assert np.array_equal(result.toarray(), expected_result.toarray())


def test_label_binarizer_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import label_binarizer_fit, label_binarizer_transform

    with pytest.raises(ValueError, match="strictly less"):
        label_binarizer_fit([1, 2], neg_label=1, pos_label=1)
    with pytest.raises(ValueError, match="strictly less"):
        LabelBinarizer(neg_label=1, pos_label=1).fit([1, 2])

    state = label_binarizer_fit([1, 2])
    y_multilabel = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="not fitted with multilabel"):
        label_binarizer_transform(y_multilabel, state)
    with pytest.raises(ValueError, match="not fitted with multilabel"):
        LabelBinarizer().fit([1, 2]).transform(y_multilabel)


def test_multi_label_binarizer_fit_transform_matches_sklearn_default_classes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import multi_label_binarizer_fit_transform

    y = [("sci-fi", "thriller"), ("comedy",), ("thriller", "comedy")]
    state, transformed = multi_label_binarizer_fit_transform(y)
    expected = MultiLabelBinarizer()
    expected_transformed = expected.fit_transform(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.array_equal(transformed, expected_transformed)


def test_multi_label_binarizer_fit_and_transform_match_sklearn_with_classes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import multi_label_binarizer_fit, multi_label_binarizer_transform

    y = [(1, 3), (2,), (1, 2, 3)]
    classes = [3, 2, 1, 4]
    state = multi_label_binarizer_fit(y, classes=classes)
    expected = MultiLabelBinarizer(classes=classes).fit(y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.array_equal(multi_label_binarizer_transform(y, state), expected.transform(y))


def test_multi_label_binarizer_sparse_output_and_unknown_warning_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import multi_label_binarizer_fit, multi_label_binarizer_transform

    train = [("red",), ("blue",)]
    test = [("red", "green"), ("blue",)]
    state = multi_label_binarizer_fit(train, sparse_output=True)
    expected = MultiLabelBinarizer(sparse_output=True).fit(train)

    with pytest.warns(UserWarning, match="unknown class"):
        result = multi_label_binarizer_transform(test, state)
    with pytest.warns(UserWarning, match="unknown class"):
        expected_result = expected.transform(test)

    assert sp.issparse(result)
    assert np.array_equal(result.toarray(), expected_result.toarray())


def test_multi_label_binarizer_inverse_transform_matches_sklearn_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import multi_label_binarizer_fit, multi_label_binarizer_inverse_transform

    y = [("a", "b"), ("c",)]
    state = multi_label_binarizer_fit(y)
    expected = MultiLabelBinarizer().fit(y)
    indicator = np.array([[1, 1, 0], [0, 0, 1]])

    assert multi_label_binarizer_inverse_transform(indicator, state) == expected.inverse_transform(indicator)
    sparse_indicator = sp.csr_matrix(indicator)
    assert multi_label_binarizer_inverse_transform(sparse_indicator, state) == expected.inverse_transform(sparse_indicator)


def test_multi_label_binarizer_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import multi_label_binarizer_fit, multi_label_binarizer_inverse_transform

    with pytest.raises(ValueError, match="duplicate"):
        multi_label_binarizer_fit([("a",)], classes=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        MultiLabelBinarizer(classes=["a", "a"]).fit([("a",)])

    state = multi_label_binarizer_fit([("a",), ("b",)])
    bad_indicator = np.array([[1, 2]])
    with pytest.raises(ValueError, match="Expected only 0s and 1s"):
        multi_label_binarizer_inverse_transform(bad_indicator, state)
    with pytest.raises(ValueError, match="Expected only 0s and 1s"):
        MultiLabelBinarizer().fit([("a",), ("b",)]).inverse_transform(bad_indicator)


def test_polynomial_features_fit_transform_matches_sklearn_dense_degree_two() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import polynomial_features_fit_transform

    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    state, transformed = polynomial_features_fit_transform(X, degree=2)
    expected = PolynomialFeatures(degree=2).fit(X)

    assert np.array_equal(state.powers, expected.powers_)
    assert state.n_output_features == expected.n_output_features_
    assert np.allclose(transformed, expected.transform(X))


def test_polynomial_features_transform_matches_sklearn_interaction_only() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import polynomial_features_fit, polynomial_features_transform

    X = np.array([[1.0, 2.0, 3.0], [2.0, 0.5, 4.0]])
    state = polynomial_features_fit(X, degree=3, interaction_only=True, include_bias=False)
    expected = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False).fit(X)

    assert np.array_equal(state.powers, expected.powers_)
    assert np.allclose(polynomial_features_transform(X, state), expected.transform(X))


def test_polynomial_features_degree_range_and_order_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import polynomial_features_fit_transform

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    state, transformed = polynomial_features_fit_transform(X, degree=(2, 3), include_bias=True, order="F")
    expected = PolynomialFeatures(degree=(2, 3), include_bias=True, order="F").fit(X)
    expected_transformed = expected.transform(X)

    assert np.array_equal(state.powers, expected.powers_)
    assert np.allclose(transformed, expected_transformed)
    assert transformed.flags["F_CONTIGUOUS"] == expected_transformed.flags["F_CONTIGUOUS"]


def test_polynomial_features_sparse_matches_sklearn_values() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import polynomial_features_fit, polynomial_features_transform

    X = sp.csr_matrix([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    state = polynomial_features_fit(X, degree=2, include_bias=True)
    expected = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    result = polynomial_features_transform(X, state)
    expected_result = expected.transform(X)

    assert sp.issparse(result)
    assert np.array_equal(state.powers, expected.powers_)
    assert np.allclose(result.toarray(), expected_result.toarray())


def test_polynomial_features_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import polynomial_features_fit, polynomial_features_transform

    X = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="empty output"):
        polynomial_features_fit(X, degree=0, include_bias=False)
    with pytest.raises(ValueError, match="empty output"):
        PolynomialFeatures(degree=0, include_bias=False).fit(X)

    state = polynomial_features_fit(X)
    with pytest.raises(Exception, match="feature count"):
        polynomial_features_transform(np.ones((2, 3), dtype=np.float64), state)
    with pytest.raises(ValueError):
        PolynomialFeatures().fit(X).transform(np.ones((2, 3), dtype=np.float64))


def test_spline_transformer_fit_transform_matches_sklearn_uniform() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import spline_transformer_fit_transform

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    state, transformed = spline_transformer_fit_transform(X, n_knots=4, degree=2, include_bias=True)
    expected = SplineTransformer(n_knots=4, degree=2, include_bias=True).fit(X)

    assert state.n_features_out == expected.n_features_out_
    assert transformed.shape == expected.transform(X).shape
    assert np.allclose(transformed, expected.transform(X))


def test_spline_transformer_transform_extrapolations_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import spline_transformer_fit, spline_transformer_transform

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    X_query = np.array([[-1.0], [0.5], [4.0]], dtype=np.float64)
    for extrapolation in ("constant", "linear", "continue", "periodic"):
        state = spline_transformer_fit(X, n_knots=3, degree=2, extrapolation=extrapolation, include_bias=False)
        expected = SplineTransformer(n_knots=3, degree=2, extrapolation=extrapolation, include_bias=False).fit(X)
        assert np.allclose(spline_transformer_transform(X_query, state), expected.transform(X_query))


def test_spline_transformer_quantile_sparse_and_missing_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import spline_transformer_fit, spline_transformer_transform

    X = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    weights = np.array([1.0, 0.5, 2.0, 1.0], dtype=np.float64)
    state = spline_transformer_fit(
        X,
        n_knots=3,
        degree=2,
        knots="quantile",
        handle_missing="zeros",
        sparse_output=True,
        sample_weight=weights,
    )
    expected = SplineTransformer(
        n_knots=3,
        degree=2,
        knots="quantile",
        handle_missing="zeros",
        sparse_output=True,
    ).fit(X, sample_weight=weights)
    result = spline_transformer_transform(X, state)
    expected_result = expected.transform(X)

    assert sp.issparse(result)
    assert result.format == expected_result.format
    assert np.allclose(result.toarray(), expected_result.toarray(), equal_nan=True)


def test_spline_transformer_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import spline_transformer_fit, spline_transformer_transform

    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="degree < n_knots"):
        spline_transformer_fit(X, n_knots=2, degree=2, extrapolation="periodic")
    with pytest.raises(ValueError, match="degree < n_knots"):
        SplineTransformer(n_knots=2, degree=2, extrapolation="periodic").fit(X)

    state = spline_transformer_fit(X, n_knots=3, degree=2, extrapolation="error")
    with pytest.raises(ValueError, match="beyond the limits"):
        spline_transformer_transform(np.array([[-1.0]], dtype=np.float64), state)
    with pytest.raises(ValueError, match="beyond the limits"):
        SplineTransformer(n_knots=3, degree=2, extrapolation="error").fit(X).transform(np.array([[-1.0]], dtype=np.float64))


def test_power_transform_matches_sklearn_yeo_johnson_standardized() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import power_transform

    X = np.array([[-2.0, 1.0], [-0.5, 2.0], [1.0, 4.0], [3.0, 8.0]], dtype=np.float64)
    assert np.allclose(power_transform(X), sklearn_power_transform(X), atol=1e-7)


def test_power_transformer_fit_transform_matches_sklearn_box_cox() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import power_transformer_fit_transform

    X = np.array([[1.0, 2.0], [3.0, 2.5], [4.0, 5.0], [8.0, 10.0]], dtype=np.float64)
    state, transformed = power_transformer_fit_transform(X, method="box-cox", standardize=False)
    expected = PowerTransformer(method="box-cox", standardize=False).fit(X)

    assert np.allclose(state.lambdas, expected.lambdas_)
    assert np.allclose(transformed, expected.transform(X))


def test_power_transformer_transform_and_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        power_transformer_fit,
        power_transformer_inverse_transform,
        power_transformer_transform,
    )

    X = np.array([[-2.0, 1.0], [-0.5, 2.0], [1.0, 4.0], [3.0, 8.0]], dtype=np.float64)
    state = power_transformer_fit(X, standardize=True)
    expected = PowerTransformer(standardize=True).fit(X)

    transformed = power_transformer_transform(X, state)
    assert np.allclose(state.lambdas, expected.lambdas_)
    assert np.allclose(transformed, expected.transform(X), atol=1e-7)
    assert np.allclose(power_transformer_inverse_transform(transformed, state), expected.inverse_transform(expected.transform(X)))


def test_power_transformer_nan_and_constant_behavior_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import power_transformer_fit, power_transformer_transform

    X = np.array([[1.0, np.nan], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], dtype=np.float64)
    state = power_transformer_fit(X, method="yeo-johnson", standardize=True)
    expected = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)

    assert np.allclose(state.lambdas, expected.lambdas_, equal_nan=True)
    assert np.allclose(power_transformer_transform(X, state), expected.transform(X), equal_nan=True)


def test_power_transformer_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import power_transformer_fit, power_transformer_transform

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="strictly positive"):
        power_transformer_fit(X, method="box-cox")
    with pytest.raises(ValueError, match="strictly positive"):
        PowerTransformer(method="box-cox").fit(X)

    state = power_transformer_fit(np.ones((3, 2), dtype=np.float64))
    with pytest.raises(Exception, match="feature"):
        power_transformer_transform(np.ones((3, 3), dtype=np.float64), state)
    with pytest.raises(ValueError):
        PowerTransformer().fit(np.ones((3, 2), dtype=np.float64)).transform(np.ones((3, 3), dtype=np.float64))


def test_quantile_transform_matches_sklearn_dense_axes_and_distributions() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import quantile_transform

    X = np.array([[0.0, 1.0, 5.0], [2.0, 1.0, 3.0], [4.0, 2.0, 1.0], [8.0, 3.0, 0.0]], dtype=np.float64)
    assert np.allclose(
        quantile_transform(X, n_quantiles=4, random_state=0),
        sklearn_quantile_transform(X, n_quantiles=4, random_state=0),
    )
    assert np.allclose(
        quantile_transform(X, axis=1, n_quantiles=3, output_distribution="normal", random_state=0),
        sklearn_quantile_transform(X, axis=1, n_quantiles=3, output_distribution="normal", random_state=0),
    )


def test_quantile_transformer_fit_transform_matches_sklearn_state() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import quantile_transformer_fit_transform

    X = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 2.0], [8.0, 3.0]], dtype=np.float64)
    state, transformed = quantile_transformer_fit_transform(
        X,
        n_quantiles=4,
        output_distribution="normal",
        random_state=0,
    )
    expected = QuantileTransformer(n_quantiles=4, output_distribution="normal", random_state=0).fit(X)

    assert np.allclose(state.quantiles, expected.quantiles_)
    assert np.allclose(state.references, expected.references_)
    assert state.n_quantiles == expected.n_quantiles_
    assert np.allclose(transformed, expected.transform(X))


def test_quantile_transformer_transform_and_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        quantile_transformer_fit,
        quantile_transformer_inverse_transform,
        quantile_transformer_transform,
    )

    X = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 2.0], [8.0, 3.0]], dtype=np.float64)
    state = quantile_transformer_fit(X, n_quantiles=4, output_distribution="uniform", random_state=0)
    expected = QuantileTransformer(n_quantiles=4, output_distribution="uniform", random_state=0).fit(X)
    query = np.array([[-1.0, 0.5], [2.0, 1.5], [10.0, 4.0]], dtype=np.float64)

    transformed = quantile_transformer_transform(query, state)
    expected_transformed = expected.transform(query)
    assert np.allclose(transformed, expected_transformed)
    assert np.allclose(quantile_transformer_inverse_transform(transformed, state), expected.inverse_transform(expected_transformed))


def test_quantile_transformer_sparse_and_nan_behavior_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import quantile_transformer_fit, quantile_transformer_transform

    X = sp.csc_matrix([[0.0, 1.0, np.nan], [2.0, 0.0, 3.0], [4.0, 2.0, 0.0]], dtype=np.float64)
    state = quantile_transformer_fit(X, n_quantiles=3, ignore_implicit_zeros=True, random_state=0)
    expected = QuantileTransformer(n_quantiles=3, ignore_implicit_zeros=True, random_state=0).fit(X)
    result = quantile_transformer_transform(X, state)
    expected_result = expected.transform(X)

    assert sp.issparse(result)
    assert result.getformat() == expected_result.getformat()
    assert np.allclose(state.quantiles, expected.quantiles_, equal_nan=True)
    assert np.allclose(result.toarray(), expected_result.toarray(), equal_nan=True)


def test_quantile_transformer_warnings_and_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import quantile_transformer_fit, quantile_transformer_transform

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.warns(UserWarning, match="n_quantiles"):
        state = quantile_transformer_fit(X, n_quantiles=5)
    with pytest.warns(UserWarning, match="n_quantiles"):
        expected = QuantileTransformer(n_quantiles=5).fit(X)
    assert state.n_quantiles == expected.n_quantiles_

    with pytest.warns(UserWarning, match="ignore_implicit_zeros"):
        quantile_transformer_fit(X, n_quantiles=2, ignore_implicit_zeros=True)
    with pytest.warns(UserWarning, match="ignore_implicit_zeros"):
        QuantileTransformer(n_quantiles=2, ignore_implicit_zeros=True).fit(X)

    with pytest.raises(ValueError, match="cannot be greater"):
        quantile_transformer_fit(X, n_quantiles=3, subsample=2)
    with pytest.raises(ValueError, match="cannot be greater"):
        QuantileTransformer(n_quantiles=3, subsample=2).fit(X)

    X_sparse_negative = sp.csc_matrix([[0.0, -1.0], [2.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative sparse"):
        quantile_transformer_fit(X_sparse_negative, n_quantiles=2, ignore_implicit_zeros=False)
    with pytest.raises(ValueError, match="non-negative sparse"):
        QuantileTransformer(n_quantiles=2, ignore_implicit_zeros=False).fit(X_sparse_negative)

    state = quantile_transformer_fit(np.ones((3, 2), dtype=np.float64), n_quantiles=3)
    with pytest.raises(Exception, match="feature"):
        quantile_transformer_transform(np.ones((3, 3), dtype=np.float64), state)
    with pytest.raises(ValueError):
        QuantileTransformer(n_quantiles=3).fit(np.ones((3, 2), dtype=np.float64)).transform(np.ones((3, 3), dtype=np.float64))


def test_scale_matches_sklearn_dense_axes_and_options() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 1.0]], dtype=np.float64)
    assert np.allclose(scale(X, axis=0), sklearn_scale(X, axis=0), equal_nan=True)
    assert np.allclose(scale(X, axis=1), sklearn_scale(X, axis=1), equal_nan=True)
    assert np.allclose(
        scale(X, axis=0, with_mean=False, with_std=True),
        sklearn_scale(X, axis=0, with_mean=False, with_std=True),
        equal_nan=True,
    )
    assert np.allclose(
        scale(X, axis=1, with_mean=True, with_std=False),
        sklearn_scale(X, axis=1, with_mean=True, with_std=False),
        equal_nan=True,
    )


def test_scale_matches_sklearn_sparse_axis0_without_centering() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = sp.csc_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    result = scale(X, axis=0, with_mean=False, with_std=True)
    expected = sklearn_scale(X, axis=0, with_mean=False, with_std=True)
    assert np.allclose(result.toarray(), expected.toarray())


def test_scale_sparse_centering_and_axis_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = sp.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        scale(X, with_mean=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        sklearn_scale(X, with_mean=True)
    with pytest.raises(ValueError, match="axis=0"):
        scale(X, axis=1, with_mean=False)
    with pytest.raises(ValueError, match="axis=0"):
        sklearn_scale(X, axis=1, with_mean=False)


def test_standard_scaler_fit_matches_sklearn_dense_with_weights() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import standard_scaler_fit

    X = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    weights = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)
    state = standard_scaler_fit(X, sample_weight=weights)
    expected = StandardScaler().fit(X, sample_weight=weights)
    expected_seen = np.asarray(expected.n_samples_seen_, dtype=np.float64)
    if expected_seen.ndim == 0:
        expected_seen = np.repeat(expected_seen, X.shape[1])

    assert np.allclose(state.mean, expected.mean_, equal_nan=True)
    assert np.allclose(state.var, expected.var_, equal_nan=True)
    assert np.allclose(state.scale, expected.scale_, equal_nan=True)
    assert np.allclose(state.n_samples_seen, expected_seen)
    assert state.with_mean is True
    assert state.with_std is True
    assert state.n_features_in == expected.n_features_in_


def test_standard_scaler_partial_fit_accumulates_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import standard_scaler_partial_fit

    X1 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    X2 = np.array([[2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    state = standard_scaler_partial_fit(X1)
    state = standard_scaler_partial_fit(X2, state)
    expected = StandardScaler().partial_fit(X1).partial_fit(X2)
    expected_seen = np.asarray(expected.n_samples_seen_, dtype=np.float64)
    if expected_seen.ndim == 0:
        expected_seen = np.repeat(expected_seen, X1.shape[1])

    assert np.allclose(state.mean, expected.mean_)
    assert np.allclose(state.var, expected.var_)
    assert np.allclose(state.scale, expected.scale_)
    assert np.allclose(state.n_samples_seen, expected_seen)


def test_standard_scaler_transform_inverse_match_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        standard_scaler_fit,
        standard_scaler_inverse_transform,
        standard_scaler_transform,
    )

    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    X_test = np.array([[4.0, 8.0], [-1.0, 2.0]], dtype=np.float64)
    state = standard_scaler_fit(X)
    scaler = StandardScaler().fit(X)
    transformed = standard_scaler_transform(X_test, state)
    expected = scaler.transform(X_test)
    assert np.allclose(transformed, expected)
    assert np.allclose(standard_scaler_inverse_transform(transformed, state), scaler.inverse_transform(expected))


def test_standard_scaler_sparse_without_mean_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        standard_scaler_fit,
        standard_scaler_inverse_transform,
        standard_scaler_transform,
    )

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    state = standard_scaler_fit(X, with_mean=False)
    scaler = StandardScaler(with_mean=False).fit(X)
    expected_seen = np.asarray(scaler.n_samples_seen_, dtype=np.float64)
    if expected_seen.ndim == 0:
        expected_seen = np.repeat(expected_seen, X.shape[1])

    assert np.allclose(state.mean, scaler.mean_)
    assert np.allclose(state.var, scaler.var_)
    assert np.allclose(state.scale, scaler.scale_)
    assert np.allclose(state.n_samples_seen, expected_seen)
    result = standard_scaler_transform(X, state)
    expected = scaler.transform(X)
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.allclose(standard_scaler_inverse_transform(result, state).toarray(), scaler.inverse_transform(expected).toarray())


def test_standard_scaler_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import standard_scaler_fit, standard_scaler_transform

    X_sparse = sp.csr_matrix([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        standard_scaler_fit(X_sparse, with_mean=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        StandardScaler(with_mean=True).fit(X_sparse)

    state = standard_scaler_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="feature count"):
        standard_scaler_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        StandardScaler().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


def test_maxabs_scale_matches_sklearn_dense_axes_and_1d() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    assert np.allclose(maxabs_scale(X, axis=0), sklearn_maxabs_scale(X, axis=0), equal_nan=True)
    assert np.allclose(maxabs_scale(X, axis=1), sklearn_maxabs_scale(X, axis=1), equal_nan=True)

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(maxabs_scale(x), sklearn_maxabs_scale(x))


def test_maxabs_scale_matches_sklearn_sparse_axes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scale

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    for axis in (0, 1):
        result = maxabs_scale(X, axis=axis)
        expected = sklearn_maxabs_scale(X, axis=axis)
        assert np.allclose(result.toarray(), expected.toarray())


def test_maxabs_scaler_fit_matches_sklearn_state_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_fit

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    expected = MaxAbsScaler().fit(X)
    assert np.allclose(state.scale, expected.scale_, equal_nan=True)
    assert np.allclose(state.max_abs, expected.max_abs_, equal_nan=True)
    assert state.n_features_in == expected.n_features_in_
    assert state.n_samples_seen == expected.n_samples_seen_

    X_sparse = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    sparse_state = maxabs_scaler_fit(X_sparse)
    sparse_expected = MaxAbsScaler().fit(X_sparse)
    assert np.allclose(sparse_state.scale, sparse_expected.scale_)
    assert np.allclose(sparse_state.max_abs, sparse_expected.max_abs_)


def test_maxabs_scaler_partial_fit_accumulates_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_partial_fit

    X1 = np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    X2 = np.array([[4.0, -1.0, -5.0]], dtype=np.float64)
    state = maxabs_scaler_partial_fit(X1)
    state = maxabs_scaler_partial_fit(X2, state)

    expected = MaxAbsScaler().partial_fit(X1).partial_fit(X2)
    assert np.allclose(state.scale, expected.scale_)
    assert np.allclose(state.max_abs, expected.max_abs_)
    assert state.n_samples_seen == expected.n_samples_seen_


def test_maxabs_scaler_transform_inverse_and_clip_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_transform,
    )

    X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    X_test = np.array([[4.0, -2.0, 4.0], [1.0, 0.5, -3.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    scaler = MaxAbsScaler().fit(X)

    transformed = maxabs_scaler_transform(X_test, state)
    expected = scaler.transform(X_test)
    assert np.allclose(transformed, expected)
    assert np.allclose(maxabs_scaler_inverse_transform(transformed, state), scaler.inverse_transform(expected))

    clipped = maxabs_scaler_transform(X_test, state, clip=True)
    expected_clipped = MaxAbsScaler(clip=True).fit(X).transform(X_test)
    assert np.allclose(clipped, expected_clipped)


def test_maxabs_scaler_sparse_transform_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_transform,
    )

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    scaler = MaxAbsScaler().fit(X)
    result = maxabs_scaler_transform(X, state)
    expected = scaler.transform(X)
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.allclose(maxabs_scaler_inverse_transform(result, state).toarray(), scaler.inverse_transform(expected).toarray())


def test_maxabs_scaler_feature_mismatch_rejected_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_fit, maxabs_scaler_transform

    state = maxabs_scaler_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="feature count"):
        maxabs_scaler_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        MaxAbsScaler().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


def test_minmax_scale_matches_sklearn_dense_axes_and_range() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    assert np.allclose(minmax_scale(X, axis=0), sklearn_minmax_scale(X, axis=0), equal_nan=True)
    assert np.allclose(
        minmax_scale(X, feature_range=(-1.0, 2.0), axis=1),
        sklearn_minmax_scale(X, feature_range=(-1.0, 2.0), axis=1),
        equal_nan=True,
    )

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(minmax_scale(x, feature_range=(-2.0, 3.0)), sklearn_minmax_scale(x, feature_range=(-2.0, 3.0)))


def test_minmax_scale_rejects_invalid_range_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scale

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(Exception):
        minmax_scale(X, feature_range=(1.0, 1.0))
    with pytest.raises(ValueError):
        sklearn_minmax_scale(X, feature_range=(1.0, 1.0))


def test_minmax_scaler_fit_matches_sklearn_state() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scaler_fit

    X = np.array([[-1.0, 2.0], [-0.5, 6.0], [0.0, 10.0], [1.0, np.nan]], dtype=np.float64)
    state = minmax_scaler_fit(X, feature_range=(-2.0, 3.0))
    expected = MinMaxScaler(feature_range=(-2.0, 3.0)).fit(X)

    assert np.allclose(state.scale, expected.scale_, equal_nan=True)
    assert np.allclose(state.min_, expected.min_, equal_nan=True)
    assert np.allclose(state.data_min, expected.data_min_, equal_nan=True)
    assert np.allclose(state.data_max, expected.data_max_, equal_nan=True)
    assert np.allclose(state.data_range, expected.data_range_, equal_nan=True)
    assert state.feature_range == (-2.0, 3.0)
    assert state.n_features_in == expected.n_features_in_
    assert state.n_samples_seen == expected.n_samples_seen_


def test_minmax_scaler_partial_fit_accumulates_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scaler_partial_fit

    X1 = np.array([[-1.0, 2.0], [0.0, 6.0]], dtype=np.float64)
    X2 = np.array([[1.0, 10.0], [-0.5, 18.0]], dtype=np.float64)
    state = minmax_scaler_partial_fit(X1, feature_range=(-1.0, 2.0))
    state = minmax_scaler_partial_fit(X2, state=state)

    expected = MinMaxScaler(feature_range=(-1.0, 2.0)).partial_fit(X1).partial_fit(X2)
    assert np.allclose(state.scale, expected.scale_)
    assert np.allclose(state.min_, expected.min_)
    assert np.allclose(state.data_min, expected.data_min_)
    assert np.allclose(state.data_max, expected.data_max_)
    assert state.n_samples_seen == expected.n_samples_seen_


def test_minmax_scaler_transform_inverse_and_clip_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        minmax_scaler_fit,
        minmax_scaler_inverse_transform,
        minmax_scaler_transform,
    )

    X = np.array([[-1.0, 2.0], [-0.5, 6.0], [0.0, 10.0], [1.0, 18.0]], dtype=np.float64)
    X_test = np.array([[2.0, 2.0], [-2.0, 20.0]], dtype=np.float64)
    state = minmax_scaler_fit(X, feature_range=(-1.0, 2.0))
    scaler = MinMaxScaler(feature_range=(-1.0, 2.0)).fit(X)

    transformed = minmax_scaler_transform(X_test, state)
    expected = scaler.transform(X_test)
    assert np.allclose(transformed, expected)
    assert np.allclose(minmax_scaler_inverse_transform(transformed, state), scaler.inverse_transform(expected))

    clipped = minmax_scaler_transform(X_test, state, clip=True)
    expected_clipped = MinMaxScaler(feature_range=(-1.0, 2.0), clip=True).fit(X).transform(X_test)
    assert np.allclose(clipped, expected_clipped)


def test_minmax_scaler_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scaler_fit, minmax_scaler_transform

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(Exception, match="feature_range"):
        minmax_scaler_fit(X, feature_range=(1.0, 1.0))
    with pytest.raises(ValueError, match="feature range"):
        MinMaxScaler(feature_range=(1.0, 1.0)).fit(X)

    X_sparse = sp.csr_matrix(X)
    with pytest.raises(TypeError, match="sparse input"):
        minmax_scaler_fit(X_sparse)
    with pytest.raises(TypeError, match="sparse input"):
        MinMaxScaler().fit(X_sparse)

    state = minmax_scaler_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="feature count"):
        minmax_scaler_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        MinMaxScaler().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


def test_robust_scale_matches_sklearn_dense_axes_options_and_1d() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [10.0, 2.0, np.nan]], dtype=np.float64)
    assert np.allclose(robust_scale(X, axis=0), sklearn_robust_scale(X, axis=0), equal_nan=True)
    assert np.allclose(
        robust_scale(X, axis=1, quantile_range=(10.0, 90.0), unit_variance=True),
        sklearn_robust_scale(X, axis=1, quantile_range=(10.0, 90.0), unit_variance=True),
        equal_nan=True,
    )
    assert np.allclose(
        robust_scale(X, axis=0, with_centering=False, with_scaling=True),
        sklearn_robust_scale(X, axis=0, with_centering=False, with_scaling=True),
        equal_nan=True,
    )

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(robust_scale(x), sklearn_robust_scale(x))


def test_robust_scale_matches_sklearn_sparse_without_centering() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    for axis in (0, 1):
        result = robust_scale(X, axis=axis, with_centering=False)
        expected = sklearn_robust_scale(X, axis=axis, with_centering=False)
        assert np.allclose(result.toarray(), expected.toarray())


def test_robust_scale_sparse_centering_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        robust_scale(X, with_centering=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        sklearn_robust_scale(X, with_centering=True)


def test_robust_scaler_fit_matches_sklearn_state_dense_options() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scaler_fit

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [10.0, 2.0, np.nan]], dtype=np.float64)
    state = robust_scaler_fit(X, quantile_range=(10.0, 90.0), unit_variance=True)
    expected = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True).fit(X)

    assert np.allclose(state.center, expected.center_, equal_nan=True)
    assert np.allclose(state.scale, expected.scale_, equal_nan=True)
    assert state.with_centering is True
    assert state.with_scaling is True
    assert state.quantile_range == (10.0, 90.0)
    assert state.unit_variance is True
    assert state.n_features_in == expected.n_features_in_

    no_center_state = robust_scaler_fit(X, with_centering=False, with_scaling=False)
    no_center_expected = RobustScaler(with_centering=False, with_scaling=False).fit(X)
    assert no_center_state.center is None
    assert no_center_state.scale is None
    assert no_center_expected.center_ is None
    assert no_center_expected.scale_ is None


def test_robust_scaler_sparse_fit_without_centering_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scaler_fit

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    state = robust_scaler_fit(X, with_centering=False)
    expected = RobustScaler(with_centering=False).fit(X)
    assert state.center is None
    assert np.allclose(state.scale, expected.scale_)


def test_robust_scaler_transform_inverse_match_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        robust_scaler_fit,
        robust_scaler_inverse_transform,
        robust_scaler_transform,
    )

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [10.0, 2.0, -3.0]], dtype=np.float64)
    X_test = np.array([[3.0, 1.5, 5.0], [-5.0, 0.5, 0.0]], dtype=np.float64)
    state = robust_scaler_fit(X, quantile_range=(20.0, 80.0))
    scaler = RobustScaler(quantile_range=(20.0, 80.0)).fit(X)
    transformed = robust_scaler_transform(X_test, state)
    expected = scaler.transform(X_test)
    assert np.allclose(transformed, expected)
    assert np.allclose(robust_scaler_inverse_transform(transformed, state), scaler.inverse_transform(expected))


def test_robust_scaler_sparse_transform_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        robust_scaler_fit,
        robust_scaler_inverse_transform,
        robust_scaler_transform,
    )

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    state = robust_scaler_fit(X, with_centering=False)
    scaler = RobustScaler(with_centering=False).fit(X)
    result = robust_scaler_transform(X, state)
    expected = scaler.transform(X)
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.allclose(robust_scaler_inverse_transform(result, state).toarray(), scaler.inverse_transform(expected).toarray())


def test_robust_scaler_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scaler_fit, robust_scaler_transform

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(Exception, match="quantile"):
        robust_scaler_fit(X, quantile_range=(-1.0, 75.0))
    with pytest.raises(ValueError, match="Invalid quantile range"):
        RobustScaler(quantile_range=(-1.0, 75.0)).fit(X)

    X_sparse = sp.csr_matrix(X)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        robust_scaler_fit(X_sparse, with_centering=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        RobustScaler(with_centering=True).fit(X_sparse)

    state = robust_scaler_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="feature count"):
        robust_scaler_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        RobustScaler().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))
