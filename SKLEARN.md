# scikit-learn Ingestion Targets

This file is the upstream target inventory for sklearn ingestion.

Execution sequencing, restart-safe handoff notes, difficulty tiers, and the
recommended first-wave targets now live in
[SKLEARN_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/SKLEARN_EXECUTION_PLAN.md).

This inventory is based on the official stable API reference for scikit-learn 1.8.0.

- API reference: https://scikit-learn.org/stable/api/index.html
- Scope for this first ingestion pass: public algorithmic estimators, transformers, kernels, and algorithmic helper functions in the selected sklearn modules below.
- Intentionally excluded for now: datasets, metrics, displays, exceptions, low-level utils, orchestration/composition (`compose`, `pipeline`), and model-selection/search APIs.
- Source locations come from each object page's official `[source]` link. `BallTree` and `KDTree` use the matching source files from the same scikit-learn revision because their generated docs pages do not expose a direct `[source]` link.
- Total targets in this inventory: 16

## Module Summary

| Module | Targets |
| --- | ---: |
| `sklearn.calibration` | 0 |
| `sklearn.cluster` | 0 |
| `sklearn.covariance` | 0 |
| `sklearn.cross_decomposition` | 0 |
| `sklearn.decomposition` | 0 |
| `sklearn.discriminant_analysis` | 0 |
| `sklearn.dummy` | 0 |
| `sklearn.ensemble` | 0 |
| `sklearn.feature_extraction` | 0 |
| `sklearn.feature_extraction.image` | 0 |
| `sklearn.feature_extraction.text` | 0 |
| `sklearn.feature_selection` | 0 |
| `sklearn.gaussian_process` | 0 |
| `sklearn.gaussian_process.kernels` | 0 |
| `sklearn.impute` | 0 |
| `sklearn.inspection` | 0 |
| `sklearn.kernel_approximation` | 0 |
| `sklearn.kernel_ridge` | 0 |
| `sklearn.linear_model` | 0 |
| `sklearn.manifold` | 1 |
| `sklearn.mixture` | 2 |
| `sklearn.multiclass` | 3 |
| `sklearn.multioutput` | 4 |
| `sklearn.naive_bayes` | 0 |
| `sklearn.neighbors` | 3 |
| `sklearn.neural_network` | 3 |
| `sklearn.preprocessing` | 0 |
| `sklearn.semi_supervised` | 0 |
| `sklearn.svm` | 0 |
| `sklearn.tree` | 0 |

## `sklearn.calibration`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.cluster`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.covariance`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.cross_decomposition`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.decomposition`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.discriminant_analysis`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.dummy`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.ensemble`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_extraction`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_extraction.image`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_extraction.text`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_selection`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.gaussian_process`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.gaussian_process.kernels`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.impute`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.inspection`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.kernel_approximation`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.kernel_ridge`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.linear_model`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.manifold`

| Target | Description | Source |
| --- | --- | --- |
| `TSNE` | T-distributed Stochastic Neighbor Embedding. | `sklearn/manifold/_t_sne.py:L560` |

## `sklearn.mixture`

| Target | Description | Source |
| --- | --- | --- |
| `BayesianGaussianMixture` | Variational Bayesian estimation of a Gaussian mixture. | `sklearn/mixture/_bayesian_mixture.py:L74` |
| `GaussianMixture` | Gaussian Mixture. | `sklearn/mixture/_gaussian_mixture.py:L556` |

## `sklearn.multiclass`

| Target | Description | Source |
| --- | --- | --- |
| `OneVsOneClassifier` | One-vs-one multiclass strategy. | `sklearn/multiclass.py:L678` |
| `OneVsRestClassifier` | One-vs-the-rest (OvR) multiclass strategy. | `sklearn/multiclass.py:L202` |
| `OutputCodeClassifier` | (Error-Correcting) Output-Code multiclass strategy. | `sklearn/multiclass.py:L1043` |

## `sklearn.multioutput`

| Target | Description | Source |
| --- | --- | --- |
| `ClassifierChain` | A multi-label model that arranges binary classifiers into a chain. | `sklearn/multioutput.py:L877` |
| `MultiOutputClassifier` | Multi target classification. | `sklearn/multioutput.py:L445` |
| `MultiOutputRegressor` | Multi target regression. | `sklearn/multioutput.py:L342` |
| `RegressorChain` | A multi-label model that arranges regressions into a chain. | `sklearn/multioutput.py:L1167` |

## `sklearn.naive_bayes`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.neighbors`

| Target | Description | Source |
| --- | --- | --- |
| `BallTree` | BallTree for fast generalized N-point problems | `sklearn/neighbors/_ball_tree.pyx.tp:L282` |
| `KDTree` | KDTree for fast generalized N-point problems | `sklearn/neighbors/_kd_tree.pyx.tp:L334` |
| `NeighborhoodComponentsAnalysis` | Neighborhood Components Analysis. | `sklearn/neighbors/_nca.py:L34` |

## `sklearn.neural_network`

| Target | Description | Source |
| --- | --- | --- |
| `BernoulliRBM` | Bernoulli Restricted Boltzmann Machine (RBM). | `sklearn/neural_network/_rbm.py:L25` |
| `MLPClassifier` | Multi-layer Perceptron classifier. | `sklearn/neural_network/_multilayer_perceptron.py:L879` |
| `MLPRegressor` | Multi-layer Perceptron regressor. | `sklearn/neural_network/_multilayer_perceptron.py:L1386` |

## `sklearn.preprocessing`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.semi_supervised`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.svm`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.tree`

| Target | Description | Source |
| --- | --- | --- |
