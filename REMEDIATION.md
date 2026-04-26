# Remediation Queue

This file tracks sklearn targets that should not be ingested as publishable
atoms until the decomposition boundary is clarified.

## Remediation decision rules

These rules are authoritative for the target groups below. They refine the
general ingestion rules from `../sciona-atoms/AGENT_INGESTION.md` and prevent
CDG fiction by making every opaque boundary explicit. Where an older section
below says to "decide whether" a wrapper, native decomposition, or helper-first
path is acceptable, apply the case rules here as the current decision.

### Case 1: native and compiled core boundaries

Targets whose public Python API delegates core behavior to compiled kernels may
be ingested at the API level only when the atom surface is honest about that
boundary.

Required treatment:

- Ingest the public Python methods as the atoms.
- Add strict `@icontract` preconditions and postconditions around data entering
  and leaving the compiled kernel boundary.
- Publish the review bundle with `review_semantic_verdict:
  "pass_with_limits"`.
- Add a `limitations` entry stating that the internal algorithmic topology is
  obscured by a compiled FFI boundary.

### Case 2: external optimizer boundaries

Targets that hide the core model math inside a closure or helper passed to
`scipy.optimize`, liblinear, or another solver must not use the orchestration
loop as the primary atom.

Required treatment:

- Ingest objective-function and gradient-function atoms wherever the source
  exposes them.
- Treat the optimizer call itself as an FFI-style boundary under Case 1.
- For models such as logistic regression, prefer atoms like logistic loss and
  logistic gradient over a monolithic `fit` wrapper.

### Case 3: meta-estimator wrappers

Targets that call arbitrary user-provided estimators through `fit`, `predict`,
`predict_proba`, `decision_function`, `score`, or importance callbacks must be
modeled as higher-order functions over explicit protocols.

Required treatment:

- Define contracts against a callable or structured object conforming to a
  Python `Protocol`, such as `SupportsPredictProba`, rather than assuming a
  specific estimator implementation.
- Keep witnesses focused on the meta-estimator combinatorics. For example, a
  voting witness models majority voting or probability averaging over abstract
  base outputs; it does not model how each base estimator produced those
  outputs.

### Case 4: cross-validation and path orchestration

Targets such as `LassoCV` are macro workflows, not mathematical atoms. They
must expand into smaller CDG nodes rather than publish as one opaque atom.

Required treatment:

- Decompose into reusable workflow atoms such as `generate_cv_folds`,
  `fit_and_score_fold`, `aggregate_cv_results`, and `refit_best_model`.
- Model `fit_and_score_fold` as a higher-order atom when it consumes a base
  estimator or scorer.
- Treat the public estimator API as a known CDG wiring template once the
  component atoms exist.

### Case 5: decomposition and matrix-factorization pipelines

Targets that mix Python/NumPy preprocessing with compiled SVD, sparse coding,
KMeans, or other heavy solvers must be sliced horizontally.

Required treatment:

- Extract pure Python or NumPy math steps as explicit, fully verifiable atoms.
- Treat solver calls such as SVD, sparse-code solvers, KMeans, or compiled
  coordinate-descent updates as FFI boundaries under Case 1.
- Preserve intermediate states as atom outputs when those states are useful to
  the matching engine.

### Case 6: clustering and spectral pipelines

Clustering pipelines should expose their graph, matrix, and embedding
intermediates instead of hiding them behind the estimator shell.

Required treatment:

- Break spectral workflows into atoms such as affinity construction, graph
  Laplacian construction, eigenvector solving, and label discretization.
- Mark eigensolver, ARPACK, KMeans, and compiled clustering kernels as
  `pass_with_limits` FFI boundaries.
- Do not publish a full estimator wrapper until the meaningful intermediate
  atoms and solver-boundary atoms exist.

### Case 7: probabilistic and mutable-kernel workflows

Targets with mutable kernels or optimizer-driven posterior state must use
state-passing style so atoms remain pure and side-effect free.

Required treatment:

- Represent updates as functions returning new state, such as
  `new_kernel_state = optimize_hyperparams(X, y, old_kernel_state)`.
- Ingest Gaussian-process likelihood, gradient, posterior linear algebra, and
  kernel-state transition atoms before publishing estimator surfaces.
- Treat L-BFGS-B, Newton loops, and other solver calls as Case 1 or Case 2
  boundaries depending on whether the objective and gradient are exposed.

### Case 8: thin wrappers and aliases

Targets that only route to another estimator or helper stay blocked until the
upstream atoms they depend on are published.

Required treatment:

- Keep the target on the blocked backlog rather than creating a phantom CDG
  node.
- Ingest the wrapper only after the delegated atoms are verified, provenanced,
  and published.
- When ingested, the wrapper should route data to existing CDG nodes rather
  than duplicate or obscure the upstream computation.

## `sklearn.linear_model` coordinate-descent solvers

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `ElasticNet` | `sklearn/linear_model/_coordinate_descent.py:L758` | Fit delegates the core L1/L2 coordinate-descent optimization to compiled `sklearn.linear_model._cd_fast` routines; publishing the estimator shell would hide the solver boundary. |
| `ElasticNetCV` | `sklearn/linear_model/_coordinate_descent.py:L2237` | Cross-validation repeatedly calls the same compiled coordinate-descent path solver before refitting, so a public estimator atom would obscure both solver and CV orchestration boundaries. |
| `enet_path` | `sklearn/linear_model/_coordinate_descent.py:L393` | The public path helper dispatches to compiled dense, sparse, Gram, and multitask coordinate-descent kernels; ingesting only the Python loop would not decompose the optimization algorithm. |
| `Lasso` | `sklearn/linear_model/_coordinate_descent.py:L1205` | This estimator is an ElasticNet specialization whose fit path delegates to compiled coordinate-descent kernels. |
| `lasso_path` | `sklearn/linear_model/_coordinate_descent.py:L199` | The helper is a thin Lasso-specialized wrapper over `enet_path`, which delegates to compiled coordinate-descent kernels. |
| `LassoCV` | `sklearn/linear_model/_coordinate_descent.py:L1970` | Cross-validated Lasso path fitting delegates to the same compiled coordinate-descent solver family and then refits the selected model. |
| `MultiTaskElasticNet` | `sklearn/linear_model/_coordinate_descent.py:L2532` | Fit delegates the mixed-norm multitask optimization to compiled `enet_coordinate_descent_multi_task`; a wrapper atom would hide the solver implementation. |
| `MultiTaskElasticNetCV` | `sklearn/linear_model/_coordinate_descent.py:L2926` | Cross-validation repeatedly calls multitask coordinate-descent paths and refits through the compiled multitask solver boundary. |
| `MultiTaskLasso` | `sklearn/linear_model/_coordinate_descent.py:L2784` | This estimator is a multitask ElasticNet specialization whose fit path delegates to compiled multitask coordinate descent. |
| `MultiTaskLassoCV` | `sklearn/linear_model/_coordinate_descent.py:L3195` | Cross-validated multitask Lasso delegates path search and refit to compiled multitask coordinate-descent kernels. |

Potential remediation path:

- Ingest the underlying `_cd_fast` coordinate-descent kernels through a native
  or FFI-backed decomposition with provenance and parity checks at the solver
  boundary.
- Or define limited estimator-state wrappers with explicit audit limitations
  only after deciding that opaque solver-backed atoms are acceptable.

## `sklearn.linear_model` optimizer and callback boundaries

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `GammaRegressor` | `sklearn/linear_model/_glm/glm.py:L606` | Fit delegates GLM optimization to SciPy L-BFGS-B or sklearn Newton solver classes; publishing only the estimator state would hide the optimizer boundary. |
| `HuberRegressor` | `sklearn/linear_model/_huber.py:L129` | Fit solves the robust objective through `scipy.optimize.minimize(method="L-BFGS-B")`; an atom around the public estimator would not decompose the optimization algorithm. |
| `LogisticRegression` | `sklearn/linear_model/_logistic.py:L735` | Fit dispatches across liblinear, SciPy L-BFGS/Newton-CG, Newton-Cholesky, SAG, and SAGA solvers; a wrapper atom would hide solver-specific training behavior. |
| `LogisticRegressionCV` | `sklearn/linear_model/_logistic.py:L1363` | Cross-validation repeatedly invokes the same logistic solver family and scorer/CV orchestration before refitting. |
| `PassiveAggressiveClassifier` | `sklearn/linear_model/_passive_aggressive.py:L17` | The estimator inherits SGD training machinery and delegates coefficient updates to compiled `_plain_sgd` routines. |
| `PassiveAggressiveRegressor` | `sklearn/linear_model/_passive_aggressive.py:L343` | The regressor inherits SGD training machinery and delegates coefficient updates to compiled `_plain_sgd` routines. |
| `Perceptron` | `sklearn/linear_model/_perceptron.py:L10` | The estimator is a `BaseSGDClassifier` specialization backed by compiled SGD update loops. |
| `PoissonRegressor` | `sklearn/linear_model/_glm/glm.py:L475` | Fit delegates GLM optimization to SciPy L-BFGS-B or sklearn Newton solver classes; a public estimator atom would hide the solver boundary. |
| `QuantileRegressor` | `sklearn/linear_model/_quantile.py:L20` | Fit formulates a linear program and delegates the solve to `scipy.optimize.linprog`; publishing the shell would not decompose the LP solver. |
| `RANSACRegressor` | `sklearn/linear_model/_ransac.py:L81` | The core workflow repeatedly fits and scores a configurable estimator, loss callable, and validity callbacks; a standalone atom would hide estimator behavior behind callback boundaries. |
| `SGDClassifier` | `sklearn/linear_model/_stochastic_gradient.py:L950` | Training delegates stochastic updates and loss gradients to Cython `_plain_sgd` and compiled loss classes. |
| `SGDOneClassSVM` | `sklearn/linear_model/_stochastic_gradient.py:L2117` | Training delegates one-class stochastic updates to Cython `_plain_sgd` and compiled loss/update machinery. |
| `SGDRegressor` | `sklearn/linear_model/_stochastic_gradient.py:L1794` | Training delegates stochastic updates and loss gradients to Cython `_plain_sgd` and compiled loss classes. |
| `TweedieRegressor` | `sklearn/linear_model/_glm/glm.py:L738` | Fit delegates GLM optimization to SciPy L-BFGS-B or sklearn Newton solver classes; a public estimator atom would hide the solver boundary. |

Potential remediation path:

- Completed helper slice: `sklearn.linear_model.glm` now publishes dense
  objective helpers for supplied parameters and data:
  `glm_linear_raw_prediction`, `glm_log_link_half_loss_gradient`, and
  `glm_dense_loss_gradient` for Poisson, Gamma, and Tweedie log-link models.
- Completed helper slice: `sklearn.linear_model.logistic` now publishes binary
  logistic helpers for supplied raw scores, parameters, and dense design
  matrices:
  `binary_logistic_positive_probability`,
  `binary_logistic_half_loss_gradient`, and
  `binary_logistic_dense_loss_gradient`.
- Completed helper slice: `sklearn.linear_model.huber` now publishes supplied
  residual and objective helpers:
  `huber_linear_residuals`, `huber_outlier_mask`, and
  `huber_loss_gradient`.
- Completed helper slice: `sklearn.linear_model.quantile` now publishes
  quantile-regression LP setup helpers for supplied dense inputs:
  `quantile_nonzero_weight_mask`, `quantile_dense_lp_problem`, and
  `quantile_solution_to_params`.
- Completed helper slice: `sklearn.linear_model.ransac` now publishes
  estimator-independent consensus bookkeeping helpers:
  `ransac_default_residual_threshold`, `ransac_loss_residuals`,
  `ransac_inlier_mask`, `ransac_consensus_is_better`, and
  `ransac_dynamic_max_trials`.
- Completed helper slice: `sklearn.linear_model.sgd` now publishes SGD and
  passive-aggressive helper atoms for learning-rate resolution,
  passive-aggressive step-size/config selection, optional `l1_ratio`
  normalization, and modified-Huber probability conversion.
- Decompose reusable, deterministic helper atoms first, such as GLM link/loss
  validation, RANSAC consensus bookkeeping from supplied residuals, or
  prediction from already-fitted coefficients.
- Decide whether optimizer-backed estimator surfaces should be represented as
  limited state wrappers, or ingest the underlying SciPy/sklearn/native solver
  boundaries with explicit provenance and parity tests.

## `sklearn.linear_model` LARS cross-validation orchestration

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `LarsCV` | `sklearn/linear_model/_least_angle.py:L1515` | Cross-validation computes per-fold LARS paths with joblib scheduling, interpolates residual curves onto a shared alpha grid, and refits the selected model; publishing the public estimator would combine path solving, CV splitting, scoring, interpolation, and refit orchestration. |
| `LassoLarsCV` | `sklearn/linear_model/_least_angle.py:L1831` | The estimator inherits the same cross-validated LARS path orchestration and adds Lasso-specific path behavior; a publishable atom should first expose the base LARS/Lasso path boundary and then model CV selection separately. |

Potential remediation path:

- Completed helper slice: `sklearn.linear_model.lars_cv` now publishes
  cross-validation helper atoms for left-out residual-path projection, shared
  alpha-grid construction, per-fold interpolated MSE evaluation, finite-row
  masking, and best-alpha selection:
  `lars_cv_residual_path`, `lars_cv_alpha_grid`,
  `lars_cv_interpolated_fold_mse`, `lars_cv_finite_row_mask`, and
  `lars_cv_best_alpha`.
- Ingest base `lars_path`/`lars_path_gram` solver atoms first.
- Then add explicit atoms for fold residual-path computation, shared-alpha
  interpolation, mean-MSE selection, and final refit, instead of publishing a
  single estimator wrapper.

## `sklearn.feature_selection` estimator-callback selectors

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `RFE` | `sklearn/feature_selection/_rfe.py:L73` | Fit repeatedly clones and trains a configurable estimator, then reads estimator-specific importance attributes or callables; publishing only the selector shell would hide the model-training and importance-extraction boundary. |
| `RFECV` | `sklearn/feature_selection/_rfe.py:L558` | Cross-validated recursive elimination wraps RFE, scorer callbacks, CV splitters, and estimator fitting per fold; a standalone atom would obscure estimator scoring and cross-validation orchestration. |
| `SelectFromModel` | `sklearn/feature_selection/_from_model.py:L95` | Selection depends on a user-provided fitted or unfitted estimator plus estimator-specific coefficient or feature-importance extraction; a publishable atom needs a first-class representation for that estimator boundary. |
| `SequentialFeatureSelector` | `sklearn/feature_selection/_sequential.py:L34` | Greedy feature selection repeatedly scores candidate feature subsets by fitting a configurable estimator under cross-validation; publishing the public class would wrap estimator callbacks rather than decompose them. |

Potential remediation path:

- Completed helper slice: `sklearn.feature_selection.selectors` now publishes
  estimator-independent selector bookkeeping atoms for importance transforms,
  `SelectFromModel` threshold and support-mask resolution, one-step RFE
  elimination, sequential candidate-mask generation, and best-feature
  selection from supplied scores.
- Completed helper slice: `sklearn.feature_selection.rfecv_aggregation` now
  publishes deterministic RFECV post-fold bookkeeping for reverse-order
  best-feature-count selection and `cv_results_` materialization from supplied
  fold score histories and a shared feature-count path.
- Completed helper slice:
  `sklearn.feature_selection.selector_mixin_postfit` now publishes support-mask
  index extraction, dense feature filtering, dense inverse transform, and
  feature-name masking from supplied selector support masks.
- Completed helper slice:
  `sklearn.feature_selection.selector_mixin_sparse_inverse_transform` now
  publishes the sparse `SelectorMixin.inverse_transform` reconstruction helper
  that expands a selected sparse matrix back to CSC layout with zero-filled
  dropped columns from a supplied support mask.
- Completed helper slice:
  `sklearn.feature_selection.selector_mixin_sparse_transform` now publishes the
  CSR `SelectorMixin._transform` helper for selected-column sparse indexing plus
  sklearn's empty dense fallback when no features are selected.
- Ingest estimator-independent helper atoms first, such as threshold parsing,
  support-mask updates from supplied importance vectors, and candidate subset
  bookkeeping.
- Decide how fitted estimator callbacks, scorer callbacks, and CV splitters
  should be represented before publishing full selector workflows.

## `sklearn.svm`

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `LinearSVC` | `sklearn/svm/_classes.py:L44` | Training is delegated to compiled liblinear internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `LinearSVR` | `sklearn/svm/_classes.py:L367` | Training is delegated to compiled liblinear internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `SVC` | `sklearn/svm/_classes.py:L623` | Training is delegated to compiled libsvm internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `NuSVC` | `sklearn/svm/_classes.py:L900` | Training is delegated to compiled libsvm internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `SVR` | `sklearn/svm/_classes.py:L1163` | Training is delegated to compiled libsvm internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `NuSVR` | `sklearn/svm/_classes.py:L1357` | Training is delegated to compiled libsvm internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |
| `OneClassSVM` | `sklearn/svm/_classes.py:L1544` | Training is delegated to compiled libsvm internals; a Python atom would be an opaque estimator wrapper rather than a decomposed algorithm. |

Potential remediation path:

- Completed helper slice: `sklearn.svm` now publishes `l1_min_c`, the
  estimator-independent lower-bound helper used to determine the smallest
  useful `C` value for L1-penalized linear classifiers from supplied training
  data and labels.
- Decide whether SVM estimators should be represented as limited
  estimator-state wrapper atoms with explicit audit limitations.
- Or ingest the underlying libsvm/liblinear source through a dedicated native
  or FFI-backed decomposition, with provenance and runtime validation at the
  solver boundary.

## `sklearn.multiclass` meta-estimator orchestration

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `OneVsOneClassifier` | `sklearn/multiclass.py:L678` | Fit clones and trains one configurable estimator for each class pair, while prediction delegates to estimator-specific `predict`, `decision_function`, or probability behavior; a publishable atom around the class would hide the base-estimator boundary. |
| `OneVsRestClassifier` | `sklearn/multiclass.py:L202` | Fit binarizes labels and trains one configurable estimator per class through joblib and metadata routing; prediction depends on estimator-specific decision or probability callbacks. |
| `OutputCodeClassifier` | `sklearn/multiclass.py:L1043` | Fit creates a random code book and trains one configurable binary estimator per code column, then prediction uses estimator-specific response methods and code-distance decoding; the estimator callbacks need a first-class boundary before publication. |

Potential remediation path:

- Completed helper slice: `sklearn.multiclass` now publishes
  estimator-independent one-vs-rest and output-code helpers:
  `one_vs_rest_multiclass_labels`, `one_vs_rest_binary_indicator`,
  `one_vs_one_decision_scores`, `one_vs_one_class_pairs`,
  `output_code_book`, and `output_code_decode`.
- Ingest estimator-independent helpers first, such as one-vs-one class-pair
  index generation, one-vs-rest score aggregation, output-code book creation,
  and code-distance decoding from supplied response matrices.
- Decide how cloned fitted estimators and metadata routing should be
  represented before publishing the full meta-estimator workflows.

## `sklearn.multioutput` estimator-callback orchestration

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `ClassifierChain` | `sklearn/multioutput.py:L877` | Fit trains a chain of configurable estimators, optionally using cross-validated predictions for previous outputs; publishing the class directly would obscure estimator and CV callback boundaries. |
| `MultiOutputClassifier` | `sklearn/multioutput.py:L445` | Fit trains one configurable classifier per output and prediction/probability methods delegate to each estimator's implementation. |
| `MultiOutputRegressor` | `sklearn/multioutput.py:L342` | Fit trains one configurable regressor per target, and prediction delegates to fitted estimators; a shell atom would not describe the underlying regression algorithm. |
| `RegressorChain` | `sklearn/multioutput.py:L1167` | Fit trains configurable regressors in a prediction-augmented chain, optionally with cross-validation; full publication needs an explicit representation of estimator training and prediction callbacks. |

Potential remediation path:

- Completed helper slice: `sklearn.multioutput` now publishes
  estimator-independent prediction/chain helpers:
  `multioutput_prediction_matrix`, `multioutput_exact_match_score`,
  `chain_order_indices`, `chain_training_features`, `chain_step_features`, and
  `chain_restore_output_order`.
- Completed helper slice:
  `sklearn.multioutput.chain_augmentation` now publishes deterministic chain
  buffer and sparse-augmentation helpers for dense CV placeholder buffers,
  sparse CV placeholder buffers, sparse cv=None target augmentation, sparse
  prediction-step augmentation, and the `cross_val_predict` column extraction
  that keeps probability column 1 for classifier-chain CV features.
- Ingest estimator-independent chain-order validation, feature augmentation
  from supplied previous predictions, independent-output stacking, and
  multioutput score aggregation.
- Decide how configured estimator clones and CV prediction callbacks should be
  represented before publishing full multioutput workflows.

## `sklearn.neighbors` native tree and optimizer boundaries

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `BallTree` | `sklearn/neighbors/_ball_tree.pyx.tp:L282` | The public data structure is implemented in Cython templates and compiled extension modules; publishing a Python-facing wrapper would hide the tree construction, metric dispatch, and query implementation. |
| `KDTree` | `sklearn/neighbors/_kd_tree.pyx.tp:L334` | The public data structure is implemented in Cython templates and compiled extension modules; a publishable atom needs native/FFI provenance at the tree-building and query boundary. |
| `NeighborhoodComponentsAnalysis` | `sklearn/neighbors/_nca.py:L34` | Fit initializes through optional PCA/LDA/random paths and delegates optimization to `scipy.optimize.minimize(method="L-BFGS-B")` with callbacks; publishing the estimator shell would hide optimizer and initialization boundaries. |

Potential remediation path:

- For `BallTree` and `KDTree`, ingest the Cython/native tree construction and
  query kernels through a native or FFI-backed decomposition with parity tests.
- Completed helper slice: `sklearn.neighbors.nca` now publishes
  estimator-independent NCA helpers for same-class masking, dense linear
  transformation, neighbor-probability construction, and objective/gradient
  evaluation from a supplied flattened transformation state.
- For NCA, publish standalone atoms for the supervised loss/gradient and
  linear transform from supplied components before deciding how to represent
  L-BFGS-B optimization and PCA/LDA initialization.

## `sklearn.neural_network` multilayer perceptron optimizers

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `MLPClassifier` | `sklearn/neural_network/_multilayer_perceptron.py:L879` | Fit combines multilayer initialization, forward/backpropagation, sample weighting, early stopping, and either SciPy L-BFGS-B, SGD, or Adam optimizer state; publishing the estimator shell would hide optimizer-specific training behavior. |
| `MLPRegressor` | `sklearn/neural_network/_multilayer_perceptron.py:L1386` | Fit shares the same optimizer framework as `MLPClassifier` and adds regression-specific output/loss behavior; a publishable atom should first expose feed-forward, loss, gradient, and optimizer update boundaries. |

Potential remediation path:

- Ingest deterministic neural-network primitives first: activation functions,
  forward pass from supplied weights, output-layer loss terms, and backprop
  gradients for fixed parameters.
- Completed helper slice: `sklearn.neural_network.mlp_primitives` now
  publishes dense activation, loss, forward-pass, per-layer gradient, and
  canonical backprop helpers for supplied weights and encoded targets:
  `mlp_activation`, `mlp_activation_derivative`, `mlp_loss`,
  `mlp_forward_pass`, `mlp_layer_gradients`, and `mlp_backprop`. It still
  defers parameter initialization, label preprocessing, batching, early
  stopping, and SGD, Adam, or L-BFGS optimizer state.
- Completed helper slice: `sklearn.neural_network.mlp_optimizers` now
  publishes state-passing SGD and Adam optimizer helpers:
  `mlp_sgd_initialize_state`, `mlp_sgd_updates`,
  `mlp_sgd_iteration_end`, `mlp_sgd_trigger_stopping`,
  `mlp_adam_initialize_state`, and `mlp_adam_updates`. It still defers
  parameter mutation, minibatch orchestration, no-improvement tracking,
  validation-based early stopping, and the outer MLP fit loop.
- Completed helper slice: `sklearn.neural_network.mlp_initialization` now
  publishes output-activation and parameter-initialization helpers:
  `mlp_output_activation_name`, `mlp_glorot_init_bound`,
  `mlp_init_layer_parameters`, and `mlp_initialize_parameters`. It still
  defers label preprocessing, stochastic-training bookkeeping, batching,
  early stopping, optimizer state, and the outer MLP fit loop.
- Completed helper slice:
  `sklearn.neural_network.mlp_classification_io` now publishes classifier-side
  target and output helpers for fit-time label-state resolution,
  first-call partial-fit label-state initialization, boolean target encoding,
  label decoding from output scores, and binary/multiclass probability
  formatting.
- Completed helper slice:
  `sklearn.neural_network.mlp_fit_bookkeeping` now publishes fit-shell helpers
  for hidden-layer normalization, first-pass detection, the `partial_fit`
  early-stopping guard, and stochastic batch-size warning and clipping
  resolution before optimizer mutation or validation-score callbacks.
- Completed helper slice:
  `sklearn.neural_network.mlp_monitoring` now publishes stochastic-monitor
  defaults plus scalar best-loss, best-validation-score, and no-improvement
  bookkeeping from supplied loss or validation score values, while score
  callbacks, best-weight copies, and optimizer-trigger stopping remain
  deferred.
- Completed helper slice:
  `sklearn.neural_network.mlp_stochastic_epoch` now publishes deterministic
  epoch-level bookkeeping after minibatch updates: epoch-loss resolution,
  sample-counter advancement, stopping-message formatting, no-improvement
  counter reset after supplied trigger outcomes, incremental-break and
  max-iteration-warning predicates, and the final early-stopping restoration
  predicate.
- Completed helper slice:
  `sklearn.neural_network.mlp_lbfgs_bookkeeping` now publishes deterministic
  LBFGS setup helpers for coefficient and intercept slice layout, flat
  parameter packing, and verbose-to-`iprint` option resolution before the
  deferred SciPy optimizer call.
- Decide whether SGD/Adam/L-BFGS optimizer state should be represented as
  separate atom families before publishing full MLP training surfaces.

## `sklearn.tree`

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `DecisionTreeClassifier` | `sklearn/tree/_classes.py:L707` | Training and prediction depend on Cython `sklearn.tree._tree`, `_splitter`, and `_criterion` internals; a Python atom would be an opaque estimator wrapper rather than a decomposed tree-building algorithm. |
| `DecisionTreeRegressor` | `sklearn/tree/_classes.py:L1114` | Training and prediction depend on Cython `sklearn.tree._tree`, `_splitter`, and `_criterion` internals; a Python atom would be an opaque estimator wrapper rather than a decomposed tree-building algorithm. |
| `ExtraTreeClassifier` | `sklearn/tree/_classes.py:L1456` | Training and prediction depend on Cython `sklearn.tree._tree`, `_splitter`, and `_criterion` internals; a Python atom would be an opaque estimator wrapper rather than a decomposed randomized tree-building algorithm. |
| `ExtraTreeRegressor` | `sklearn/tree/_classes.py:L1745` | Training and prediction depend on Cython `sklearn.tree._tree`, `_splitter`, and `_criterion` internals; a Python atom would be an opaque estimator wrapper rather than a decomposed randomized tree-building algorithm. |

Potential remediation path:

- Decide whether tree estimators should be represented as limited
  estimator-state wrapper atoms with explicit audit limitations.
- Or ingest the Cython/native tree builder, splitter, and criterion internals
  through a dedicated native or FFI-backed decomposition with solver-boundary
  provenance and parity tests.

## `sklearn.gaussian_process.kernels` pairwise wrapper

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `PairwiseKernel` | `sklearn/gaussian_process/kernels.py:L2248` | The public class is explicitly a thin wrapper around `sklearn.metrics.pairwise.pairwise_kernels`; publishing a Gaussian-process kernel atom here would hide the delegated pairwise metric implementation rather than decompose it. |

Potential remediation path:

- Completed helper slice: `sklearn.gaussian_process.kernels` now publishes
  dense kernel and kernel-composition atoms for constant, white, dot-product,
  RBF, rational-quadratic, Matern, and exp-sine-squared kernels, plus sum,
  product, exponentiation, and compound-kernel stack helpers.
- Completed helper slice: `sklearn.metrics.pairwise_kernels` now publishes
  dense upstream pairwise-kernel helpers for default-gamma resolution plus
  linear, polynomial, laplacian, sigmoid, and cosine kernels, which are the
  actual math surfaces a deferred `PairwiseKernel` wrapper would delegate to.
- Completed helper slice:
  `sklearn.gaussian_process.kernel_hyperparameters` now publishes
  hyperparameter-bookkeeping helpers for flattening supplied non-fixed kernel
  value blocks into `theta`, reconstructing value blocks from `theta`,
  flattening non-fixed bounds, and generating bound-warning records when a
  fitted `theta` is close to lower or upper bounds.
- Ingest selected pairwise metrics and kernels directly from
  `sklearn.metrics.pairwise` as first-class atoms, then decide whether a
  limited adapter atom is useful.

## `sklearn.impute` iterative estimator loop

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `IterativeImputer` | `sklearn/impute/_iterative.py:L60` | The estimator is experimental and its core loop repeatedly fits and predicts with a configurable estimator per feature; publishing only the public estimator shell would hide the model-training boundary and the broader orchestration around feature ordering, posterior sampling, and convergence. |

Potential remediation path:

- Completed helper slice: `sklearn.impute.iterative` now publishes
  deterministic iterative-imputer helpers for feature update ordering,
  normalized absolute-correlation matrices, neighbor-feature selection, limit
  vector expansion, and convergence detection.
- Completed helper slice: `sklearn.impute.iterative_postprocessing` now
  publishes deterministic one-feature postprocessing for supplied posterior
  means and standard deviations, clipped deterministic predictions, dense
  feature assignment, and ndarray observed-value restoration.
- Completed helper slice: `sklearn.impute.iterative_initial_bookkeeping` now
  publishes deterministic empty-feature bookkeeping for fit-time all-missing
  detection, dense nonempty-column filtering, keep-empty mask clearing, and
  empty-column restoration from supplied filled values.
- Decompose the deterministic helper boundaries first, including feature order,
  correlation-based neighbor selection, limit validation, one-feature
  prediction postprocessing, and any remaining initial-imputation bookkeeping.
- Decide how fitted per-feature estimators should be represented before
  publishing the full fit/transform state surface.

## `sklearn.feature_extraction` native hashing boundary

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `FeatureHasher` | `sklearn/feature_extraction/_hash.py:L21` | The public transform delegates core index/sign/value construction to compiled `sklearn.feature_extraction._hashing_fast.transform`; publishing the estimator shell would hide the Murmurhash feature hashing implementation. |

Potential remediation path:

- Ingest the hashing transform through a native or FFI-backed decomposition, or
  reimplement the exact signed Murmurhash3 feature-index logic with provenance
  and parity tests for dict, pair, and string input modes.

## `sklearn.inspection` estimator callback workflows

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `partial_dependence` | `sklearn/inspection/_partial_dependence.py:L350` | Public computation is defined around fitted estimator prediction methods, estimator-type dispatch, recursion support for selected tree estimators, and brute-force prediction callbacks; publishing a standalone atom would hide estimator behavior behind callback boundaries. |
| `permutation_importance` | `sklearn/inspection/_permutation_importance.py:L114` | Public computation depends on fitted estimator scoring callbacks, scorer validation, joblib column parallelism, and optional subsampling; a shell atom would wrap estimator-specific scoring rather than decompose a stable algorithmic core. |

Potential remediation path:

- Completed helper slice: `sklearn.inspection.permutation` now publishes
  permutation-importance aggregation helpers for baseline-minus-permuted score
  values, featurewise means, featurewise standard deviations, and the combined
  summary tuple from supplied score arrays.
- Completed helper slice:
  `sklearn.inspection.partial_dependence_grid` now publishes dense numeric
  grid-parameter validation, per-feature axis construction, and Cartesian grid
  construction before the deferred estimator prediction paths.
- Completed helper slice:
  `sklearn.inspection.partial_dependence_postprocessing` now publishes dense
  grid-value assignment plus brute-mode response averaging, response stacking,
  and final regression or binary-classification reshape helpers around supplied
  prediction arrays, while estimator prediction callbacks remain deferred.
- Completed helper slice:
  `sklearn.inspection.partial_dependence_preflight` now publishes the explicit
  regressor response-method guard, kind-versus-method resolution, recursion
  sample-weight restriction, auto-method resolution from supplied recursion
  support, and the final recursion-support guard before estimator callbacks.
- Ingest small helper atoms only where the boundary is explicit, such as
  permutation score aggregation from baseline and permuted score arrays.
- Decide how estimator callback boundaries should be represented before
  publishing full inspection workflows.

## `sklearn.gaussian_process` estimator optimizer boundaries

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `GaussianProcessRegressor` | `sklearn/gaussian_process/_gpr.py:L32` | Fit and prediction combine mutable kernel objects, optional L-BFGS-B hyperparameter optimization, Cholesky factorization, log-marginal likelihood evaluation, and posterior sampling; a public estimator atom would hide the optimizer and linear-algebra state boundaries. |
| `GaussianProcessClassifier` | `sklearn/gaussian_process/_gpc.py:L516` | Classification wraps binary Laplace-approximation estimators, kernel optimization, Newton posterior-mode iterations, and one-vs-rest/one-vs-one multiclass orchestration; a shell atom would obscure the posterior solver and meta-estimator boundaries. |

Potential remediation path:

- Completed helper slice: `sklearn.gaussian_process.regression` now publishes
  state-passing dense regression linear-algebra atoms for kernel diagonal
  regularization, Cholesky factorization, dual coefficient solves,
  log-marginal likelihood, posterior mean, posterior cross solves, posterior
  covariance, and posterior standard deviation. This covers
  `GaussianProcessRegressor` deterministic matrix algebra after kernel matrices
  are supplied.
- Completed helper slice: `sklearn.gaussian_process.classification` now
  publishes deterministic binary Laplace helpers for one Newton step,
  Laplace log-marginal-likelihood scoring, posterior mean/cross-solve/variance
  evaluation, and predictive probability stacking from supplied kernel blocks.
  Kernel optimization, the outer Newton loop, mutable kernel state, and
  multiclass one-vs-rest/one-vs-one orchestration remain deferred.
- Completed helper slice:
  `sklearn.gaussian_process.classification_posterior_mode` now publishes
  binary Laplace posterior-mode helpers for warm-start latent initialization,
  the log-marginal-likelihood improvement stopping rule, and the fixed-kernel
  Newton loop returning the cached latent vector and final Newton temporaries.
- Completed helper slice:
  `sklearn.gaussian_process.regression_gradients` now publishes dense
  log-marginal-likelihood gradient helpers for the shared
  alpha-alpha-transpose-minus-kernel-inverse inner tensor, per-parameter
  per-output kernel-gradient contractions, and final output-axis reduction.
- Completed helper slice:
  `sklearn.gaussian_process.regression_sampling` now publishes predictive
  sampling helpers for single-output and multi-output Gaussian-process draws
  from supplied predictive means and covariance tensors, including sklearn's
  per-target stacking behavior.
- Completed helper slice:
  `sklearn.gaussian_process.regression_preprocessing` now publishes fit-time
  helper atoms for observed target counting, configured target-count
  validation, target mean and scale setup, target normalization, and alpha
  shape resolution before the optimizer and linear-algebra stages.
- Completed helper slice:
  `sklearn.gaussian_process.regression_optimizer_bookkeeping` now publishes
  optimizer-side helper atoms for restart-bounds validation, restart-theta
  sampling from supplied finite bounds, and best-optimum selection from
  accumulated negative log-marginal-likelihood objective values.
- Completed helper slice:
  `sklearn.gaussian_process.regression_prior_prediction` now publishes
  unfitted-prior helper atoms for target-count defaulting, zero-mean output
  shaping, covariance and variance formatting from supplied kernel outputs,
  and standard-deviation conversion.
- Ingest standalone Gaussian-process linear algebra primitives first, such as
  kernel regularization, Cholesky solve, posterior mean/covariance, and
  log-marginal-likelihood components. The first regression slice is complete;
  remaining work should target kernel-state transitions, full optimizer
  execution, validate-data and default-kernel handling, and classifier
  multiclass or estimator-state boundaries.
- Decide how optimizer and Laplace posterior-mode loops should be represented
  before publishing full estimator state atoms.

## `sklearn.manifold` t-SNE optimization boundaries

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `TSNE` | `sklearn/manifold/_t_sne.py:L560` | The public estimator combines pairwise or nearest-neighbor probability construction, PCA/random initialization, a two-stage gradient-descent schedule, and the default Barnes-Hut path delegates the core gradient to compiled `sklearn.manifold._barnes_hut_tsne.gradient`; publishing the class as a single atom would hide native and optimizer boundaries. |

Potential remediation path:

- Completed helper slice: `sklearn.manifold.tsne` now publishes exact-method
  helper atoms for dense joint-probability normalization, KL objective/gradient,
  and one deterministic momentum/gain update step.
- Completed helper slice: `sklearn.manifold.tsne_initialization` now publishes
  initialization and scheduling helpers for automatic learning-rate selection,
  Barnes-Hut neighbor-count selection, random Gaussian initialization,
  PCA-embedding rescaling, and Student-t degrees of freedom before the
  deferred optimization loop.
- Completed helper slice: `sklearn.manifold.tsne_schedule` now publishes
  deterministic optimization bookkeeping helpers for gradient-descent buffer
  initialization, scalar-error scheduling, convergence checks, dense/CSR early
  exaggeration scaling and unscaling, and the stage-two continuation predicate.
- Additional t-SNE remediation should cover remaining estimator orchestration
  such as binary perplexity search boundaries, full optimization-loop
  execution, and native Barnes-Hut gradient handling.
- Treat Barnes-Hut t-SNE as a native/FFI-backed ingestion target around the
  compiled gradient kernel before publishing the default estimator surface.

## `sklearn.cluster` agglomerative hierarchy

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `AgglomerativeClustering` | `sklearn/cluster/_agglomerative.py:L781` | Fit delegates tree construction and early-cut labeling to `ward_tree`, linkage builders, and compiled `sklearn.cluster._hierarchical` helpers; a Python atom would be an estimator wrapper rather than a fully decomposed hierarchical clustering algorithm. |
| `FeatureAgglomeration` | `sklearn/cluster/_agglomerative.py:L1121` | Feature clustering inherits the same agglomerative tree builder and compiled hierarchical helpers, so ingesting only fit/transform would hide the core merge algorithm. |
| `ward_tree` | `sklearn/cluster/_agglomerative.py:L184` | Structured Ward linkage uses compiled `_hierarchical.compute_ward_dist` and parent traversal helpers, while the unstructured path delegates to SciPy hierarchy; a publishable atom needs a direct decomposition or native/FFI provenance at that boundary. |

Potential remediation path:

- Completed helper slice: `sklearn.cluster.agglomerative` now publishes
  hierarchy root lookup, descendant-leaf expansion, and `_hc_cut` label
  extraction for already-built binary children arrays.
- Completed helper slice: `sklearn.cluster.agglomerative_connectivity` now
  publishes deterministic connectivity preprocessing for symmetry,
  dense-or-sparse normalization to LIL, connected-component counting,
  disconnected-graph completion, and nearest cross-component bridge insertion.
- Completed helper slice:
  `sklearn.cluster.agglomerative_fit_bookkeeping` now publishes
  compute-full-tree resolution, effective tree `n_clusters` resolution,
  distance-output gating, distance-threshold cluster-count derivation, and
  supplied parent-head relabeling.
- Completed helper slice:
  `sklearn.cluster.agglomerative_fit_preflight` now publishes the explicit
  `_fit` parameter guards for exactly-one-of `n_clusters` or
  `distance_threshold`, distance-threshold/full-tree compatibility, and Ward's
  literal-`"euclidean"` metric requirement before connectivity preparation and
  tree-builder selection.
- Completed helper slice:
  `sklearn.cluster.agglomerative_fit_setup` now publishes tree-builder lookup
  from linkage plus optional connectivity callable execution and
  `check_array(..., accept_sparse=["csr", "coo", "lil"])` normalization
  before the later compute-full-tree bookkeeping and deferred tree
  construction.
- Tree construction, Ward distances, linkage builders, SciPy hierarchy calls,
  compiled `_hierarchical` helpers, estimator state, and
  `FeatureAgglomeration` transform behavior remain deferred.
- Decide whether these targets should be represented as limited hierarchy-state
  wrapper atoms with explicit audit limitations.
- Or ingest the compiled hierarchical helpers and SciPy linkage boundary
  through a dedicated native or FFI-backed decomposition with parity tests for
  structured and unstructured trees.

## `sklearn.cluster` density and KMeans native cores

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `DBSCAN` | `sklearn/cluster/_dbscan.py:L201` | Fit computes neighborhoods in Python but delegates connected density expansion to compiled `sklearn.cluster._dbscan_inner.dbscan_inner`; wrapping the estimator would hide the core cluster-label propagation. |
| `dbscan` | `sklearn/cluster/_dbscan.py:L22` | Public helper delegates to `DBSCAN.fit`, which relies on compiled `dbscan_inner` for cluster expansion. |
| `BisectingKMeans` | `sklearn/cluster/_bisect_k_means.py:L83` | Recursive splits rely on sklearn KMeans routines whose Lloyd/Elkan update kernels are compiled; a direct atom would be a wrapper over native optimization loops. |
| `k_means` | `sklearn/cluster/_kmeans.py:L296` | Public helper delegates to `KMeans.fit`, which calls compiled Lloyd or Elkan single-run kernels for centroid updates. |
| `KMeans` | `sklearn/cluster/_kmeans.py:L1192` | Fit delegates the optimization loop to compiled `_kmeans_single_lloyd` or `_kmeans_single_elkan`; ingesting the estimator shell would not decompose the clustering algorithm. |
| `MiniBatchKMeans` | `sklearn/cluster/_kmeans.py:L1684` | Fit delegates minibatch centroid updates to compiled k-means kernels and OpenMP-threaded internals; a Python atom would hide the core update rule implementation. |

Potential remediation path:

- Completed helper slice: `sklearn.cluster.dbscan` now publishes limited
  public-boundary DBSCAN fit and public-helper output atoms for dense finite
  arrays with supported string metrics.
- Completed helper slice: `sklearn.cluster.kmeans_plusplus` now publishes
  dense k-means++ seeding helpers for default local-trial counts, first-center
  weighted sampling, greedy candidate-id sampling, candidate-potential
  evaluation, and full dense center initialization before the deferred KMeans
  optimization loop.
- Nearest-neighbor search, sparse precomputed graph handling, callable metrics,
  and compiled `_dbscan_inner.dbscan_inner` connected expansion remain
  explicit native/FFI boundaries.
- Keep `kmeans_plusplus` separate because its seeding logic can be considered
  for Python-level ingestion without claiming to ingest KMeans optimization.
- Decide whether DBSCAN and KMeans families should be ingested through native
  or FFI-backed kernels with explicit solver-boundary provenance and parity
  tests.

## `sklearn.cluster` HDBSCAN native hierarchy

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `HDBSCAN` | `sklearn/cluster/_hdbscan/hdbscan.py:L423` | Fit dispatches minimum-spanning-tree construction to `_hdbscan_brute` or `_hdbscan_prims` and tree backends, then condenses/extracts a density hierarchy; ingesting only the estimator shell would hide the core hierarchical density algorithm. |

Potential remediation path:

- Completed helper slice: `sklearn.cluster.hdbscan` now publishes limited
  public-boundary HDBSCAN fit and fit-predict atoms for dense finite arrays
  with supported string metrics.
- MST construction, tree backends, hierarchy condensation/extraction,
  precomputed and sparse inputs, callable metrics, non-finite remapping, and
  stored centers remain explicit native/private sklearn boundaries.
- Ingest HDBSCAN through the MST and hierarchy construction boundary with
  native/FFI provenance, including parity tests for brute-force, KD-tree,
  BallTree, sparse, and precomputed-distance modes.

## `sklearn.cluster` BIRCH global clustering boundary

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `Birch` | `sklearn/cluster/_birch.py:L359` | The CF-tree insertion logic is Python-level, but the public estimator's default final labeling uses `AgglomerativeClustering(n_clusters=...)`; publishing the estimator shell would hide the deferred hierarchical-clustering boundary. |

Potential remediation path:

- Completed helper slice: `sklearn.cluster.birch` now publishes BIRCH
  no-global-clustering fit, predict, and transform atoms with `n_clusters=None`
  and immutable subcluster state.
- Default `AgglomerativeClustering`, custom global clusterers, sparse inputs,
  `partial_fit`, and live mutable CF-tree object surfaces remain deferred.
- Ingest the CF-tree insertion, subcluster merge, and node-splitting logic as
  separate atoms with an explicit `n_clusters=None` or no-global-clustering
  boundary.
- Or ingest the agglomerative global-clustering dependency first, then publish
  the full Birch fit/predict state surface with parity coverage for default and
  custom clusterer modes.

## `sklearn.cluster` spectral clustering and biclustering solver boundaries

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `spectral_clustering` | `sklearn/cluster/_spectral.py:L190` | Public helper delegates to `SpectralClustering.fit`; the default `assign_labels="kmeans"` route depends on deferred KMeans optimization kernels. |
| `SpectralClustering` | `sklearn/cluster/_spectral.py:L379` | Fit builds an affinity matrix and spectral embedding, then defaults to KMeans label assignment; publishing only the estimator shell would hide the deferred KMeans core and eigensolver boundary. |
| `SpectralCoclustering` | `sklearn/cluster/_bicluster.py:L202` | Fit normalizes data and computes singular vectors, then labels rows and columns through `BaseSpectral._k_means`, which delegates to KMeans or MiniBatchKMeans. |
| `SpectralBiclustering` | `sklearn/cluster/_bicluster.py:L360` | Fit relies on SVD projection and repeated KMeans/MiniBatchKMeans labeling for row and column clusters, so an estimator atom would hide the deferred KMeans core. |

Potential remediation path:

- Completed helper slice: `sklearn.cluster.spectral_labeling` publishes
  non-KMeans label assignment atoms for `cluster_qr` and `discretize` from
  precomputed spectral embeddings.
- Completed helper slice: `sklearn.cluster.bicluster` publishes dense
  spectral biclustering preprocessing atoms for scale, bistochastic, and
  log-interaction normalization before the SVD and KMeans stages.
- Completed helper slice: `sklearn.cluster.bicluster_structure` publishes
  deterministic coclustering and biclustering bookkeeping helpers for
  coclustering singular-vector counts, stacked SVD embeddings, label splits,
  coclustering indicator matrices, biclustering SVD-dimension resolution,
  row/column cluster-count resolution, and checkerboard indicator-grid
  construction from supplied labels.
- Completed helper slice:
  `sklearn.cluster.bicluster_piecewise_projection` now publishes supplied-state
  helpers for piecewise vector reconstruction from KMeans outputs, residual
  norm scoring against those piecewise approximations, best-vector selection,
  and dense projection before the deferred KMeans assignment step.
- Sparse biclustering normalization branches, SVD/randomized-SVD projection,
  piecewise-vector selection, projection-and-cluster helpers, KMeans and
  MiniBatchKMeans label assignment, and full estimator state surfaces remain
  deferred.
- Decide how to represent eigensolver/SVD boundaries and KMeans assignment
  before publishing the full spectral clustering and biclustering estimators.

## `sklearn.covariance` sparse precision and robust covariance solvers

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `graphical_lasso` | `sklearn/covariance/_graph_lasso.py:L230` | Public helper delegates to `GraphicalLasso.fit`; the default `mode="cd"` path calls compiled `cd_fast.enet_coordinate_descent_gram`, so a Python atom would hide the sparse precision solver. |
| `GraphicalLasso` | `sklearn/covariance/_graph_lasso.py:L399` | Fit computes or accepts empirical covariance, then delegates the core sparse inverse-covariance optimization to `_graphical_lasso`, whose default coordinate-descent inner loop is compiled. |
| `GraphicalLassoCV` | `sklearn/covariance/_graph_lasso.py:L722` | Cross-validation repeatedly runs graphical lasso paths and then refits the same sparse precision solver, so publishing the estimator shell would hide both solver and CV orchestration boundaries. |
| `MinCovDet` | `sklearn/covariance/_robust_covariance.py:L621` | Fit delegates raw robust location/covariance search to the FastMCD candidate-selection algorithm and then applies correction/reweighting; a state wrapper would hide the robust subset search. |
| `EllipticEnvelope` | `sklearn/covariance/_elliptic_envelope.py:L15` | Fit inherits `MinCovDet.fit` and only adds an outlier offset, so it should wait until the FastMCD robust covariance boundary is decomposed. |

Potential remediation path:

- Ingest graphical lasso by decomposing `_graphical_lasso` and deciding whether
  native coordinate-descent/LARS solver boundaries should be represented by FFI
  atoms or limited solver-boundary atoms.
- Completed helper slice: `sklearn.covariance.graphical_lasso` now publishes
  supplied-state scoring atoms for the off-diagonal L1 penalty, Gaussian
  covariance log-likelihood, graphical-lasso objective value, and dual-gap
  convergence score.
- Graphical-lasso coordinate-descent and LARS inner solvers, path fitting,
  cross-validation orchestration, public estimator mutation, and alpha
  selection remain deferred.
- Completed helper slice: `sklearn.covariance.robust` now publishes
  post-FastMCD deterministic helpers for MCD consistency scaling, corrected
  covariance/distance updates, chi-square support reweighting, reweighted
  location/covariance recomputation, and squared Mahalanobis scoring.
- Completed helper slice:
  `sklearn.covariance.robust_fastmcd_1d` now publishes one-dimensional
  FastMCD helpers for support-size resolution, raw location, support-mask
  selection, raw covariance, and squared distances from the deterministic
  one-dimensional shortcut branch.
- Completed helper slice:
  `sklearn.covariance.robust_fastmcd_selection` now publishes multivariate
  FastMCD helper atoms for trial-plan resolution from random starts versus
  supplied estimates, determinant-based best-candidate ranking and gather,
  large-sample subset scheduling constants, and merged-result scattering back
  to full-length support and distance arrays.
- Completed helper slice:
  `sklearn.covariance.robust_fastmcd_c_step` now publishes the deterministic
  FastMCD c-step subset-search loop plus random-start support initialization,
  estimate-start support selection, and support-statistic updates.
- Completed helper slice:
  `sklearn.covariance.robust_fastmcd_candidates` now publishes multivariate
  FastMCD candidate-pool generation from repeated c-step runs over random
  starts or supplied estimate stacks.
- Ingest robust covariance by decomposing FastMCD helpers (`fast_mcd`,
  candidate selection, correction, reweighting, and Mahalanobis scoring) before
  publishing `MinCovDet` or `EllipticEnvelope` fit states. Correction,
  reweighting, scoring, the one-dimensional shortcut, and deterministic
  multivariate trial-planning and scheduling helpers are complete; the `_c_step`
  subset-search loop, random subset generation, and multivariate candidate
  evaluation are now complete, while estimator state publication remains
  deferred.

## `sklearn.decomposition` sparse coding and matrix factorization solvers

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `sparse_encode` | `sklearn/decomposition/_dict_learning.py:L226` | The public helper multiplexes LARS, Lasso-LARS, coordinate descent, OMP, thresholding, joblib slicing, Gram/covariance precomputation, and positivity constraints; publishing a single wrapper atom would hide the selected sparse-code solver. |
| `SparseCoder` | `sklearn/decomposition/_dict_learning.py:L1182` | The estimator is a transform/inverse shell over `sparse_encode`, so it inherits the same deferred sparse-code solver boundary. |
| `dict_learning` | `sklearn/decomposition/_dict_learning.py:L892` | The public helper delegates to `DictionaryLearning.fit_transform`, which alternates sparse-code solves with dictionary updates and optional callbacks. |
| `dict_learning_online` | `sklearn/decomposition/_dict_learning.py:L673` | The public helper delegates to `MiniBatchDictionaryLearning.fit` and combines mini-batch iteration, sparse-code solves, sufficient-statistic updates, and callbacks. |
| `DictionaryLearning` | `sklearn/decomposition/_dict_learning.py:L1417` | Fit is an alternating optimization loop over sparse coding and dictionary updates; an estimator state atom would obscure the LARS/Lasso/OMP solver choices and callback boundary. |
| `MiniBatchDictionaryLearning` | `sklearn/decomposition/_dict_learning.py:L1760` | Fit adds mini-batch scheduling, sufficient statistics, early stopping, and the same sparse-code solver dependency. |
| `SparsePCA` | `sklearn/decomposition/_sparse_pca.py:L162` | Fit delegates sparse component extraction to dictionary learning, then uses ridge regression for transform; it should wait for the dictionary-learning solver boundary. |
| `MiniBatchSparsePCA` | `sklearn/decomposition/_sparse_pca.py:L342` | Fit delegates to mini-batch dictionary learning and inherits its sparse-code and mini-batch solver boundaries. |
| `non_negative_factorization` | `sklearn/decomposition/_nmf.py:L897` | The public helper dispatches between coordinate descent and multiplicative-update solvers, with the coordinate-descent path calling compiled `_update_cdnmf_fast`. |
| `NMF` | `sklearn/decomposition/_nmf.py:L1317` | Fit wraps `non_negative_factorization` and inherits the compiled coordinate-descent or iterative multiplicative-update solver boundary. |
| `MiniBatchNMF` | `sklearn/decomposition/_nmf.py:L1758` | Fit adds mini-batch scheduling and online updates over the NMF solver surface, so a public atom would hide both the update loop and solver choice. |
| `LatentDirichletAllocation` | `sklearn/decomposition/_lda.py:L160` | Fit/partial_fit/transform rely on online variational Bayes loops and compiled `_online_lda_fast` routines for Dirichlet expectation and mean-change inner updates. |

Potential remediation path:

- Ingest small explicit helper kernels separately where they are not solver
  shells, such as NMF `trace_dot`, beta-divergence, NNDSVD initialization, or
  dictionary-update normalization.
- Completed helper slice: `sklearn.decomposition.nmf` now publishes
  beta-loss normalization, trace-dot, beta-divergence, dense random
  initialization, NNDSVD reconstruction from a supplied SVD triplet, and
  initialization-matrix validation. It still stops short of default init-mode
  dispatch, randomized SVD publication, multiplicative updates, coordinate
  descent, minibatch scheduling, and estimator state.
- Completed helper slice: `sklearn.decomposition.dictionary_update` publishes
  dense dictionary-learning sufficient statistics and active-atom update
  helpers with unit-norm projection.
- Completed helper slice:
  `sklearn.decomposition.sparse_encode_preprocessing` publishes sparse-encode
  regularization selection, Gram and covariance precompute helpers, and the
  threshold-only encoding branch.
- Completed helper slice: `sklearn.decomposition.nmf_minibatch` now publishes
  MiniBatchNMF scheduling and convergence helpers for batch-size clamping,
  forgetting-rate exponent calculation, MM gamma selection, transform-iteration
  defaulting, exponentially weighted cost updates, small-H-change stopping, and
  no-improvement bookkeeping before the deferred multiplicative updates and fit
  loop.
- Completed helper slice:
  `sklearn.decomposition.dictionary_learning_minibatch` now publishes
  MiniBatchDictionaryLearning parameter resolution, sufficient-stat decay and
  update helpers, warmup gating, exponentially weighted cost tracking, and
  dictionary-change and no-improvement convergence bookkeeping.
- Completed helper slice:
  `sklearn.decomposition.dictionary_learning_loop` now publishes dense SVD
  initialization, factor resize helpers, objective evaluation, cost-delta
  stopping, and callback cadence bookkeeping for the plain dictionary-learning
  loop.
- Sparse-code solving, LARS/Lasso/OMP/coordinate-descent branches, joblib
  parallel sparse encoding, dictionary-learning callbacks, mini-batch
  scheduling, sklearn's random unused-atom resampling branch, and non-threshold
  sparse-code solvers remain deferred.
- Decide whether sparse-code solvers, compiled NMF coordinate descent, and
  online-LDA Cython helpers should be represented through native/FFI-backed
  atoms or through limited solver-boundary atoms with direct parity tests.
- Publish dictionary learning, SparsePCA, NMF, and LDA estimator states only
  after their inner solver boundaries have first-class provenance and tests.

## `sklearn.ensemble` tree, estimator-callback, and native solver boundaries

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `AdaBoostClassifier` | `sklearn/ensemble/_weight_boosting.py:L321` | Fit is a boosting loop around a mutable base estimator, defaulting to `DecisionTreeClassifier`; a publishable atom would need the classifier's fit/predict behavior and the SAMME weight update decomposed separately. |
| `AdaBoostRegressor` | `sklearn/ensemble/_weight_boosting.py:L823` | Fit wraps repeated base-regressor training and AdaBoost.R2 weighted-error updates; default tree training and arbitrary estimator callbacks are hidden behind the public estimator shell. |
| `BaggingClassifier` | `sklearn/ensemble/_bagging.py:L741` | Fit clones and trains arbitrary classifiers over sampled rows/features in joblib workers, so a wrapper atom would hide estimator-specific fit/predict behavior. |
| `BaggingRegressor` | `sklearn/ensemble/_bagging.py:L1253` | Fit/predict aggregate arbitrary regressors over bootstrap samples and feature subsets; the core learned behavior lives in estimator callbacks. |
| `RandomForestClassifier` | `sklearn/ensemble/_forest.py:L1174` | Fit delegates each base learner to `DecisionTreeClassifier._fit`, whose tree-building core is native/compiled; a forest shell would obscure the tree growth boundary. |
| `RandomForestRegressor` | `sklearn/ensemble/_forest.py:L1572` | Fit delegates to compiled decision-tree growth through `DecisionTreeRegressor._fit`, with only bootstrap orchestration visible in Python. |
| `ExtraTreesClassifier` | `sklearn/ensemble/_forest.py:L1944` | Fit delegates randomized tree construction to `ExtraTreeClassifier._fit`, which depends on the native tree builder. |
| `ExtraTreesRegressor` | `sklearn/ensemble/_forest.py:L2328` | Fit delegates randomized regression-tree construction to the native tree builder through `ExtraTreeRegressor._fit`. |
| `RandomTreesEmbedding` | `sklearn/ensemble/_forest.py:L2679` | Fit builds an ensemble of totally random trees and one-hot encodes leaf indices; the tree induction boundary is the same native forest dependency. |
| `IsolationForest` | `sklearn/ensemble/_iforest.py:L55` | Fit uses random ExtraTreeRegressor ensembles and scoring depends on learned tree paths; helper kernels like average path length can be separated, but the public estimator hides native tree construction. |
| `GradientBoostingClassifier` | `sklearn/ensemble/_gb.py:L1134` | Fit repeatedly trains regression trees and updates terminal regions; the tree training and loss-specific boosting loop need separate decomposition before publishing an estimator state. |
| `GradientBoostingRegressor` | `sklearn/ensemble/_gb.py:L1746` | Fit uses staged regression-tree training and terminal-region updates; a public atom would hide the tree solver and staged loss update boundary. |
| `HistGradientBoostingClassifier` | `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:L1875` | Fit uses histogram binning, native grower/predictor internals, and loss-specific boosting updates; the compiled histogram-tree solver is not decomposed. |
| `HistGradientBoostingRegressor` | `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:L1472` | Fit shares the compiled histogram-tree grower and predictor boundary with the classifier variant. |
| `StackingClassifier` | `sklearn/ensemble/_stacking.py:L422` | Fit trains arbitrary base estimators, obtains cross-validated predictions, and trains a final estimator; publishing a shell would hide both estimator callbacks and CV orchestration. |
| `StackingRegressor` | `sklearn/ensemble/_stacking.py:L841` | Fit trains arbitrary regressors and a final estimator over generated meta-features, so the algorithmic behavior is estimator-callback orchestration. |
| `VotingClassifier` | `sklearn/ensemble/_voting.py:L194` | Fit clones arbitrary classifiers and prediction aggregates their labels/probabilities; the publishable pieces are aggregation helpers, not the unfitted-estimator meta-estimator itself. |
| `VotingRegressor` | `sklearn/ensemble/_voting.py:L542` | Fit clones arbitrary regressors and prediction averages callback outputs; a state atom would hide estimator-specific fit/predict behavior. |

Potential remediation path:

- Ingest small standalone helper kernels separately, such as bootstrap index
  generation, weighted voting/probability aggregation, forest prediction
  averaging, and IsolationForest average path length.
- Completed helper slice: `sklearn.ensemble.voting` publishes weighted hard
  voting, soft probability averaging, and regression averaging over
  already-computed estimator outputs.
- Completed helper slice: `sklearn.ensemble.isolation_forest` publishes
  average path length, per-tree leaf depth accumulation, and raw anomaly-score
  conversion while tree growth remains deferred.
- Completed helper slice: `sklearn.ensemble.bagging_sampling` publishes
  bootstrap and sampling-without-replacement feature/sample index draws for
  one bagging estimator.
- Completed helper slice: `sklearn.ensemble.bagging_fit_bookkeeping`
  publishes fit-shell helper atoms for integer `max_samples` and
  `max_features` resolution, out-of-bag preflight guards, and warm-start
  additional-estimator counting before the deferred estimator-build workers.
- Completed helper slice: `sklearn.ensemble.bagging_fit_scheduling`
  publishes deterministic helper atoms for balanced job partitioning and
  warm-start per-estimator seed generation before the deferred bagging worker
  builds.
- Completed helper slice: `sklearn.ensemble.bagging_index_reconstruction`
  publishes deterministic helper atoms for reconstructing per-estimator
  feature/sample draw pairs plus feature-only and sample-only index blocks
  from stored bagging seed vectors and validated sampling parameters.
- Completed helper slice: `sklearn.ensemble.bagging_classifier_io`
  publishes deterministic helper atoms for BaggingClassifier fit-time class
  state plus encoded targets, and predict-time label decoding from already
  averaged class probabilities.
- Completed helper slice: `sklearn.ensemble.bagging_aggregation` publishes
  class-aligned probability and log-probability averaging plus decision and
  regression prediction averaging over already-computed bagging estimator
  outputs.
- Completed helper slice: `sklearn.ensemble.bagging_oob` publishes
  out-of-bag classifier probability and vote accumulation, OOB decision
  normalization, OOB argmax label selection, and regressor OOB prediction
  averaging from supplied in-bag sample indices and already-computed
  estimator outputs.
- Completed helper slice: `sklearn.ensemble.bagging_oob_scoring` publishes
  deterministic helper atoms for the bagging OOB uncovered-sample warning
  mask, classifier OOB accuracy from class totals, and regressor OOB `r2`
  scoring from already-averaged OOB predictions.
- Completed helper slice: `sklearn.ensemble.forest_sampling` publishes
  validated bootstrap sample-count resolution plus sampled and unsampled index
  draws for one forest tree before tree growth or OOB scoring.
- Completed helper slice: `sklearn.ensemble.forest_fit_bookkeeping`
  publishes deterministic helper atoms for fit-time bootstrap draw-count
  bookkeeping, OOB bootstrap and target-type preflight checks, warm-start
  additional-estimator counting, and the Boolean gate for recomputing OOB
  attributes before deferred tree growth and OOB prediction callbacks.
- Completed helper slice: `sklearn.ensemble.forest_index_reconstruction`
  publishes deterministic helper atoms for reconstructing one forest tree's
  in-bag sample indices and the full `estimators_samples_` property from
  stored per-tree seeds and validated bootstrap bookkeeping.
- Completed helper slice: `sklearn.ensemble.forest_oob_predictions`
  publishes deterministic helper atoms for formatting per-tree classifier and
  regressor OOB prediction blocks, accumulating OOB prediction totals and
  counts from supplied unsampled index blocks, detecting uncovered samples,
  and averaging the OOB prediction tensor with sklearn's zero-count
  safeguard.
- Completed helper slice: `sklearn.ensemble.forest_oob_postprocessing`
  publishes deterministic helper atoms for converting the averaged forest OOB
  tensor into sklearn's public classifier decision-function or regressor
  prediction shapes, plus final classifier accuracy and regressor `r2`
  scoring from those already-postprocessed arrays.
- Completed helper slice: `sklearn.ensemble.forest_classifier_targets`
  publishes deterministic helper atoms for per-output ForestClassifier class
  extraction and encoded targets, class_weight string-preset validation,
  warm-start warning gating, and expanded per-sample class-weight resolution
  from original targets before deferred warning side effects and tree
  fitting.
- Completed helper slice: `sklearn.ensemble.forest_feature_importances`
  publishes deterministic helper atoms for deciding which forest trees
  contribute impurity feature importances, sklearn's zero-vector fallback
  when none contribute, mean aggregation across contributing tree importance
  vectors, and final normalization of the averaged importances.
- Completed helper slice: `sklearn.ensemble.forest_classifier_outputs`
  publishes deterministic helper atoms for single-output forest
  log-probability conversion, multioutput forest per-output log-probability
  conversion, and multioutput label decoding from already-aggregated
  probability matrices and class vectors.
- Completed helper slice: `sklearn.ensemble.forest_predict_preflight`
  publishes deterministic helper atoms for the pure-Python preflight inside
  `BaseForest._validate_X_predict`: missing-value validation mode selection
  and the CSR sparse-index dtype guard before deferred `validate_data`
  execution and fitted-estimator checks.
- Completed helper slice: `sklearn.ensemble.forest_path_outputs`
  publishes deterministic helper atoms for the final `BaseForest.apply` and
  `BaseForest.decision_path` output formatting: transposing per-tree leaf
  vectors, building cumulative node-pointer offsets, and horizontally stacking
  supplied sparse indicator blocks into CSR output after deferred tree
  callbacks.
- Completed helper slice: `sklearn.ensemble.adaboost` publishes
  estimator-independent SAMME decision aggregation, decision-to-probability
  conversion, and AdaBoost.R2 weighted-median regression aggregation over
  already-computed estimator outputs.
- Completed helper slice: `sklearn.ensemble.adaboost_weight_updates`
  publishes deterministic training-stage helper atoms for classifier stage
  error, SAMME estimator weights, nonterminal classifier sample-weight
  updates, and AdaBoost.R2 loss-vector, beta, estimator-weight, and
  nonterminal sample-weight update math from supplied predictions or error
  vectors and sample weights.
- Completed helper slice: `sklearn.ensemble.forest_aggregation` publishes
  mean class-probability aggregation, class selection from aggregated
  probabilities, and regression prediction averaging over already-computed
  tree outputs.
- Completed helper slice: `sklearn.ensemble.gradient_boosting` publishes
  deterministic helper math for numerically safe line-search division and
  Huber delta selection from residual magnitudes and sample weights.
- Completed helper slice: `sklearn.ensemble.stacking_meta_features` publishes
  deterministic stacking helper atoms for sklearn-style prediction-output
  normalization into meta-feature matrices, per-block output-width
  bookkeeping, and transform feature-name materialization from supplied
  estimator names and optional passthrough feature names.
- Completed helper slice: `sklearn.ensemble.stacking_classifier_outputs`
  publishes classifier-side postprocessing helpers for single-output encoded
  label decoding, multilabel encoded-column decoding, and the multilabel
  probability-block conversion used after deferred final-estimator prediction
  callbacks.
- Decide whether sklearn tree builders and histogram-gradient-boosting native
  internals should be represented through FFI/native atoms before publishing
  random forest, extra trees, isolation forest, and gradient boosting states.
- Define a general policy for arbitrary-estimator meta-estimators before
  publishing bagging, stacking, voting, or AdaBoost wrappers.
