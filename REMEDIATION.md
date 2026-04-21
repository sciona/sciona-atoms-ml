# Remediation Queue

This file tracks sklearn targets that should not be ingested as publishable
atoms until the decomposition boundary is clarified.

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

- Decide whether SVM estimators should be represented as limited
  estimator-state wrapper atoms with explicit audit limitations.
- Or ingest the underlying libsvm/liblinear source through a dedicated native
  or FFI-backed decomposition, with provenance and runtime validation at the
  solver boundary.

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

- Ingest selected pairwise metrics and kernels directly from
  `sklearn.metrics.pairwise` as first-class atoms, then decide whether a
  limited adapter atom is useful.

## `sklearn.impute` iterative estimator loop

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `IterativeImputer` | `sklearn/impute/_iterative.py:L60` | The estimator is experimental and its core loop repeatedly fits and predicts with a configurable estimator per feature; publishing only the public estimator shell would hide the model-training boundary and the broader orchestration around feature ordering, posterior sampling, and convergence. |

Potential remediation path:

- Decompose the deterministic helper boundaries first, including feature order,
  correlation-based neighbor selection, limit validation, and one-feature
  imputation.
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

- Ingest standalone Gaussian-process linear algebra primitives first, such as
  kernel regularization, Cholesky solve, posterior mean/covariance, and
  log-marginal-likelihood components.
- Decide how optimizer and Laplace posterior-mode loops should be represented
  before publishing full estimator state atoms.

## `sklearn.cluster` agglomerative hierarchy

Deferred targets:

| Target | Source | Reason |
| --- | --- | --- |
| `AgglomerativeClustering` | `sklearn/cluster/_agglomerative.py:L781` | Fit delegates tree construction and early-cut labeling to `ward_tree`, linkage builders, and compiled `sklearn.cluster._hierarchical` helpers; a Python atom would be an estimator wrapper rather than a fully decomposed hierarchical clustering algorithm. |
| `FeatureAgglomeration` | `sklearn/cluster/_agglomerative.py:L1121` | Feature clustering inherits the same agglomerative tree builder and compiled hierarchical helpers, so ingesting only fit/transform would hide the core merge algorithm. |
| `ward_tree` | `sklearn/cluster/_agglomerative.py:L184` | Structured Ward linkage uses compiled `_hierarchical.compute_ward_dist` and parent traversal helpers, while the unstructured path delegates to SciPy hierarchy; a publishable atom needs a direct decomposition or native/FFI provenance at that boundary. |

Potential remediation path:

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

- Ingest HDBSCAN through the MST and hierarchy construction boundary with
  native/FFI provenance, including parity tests for brute-force, KD-tree,
  BallTree, sparse, and precomputed-distance modes.

## `sklearn.cluster` BIRCH global clustering boundary

Deferred target:

| Target | Source | Reason |
| --- | --- | --- |
| `Birch` | `sklearn/cluster/_birch.py:L359` | The CF-tree insertion logic is Python-level, but the public estimator's default final labeling uses `AgglomerativeClustering(n_clusters=...)`; publishing the estimator shell would hide the deferred hierarchical-clustering boundary. |

Potential remediation path:

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

- Ingest non-KMeans spectral label assignment helpers such as `cluster_qr` or
  `discretize` as separate atoms if they become first-class targets.
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
- Ingest robust covariance by decomposing FastMCD helpers (`fast_mcd`,
  candidate selection, correction, reweighting, and Mahalanobis scoring) before
  publishing `MinCovDet` or `EllipticEnvelope` fit states.
