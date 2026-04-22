# Solution CDG Gap Analysis

Systematic record of gaps found while converting Kaggle winning solutions
into CDGs backed by existing atoms. Each solution is a test of whether
the architect tooling can reconstruct atom bindings from conceptual nodes.

## Summary Dashboard

| Solution | Stages | Bound | Partial | Gaps | Coverage |
|----------|--------|-------|---------|------|----------|
| Connectomics 1st | 9 | 1 | 0 | 8 | 11% |
| Cause-Effect 2nd | 17 | 16 | 0 | 1 | 94% |
| TrackML 5th | 11 | 4 | 0 | 7 | 36% |
| DSB2017 1st | — | — | — | — | — |
| Adversarial 1st | — | — | — | — | — |
| Barachant EEG (x4) | — | — | — | — | — |
| Flavours Physics 1st | — | — | — | — | — |
| **Totals (3 done)** | **37** | **21** | **0** | **16** | **57%** |

---

## Gap Taxonomy

| Category | Code | Description |
|----------|------|-------------|
| **Missing atom** | `MISSING_ATOM` | Novel technique with no existing atom. Requires ingestion. |
| **Partial binding** | `PARTIAL_BIND` | An atom covers part of the operation but not all (e.g., fit exists but state-query doesn't). |
| **Semantic mismatch** | `SEM_MISMATCH` | Keyword search returns an atom that looks right but does something semantically different. |
| **Orchestration gap** | `ORCH_GAP` | Higher-order pattern (loop, sweep, branch) not representable as a single atom. |
| **Trivial unbound** | `TRIVIAL` | One-liner numpy/scipy call; doesn't justify a standalone atom but has no binding target. |

---

## 1. Connectomics 1st Place (Graph Inference from Calcium Fluorescence)

**CDG:** `data/solution_cdgs/connectomics_1st.json` | **Coverage:** 1/9 (11%)

| Stage | Binding | Confidence | Category | Notes |
|-------|---------|------------|----------|-------|
| calcium_lowpass_filter | gap | 0.0 | `MISSING_ATOM` | Circular-shift moving average with multi-kernel variants. |
| first_difference | gap | 0.3 | `TRIVIAL` | `np.diff`. |
| fluorescence_hard_threshold | gap | 0.15 | `MISSING_ATOM` | Element-wise value clipping. VarianceThreshold is column-wise (wrong). |
| global_activity_sample_reweighting | gap | 0.0 | `MISSING_ATOM` | Key innovation. `(x+1)^(1+1/S)` with regime exponents. |
| pca_precision_matrix | `pca_fit` | 0.7 | `PARTIAL_BIND` | PCA fit covered. `get_precision()` not covered. |
| score_matrix_normalization | gap | 0.2 | `TRIVIAL` | Matrix min-max + diagonal zeroing. |
| threshold_sweep_ensemble | gap | 0.0 | `ORCH_GAP` | MAP_OVER: 120 thresholds x 4 filters. |
| temporal_precedence_directivity | gap | 0.0 | `MISSING_ATOM` | Biophysics-tuned temporal lag detection. |
| weighted_score_combination | gap | 0.1 | `TRIVIAL` | Weighted sum (0.997/0.003). |

**Findings:** C-1 (state-query gap), C-2 (VarianceThreshold keyword collision), C-3 (no composition primitives).

---

## 2. Cause-Effect 2nd Place (Causal Direction from Bivariate Data)

**CDG:** `data/solution_cdgs/cause_effect_2nd.json` | **Coverage:** 16/17 (94%)

| Stage | Binding | Confidence | Category | Notes |
|-------|---------|------------|----------|-------|
| discretize_and_bin | `...feature_primitives.discretize_and_bin` | 1.0 | bound | |
| igci_asymmetry_score | `...feature_primitives.igci_asymmetry_score` | 1.0 | bound | |
| hsic_independence_test | `...feature_primitives.hsic_independence_test` | 1.0 | bound | |
| normalized_error_probability | `...feature_primitives.normalized_error_probability` | 1.0 | bound | |
| uniform_divergence | `...feature_primitives.uniform_divergence` | 1.0 | bound | |
| knn_entropy_estimator | `...feature_primitives.knn_entropy_estimator` | 1.0 | bound | |
| polyfit_nonlinearity_asymmetry | `...feature_primitives.polyfit_nonlinearity_asymmetry` | 1.0 | bound | |
| polyfit_residual_error | `...feature_primitives.polyfit_residual_error` | 1.0 | bound | |
| conditional_noise_entropy_variance | `...conditional_statistics.conditional_noise_entropy_variance` | 1.0 | bound | |
| conditional_noise_skewness_variance | `...conditional_statistics.conditional_noise_skewness_variance` | 1.0 | bound | |
| conditional_noise_kurtosis_variance | `...conditional_statistics.conditional_noise_kurtosis_variance` | 1.0 | bound | |
| conditional_distribution_similarity | `...conditional_statistics.conditional_distribution_similarity` | 1.0 | bound | |
| **asymmetric_feature_difference** | **gap** | **0.0** | **`ORCH_GAP`** | **compute-both-directions-then-subtract meta-pattern** |
| symmetrized_prediction_fusion | `...estimators.symmetrized_prediction_fusion` | 1.0 | bound | |
| two_stage_independence_direction | `...estimators.two_stage_independence_direction` | 1.0 | bound | |
| left_right_decomposed_prediction | `...estimators.left_right_decomposed_prediction` | 1.0 | bound | |
| weighted_ensemble_combination | `...estimators.weighted_ensemble_combination` | 1.0 | bound | |

**Findings:**

**Finding CE-1: concept_type='custom' blocks category search (SEM_MISMATCH)**
All 16 causal inference atoms have `concept_type=custom` in their CDG files.
`PrimitiveCatalog.search_by_category()` will not find them via concept_type
matching. Retrieval must fall through to keyword/embedding search. This is the
same issue as G-006 and affects ALL Kaggle-ingested atoms.

**Finding CE-2: asymmetric_feature_difference is a cross-solution pattern (ORCH_GAP)**
The compute-both-directions-then-subtract pattern also appears in Connectomics
(temporal_precedence_directivity applies it to lag counts). This is a reusable
template that should be a MAP_OVER or similar composition primitive, not a
single atom.

---

## 3. TrackML 5th Place (Physics-Geometric Track Reconstruction)

**CDG:** `data/solution_cdgs/trackml_5th.json` | **Coverage:** 4/11 (36%)

| Stage | Binding | Confidence | Category | Notes |
|-------|---------|------------|----------|-------|
| detector_geometry_autodiscovery | gap | 0.3 | `PARTIAL_BIND` | Uses KMeans+MeanShift but as a geometry discovery workflow. |
| coordinate_rescaling_for_knn | gap | 0.0 | `MISSING_ATOM` | Physics-informed coordinate warping for kNN. |
| per_layer_knn_search | `nearest_neighbors_fit` | 0.85 | bound | Direct sklearn NearestNeighbors use. |
| circle_from_three_points | `...helix_geometry.circle_from_three_points` | 1.0 | bound | |
| helix_pitch_least_squares | `...helix_geometry.helix_pitch_least_squares` | 1.0 | bound | |
| helix_cylinder_intersection | gap | 0.0 | `MISSING_ATOM` | Analytic helix-cylinder intersection. In catalog, not ingested. |
| helix_cap_intersection | gap | 0.0 | `MISSING_ATOM` | Helix-cap intersection. In catalog, not ingested. |
| perturbative_helix_correction | gap | 0.0 | `MISSING_ATOM` | Learned displacement maps with physics scaling. |
| helix_nearest_point_distance | `...helix_geometry.helix_nearest_point_distance` | 1.0 | bound | Kepler equation reduction. |
| bayesian_neighbor_evaluation | gap | 0.0 | `MISSING_ATOM` | Adaptive Bayesian signal/background thresholding. |
| greedy_track_commit | gap | 0.0 | `MISSING_ATOM` | Greedy assignment with overlap resolution. |

**Findings:**

**Finding T-1: Cross-repo binding required**
Bindings span two repos: `sciona-atoms` (helix_geometry, 3 atoms) and
`sciona-atoms-ml` (sklearn.neighbors, 1 atom). The architect must search
multiple provider repos. Currently `PrimitiveCatalog` discovers atoms from
`SCIONA_ATOM_PROVIDER_ROOTS` — verify this includes all sibling repos.

**Finding T-2: helix_geometry atoms also have concept_type='custom' (G-006 again)**
Same issue as causal inference. The helix atoms won't be found by category
search. The pattern is now confirmed across 3 repos worth of Kaggle-ingested
atoms.

**Finding T-3: 'analytic model + learned residual' is a reusable meta-pattern**
`perturbative_helix_correction` applies physics-informed scaling to learned
displacement maps. This pattern (approximate analytic prediction, then add a
learned correction) appears in many domains (weather, materials, robotics).
Should be recognized as a composition template.

---

## Cross-Solution Patterns (3 solutions analyzed)

### Recurring Gap Categories

| Category | Count | Solutions | Example |
|----------|-------|-----------|---------|
| `MISSING_ATOM` | 10 | All 3 | global_activity_reweighting, helix_cylinder_intersection |
| `PARTIAL_BIND` | 2 | Connectomics, TrackML | pca_fit exists but get_precision() doesn't |
| `SEM_MISMATCH` | 1 | Connectomics | VarianceThreshold vs element-wise threshold |
| `ORCH_GAP` | 2 | Connectomics, Cause-Effect | threshold_sweep, asymmetric_feature_difference |
| `TRIVIAL` | 3 | Connectomics | np.diff, weighted sum |

### Systemic Issues (affect all solutions)

**Issue S-1: concept_type='custom' on all Kaggle-ingested atoms**
Every atom ingested from Kaggle solutions has `concept_type=custom` in its
CDG. This means `PrimitiveCatalog.search_by_category()` will never match
them by concept type. The architect must rely entirely on keyword/embedding
fallback for these atoms. Confirmed across 3 repos, 21+ atoms.

**Fix:** Assign meaningful concept_types during ingestion. The solution CDGs
already have good concept_types on their stages — these should propagate to
the atom CDGs. Mapping:
- Information-theoretic atoms → `information_theory`
- Geometric atoms → `geometry`
- Signal processing atoms → `signal_filter` or `signal_transform`
- Statistical analysis atoms → `analysis`
- Searching/matching atoms → `searching`
- Greedy assignment atoms → `greedy`
- Clustering-based atoms → `clustering`

**Issue S-2: No composition/orchestration primitives**
Two solutions independently surfaced the need for higher-order patterns:
- Connectomics: threshold_sweep_ensemble (MAP_OVER × parameter grid)
- Cause-Effect: asymmetric_feature_difference (MAP_OVER × direction)
The atom catalog has individual primitives but no way to express "run this
sub-pipeline N times with different parameters." The ConceptType enum has
`MAP_OVER` and `FIXED_POINT` — these should be leveraged.

**Issue S-3: State-query atoms missing for sklearn**
PCA's `get_precision()` is not the only state query missing. Many sklearn
estimators expose useful state beyond fit/predict (e.g., `coef_`, `components_`,
`feature_importances_`). These are not covered by existing atoms.

### Engineering Priorities

| Priority | Issue | Impact | Fix Effort |
|----------|-------|--------|------------|
| **P0** | S-1: concept_type='custom' | Blocks category-based retrieval for 21+ atoms | Low: update CDG files |
| **P1** | S-2: No orchestration primitives | Blocks representation of parameterized sub-pipelines | Medium: design MAP_OVER template |
| **P2** | S-3: State-query atoms | Partial bindings where fit exists but query doesn't | Medium: ingest sklearn state accessors |
| P3 | MISSING_ATOM backlog | 10 novel atoms across 3 solutions | High: individual ingestion per atom |
