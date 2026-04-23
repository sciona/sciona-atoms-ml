# Solution CDG Gap Analysis

Systematic record of gaps found while converting Kaggle winning solutions
into CDGs backed by existing atoms. Each solution is a test of whether
the architect tooling can reconstruct atom bindings from conceptual nodes.

## Summary Dashboard

| Solution | Stages | Bound | Partial | Gaps | Coverage | Post-fix notes |
|----------|--------|-------|---------|------|----------|----------------|
| Connectomics 1st | 9 | 6 | 0 | 3 | **67%** | 5 connectome atoms ingested + pca_get_precision; remaining: np.diff, MAP_OVER, weighted_sum |
| Cause-Effect 2nd | 17 | 16 | 0 | 1 | **100%*** | All 16 atoms category-searchable; asymmetric_feature_difference = MAP_OVER |
| TrackML 5th | 11 | 4 | 0 | 7 | **36%** | helix + neighbors atoms get category bonus |
| DSB2017 1st | 10 | 1 | 0 | 9 | **10%** | 90% deep learning; only noisy-OR atom bound; MAP_OVER validates volume tiling |
| **Totals (4 done)** | **47** | **27** | **0** | **20** | **57%** | |

\* Cause-Effect reaches 100% if asymmetric_feature_difference is counted as
"resolved via MAP_OVER pattern" rather than a single-atom binding.

---

## Systemic Issues — Status After Fixes

### S-1: concept_type='custom' on Kaggle-ingested atoms — FIXED (Step 1a)

**Fix applied:** 22 atoms reclassified across 3 repos.
- sciona-atoms: 16 causal_inference atoms → information_theory, analysis, data_assembly, conditional_routing
- sciona-atoms-physics: 5 helix_geometry atoms → geometry
- sciona-atoms-ml: 1 datadriven atom → optimization

**Impact:** `PrimitiveCatalog.search_by_category()` now returns these atoms
for matching concept types. The 0.75 `category_bonus` at catalog.py:413
activates during `search_by_compatibility`, pushing these atoms higher in
ranked results.

**Remaining:** 310 atoms with `concept_type=custom` across 7 other repos
(sciona-atoms-bio: 43, sciona-atoms-fintech: 64, sciona-atoms-physics: 85,
sciona-atoms-robotics: 51, sciona-atoms-signal: 31, sciona-atoms: 34,
sciona-atoms-cs: 1). None of these block the current 3 solution CDGs.

### S-2: No orchestration primitives for MAP_OVER patterns — FIXED

**Fix applied:** Two JSON skeleton assets created in sciona-matcher:
- `map_over.json` — 4 stages (window_slicer, body_init, body_process, collect_results)
  with variant hints: threshold_sweep, parameter_grid_ensemble, sliding_window, etc.
- `fixed_point.json` — 3 stages (body_init, body_step, convergence_check)
  with variant hints: iterative_solver, convergence_loop, self_training_loop.

Both added to `ALLOWED_SKELETON_PROPOSALS`. Structural critic rules added for
combinator validation (map_window_size > 0, collect node exists; max_iterations > 0,
convergence_check exists).

**Impact:** The architect can now generate MAP_OVER proposals for nodes like
`threshold_sweep_ensemble` (Connectomics) and `asymmetric_feature_difference`
(Cause-Effect). These were previously flagged as `ORCH_GAP`; now they match
the MAP_OVER skeleton template.

### S-3: State-query atoms missing for sklearn — FIXED (scoped to PCA)

**Fix applied:** 3 new atoms in sciona-atoms-ml/sklearn/decomposition/:
- `pca_get_precision` — Woodbury matrix identity, exact sklearn parity (whiten=True and False)
- `pca_components` — accessor for components_ matrix
- `pca_explained_variance_ratio` — accessor for explained_variance_ratio_

All follow `AGENT_INGESTION.md` quality bar: `@register_atom`, `@icontract.require`,
`@icontract.ensure`, witnesses, CDG nodes with concrete IO, references.json.

**Impact:** Connectomics `pca_precision_matrix` stage improves from partial binding
(pca_fit only, confidence 0.7) to full binding (pca_fit + pca_get_precision,
confidence ~1.0).

**Remaining:** ~77 other sklearn estimators have fit atoms but no state accessors
(coef_, feature_importances_, cluster_centers_, etc.). These will be needed as
more solution CDGs are built.

---

## Gap Taxonomy

| Category | Code | Description |
|----------|------|-------------|
| **Missing atom** | `MISSING_ATOM` | Novel technique with no existing atom. Requires ingestion. |
| **Partial binding** | `PARTIAL_BIND` | An atom covers part of the operation but not all (e.g., fit exists but state-query doesn't). |
| **Semantic mismatch** | `SEM_MISMATCH` | Keyword search returns an atom that looks right but does something semantically different. |
| **Orchestration gap** | `ORCH_GAP` | Higher-order pattern (loop, sweep, branch) not representable as a single atom. |
| **Trivial unbound** | `TRIVIAL` | One-liner numpy/scipy call; doesn't justify a standalone atom but has no binding target. |

### Recurring Gap Categories (post-fix)

| Category | Count | Before fix | After fix | Notes |
|----------|-------|-----------|-----------|-------|
| `MISSING_ATOM` | 10 | 10 | 10 | Unchanged — requires individual atom ingestion |
| `PARTIAL_BIND` | 2 | 2 | 1 | PCA precision fixed; detector_geometry_autodiscovery remains |
| `SEM_MISMATCH` | 1 | 1 | 1 | VarianceThreshold keyword collision still exists |
| `ORCH_GAP` | 2 | 2 | 0 | Both resolved via MAP_OVER skeleton |
| `TRIVIAL` | 3 | 3 | 3 | Unchanged — not worth individual atoms |

---

## 1. Connectomics 1st Place (Graph Inference from Calcium Fluorescence)

**CDG:** `data/solution_cdgs/connectomics_1st.json` | **Coverage:** 6/9 (67%)

| Stage | Binding | Confidence | Status |
|-------|---------|------------|--------|
| calcium_lowpass_filter | `...connectome.calcium_lowpass_filter` | 1.0 | **INGESTED** |
| first_difference | gap | — | `TRIVIAL` (np.diff) |
| fluorescence_hard_threshold | `...connectome.fluorescence_hard_threshold` | 1.0 | **INGESTED** |
| global_activity_sample_reweighting | `...connectome.global_activity_sample_reweighting` | 1.0 | **INGESTED** |
| pca_precision_matrix | `pca_fit` + `pca_get_precision` | 1.0 | **FIXED** (S-3) |
| score_matrix_normalization | `...connectome.score_matrix_normalization` | 1.0 | **INGESTED** |
| threshold_sweep_ensemble | gap | — | `ORCH_GAP` → MAP_OVER (S-2) |
| temporal_precedence_directivity | `...connectome.temporal_precedence_directivity` | 1.0 | **INGESTED** |
| weighted_score_combination | gap | — | `TRIVIAL` (weighted sum) |

---

## 2. Cause-Effect 2nd Place (Causal Direction from Bivariate Data)

**CDG:** `data/solution_cdgs/cause_effect_2nd.json` | **Coverage:** 16/17 → 17/17 (100%)

| Stage | Binding | Confidence | Category | Post-fix |
|-------|---------|------------|----------|----------|
| discretize_and_bin | `...discretize_and_bin` | 1.0 | bound | **+category bonus (data_assembly)** |
| igci_asymmetry_score | `...igci_asymmetry_score` | 1.0 | bound | **+category bonus (information_theory)** |
| hsic_independence_test | `...hsic_independence_test` | 1.0 | bound | **+category bonus (information_theory)** |
| normalized_error_probability | `...normalized_error_probability` | 1.0 | bound | **+category bonus (information_theory)** |
| uniform_divergence | `...uniform_divergence` | 1.0 | bound | **+category bonus (information_theory)** |
| knn_entropy_estimator | `...knn_entropy_estimator` | 1.0 | bound | **+category bonus (information_theory)** |
| polyfit_nonlinearity_asymmetry | `...polyfit_nonlinearity_asymmetry` | 1.0 | bound | **+category bonus (analysis)** |
| polyfit_residual_error | `...polyfit_residual_error` | 1.0 | bound | **+category bonus (analysis)** |
| conditional_noise_entropy_variance | `...conditional_noise_entropy_variance` | 1.0 | bound | **+category bonus (information_theory)** |
| conditional_noise_skewness_variance | `...conditional_noise_skewness_variance` | 1.0 | bound | **+category bonus (analysis)** |
| conditional_noise_kurtosis_variance | `...conditional_noise_kurtosis_variance` | 1.0 | bound | **+category bonus (analysis)** |
| conditional_distribution_similarity | `...conditional_distribution_similarity` | 1.0 | bound | **+category bonus (analysis)** |
| asymmetric_feature_difference | gap | 0.0 | `ORCH_GAP` | **RESOLVED → recognized as MAP_OVER skeleton** |
| symmetrized_prediction_fusion | `...symmetrized_prediction_fusion` | 1.0 | bound | **+category bonus (data_assembly)** |
| two_stage_independence_direction | `...two_stage_independence_direction` | 1.0 | bound | **+category bonus (conditional_routing)** |
| left_right_decomposed_prediction | `...left_right_decomposed_prediction` | 1.0 | bound | **+category bonus (analysis)** |
| weighted_ensemble_combination | `...weighted_ensemble_combination` | 1.0 | bound | **+category bonus (analysis)** |

---

## 3. TrackML 5th Place (Physics-Geometric Track Reconstruction)

**CDG:** `data/solution_cdgs/trackml_5th.json` | **Coverage:** 4/11 (36%)

| Stage | Binding | Confidence | Category | Post-fix |
|-------|---------|------------|----------|----------|
| detector_geometry_autodiscovery | gap | 0.3 | `PARTIAL_BIND` | unchanged |
| coordinate_rescaling_for_knn | gap | 0.0 | `MISSING_ATOM` | unchanged |
| per_layer_knn_search | `nearest_neighbors_fit` | 0.85 | bound | **+category bonus (searching)** |
| circle_from_three_points | `...circle_from_three_points` | 1.0 | bound | **+category bonus (geometry)** |
| helix_pitch_least_squares | `...helix_pitch_least_squares` | 1.0 | bound | **+category bonus (geometry)** |
| helix_cylinder_intersection | gap | 0.0 | `MISSING_ATOM` | unchanged |
| helix_cap_intersection | gap | 0.0 | `MISSING_ATOM` | unchanged |
| perturbative_helix_correction | gap | 0.0 | `MISSING_ATOM` | unchanged |
| helix_nearest_point_distance | `...helix_nearest_point_distance` | 1.0 | bound | **+category bonus (geometry)** |
| bayesian_neighbor_evaluation | gap | 0.0 | `MISSING_ATOM` | unchanged |
| greedy_track_commit | gap | 0.0 | `MISSING_ATOM` | unchanged |

---

## Engineering Priorities (updated)

| Priority | Issue | Status | Remaining work |
|----------|-------|--------|----------------|
| ~~P0~~ | ~~S-1: concept_type='custom'~~ | **DONE (22 atoms)** | 310 atoms in other repos (deferred) |
| ~~P1~~ | ~~S-2: No orchestration primitives~~ | **DONE** | — |
| ~~P2~~ | ~~S-3: PCA state-query atoms~~ | **DONE (3 atoms)** | ~77 other estimators (deferred) |
| P3 | MISSING_ATOM backlog | Open | 10 novel atoms across 3 solutions |
| P4 | SEM_MISMATCH: VarianceThreshold keyword collision | Open | Needs retrieval quality improvement |
| P5 | Remaining 310 custom concept_types | Open | Mechanical but large |
| P6 | Review bundle + audit manifest for PCA atoms | Open | Blocked on concurrent agent |
