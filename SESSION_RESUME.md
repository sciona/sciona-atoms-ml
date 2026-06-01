# Session Resume

This file is the operational handoff for a clean agent session. `REMEDIATION.md`
remains the authoritative backlog ledger. This file captures the current
frontier, execution workflow, and repo-specific constraints needed to resume
work safely.

## Current Status

- Branch target: `main`
- Current remote state: `main` synced with `origin/main` at the time of writing
- Active remediation frontier:
  - `sklearn.linear_model` optimizer and callback boundaries
  - coordinate-descent has many landed shells; the latest frontier audit
    recommended moving to non-coordinate-descent callback seams with lower
    duplicate risk
- `REMEDIATION.md` is up-to-date through:
  - `sklearn.linear_model.glm_score_deviance_tail`

## Known Unrelated Local Modification

No unrelated local modifications were observed at the time of writing.

## Hard Constraints

- Do not touch `audit_manifest.json`.
- Do not revert unrelated user/local changes.
- Use `apply_patch` for manual edits.
- Keep edits narrowly scoped to the active family.
- For each new family, update `REMEDIATION.md`.

## Validation Commands

Use these exact command shapes for new families.

Focused tests:

```bash
PYTHONPATH=src:../sciona-atoms/src:../sciona-matcher python -m pytest -q \
  tests/test_ml_sklearn_<family>_behavior.py \
  tests/test_ml_sklearn_<family>_references_metadata.py \
  tests/test_ml_sklearn_<family>_review_bundle.py
```

Publishability:

```bash
../sciona-matcher/.venv/bin/python \
  ../sciona-atoms/scripts/verify_publishability.py \
  src/sciona/atoms/ml/sklearn/<path-to-family>
```

Dejargon:

```bash
../sciona-matcher/.venv/bin/python \
  ../sciona-atoms/scripts/validate_dejargon.py --root . --ci
```

## Required File Set For Each New Family

Under `src/sciona/atoms/ml/sklearn/.../<family>/`:

- `__init__.py`
- `atoms.py`
- `witnesses.py`
- `cdg.json`
- `references.json`

Tests:

- `tests/test_ml_sklearn_<family>_behavior.py`
- `tests/test_ml_sklearn_<family>_references_metadata.py`
- `tests/test_ml_sklearn_<family>_review_bundle.py`

Review bundle:

- `data/review_bundles/ml_sklearn_<family>.review_bundle.json`

Ledger update:

- add one completed-slice bullet to `REMEDIATION.md`

## Commit Workflow

1. Validate the family with the commands above.
2. Stage only:
   - the family directory
   - its 3 tests
   - its review bundle
   - `REMEDIATION.md`
   - `SESSION_RESUME.md`
3. Commit with a message of the form:

```bash
git commit -m "add <family description> atoms"
```

4. Push:

```bash
git push origin main
```

5. Re-check status:

```bash
git status --short --branch
git show --stat --oneline --name-only HEAD
```

## Current Coordinate-Descent Coverage

Already landed in this section:

- `coordinate_descent_alpha_grid_math`
- `coordinate_descent_alpha_grid_prelude_shell`
- `coordinate_descent_enet_path_bookkeeping`
- `coordinate_descent_enet_path_input_shell`
- `coordinate_descent_enet_path_params_shell`
- `coordinate_descent_path_deprecation_prelude_shell`
- `coordinate_descent_enet_path_screening_shell`
- `coordinate_descent_enet_path_prefit_grid_payload_shell`
- `coordinate_descent_enet_path_prefit_grid_callback_shell`
- `coordinate_descent_enet_path_solver_dispatch`
- `coordinate_descent_enet_path_state_setup`
- `coordinate_descent_enet_path_loop_tail`
- `coordinate_descent_lasso_path_wrapper`
- `coordinate_descent_estimator_postfit_shell`
- `coordinate_descent_cv_postfit_shell`
- `coordinate_descent_cv_api_shell`
- `coordinate_descent_multitask_estimator_shell`
- `coordinate_descent_cv_target_guards`
- `coordinate_descent_cv_alpha_bookkeeping`
- `coordinate_descent_cv_deprecation_prelude_shell`
- `coordinate_descent_cv_routing_guards`
- `coordinate_descent_cv_path_params_shell`
- `coordinate_descent_cv_mse_selection_shell`
- `coordinate_descent_cv_refit_setup_shell`
- `coordinate_descent_path_residuals_prelude`
- `coordinate_descent_path_residuals_copy_isolation_shell`
- `coordinate_descent_path_residuals_split_slicing_shell`
- `coordinate_descent_path_residuals_path_params_shell`
- `coordinate_descent_path_residuals_writeable_array_shell`
- `coordinate_descent_path_residuals_callback_shell`
- `coordinate_descent_path_residuals_mono_output_normalization`
- `coordinate_descent_path_residuals_projection_shell`
- `coordinate_descent_path_residuals_error_aggregation`
- `coordinate_descent_cv_parallel_setup_shell`
- `coordinate_descent_cv_parallel_callback_shell`
- `coordinate_descent_cv_best_update_shell`
- `coordinate_descent_cv_refit_callback_shell`
- `coordinate_descent_cv_splitter_callback_shell`
- `coordinate_descent_cv_unweighted_refit_callback_shell`
- `coordinate_descent_cv_metadata_routing_callback_shell`
- `coordinate_descent_cv_nonrouting_fallback_shell`
- `coordinate_descent_cv_alpha_packaging_tail`
- `coordinate_descent_cv_validation_prelude_shell`
- `coordinate_descent_cv_validation_callback_shell`
- `coordinate_descent_cv_target_callback_shell`
- `coordinate_descent_cv_estimator_params_callback_shell`
- `coordinate_descent_cv_alpha_validation_callback_shell`
- `coordinate_descent_cv_subclass_api_shell`
- `coordinate_descent_estimator_sample_weight_shell`
- `coordinate_descent_estimator_validation_prelude_shell`
- `coordinate_descent_estimator_prefit_shell`
- `coordinate_descent_estimator_prefit_callback_shell`
- `coordinate_descent_estimator_loop_setup_shell`
- `coordinate_descent_estimator_loop_tail_shell`
- `coordinate_descent_estimator_multitarget_postfit_shell`
- `coordinate_descent_lasso_estimator_api_shell`
- `coordinate_descent_multitask_lasso_estimator_api_shell`
- `coordinate_descent_elastic_net_api_shell`
- `coordinate_descent_elastic_net_dense_decision_callback_shell`
- `coordinate_descent_elastic_net_class_api_shell`
- `coordinate_descent_multitask_elastic_net_api_shell`
- `coordinate_descent_multitask_validation_prelude_shell`
- `coordinate_descent_multitask_solver_setup_shell`
- `coordinate_descent_multitask_cv_api_shell`
- `coordinate_descent_set_order_helper_shell`
- `coordinate_descent_enet_path_validation_callback_shell`
- `coordinate_descent_enet_path_solver_payload_shell`
- `coordinate_descent_estimator_intercept_callback_shell`
- `coordinate_descent_multitask_solver_result_shell`
- `coordinate_descent_lasso_cv_init_shell`
- `coordinate_descent_elastic_net_cv_init_shell`
- `coordinate_descent_cv_base_init_shell`
- `coordinate_descent_cv_base_constraints_shell`
- `coordinate_descent_cv_base_abstract_api_shell`
- `coordinate_descent_cv_base_fit_context_shell`
- `coordinate_descent_multitask_fit_context_shell`
- `coordinate_descent_multitask_cv_sample_weight_absence_shell`
- `coordinate_descent_path_validation_decorator_shell`
- `coordinate_descent_enet_path_return_shell`
- `coordinate_descent_cv_subclass_fit_return_shell`
- `coordinate_descent_cv_metadata_router_payload_shell`
- `coordinate_descent_cv_metadata_router_check_cv_callback_shell`
- `coordinate_descent_cv_metadata_router_method_mapping_callback_shell`
- `coordinate_descent_cv_metadata_router_add_callback_shell`
- `coordinate_descent_cv_tags_super_callback_shell`
- `coordinate_descent_elastic_net_tags_super_callback_shell`
- `coordinate_descent_multitask_cv_tags_super_callback_shell`

## Recent Non-Coordinate-Descent Coverage

- `glm_fit_optimizer_shell`
- `glm_score_deviance_tail`
- `glm_tags_loss_callback_shell`
- `huber_fit_optimizer_shell`
- `huber_tags_super_callback_shell`
- `lars_cv_orchestration_shell`
- `lars_cv_refit_callback_shell`
- `logistic_cv_best_refit_selection_shell`
- `logistic_cv_final_array_packaging_tail`
- `logistic_cv_l1_axis_packaging_tail`
- `logistic_cv_path_result_packaging_shell`
- `logistic_cv_refit_callback_payload_shell`
- `logistic_fit_postpath_packaging_shell`
- `quantile_linprog_callback_shell`
- `quantile_sparse_lp_matrix_shell`
- `quantile_solver_guard_shell`
- `quantile_linprog_failure_message_shell`
- `ransac_callback_orchestration_shell`
- `ransac_fit_prelude_termination_shell`
- `ransac_metadata_routing_shell`
- `ransac_predict_score_callback_shell`
- `sgd_classifier_fit_callback_shell`
- `sgd_regressor_fit_callback_shell`
- `sgd_tags_super_callback_shell`

## Next Likely Seams

Coordinate-descent remediation is closed for the audited sklearn 1.6.1 source.
The final duplicate-coverage audit found no remaining deterministic, bounded,
non-duplicative coordinate-descent seam. A tiny multitask tag-super package was
identified as possible but intentionally skipped because tag-only seams are
low value and tag values are already represented elsewhere.

The next best bounded candidates should come from non-coordinate
`sklearn.linear_model` optimizer and callback boundaries:

1. Re-read the non-coordinate deferred-target ledger before choosing a seam
   - prioritize non-tag optimizer/callback shells with clear source-local
     behavior
   - avoid broad estimator wrappers that hide SciPy, native, or Cython solver
     boundaries
   - avoid tag-only families unless they unblock a larger non-tag remediation

Best audited next candidates after the current wave:

- `sgd_classifier_partial_fit_callback_shell`
  - source: sklearn 1.6.1 `_stochastic_gradient.py` lines 871-899
  - likely scope: `partial_fit(...)->_partial_fit(...)` callback payload,
    first-call predicate, validate-params payload, balanced class-weight
    rejection message, and callback result identity
  - keep `_partial_fit` internals and compiled `_plain_sgd` outside the slice

- `logistic_scoring_path_callback_shell`
  - source: sklearn 1.6.1 `_logistic.py` lines 735-804
  - likely scope: `_log_reg_scoring_path` fold slicing, delegated path-call
    payloads, temporary estimator state from supplied coefficients, and scorer
    callback payloads
  - keep scorer lookup/execution and `_logistic_regression_path` execution
    outside the slice

- `logistic_fit_path_dispatch_payload_shell`
  - source: sklearn 1.6.1 `_logistic.py` lines 1193-1373
  - likely scope: non-liblinear `LogisticRegression.fit` path dispatch payloads
    around warm-start expansion, class iteration, `prefer`, C/penalty
    normalization, and `n_threads`
  - avoid absorbing the direct liblinear branch or solver execution

Completed current wave:

- Final coordinate-descent duplicate audit
  - source-side and coverage-matrix audits both rejected a broad final
    coordinate-descent atom wave
  - no new atoms were added because remaining coordinate-descent text is
    compiled solver boundary or duplicate of existing validation, routing,
    path, parallel, MSE, refit, postfit, metadata-routing, callback, or tag
    slices

- `coordinate_descent_path_residuals_copy_isolation_shell`
  - publishes the single remaining narrow `_path_residuals` seam:
    shallow `path_params.copy()` isolation before local per-fold mutation
  - leaves subsequent `Xy`, `X_offset`, `X_scale`, `precompute`, `copy_X`,
    `alphas`, `sample_weight`, and `l1_ratio` updates with the existing
    `coordinate_descent_path_residuals_path_params_shell`

- `ransac_fit_prelude_termination_shell`
  - publishes deterministic `RANSACRegressor.fit` helpers for `min_samples`
    resolution, the `min_samples > n_samples` guard payload, accepted-consensus
    stop conditions, and the valid-consensus max-skips warning payload
  - leaves residual/loss/inlier/consensus math, dynamic-trial math, subset
    extraction, callback payloads, aggregate skip-limit guards, no-consensus
    messages, warning emission, estimator fitting, and final refit payloads
    outside this slice

- `ransac_predict_score_callback_shell`
  - publishes deterministic `RANSACRegressor.predict` and
    `RANSACRegressor.score` public callback-boundary helpers: fixed
    `validate_data` kwargs, non-routing empty parameter fallbacks,
    estimator `.predict` payloads, and estimator `.score` payloads
  - leaves fitted checks, validation execution, `_raise_for_params`,
    metadata-routing execution, arbitrary base-estimator behavior, and
    estimator mutation outside the slice

- `ransac_metadata_routing_shell`
  - publishes deterministic `RANSACRegressor.get_metadata_routing` helpers:
    router owner name, fixed estimator caller/callee method-mapping pairs,
    `MethodMapping.add` kwargs, `MetadataRouter.add` estimator payload, and
    final router return identity
  - leaves MetadataRouter/MethodMapping construction, metadata-routing
    execution, arbitrary base-estimator behavior, tags, and estimator mutation
    outside the slice

- `quantile_linprog_callback_shell`
  - publishes deterministic `QuantileRegressor.fit` external solver callback
    helpers for exact `scipy.optimize.linprog` keyword payload preservation and
    raw `result.x` solution extraction
  - leaves LP construction, solver guards/options, warning behavior, solution
    decoding, solver execution, and estimator mutation outside the slice

- `quantile_sparse_lp_matrix_shell`
  - publishes deterministic `QuantileRegressor.fit` HiGHS sparse
    equality-matrix helpers: CSC identity block construction, CSC
    intercept-column construction, and branch-specific CSC `A_eq` assembly
    with and without an intercept
  - leaves zero-weight row filtering, objective-vector construction, solver
    guards/options, `linprog` execution, solution decoding, warnings, and
    estimator mutation outside the slice

- `sgd_one_class_fit_shell`
  - publishes deterministic `SGDOneClassSVM._fit_one_class` / `_partial_fit`
    helpers for artificial one-class targets, positive-weight validation masks,
    fixed one-class solver context, offset/intercept conversion, `t_`
    advancement, averaging threshold and buffer allocation, one-class parameter
    allocation payloads, `_fit_one_class` delegation payloads, and `_partial_fit`
    return identity
  - leaves compiled `_plain_sgd`, stochastic updates, convergence, dataset
    construction, seed generation, and generic SGD formulas outside the slice

- `lars_cv_orchestration_shell`
  - publishes the deterministic `_lars_path_residues` keyword callback payload
    assembled by `LarsCV.fit` and inherited by `LassoLarsCV.fit`
  - leaves LARS/Lasso-LARS path solving, residual projection, shared-alpha
    interpolation, joblib scheduling, metadata routing, precompute warning
    behavior, estimator mutation, and final refit outside the slice

- `lars_cv_refit_callback_shell`
  - publishes deterministic `LarsCV.fit` helpers after CV alpha selection:
    selected alpha/CV-grid/MSE-path state payloads, final `_fit(...)` keyword
    payload, final `_fit` call payload, and fit return identity
  - leaves path solving, residual projection, shared-alpha interpolation, CV
    splitting, joblib scheduling, metadata routing, `_fit` execution,
    lasso-specific solver behavior, and estimator mutation outside the slice

- `glm_fit_optimizer_shell`
  - publishes deterministic `_GeneralizedLinearRegressor.fit` optimizer-boundary
    helpers for GLM initial coefficient setup, cold-start intercept
    initialization, L-BFGS-B payloads, Newton solver constructor payloads, and
    final coefficient/intercept unpacking
  - leaves raw-prediction and dense objective math with `glm`, tags and
    `_get_loss` callbacks with `glm_tags_loss_callback_shell`, and optimizer
    execution/convergence outside the slice

- `glm_score_deviance_tail`
  - publishes deterministic `_GeneralizedLinearRegressor.score` helpers for
    target validation kwargs, sample-weight validation payloads, invalid
    target-range message, weighted constant averaging from supplied loss
    callbacks, null raw-prediction tiling from a supplied linked mean, and final
    D2 from supplied model/null deviances
  - leaves `_linear_predictor`, validation execution, base-loss callbacks, link
    callbacks, optimizer work, and estimator mutation outside the slice

- `logistic_fit_postpath_packaging_shell`
  - publishes deterministic `LogisticRegression.fit` post-path packaging after
    `_logistic_regression_path` returns: path-result unzipping, `n_iter_`
    slicing, multinomial/OvR coefficient matrix layout, final `coef_`
    extraction, and final `intercept_` extraction or zero initialization
  - preserves coefficient dtype from path results, including float32-capable
    solver paths, and models sklearn's effective binary OvR tail with
    `n_classes=1`
  - leaves solver dispatch, validation, class encoding, liblinear,
    `_logistic_regression_path`, joblib scheduling, and
    `LogisticRegressionCV.fit` reshaping/refit outside the slice

- `logistic_cv_path_result_packaging_shell`
  - publishes deterministic `LogisticRegressionCV.fit` path-result packaging
    after `_log_reg_scoring_path` returns: 4-tuple unzipping, public `Cs_`
    selection, multinomial/OvR coefficient-path layout, branch-specific
    `n_iter_` layout, multinomial score tiling, score layout, and class-keyed
    `scores_` / `coefs_paths_` dictionaries
  - preserves coefficient, score, and iteration dtypes from scoring-path
    outputs and leaves the later l1-ratio public-axis expansion outside this
    first CV package
  - leaves solver dispatch, scorer callbacks, CV splitter construction,
    metadata routing, best-C/refit selection, final `coef_`/`intercept_`, and
    estimator mutation outside the slice

- `logistic_cv_l1_axis_packaging_tail`
  - publishes deterministic `LogisticRegressionCV.fit` public l1-ratio axis
    packaging when `self.l1_ratios is not None`: branch enablement,
    class-keyed `coefs_paths_` reshape/transpose, class-keyed `scores_`
    reshape/transpose, and `n_iter_` reshape/transpose with an inferred class
    axis
  - preserves coefficient, score, and iteration dtypes while retaining
    singleton C/l1 axes
  - leaves path-result unzipping with `logistic_cv_path_result_packaging_shell`
    and leaves solver dispatch, scorer callbacks, CV splitter construction,
    metadata routing, best-C/refit selection, final `coef_`/`intercept_`, and
    estimator mutation outside the slice

- `logistic_cv_best_refit_selection_shell`
  - publishes deterministic `LogisticRegressionCV.fit` best/refit selection
    helpers: per-loop OvR/multinomial path views, summed-score best-index
    selection, flattened C/l1 lookup, refit coefficient initialization,
    non-refit per-fold best-index selection, non-refit coefficient/C/l1
    averaging, and multinomial/OvR final component extraction from supplied
    weight results
  - handles non-elastic-net `None` l1-ratio values in the selection branch
    while requiring numeric l1 grids for elastic-net l1 averaging
  - leaves `_logistic_regression_path` refit execution and payload execution,
    scorer callbacks, CV splitter construction, metadata routing, validation,
    joblib scheduling, solver dispatch, and estimator mutation outside the
    slice

- `logistic_cv_refit_callback_payload_shell`
  - publishes deterministic `LogisticRegressionCV.fit` refit callback payload
    helpers for `_logistic_regression_path`: single-C grid packaging,
    `verbose=max(0, self.verbose - 1)`, exact keyword payload assembly with
    `check_input=False`, positional `X`/`y` call payload preservation, and
    first returned weight extraction after the solver boundary
  - leaves best-index/C/l1 selection and coefficient initialization with
    `logistic_cv_best_refit_selection_shell`
  - leaves solver execution, convergence, objective math, scorer callbacks,
    CV splitter construction, metadata routing, validation, joblib scheduling,
    path-result packaging, public l1-axis expansion, and estimator mutation
    outside the slice

- `logistic_cv_final_array_packaging_tail`
  - publishes deterministic `LogisticRegressionCV.fit` final `np.asarray`
    packaging for selected `C_`, selected `l1_ratio_`, and public
    `l1_ratios_` immediately before the existing public l1-axis reshaping
  - preserves object arrays for non-elastic-net `None` l1-ratio values and
    numeric arrays for selected C values
  - leaves path-result packaging, best/refit selection, refit callback
    payloads, final coefficient/intercept extraction, public l1-axis
    reshaping, solver execution, validation, scorer callbacks, CV splitter
    construction, metadata routing, joblib scheduling, and estimator mutation
    outside the slice

Pick the next seam by re-reading the immediate source region rather than
assuming this ordering is still optimal.

## Practical Notes

- Prefer explicit lambdas in `icontract` predicates when argument names are
  easy to mismatch.
- Avoid zero-input atoms unless they are clearly acceptable under current
  publishability rules; fixed constants often need to be folded into a
  surrounding atom.
- When validating tests, always include the full `PYTHONPATH` above. `src`
  alone is not enough because these families import shared `sciona.ghost`
  pieces from `../sciona-atoms/src`.
