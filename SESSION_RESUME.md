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
  - `sklearn.linear_model.logistic_fit_postpath_packaging_shell`

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
- `glm_tags_loss_callback_shell`
- `huber_fit_optimizer_shell`
- `huber_tags_super_callback_shell`
- `lars_cv_orchestration_shell`
- `logistic_fit_postpath_packaging_shell`
- `quantile_solver_guard_shell`
- `quantile_linprog_failure_message_shell`
- `ransac_callback_orchestration_shell`
- `ransac_fit_prelude_termination_shell`
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

Best audited next candidate after the current wave:

- `logistic_cv_path_result_packaging_shell`
  - basis: `/tmp/sciona-logistic-next-audit/report.md` identified
    `LogisticRegressionCV.fit` post-parallel result reshaping as the next
    distinct logistic seam after the basic `LogisticRegression.fit` tail
  - recommended scope: `_logistic.py` around lines 2035-2061 only, covering
    `coefs_paths`, `scores`, and `n_iter_` shape normalization and
    class-keyed packaging
  - keep best-C/refit selection, l1-ratio axis reshaping, solver dispatch,
    scorer callbacks, CV splitter construction, and metadata routing outside
    that first CV package

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

- `glm_fit_optimizer_shell`
  - publishes deterministic `_GeneralizedLinearRegressor.fit` optimizer-boundary
    helpers for GLM initial coefficient setup, cold-start intercept
    initialization, L-BFGS-B payloads, Newton solver constructor payloads, and
    final coefficient/intercept unpacking
  - leaves raw-prediction and dense objective math with `glm`, tags and
    `_get_loss` callbacks with `glm_tags_loss_callback_shell`, and optimizer
    execution/convergence outside the slice

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
