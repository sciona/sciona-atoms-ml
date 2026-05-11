# Session Resume

This file is the operational handoff for a clean agent session. `REMEDIATION.md`
remains the authoritative backlog ledger. This file captures the current
frontier, execution workflow, and repo-specific constraints needed to resume
work safely.

## Current Status

- Branch target: `main`
- Current remote state: `main` synced with `origin/main` at the time of writing
- Active remediation frontier:
  - `sklearn.linear_model` coordinate-descent solvers
  - specifically the deterministic decomposition of multitask
    coordinate-descent estimators and CV wrappers in
    `sklearn.linear_model._coordinate_descent`
- `REMEDIATION.md` is up-to-date through:
  - `sklearn.linear_model.coordinate_descent_cv_metadata_router_check_cv_callback_shell`

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

## Next Likely Seams

The latest pass added a bounded `check_cv(self.cv)` callback-boundary shell for
`LinearModelCV.get_metadata_routing`. That shell is now covered by
`coordinate_descent_cv_metadata_router_check_cv_callback_shell`.

The next best bounded candidates are:

1. another coordinate-descent class/helper seam
   - move to the next blocked coordinate-descent source region with a small
     deterministic boundary rather than duplicating the `LinearModelCV.fit`
     shells, the CV subclass API shell, or the ElasticNet sample-weight
     validation-prelude, pre-fit, loop-setup, and loop-tail shells already
     landed, or the Lasso and MultiTaskLasso estimator API shells already
     landed, or the ElasticNet API and sparse-prediction shell already
     landed, or the MultiTaskElasticNet API and validation-prelude shells
     already landed, or the MultiTaskElasticNet solver-setup shell already
     landed, or the multitask CV API shell already landed, or the
     `_set_order` helper shell already landed, or the `enet_path`
     validation callback shell already landed, the `enet_path` solver
     payload shell already landed, the ElasticNet post-loop intercept
     callback shell already landed, the MultiTaskElasticNet solver-result
     tail shell already landed, the LassoCV constructor-forwarding shell
     already landed, the ElasticNetCV constructor shell already landed, or
     the LinearModelCV base constructor shell already landed, the
     `enet_path` params pop/default shell already landed, the `_alpha_grid`
     prelude shell already landed, or the `ElasticNet` class-level fit
     metadata and base parameter-constraint shell already landed, the
     `LinearModelCV.fit` n_alphas/alphas deprecation prelude already landed,
     the path-helper n_alphas/alphas deprecation prelude already landed, or
     the `enet_path` do_screening parameter and payload delta already landed,
     the sklearn 1.8 `enet_path` prefit/grid payload delta already landed,
     or the sklearn 1.8 `enet_path` prefit/grid callback output delta
     already landed, or the `ElasticNet.fit` prefit callback output delta
     already landed, or the `ElasticNet.fit` multi-target postfit
     passthrough already landed, or the `ElasticNet._decision_function`
     dense fallback already landed, or the `LinearModelCV` base
     parameter-constraint schema already landed, or the `LinearModelCV`
     abstract base API contract already landed, or the `LinearModelCV.fit`
     decorator context already landed, or the `MultiTaskElasticNet.fit`
     decorator context already landed, or the multitask CV signature-level
     sample-weight absence seam already landed, or the `lasso_path` and
     `enet_path` validate_params decorator schemas already landed, or the
     final public `enet_path` return tuple branch already landed, or the
     coordinate-descent CV subclass fit return passthrough already landed, or
     the `LinearModelCV.get_metadata_routing` owner/caller/callee spec and
     payload/return identities already landed, or the
     `LinearModelCV.get_metadata_routing` `check_cv(self.cv)` callback
     boundary already landed

2. next coordinate-descent seam after CV metadata-router check_cv callback shell
   - a worker audit is scouting ahead under
     `/tmp/sciona-worker-coordinate-next-after-cv-metadata-router-check-cv`
   - if present, review its recommended family before copying anything into
     the repo
   - duplicate-risk check should confirm the candidate is separate from the
     landed constructor, class-API, validation, path, callback, and solver
     payload shells listed above
   - do not copy
     `/tmp/sciona-worker-coordinate-next-after-cv-base-fit-context/coordinate_descent_cv_base_metadata_tags_shell`
     without redesign; it duplicates `coordinate_descent_cv_api_shell`
   - treat signature-only seams as low priority unless they add source-level
     evidence that is not already expressed by callback payload families

3. `_path_residuals` duplicate-coverage check
   - split slicing and projection are now covered alongside the existing
     sample-weight slicing, writeable-array, callback, mono-output
     normalization, path-params, and residual aggregation families
   - re-read the whole helper before adding any further `_path_residuals`
     family, because the obvious deterministic body seams are now likely
     exhausted

4. final duplicate-coverage check
   - before leaving `LinearModelCV.fit` permanently, verify that no new seam
     duplicates existing alpha, validation, routing, path, parallel, MSE,
     refit, or postfit slices

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
