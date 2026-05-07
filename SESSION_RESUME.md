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
  - specifically the deterministic decomposition of
    `sklearn.linear_model._coordinate_descent.LinearModelCV.fit`
- `REMEDIATION.md` is up-to-date through:
  - `sklearn.linear_model.coordinate_descent_cv_alpha_validation_callback_shell`

## Known Unrelated Local Modification

Do not touch this unless the user explicitly asks:

- `data/expansions/ml_model_selection.json`

It is currently a local modification outside the remediation slices being
landed.

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
- `coordinate_descent_enet_path_bookkeeping`
- `coordinate_descent_enet_path_input_shell`
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
- `coordinate_descent_cv_routing_guards`
- `coordinate_descent_cv_path_params_shell`
- `coordinate_descent_cv_mse_selection_shell`
- `coordinate_descent_cv_refit_setup_shell`
- `coordinate_descent_path_residuals_prelude`
- `coordinate_descent_path_residuals_path_params_shell`
- `coordinate_descent_path_residuals_writeable_array_shell`
- `coordinate_descent_path_residuals_callback_shell`
- `coordinate_descent_path_residuals_mono_output_normalization`
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

## Next Likely Seams

The latest pass re-read `LinearModelCV.fit` end to end and found one
remaining small deterministic callback shell around user-provided alpha
validation. That shell is now covered by
`coordinate_descent_cv_alpha_validation_callback_shell`.

The next best bounded candidates are:

1. another coordinate-descent class/helper seam
   - move to the next blocked coordinate-descent source region with a small
     deterministic boundary rather than duplicating the `LinearModelCV.fit`
     shells already landed

2. final duplicate-coverage check
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
