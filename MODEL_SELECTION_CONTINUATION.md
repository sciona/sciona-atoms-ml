# Model Selection Heuristics — Continuation Notes

## Status

**Family 1 (diagnostics): COMPLETE — 16 atoms, tested, published**

All 16 diagnostic atoms are implemented, tested, and published at:
`src/sciona/atoms/ml/model_selection/diagnostics/atoms.py`

**Family 2 (recommendations): COMPLETE — 8 atoms, tested, published**

All 8 recommendation atoms are implemented, tested, and published at:
`src/sciona/atoms/ml/model_selection/recommendations/atoms.py`

**Publishing artifacts: COMPLETE for both families**

## What's Done

- Directory structure created at `src/sciona/atoms/ml/model_selection/`
- `diagnostics/atoms.py` — 16 atoms, all tested
- `diagnostics/witnesses.py` — 16 witnesses
- `diagnostics/__init__.py`
- `recommendations/__init__.py`

## What Remains

### 1. Recommendations Family (8 atoms)

Write `recommendations/atoms.py` with these 8 decision atoms:

| Atom | Inputs (all scalars/bools from diagnostics) | Section |
|------|---------------------------------------------|---------|
| `recommend_regularization` | condition_number, n_p_ratio, mutual_incoherence, lasso_viable | A |
| `recommend_loss_function` | excess_kurtosis, residual_kurtosis | B |
| `recommend_linear_model` | n, p, is_sparse, dispersion_index, tweedie_power | C |
| `recommend_tree_ensemble` | n, n_categorical, noise_level | D |
| `recommend_preprocessing` | skewness_array, vif_array, is_sparse, model_requires_scaling | E |
| `recommend_dimensionality_reduction` | condition_number, is_sparse, explained_variance | F |
| `recommend_hyperparameter_ranges` | model_type, n, p | G |
| `recommend_cv_strategy` | n, is_classification, is_timeseries, has_groups | H |

Each returns a dict with: `recommendation`, `confidence`, `sklearn_class`,
`reasoning`, `alternatives`, `thresholds_applied`, `source_sections`.

Named threshold constants (at module level):
```python
OLS_INSTABILITY_LIMIT = 30
LASSO_MUTUAL_INCOHERENCE_LIMIT = 1.0
KURTOSIS_ROBUST_LOSS_THRESHOLD = 1.0
DISPERSION_INDEX_POISSON = 1.0
DISPERSION_OVERDISPERSION = 1.1
SKEWNESS_TRANSFORM_THRESHOLD = 1.0
VIF_MODERATE_COLLINEARITY = 5
VIF_SEVERE_COLLINEARITY = 10
HIST_GB_SAMPLE_BOUNDARY = 10_000
EXPLAINED_VARIANCE_DEFAULT = 0.95
SOLVER_FEATURE_LIMIT = 1000
BOOSTING_DIMINISHING_LR = 0.125
BOOSTING_FLATLINE_ESTIMATORS = 200
LOOCV_VIABILITY_BOUNDARY = 100
ALPHA_FLOOR = 1e-5
```

Decision tables are in `heuristics.pdf` and in the raw text provided to
the agent in the conversation. The full text was pasted as a user message
— search for "Decision Heuristics for Deterministic Model Selection".

Also write `recommendations/witnesses.py`.

### 2. CDG files (both families)

Write `diagnostics/cdg.json` and `recommendations/cdg.json` with:
- snake_case node names (not human-readable)
- Full inputs/outputs on every atomic node
- All required fields per PUBLISHING.md

### 3. References

Add to `data/references/registry.json` in sciona-atoms-ml:
- `hoerl1970ridge`, `tibshirani1996lasso`, `zou2005elasticnet`
- `wainwright2009lasso`, `huber1964robust`, `friedman1999boosting`
- `varma2006nested`, `hastie2009esl`

Write `diagnostics/references.json` and `recommendations/references.json`
with fully-qualified atom keys and line numbers.

### 4. Heuristic Registry

Create `data/heuristics/families/model_selection.json` following the
schema in `sciona-atoms/data/heuristics/families/sequential_filter.json`.
One heuristic_id per recommendation atom.

### 5. Heuristic Metadata

Create `src/sciona/atoms/ml/model_selection/heuristic_metadata.json`
binding each recommendation atom's output to the registry.

### 6. Publishing (per PUBLISHING.md)

For EACH family:
- Create review bundle in `data/review_bundles/`
- Write 3 focused tests (review bundle, references metadata, behavior)
- Merge review bundles into `data/audit_manifest.json`
- All tests must pass

DB-compatibility rules:
- `risk_score` and `acceptability_score`: integers 0-100
- `acceptability_band`: use `"review_ready"` (from DB enum)
- `parity_coverage_level`: use `"positive_path"` (from DB enum)
- CDG node names: snake_case (not human-readable titles)
- References: schema v1.1, fully-qualified keys with `@file:line`
- Registry: local to this repo (`data/references/registry.json`)

### 7. Validation

- `verify_contribution_rules.py --repo-root .`
- `validate_dejargon.py --root .`
- Functional tests with synthetic data verifying recommendations match
  expected outcomes (e.g., high condition number → Ridge, heavy tails →
  Huber)

## Key Context

- The plan file is at `/Users/conrad/.claude/plans/mellow-mixing-moth.md`
- The research document is at `heuristics.pdf` in this repo
- The full raw text of the research was provided in the conversation
- Follow `../sciona-atoms/AGENT_INGESTION.md` and `PUBLISHING.md`
- The repo is on branch `kaggle-ingest-batch-1`
- Python venv: `../sciona-matcher/.venv/bin/python`
- Validation scripts: `../sciona-atoms/scripts/`

## Related Work in This Session

This session also produced:
- `sciona.tools` package in sciona-matcher (19 agent-callable tools)
- `AGENT_INGESTION.md` and `PUBLISHING.md` in sciona-atoms
- 21 Kaggle atoms across 4 families (3 causal inference + 1 helix geometry)
- 3 ML skeleton CDGs + 4 expansion rules in this repo
- Per-repo registry migration (all 8 sibling repos)
- Backfill fix for multi-repo registry discovery
- Line number validation in verify_contribution_rules.py
