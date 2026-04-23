# Remediation Continuation Notes

## Session State

- Repo: `/Users/conrad/personal/sciona-atoms-ml`
- Branch: `kaggle-ingest-batch-1`
- Normal sklearn ingestion inventory: complete. `SKLEARN.md` reports
  `Total targets in this inventory: 0`.
- Remaining sklearn work is the remediation backlog in `REMEDIATION.md`.

## Required Reading

Start a new remediation session by reading these files:

- `../sciona-atoms/AGENT_INGESTION.md`
- `REMEDIATION.md`
- `REMEDIATION_PROMPT.md`
- `SKLEARN.md`
- `PUBLISHING.md` if present in the active repo or shared atom repo

`REMEDIATION.md` contains the authoritative Case 1-8 remediation decision
rules. Apply those top-level case rules even where older per-section text says
to "decide whether" a wrapper, helper-first path, or native decomposition is
acceptable.

## Dirty Files To Avoid Unless Relevant

At the time this handoff was written, the worktree had pre-existing changes:

- `data/audit_manifest.json`
- `data/references/registry.json`
- `heuristics.pdf`
- `heuristics.txt`

Do not stage, revert, or rewrite those files unless the current remediation
task specifically needs them. If publishing a remediation family legitimately
requires changing `data/audit_manifest.json` or `data/references/registry.json`,
inspect the existing diffs first and preserve unrelated user changes.

## Remediation Policy Summary

- Case 1 native/compiled boundaries: ingest public Python API methods only with
  strict contracts around compiled-kernel inputs and outputs, publish as
  `pass_with_limits`, and document the compiled FFI topology limitation.
- Case 2 external optimizers: ingest objective and gradient functions as the
  core atoms; treat optimizer calls as FFI-style boundaries.
- Case 3 meta-estimators: model wrappers as higher-order functions over
  explicit `Protocol` boundaries; witnesses model combinatorics over abstract
  base outputs.
- Case 4 cross-validation/path orchestration: decompose into workflow atoms
  such as fold generation, fold fit/score, aggregation, and refit templates.
- Case 5 decomposition/matrix factorization: slice horizontally, exposing pure
  Python/NumPy steps and marking compiled solvers as boundaries.
- Case 6 clustering/spectral pipelines: expose affinity, Laplacian,
  embeddings, and label-assignment intermediates before estimator surfaces.
- Case 7 probabilistic/mutable-kernel workflows: use state-passing style;
  publish likelihood, gradient, posterior, and kernel-state transition atoms.
- Case 8 thin wrappers: keep blocked until upstream atoms are verified,
  provenanced, and published.

## Suggested First Remediation Slice

Start with a small Case 3 meta-estimator helper family rather than a large
native solver family. Good candidates from `REMEDIATION.md`:

- Voting aggregation helpers from `VotingClassifier` / `VotingRegressor`.
- One-vs-rest or output-code aggregation helpers from `sklearn.multiclass`.
- Feature-selection mask bookkeeping from `RFE` / `SelectFromModel`.

These should be publishable without solving arbitrary estimator training:

- Define protocols only for the callback boundary where needed.
- Make atoms consume already-computed labels, probabilities, scores,
  importances, or masks.
- Keep witnesses limited to shape and combinatoric behavior over abstract
  arrays.

## Required Completion Criteria Per Family

Follow `../sciona-atoms/AGENT_INGESTION.md` end to end:

- Implement typed atoms with `@register_atom`, meaningful `@icontract.require`,
  and meaningful `@icontract.ensure`.
- Add pure witnesses that mirror atom signatures.
- Add CDG JSON with concrete `inputs` and `outputs` for every atomic node.
- Add `__init__.py` files for new package levels.
- Add local provenance in `data/references/registry.json` and family
  `references.json`.
- Run contribution validation and dejargon validation.
- Add functional tests and review-bundle tests.
- Complete publishing artifacts: IO specs, parameters, descriptions, audit
  rollups, review bundles, and references.
- For limited boundaries, set semantic review verdicts to `pass_with_limits`
  and include explicit limitations.
- Commit and push only the files belonging to the completed remediation family.

## Useful Commands

```bash
git status --short
sed -n '1,180p' ../sciona-atoms/AGENT_INGESTION.md
sed -n '1,180p' REMEDIATION.md
python -m pytest -q
```

Use the matcher venv documented in `../sciona-atoms/AGENT_INGESTION.md` for
Sciona tooling and validation scripts.
