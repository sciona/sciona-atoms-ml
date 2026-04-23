# Remediation Worker Prompt

Use this file as the default prompt and operating procedure for any worker
continuing sklearn remediation work in this repo. The goal is to stop
re-deriving the remediation rules, backlog interpretation, and packaging
requirements in every session.

This file does not replace `REMEDIATION.md`, `PUBLISHING.md`, or
`../sciona-atoms/AGENT_INGESTION.md`. It tells you how to work with them
efficiently and consistently.

## Mission

Continue the remediation backlog in `REMEDIATION.md` by publishing honest,
bounded sklearn atom families without inventing CDG structure, hiding opaque
solver boundaries, or redoing analysis that the repo already captures.

## Required Reading

Before choosing a remediation target, read:

1. `../sciona-atoms/AGENT_INGESTION.md`
2. `REMEDIATION.md`
3. `REMEDIATION_CONTINUATION.md`
4. `SKLEARN.md`
5. `PUBLISHING.md` if present in this repo or the shared atoms repo

Treat `REMEDIATION.md` Case 1-8 rules as authoritative even if an older
section still says to "decide whether" a wrapper or helper-first path is
acceptable.

## Core Working Rule

Do not trust `REMEDIATION.md` blindly as a current ledger of what has or has
not already landed. Trust the codebase first.

Before starting a new family, check whether related helper atoms, tests,
review bundles, or references already exist under:

- `src/sciona/atoms/ml/sklearn/...`
- `tests/...`
- `data/review_bundles/...`
- `data/audit_manifest.json`

If the codebase already contains a helper slice that `REMEDIATION.md` does not
acknowledge, update `REMEDIATION.md` first. Do not spend a session re-mining a
family that is already partially remediated.

## What Counts as Success

A remediation family is complete only when all of the following are done in
the same logical round:

- atoms, witnesses, CDG, and package files are added or updated
- provenance is added locally
- functional tests are added
- review-bundle coverage is added
- `data/audit_manifest.json` is updated if a new review bundle was created
- `REMEDIATION.md` is updated so the backlog reflects the new state
- contribution and dejargon validation pass
- focused tests pass

Do not leave `REMEDIATION.md` stale after a remediation commit.

## Decision Rules

Apply the top-level remediation cases directly:

### Case 1: native or compiled boundaries

- Publish public Python API surfaces only when the contracts are honest about
  the compiled boundary.
- Use `pass_with_limits`.
- State the limitation explicitly in the review bundle.

### Case 2: external optimizer boundaries

- Prefer objective, loss, gradient, or closure-helper atoms over `fit`
  wrappers.
- Treat the optimizer call as an FFI-style boundary.

### Case 3: meta-estimators

- Use explicit protocols or higher-order boundaries where needed.
- Keep helper atoms estimator-independent whenever possible.
- Witnesses should model combinatorics of supplied outputs, not estimator
  training.

### Case 4: CV and path orchestration

- Do not publish monolithic public estimators.
- Break workflows into folds, fit/score, aggregation, selection, and refit
  atoms when the boundary is explicit.

### Case 5: decomposition and factorization

- Slice horizontally.
- Publish pure Python or NumPy helpers first.
- Treat SVD, sparse-code solving, KMeans, coordinate descent, and similar
  heavy solver calls as boundaries unless explicitly decomposed.

### Case 6: clustering and spectral workflows

- Publish graph, Laplacian, embedding, normalization, and label-assignment
  helpers before estimator shells.
- Mark eigensolvers and clustering kernels as limited boundaries when exposed.

### Case 7: probabilistic and mutable-kernel workflows

- Use state-passing style.
- Publish likelihood, gradient, posterior, and kernel-state helpers before
  estimator surfaces.

### Case 8: thin wrappers

- Keep blocked until upstream atoms they route to already exist and are
  published honestly.

## Preferred Target Selection

When several backlog groups are available, prefer the next target that meets
all of these:

1. It has a real helper-first path under the case rules.
2. The helper slice can be honest without hiding training or solver behavior.
3. The family is not already partially landed in the repo unless you are
   extending that exact helper slice.
4. The scope is large enough to be meaningful, but still bounded enough to
   finish with tests, review bundle, manifest merge, and `REMEDIATION.md`
   update in one round.

Avoid defaulting to the next section in textual order if another section has a
clearer publishable helper family.

## Pre-Flight Checklist Before Any New Family

Before writing code:

1. Check `git status --short`.
2. Identify unrelated dirty files and avoid staging or reverting them.
3. Search for existing atoms in the target family.
4. Search for existing focused tests and review-bundle tests.
5. Check whether `REMEDIATION.md` already documents the landed helper slice
   accurately.
6. Check whether the family already has a review bundle and whether its rows
   are present in `data/audit_manifest.json`.

If the pre-flight check reveals that the family is already partially landed but
the backlog is stale, fix the backlog first.

## Bookkeeping Rules

These are mandatory process rules for remediation work:

1. Update `REMEDIATION.md` in the same commit as every remediation family.
2. If you add or regenerate a review bundle, merge it into
   `data/audit_manifest.json` in the same commit.
3. Do not stage unrelated user changes.
4. Do not rewrite or normalize large unrelated files just because you touched
   a nearby section.
5. Keep the commit scoped to the remediation family plus required ledger and
   manifest updates.

## How to Write `REMEDIATION.md` Updates

Every completed helper slice entry should be narrow and explicit.

Use this pattern:

- `Completed helper slice: <family> now publishes <brief scope>.`
- List exact atom names when the surface is small enough.
- State what remains deferred, especially solver calls, estimator mutation,
  CV orchestration, callbacks, or native kernels.

Good example shape:

- Completed helper slice: `sklearn.example.family` now publishes dense helper
  atoms for supplied scores and labels: `atom_a`, `atom_b`, and `atom_c`. It
  still defers estimator fitting, cross-validation orchestration, and the
  compiled solver boundary.

Avoid vague entries like "partially complete" or "added some helpers."

## Implementation Rules

Follow `../sciona-atoms/AGENT_INGESTION.md` strictly:

- `@register_atom` on public atoms
- meaningful `@icontract.require`
- meaningful `@icontract.ensure`
- typed public interfaces
- honest docstrings
- pure witnesses
- CDG nodes with concrete `inputs` and `outputs`
- package `__init__.py` files at every new level

Do not create phantom orchestration nodes. Do not invent helper atoms that the
source does not justify.

## Publishing Rules

For every remediation family:

- add or update local provenance in `data/references/registry.json`
- add family `references.json`
- add publishing artifacts required by `PUBLISHING.md`
- add review-bundle coverage
- ensure the new review bundle is reachable through `data/audit_manifest.json`

If the family is limited by a solver or compiled boundary, encode that
explicitly in semantic review output rather than hiding it in prose.

## Validation Rules

At minimum, run:

1. focused tests for the family
2. contribution validation
3. dejargon validation

Run broader tests when the blast radius justifies it.

Do not report a remediation family as complete until these pass.

## Commit and Push Rules

- Stage only the files for the remediation family plus the required
  `REMEDIATION.md`, manifest, provenance, and test updates.
- Commit with a narrow message describing the family.
- Push after validation passes.

If a round is docs-only, say that directly and do not pretend code validation
was needed.

## When to Stop and Ask the Human

Ask instead of guessing when:

- decomposition boundaries are genuinely ambiguous
- licensing or provenance is unclear
- a helper-first path would still hide arbitrary estimator behavior
- a family could be modeled either as a limited wrapper or as a solver-backed
  decomposition and the tradeoff is not obvious
- existing user changes make the target family hard to modify safely

## Suggested Worker Prompt

Use this prompt when delegating remediation work:

> Read `../sciona-atoms/AGENT_INGESTION.md`, `REMEDIATION.md`,
> `REMEDIATION_CONTINUATION.md`, `REMEDIATION_PROMPT.md`, `SKLEARN.md`, and
> `PUBLISHING.md` if present. Apply the Case 1-8 rules from `REMEDIATION.md`
> as authoritative. Before choosing a target, audit the repo for already
> landed helper slices so you do not duplicate prior remediation work. If
> `REMEDIATION.md` is stale for the target area, update it first. For any new
> remediation family, complete atoms, witnesses, CDG, tests, provenance,
> review bundle, audit-manifest merge, and a same-commit `REMEDIATION.md`
> update. Keep solver boundaries honest, avoid phantom CDG structure, do not
> stage unrelated changes, and report exactly what remains deferred.

## Useful Commands

```bash
git status --short
rg --files src/sciona/atoms/ml/sklearn
rg -n "Completed helper slice|sklearn\\." REMEDIATION.md
rg -n "register_atom\\(" src/sciona/atoms/ml/sklearn
python -m pytest -q
```
