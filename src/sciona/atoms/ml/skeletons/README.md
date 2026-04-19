# ML Solution Skeletons

CDG templates for common ML solution structures. These are not executable
atoms — they are structural patterns that the architect agent can
instantiate, expand, and refine.

## How Skeletons Work

A skeleton is a CDG with **placeholder nodes** that reference atom families
or other CDGs rather than specific atoms. The architect:

1. Discovers a skeleton that matches the problem structure
2. Instantiates it by binding placeholder nodes to concrete atoms
3. Expands nodes using expansion rules (e.g., "replace single_model with
   k-fold ensemble")
4. Refines by adding domain-specific nodes (feature engineering, constraints)

## Skeleton Registry

| Skeleton | File | Covers |
|----------|------|--------|
| `tabular_pipeline` | `tabular_pipeline.cdg.json` | Feature eng → model → ensemble (60%+ of Kaggle wins) |
| `multi_instance_detection` | `multi_instance_detection.cdg.json` | Detect → classify → aggregate |
| `graph_inference_pipeline` | `graph_inference_pipeline.cdg.json` | Preprocess → pairwise score → graph construction |

## Expansion Rules

| Rule | File | Transform |
|------|------|-----------|
| `kfold_ensemble` | `expansions/kfold_ensemble.cdg.json` | Single model → k-fold cross-validated ensemble |
| `dl_backbone_substitution` | `expansions/dl_backbone_substitution.cdg.json` | Feature eng + GBDT → pretrained backbone + finetune |
| `constraint_injection` | `expansions/constraint_injection.cdg.json` | Add decorrelation/fairness constraint node |
| `stacking` | `expansions/stacking.cdg.json` | Flat ensemble → two-level stacking |

## Relationship to Atoms

Skeletons reference atom families by namespace, not by specific atom. For
example, `tabular_pipeline` has a node `feature_engineering` that can be
bound to any atom family under `sciona.atoms.ml.sklearn.preprocessing` or
to domain-specific feature atoms like `sciona.atoms.causal_inference.feature_primitives`.

The binding is the architect's job — skeletons provide structure, atoms
provide computation.
