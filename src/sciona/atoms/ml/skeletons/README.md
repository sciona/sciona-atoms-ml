# ML Solution Skeletons

CDG templates for common ML solution structures. These are not executable
atoms — they are structural patterns that the architect agent can
instantiate, expand, and refine.

## Patterns

From analysis of 230+ Kaggle competition winning solutions on GitHub,
most victories cluster into 5-6 recurring structural patterns. These
reduce to 3 core skeletons and 4 expansion rules.

### Pattern 1: Feature-Engineer → Ensemble-GBDT (~60% of wins)

The dominant tabular competition pattern. Engineer features from raw data,
train one or more gradient-boosted models, blend predictions. Variations
include target encoding, adversarial validation, and pseudo-labeling.
The cleverness is in *which* features, not in the pipeline structure.

**Skeleton**: `tabular_pipeline`

### Pattern 2: Detect → Classify → Aggregate

Multi-instance learning. A detector finds candidates, a classifier scores
each one, and an aggregation layer (noisy-OR, max, mean) produces the
final prediction. The canonical example is DSB2017 lung cancer detection:
detect nodules → classify each → aggregate via noisy-OR with learned
baseline.

**Skeleton**: `multi_instance_detection`

### Pattern 3: Pairwise Score → Graph Construction

Infer a graph structure from multivariate observations. Preprocess
signals, compute pairwise scores (partial correlation, HSIC, IGCI), then
threshold or optimize to produce edges. Covers connectomics (calcium
imaging → neural circuit), causal discovery, and correlation networks.

**Skeleton**: `graph_inference_pipeline`

### Pattern 4: Pretrain → Finetune → Ensemble

Transfer learning from large pretrained backbones (ImageNet, BERT).
Finetune on competition data, ensemble multiple architectures and folds,
optionally apply test-time augmentation.

**Not a separate skeleton.** This is a **rewrite** of Pattern 1 where
`feature_engineering` is replaced by `load_pretrained` and
`model_training` is replaced by `finetune`.

**Expansion rule**: `dl_backbone_substitution` applied to `tabular_pipeline`

### Pattern 5: Constraint-Aware Training

Standard ML pipeline with an additional node that enforces decorrelation,
fairness, or physics-invariance. Predictions must pass a statistical
test (KS, CvM) against a protected variable. The Flavours of Physics
competition demonstrated that noise injection on a powerful full-feature
model beats a feature-restricted model.

**Not a separate skeleton.** This is an **insertion** into Pattern 1
where a constraint verification + decorrelation node is added between
training and prediction.

**Expansion rule**: `constraint_injection` applied to `tabular_pipeline`

### Pattern 6: Combinatorial Optimization

Heuristic search: initial solution → local search (swap, 2-opt, or-opt) →
acceptance criterion → iterate. Structurally different from all ML
patterns. Examples: Santa's Stolen Sleigh (vehicle routing), traveling
salesman variants.

**Not yet covered.** Rare enough in Kaggle that a skeleton is not
justified for the first pass. More relevant for operations research atoms.

## Skeleton Registry

| Skeleton | File | Derived Patterns |
|----------|------|-----------------|
| `tabular_pipeline` | `tabular_pipeline.cdg.json` | 1, 4 (via rewrite), 5 (via insertion) |
| `multi_instance_detection` | `multi_instance_detection.cdg.json` | 2 |
| `graph_inference_pipeline` | `graph_inference_pipeline.cdg.json` | 3 |

## Expansion Rules

| Rule | File | Type | Transform |
|------|------|------|-----------|
| `kfold_ensemble` | `expansions/kfold_ensemble.cdg.json` | replace | Single model → k-fold CV with OOF predictions |
| `dl_backbone_substitution` | `expansions/dl_backbone_substitution.cdg.json` | rewrite | Features + GBDT → pretrained backbone + finetune |
| `constraint_injection` | `expansions/constraint_injection.cdg.json` | insert | Add decorrelation/fairness constraint between training and prediction |
| `stacking` | `expansions/stacking.cdg.json` | replace | Flat ensemble → two-level stacking with meta-learner |

Expansion types:
- **replace**: swap one node for a sub-CDG
- **rewrite**: swap multiple nodes for a different structure
- **insert**: add new nodes between existing ones

Some expansions have prerequisites: `stacking` requires `kfold_ensemble`
to have been applied first (it needs OOF predictions).

## How Skeletons Work

A skeleton is a CDG with **placeholder nodes** that reference atom families
by namespace rather than specific atoms. The architect agent:

1. Discovers a skeleton that matches the problem structure
2. Instantiates it by binding placeholder nodes to concrete atoms
3. Expands nodes using expansion rules as needed
4. Refines by adding domain-specific nodes

## Relationship to Atoms

Placeholder nodes reference atom families by namespace:
- `sciona.atoms.ml.sklearn.preprocessing` for feature engineering
- `sciona.atoms.ml.sklearn.ensemble` for model training
- `sciona.atoms.causal_inference.feature_primitives` for domain features
- `sciona.atoms.medical_imaging_3d.aggregation` for noisy-OR aggregation
- `sciona.atoms.graph_inference.connectome` for pairwise scoring

The binding to concrete atoms is the architect's job. Skeletons provide
structure; atoms provide computation.

## What's Needed to Make These Executable

1. **sklearn atoms** in `sciona-atoms-ml` — the 272-target list in SKLEARN.md.
   Core pipeline atoms (train_test_split, StandardScaler, GBDT, etc.)
   are the prerequisite for the tabular skeleton.
2. **LightGBM/XGBoost/CatBoost atoms** — the actual Kaggle workhorses
   (sklearn's GBDT implementations are rarely used in winning solutions).
3. **Architect agent** must understand the skeleton/expansion format and
   be able to discover, instantiate, and compose them.
