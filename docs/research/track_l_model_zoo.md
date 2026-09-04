# Track L — the resource-constrained tabular model zoo

Thirty-one algorithms from six families, trained under one 1,000-row budget on
one CPU, evaluated on the same held-out 13,376 rows as Track K.

**Track L cannot supersede Track K, and is not trying to.** Read the evidence
class before the numbers.

## Why this track exists

Track K asked whether deep tabular models beat classical baselines and answered
no, at two training budgets, with three architectures. Its most durable finding
was not about deep learning at all: the families on this dataset are strongly
**sample-dependent**. XGBoost was the worst model in the study at 500 training
rows, 0.035 ROC-AUC behind logistic regression, and the best classical model at
40,125. Logistic regression was the reverse.

That leaves two open questions Track K could not answer with four models:

1. **Is the plateau really the feature set?** Track K inferred an information
   bottleneck from four models making the same mistakes. Four is a small sample
   for that claim. Thirty spanning linear, probabilistic, kernel, distance,
   tree, boosting and deep designs is a much stronger test — and if they all
   fail on the same patients, the inference becomes hard to avoid.
2. **What does the constrained regime actually look like?** Track K measured
   500, 1,000, 2,500 and 5,000 rows for five families. Track L measures 1,000
   rows for thirty, which is a different cut through the same space.

There is also an engineering reason. The development machine is CPU-only. A
platform that can train, evaluate, serialize and describe thirty heterogeneous
models under a fixed budget is more useful to this repository than one more
expensive run of four.

## What Track L is not

- **Not a promotion track.** No model here is a production candidate, whatever
  it scores. Every result is stamped `RESOURCE_CONSTRAINED_EXPLORATORY`.
- **Not a replacement for Track K.** For any model Track K tested, Track K's
  numbers are the stronger evidence: more training data, a real hyperparameter
  search, and a paired bootstrap with promotion gates. Track L has none of
  those.
- **Not tuned.** Configurations are frozen sensible defaults. A model that does
  badly here may simply be poorly configured for 1,000 rows, and the report
  says so rather than concluding the algorithm is weak.

## The training budget

| | |
| --- | --- |
| Training rows | 1,000, drawn from the TRAIN partition only |
| Selection | deterministic, stratified, fingerprinted, nested |
| Validation | full partition, used for thresholds only |
| Test | full 13,376 rows, read once at the end |
| Hardware | CPU only; no CUDA anywhere |
| Per-model ceiling | 300 s wall clock, then `RESOURCE_LIMIT` |
| Search | none — frozen defaults |

The subset is built by Track K's `research.track_k.subsets`, not a second
sampler, so Track L's 1,000 rows are literally a rung of the ladder Track K's
sample-efficiency study used. That makes the two directly comparable at the
size they share, and the fail-closed verification comes along for free.

Training is what is constrained. Evaluation is cheap, so every model is scored
on the whole test partition — a small training budget buys no weakening of the
uncertainty in the measurement.

## The model contract

Six families with genuinely different capabilities cannot share a benchmark
runner full of `if model_id == ...`. So each model **declares** what it can do,
and `tests/test_model_zoo_registry.py` asserts every declaration against the
built model.

The declaration that matters most is `ProbabilityBehavior`, because three very
different things get called "probability":

| Behaviour | Meaning | Examples |
| --- | --- | --- |
| `NATIVE_PROBABILISTIC` | fitted by maximum likelihood; Brier and ECE are meaningful | logistic regression, LDA, the neural models |
| `NATIVE_UNCALIBRATED` | a real distribution from a model not fitted to calibrate it | random forest vote share, boosted margin, naive Bayes |
| `REQUIRES_EXTERNAL_CALIBRATION` | only a decision function exists | both SVMs |
| `HARD_LABELS_ONLY` | no score at all; threshold-free metrics are **undefined** | nearest centroid |

Averaging those into one column headed "probability" would be the most
misleading thing the zoo could do. Instead `decision_scores()` resolves a
ranking by whichever route a model actually offers, a hard-label model's
ROC-AUC is reported as undefined rather than computed from its 0/1 output, and
every card states what its model's probabilities mean.

Preprocessing is declared the same way — `STANDARDIZED`, `ROBUST_SCALED`,
`RAW_NUMERIC` or `MODEL_NATIVE` — and fitted inside an sklearn `Pipeline` so
that fitting the pipeline fits the transformer on exactly the rows the model
saw.

## The algorithm taxonomy

| Family | Models | The question it answers |
| --- | --- | --- |
| Linear | logistic L2/L1/elastic net, SGD logistic, SGD modified Huber | Which part of logistic regression's strength is the penalty, the loss, or the boundary? |
| Probabilistic | LDA, QDA, Gaussian naive Bayes | Does a generative story, a curved boundary or dropping feature correlation change anything? |
| Kernel | linear SVM, RBF SVM | Does a margin-fitted boundary differ from a likelihood-fitted one? |
| Distance | k-NN, nearest centroid | How much can purely local structure recover — and what is the floor? |
| Tree | decision tree, random forest, extra trees | How much of a forest's advantage is variance reduction versus the split search? |
| Boosting | AdaBoost, gradient boosting, hist gradient boosting, XGBoost (+ LightGBM, CatBoost optional) | Does Track K's sample-hunger result reproduce across four implementations? |
| Deep | MLP, tabular ResNet, FT-Transformer (reused from Track K) + TabTransformer, Deep & Cross, Wide & Deep, gated residual MLP, feature-token mixer, neural additive model, feature CNN | Does any structural prior find signal the others miss? |

Track K's three networks are **imported**, not copied, so the zoo cannot drift
from the evidence Track K published.

### The most informative model in the zoo

The **neural additive model** learns an arbitrary shape function per feature
and sums them. It is structurally incapable of representing any interaction
between two features. If it matches the unconstrained networks, then
interactions are not where the remaining signal lives — a direct, falsifiable
answer to the question Track K left open. It is also fully interpretable: each
feature's contribution can be plotted.

### The experimental control

The **feature CNN** is included as an `EXPERIMENTAL_INDUCTIVE_BIAS_BASELINE`
and nothing more. A 1D convolution assumes adjacent positions are related; the
"positions" here are the ten contract features in the order the contract lists
them, which is arbitrary. Permuting the columns would change its predictions
and nothing about the problem. It is in the table as a case where the
architecture demonstrably does not match the data, and it is **not** a
production candidate.

## Failure is data

A model that cannot be installed, cannot converge, cannot serialize or exceeds
its time budget stays in the results table with an outcome and a reason:

`COMPLETED` · `FAILED` · `SKIPPED` · `RESOURCE_LIMIT` · `UNSUITABLE`

Removing a failed model would turn an honest table into a flattering one.

## Optional dependencies

LightGBM and CatBoost are registered but absent from both lockfiles. The
registry detects them at import, downgrades them to `OPTIONAL`, and the
benchmark records them as `SKIPPED` with the missing package named. Their
imports live inside their builder functions, so `import
research.model_zoo.registry` never needs them.

To include them:

```powershell
pip install lightgbm catboost
python -m research.model_zoo.run --train-rows 1000
```

The core install stays small, a fresh clone stays reliable, and
`tests/test_model_zoo_registry.py` covers the absent case.

## Guarding the harness

Thirty models sharing one harness is thirty chances for one harness bug to look
like a finding.

**The negative control** is the most important test in the track. On
`NOISE_ONLY` — labels independent of every feature — no model can beat chance
on held-out rows. A model that appears to is being scored on rows it was fitted
on, and that failure is invisible in the results table because the metrics
simply look good.

**Leakage tests** assert the properties directly: the scaler's learned
statistics match a fit on train alone and *not* a fit on everything; the deep
adapter's early-stopping split is carved from the training rows it was handed,
so the zoo's validation partition stays untouched until thresholds are chosen.

**Synthetic behaviour tests** check that models learn a trivially learnable
problem — because a harness that trains nothing would also pass the negative
control.

## Running it

```powershell
# The full zoo. Minutes, CPU only.
python -m research.model_zoo.run --train-rows 1000

# A subset, for iteration.
python -m research.model_zoo.run --train-rows 1000 --models logistic_l2 mlp xgboost

# A different budget.
python -m research.model_zoo.run --train-rows 500
```

Output lands in `research_artifacts/model_zoo/<run-id>/`, gitignored: runs are
reproducible from the committed registry and seeds rather than from committed
weights. Each run writes the manifest **last**, once every artifact it
describes exists, plus per-row test predictions so any metric can be recomputed
and the agreement analysis re-run without refitting anything.

## Adding a model

1. Write a builder in the right `research/model_zoo/families/` module.
2. `register(ModelSpec(...))` with its capabilities, preprocessing,
   probability behaviour, resource class and a one-sentence rationale.
3. Run `tests/test_model_zoo_registry.py`. It builds your model, fits it, and
   asserts every declaration against what the model actually does.

There is no second place to update. The benchmark, the capability matrix and
the cards all read the registry.

## The production boundary

Track L changed no production behaviour, and continues Track K's protections:
`model_artifacts/` and `provenance/` are hashed before and after every run and
must be byte-identical. Nothing in `research/model_zoo/` writes outside
`research_artifacts/`.

## Results

See [track_l_results.md](track_l_results.md).
