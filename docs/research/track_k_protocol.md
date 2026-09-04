# Track K benchmark protocol

**Protocol version 1.0.0. Frozen before any model was evaluated on the test set.**

This document and `research/track_k/protocol.py` are the two halves of one
contract — prose here, machine-readable constants there, with a test asserting
they agree. Both were committed before the final benchmark ran, so the metric,
the split and the promotion thresholds could not be chosen to fit a result.

## Research question

Do modern deep-learning tabular models — an MLP and an FT-Transformer — materially
improve diabetes-risk prediction on this dataset over strong classical baselines
(logistic regression, gradient-boosted trees) trained under identical conditions?

The question is genuinely open. A negative result is a result: if the classical
baselines win, that is the finding, and it will be published here unchanged.

## Why fresh baselines rather than the deployed artifacts

The models in `model_artifacts/` predate this project's provenance system. Their
`provenance/legacy_artifact_attestation.json` records their training lineage as
explicitly unknown — no run record, no dataset fingerprint, no seed. Benchmarking
a newly trained network against them would compare two things that differ in
more ways than the architecture.

Track K therefore trains **fresh research instances of all four families** under
one protocol: same dataset, same feature contract, same split, same evaluation,
recorded seeds, known lineage. The deployed artifacts are never read for
comparison and never overwritten.

## Dataset

| Property | Value |
| --- | --- |
| File | `cleaned_data.csv` (committed) |
| Rows | 66,877 |
| Columns | 22 (10 served features + target + 11 unused) |
| Target | `Diabetes_binary` |
| Class balance | 33,474 negative / 33,403 positive — **49.95% positive** |
| Missing values | 0 |
| Duplicate rows | 0 |

### The class balance is a first-class limitation

**Directly measured from the committed file:** 66,877 rows, 33,474 negative and
33,403 positive, a 49.95% positive rate, a class ratio of approximately 1.002:1,
no missing values and no duplicate rows.

**Inferred, and labelled as inference:** a near-exact 50/50 split is not what an
unselected screening population produces, so this file is consistent with a
deliberately balanced or resampled construction rather than a natural sample.
The repository contains no dataset card, provenance record or source citation
for `cleaned_data.csv`, so its exact construction is **not established here**.
This document deliberately states no external prevalence figure, because none is
evidenced inside this repository; establishing one would require introducing and
citing a specific authoritative source, which Track K does not do.

**Consequence, which holds regardless of how the file was built.** Every
probability in this study is conditional on the study base rate of roughly 50%,
not on any population prevalence. A model reporting "62%" here means 62% under
this dataset's prior. Two things follow:

1. **Raw probabilities must not be read as population disease probabilities.**
2. **Calibration metrics measure internal consistency on this dataset**, not
   real-world correctness. A perfectly calibrated model here could still be
   badly calibrated against a differently distributed population.

This applies to the classical and the deep models equally, and to the deployed
production model, which was trained on the same file. Track K records the
limitation and does NOT attempt to correct probabilities toward a population
prevalence: that would need a prespecified recalibration study with a sourced
base rate, which is out of scope here.

### Features

The ten features of the canonical served contract, reused verbatim from
`ml_core.feature_contract` so results transfer to the deployed problem:

| Feature | Kind | Range |
| --- | --- | --- |
| `GenHlth` | ordinal | 1–5 |
| `HighBP` | binary | 0–1 |
| `BMI` | continuous | 10–80 (observed 12–45) |
| `HighChol` | binary | 0–1 |
| `Age` | ordinal | 1–13 (age bands) |
| `DiffWalk` | binary | 0–1 |
| `HeartDiseaseorAttack` | binary | 0–1 |
| `PhysHlth` | ordinal | 0–30 (days) |
| `Education` | ordinal | 1–6 |
| `PhysActivity` | binary | 0–1 |

Eleven further columns exist in the file (`Income`, `CholCheck`, `Smoker`,
`Stroke`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`,
`NoDocbcCost`, `MentHlth`, `Sex`) and are **excluded**, because the served
contract excludes them. Whether they would improve accuracy is a separate
question this benchmark does not answer.

## Split

Derived once by `research/track_k/split.py` and shared by all four families.

| Partition | Rows | Positive rate |
| --- | --- | --- |
| Train | 40,125 | 0.4995 |
| Validation | 13,376 | 0.4995 |
| Test | 13,376 | 0.4995 |

Stratified on the target, seed **42**, 20% test then 25% of the remainder as
validation — the same convention the production pipelines use. Row membership is
fingerprinted by SHA-256 of the index sets; `load_frozen_split` refuses to
proceed if the dataset bytes or the derived membership differ from a recorded
run.

## Training profiles

**Amendment, recorded after the reference run and before the constrained one.**
The development machine is CPU-only. A full-partition run costs about ninety
minutes, which is not a sensible unit of iteration, so Track K has two arms.

| | `full_reference` | `cpu_constrained` |
| --- | --- | --- |
| Training rows | all 40,125 | a fixed 5,000-row subset |
| Trials — LR / XGB | 20 / 30 | 8 / 8 |
| Trials — MLP / FT / ResNet | 20 / 15 / 15 | 6 / 4 / 6 |
| Search epoch cap | 30 | 10 |
| Final epoch cap | 80 | 30 |

**A profile fixes the training budget and nothing else.** The split, the
validation partition, the single read of the test partition, the metrics, the
bootstrap and the promotion policy are identical in both arms — which is exactly
what makes them comparable on training budget alone. `PROTOCOL_VERSION` is
therefore **not** bumped: no frozen decision above changed, and a run records
which arm produced it.

### Budgets came from measurement

Measured on the 5,000-row subset, one CPU, before the budgets were chosen:

| Family | Cost |
| --- | --- |
| Logistic regression | 0.1 s per fit |
| XGBoost | 2.2 s per fit |
| Tabular ResNet | 2.6 s per epoch |
| MLP | 3.3 s per epoch |
| FT-Transformer | **25.0 s per epoch** |

The FT-Transformer is roughly eight times the per-epoch cost of the other two
networks. It therefore receives fewer trials rather than a longer wall clock:
the trial counts are uneven **by compute, not by favour**, and every family
still trains on the same rows, selects on the same validation partition and is
judged on the same test partition.

Budgets were fixed before any constrained-arm test evaluation and are not
raised because a model is losing.

### Training is constrained; evaluation is not

Training is expensive and scoring is cheap, so a model fitted on 5,000 rows is
still evaluated on all 13,376 test rows. A smaller training budget buys no
weakening of the uncertainty estimates.

## The training subsets

One subset, drawn once, reused by every model and every re-run. Four properties,
each load-bearing:

- **Drawn only from train.** Validation still selects and test is still read
  once. A subset borrowing from either would invalidate the study.
- **Nested.** 500 ⊂ 1,000 ⊂ 2,500 ⊂ 5,000, so a change between two points is
  caused by having more data rather than different data.
- **Stratified.** Each subset preserves the train partition's positive rate;
  500 rows drawn without stratification can drift several points by chance, and
  a drifted base rate would move calibration for reasons unrelated to size.
- **Fingerprinted and fail-closed.** Membership, parent partition and dataset
  hashes are recorded, and every entry point refuses to proceed if they no
  longer match.

Seed 20260829, deliberately distinct from the split seed so that changing which
rows a subset holds can never be mistaken for changing the split.

## The third challenger

Both original networks reach within 0.002 of their final validation ROC-AUC in a
single epoch. That has two possible explanations — the features are exhausted,
or a shallow feed-forward stack is the wrong shape for what remains — and a
**residual tower** (`tabular_resnet`) is the standard way to give a tabular
network real depth without the optimisation problems depth usually brings. It is
included because it discriminates between those two explanations, not to raise
the model count. It differs from the MLP by having skip connections and
pre-normalised blocks, and from the FT-Transformer by having no attention and no
per-feature tokens.

Comparisons are derived from the family roster rather than listed, so adding a
challenger cannot silently omit its comparisons.

## Sample-efficiency study

Which family extracts the most from the least data. Every family trains a
**frozen, conventional configuration** — no per-size search, which would confound
"more data" with "a differently shaped model" — at each rung of the nested
ladder, and every model is scored on the full test partition.

No calibrator is fitted in this arm. Calibrators are themselves data-hungry, and
fitting one per size would mix "the model learned more" with "the calibrator had
more to work with", so the curve is reported on raw probabilities and the
threshold-free metrics are the ones to read.

## Recomputation without retraining

Every run persists per-row test predictions, so a corrected metric can be
re-derived from a finished run rather than by retraining it. A recomputation
never retrains, never rewrites the original manifest, and fails closed if a
saved artifact no longer matches its hash or the frozen split has moved. It
records the run it came from and the reason it was made.

This is a correctness tool, not a second look at the test set: it cannot change
a model, a hyperparameter, a threshold or a calibrator.

## Anti-overfitting rule

**The test set is evaluated once, after this protocol is committed.**

All model selection — architecture choices, hyperparameter search, early
stopping, calibration fitting — uses train and validation only. No result below
was produced by inspecting test metrics and revising a model.

## Metrics

### Primary: ROC-AUC

Chosen deliberately, not by default. PR-AUC is the right primary metric when
positives are rare, because precision-recall stays informative where ROC's
false-positive axis is swamped by a large negative class. **This dataset is
balanced**, so that advantage does not apply: the PR baseline is 0.5, and PR-AUC
carries no information ROC-AUC lacks here while being harder to compare against
the figures this repository already publishes.

ROC-AUC is therefore primary as a threshold-free measure of ranking quality.
PR-AUC is computed and reported as a secondary metric so the decision remains
auditable.

Discrimination alone is not the whole story: this product shows a visitor a
percentage, so the promotion policy adds calibration and recall guardrails.

### Secondary

PR-AUC, accuracy, balanced accuracy, precision, recall (sensitivity),
specificity, F1, Brier score, log loss, confusion matrix.

### Calibration

Expected calibration error over 10 equal-width bins, plus calibration slope and
intercept from a logistic fit of outcomes on the logit of predicted probability.
Reliability-bin data is persisted so the curves can be re-plotted.

## Uncertainty and comparison

Every model predicts the **same** test rows, so comparisons are paired. One
bootstrap replicate draws a single resample of row indices and applies it
identically to every model, then metric deltas are computed within that
replicate. Pairing removes variance that comes purely from which rows landed in
the test split — the variance an unpaired comparison would mistake for a
difference between models.

2,000 resamples, 95% percentile intervals, seed 20260828.

A comparison is reported as one of:

- **CLEAR IMPROVEMENT** — the paired interval for (challenger − baseline) lies
  entirely above 0
- **CLEAR REGRESSION** — entirely below 0
- **INCONCLUSIVE** — the interval contains 0

No p-values are reported. A percentile bootstrap interval is not a hypothesis
test, and describing one as significant would overclaim.

## Promotion policy

Frozen here before any test result existed. A challenger must clear **every**
gate against the strongest classical baseline under this protocol:

| Gate | Threshold | Why |
| --- | --- | --- |
| Primary metric | paired 95% CI for ΔROC-AUC entirely above **+0.005** | An interval above 0 alone would promote an arbitrarily small gain; 0.005 is the smallest gap worth serving a neural network for |
| Calibration | ECE no more than **+0.01** worse | The product shows a probability; better ranking with worse numbers is not an improvement |
| Recall | no more than **0.02** below baseline at the operating threshold | Screening context — sensitivity may not be traded away for discrimination |
| Latency | no more than **10×** baseline median single-row CPU latency | A small gain does not justify unbounded serving cost |

Verdicts: **PROMOTE**, **REJECT**, **INCONCLUSIVE**.

**Promotion means "earned consideration", not deployment.** Track K does not
change production serving. Wiring any challenger into `/predict` would be a
separate track with its own review.

## Reproducibility

Seeds derive from the split seed (42): logistic regression 1042, XGBoost 2042,
MLP 3042, FT-Transformer 4042. Each artifact records the dataset fingerprint,
feature-contract hash, split fingerprint, seed, hyperparameters, library
versions, git SHA and checkpoint hash.

## Limitations

- **Base-rate distortion.** Probabilities are conditional on an engineered 50%
  prevalence (see above). This is the most significant limitation of the study.
- **Ten features.** Eleven available columns are excluded to match the served
  contract, so this is not the best achievable model on this file.
- **One dataset, one split.** No cross-validation across multiple splits and no
  external validation set, so the intervals describe sampling variability within
  this test set, not generalisation to another population.
- **Ordinal encoding.** `GenHlth`, `Age`, `PhysHlth` and `Education` are ordinal
  codes. The treatment each model gives them is documented per model rather than
  assumed equivalent.
- **Single hardware profile.** Latency figures come from one CPU machine and are
  comparative, not absolute.
