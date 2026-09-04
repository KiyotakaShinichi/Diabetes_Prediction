# Track K results — deep learning challengers vs classical baselines

**Protocol 1.0.0.** The metric, the split, the search budgets, the calibration
procedure, the bootstrap and the promotion thresholds were committed in
`docs/research/track_k_protocol.md` and `research/track_k/protocol.py` **before**
the test partition was read. This document reports what happened.

Two arms were run, differing in training budget and in nothing else:

| Arm | Training rows | Families | Status |
| --- | --- | --- | --- |
| `full_reference` | 40,125 (all of train) | 4 | run once, preserved as historical evidence |
| `cpu_constrained` | a fixed 5,000-row subset | 5 | the working arm |

## The finding

**No deep model earned promotion in either arm.** Every challenger was rejected
against the strongest classical baseline available to it, at both training
budgets, on the same held-out 13,376 rows.

The differences between families are small enough that the ordering changes with
the training budget while the conclusion does not:

| | Best classical | Best deep | Gap | Verdict |
| --- | --- | --- | --- | --- |
| 40,125 rows | XGBoost 0.82229 | MLP 0.82308 | +0.00079 [−0.00042, +0.00198] | INCONCLUSIVE → REJECT |
| 5,000 rows | Logistic regression 0.81767 | MLP 0.82041 | +0.00272 [+0.00117, +0.00414] | CLEAR IMPROVEMENT → REJECT |

At the smaller budget the MLP's advantage is real — the interval clears zero —
and it is still rejected, because the protocol asked for **+0.005** before
accepting the cost of serving a neural network, and because the MLP gives up
3.9 points of recall to get there. A gain that is measurable is not
automatically a gain that is worth having, and the bar that says so was written
before the number existed.

### Why the numbers are this close

The learning curves answer it. Both networks in the reference arm reach within
0.002 of their final validation ROC-AUC in a **single epoch**, while training
loss barely moves (0.539 → 0.520 over twenty epochs). A third architecture was
added specifically to test whether depth was the missing ingredient — a residual
tower, the standard way to give a tabular network real depth — and it landed in
the same place as the other two.

Three architectures, two training budgets, one answer: the ceiling is in the ten
features, not in the model.

## Arm 1 — full reference (40,125 training rows)

Run once. Preserved rather than repeated: it costs about ninety minutes of
CPU, and re-running it to answer follow-up questions would burn that time
reproducing predictions already on disk.

| Metric | Logistic regression | XGBoost | MLP | FT-Transformer |
| --- | --- | --- | --- | --- |
| **ROC-AUC** (primary) | 0.81835 | 0.82229 | **0.82308** | 0.82302 |
| PR-AUC | 0.78442 | 0.79098 | 0.78924 | **0.79729** |
| Accuracy | 0.74200 | 0.74611 | **0.74723** | 0.74208 |
| Precision | 0.71522 | 0.72265 | 0.74583 | **0.74827** |
| Recall | **0.80332** | 0.79793 | 0.74929 | 0.72878 |
| Specificity | 0.68081 | 0.69440 | 0.74518 | **0.75534** |
| F1 | 0.75671 | **0.75843** | 0.74755 | 0.73840 |
| Brier | 0.17264 | 0.17106 | 0.17034 | **0.17029** |
| Log loss | 0.51884 | 0.51543 | 0.51365 | **0.51136** |
| ECE (10 bins) | 0.01117 | 0.01326 | 0.01221 | **0.00633** |
| Calibration slope | 0.99512 | 0.99016 | 0.99496 | **1.00971** |
| Threshold | 0.45690 | 0.48526 | 0.51579 | 0.55636 |

Confusion matrices at each model's validation-chosen threshold, `[[TN, FP], [FN, TP]]`:

| Model | Matrix |
| --- | --- |
| Logistic regression | `[[4558, 2137], [1314, 5367]]` |
| XGBoost | `[[4649, 2046], [1350, 5331]]` |
| MLP | `[[4989, 1706], [1675, 5006]]` |
| FT-Transformer | `[[5057, 1638], [1812, 4869]]` |

### Paired comparisons

Each replicate resamples row indices once and applies that same index set to
every model, so a delta is always measured on identical rows. 2,000 resamples,
95% percentile intervals, seed 20260828.

| Comparison | ΔROC-AUC | 95% interval | Outcome |
| --- | --- | --- | --- |
| MLP − logistic regression | +0.00469 | [+0.00301, +0.00634] | **CLEAR IMPROVEMENT** |
| FT-Transformer − logistic regression | +0.00464 | [+0.00301, +0.00636] | **CLEAR IMPROVEMENT** |
| MLP − XGBoost | +0.00079 | [−0.00042, +0.00198] | INCONCLUSIVE |
| FT-Transformer − XGBoost | +0.00073 | [−0.00042, +0.00189] | INCONCLUSIVE |
| FT-Transformer − MLP | −0.00005 | [−0.00137, +0.00129] | INCONCLUSIVE |

Pairing earns its keep here. The per-model intervals all overlap — read alone
they cannot separate any two of these models — yet a +0.0047 gain over logistic
regression is cleanly resolvable, while a +0.0008 gain over XGBoost is not
distinguishable from zero.

| Model | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- |
| Logistic regression | 0.81835 [0.81122, 0.82572] | 0.78442 [0.77375, 0.79569] | 0.17264 [0.16910, 0.17597] |
| XGBoost | 0.82229 [0.81532, 0.82940] | 0.79098 [0.78053, 0.80149] | 0.17106 [0.16765, 0.17442] |
| MLP | 0.82308 [0.81606, 0.83025] | 0.78924 [0.77878, 0.79983] | 0.17034 [0.16685, 0.17366] |
| FT-Transformer | 0.82302 [0.81589, 0.83010] | 0.79729 [0.78649, 0.80819] | 0.17029 [0.16682, 0.17369] |

No p-values are reported. A percentile bootstrap interval is not a hypothesis
test, and calling one significant would overclaim.

### Promotion verdicts — baseline XGBoost

| Gate | MLP | FT-Transformer |
| --- | --- | --- |
| ΔROC-AUC interval entirely above +0.005 | ✗ lower bound −0.00042 | ✗ lower bound −0.00042 |
| ECE no more than +0.010 worse | ✓ −0.00105 | ✓ −0.00693 |
| Recall no more than 0.020 below | ✗ **−0.04865** | ✗ **−0.06915** |
| Latency no more than 10× | ✓ 1.13× | ✓ 1.62× |
| **Verdict** | **REJECT** | **REJECT** |

Both fail on discrimination and on sensitivity. Neither fails on cost, and both
are *better* calibrated than the baseline. Had the bar been calibration alone,
the FT-Transformer would have cleared it — it was not, and the bar was set
before the numbers existed.

## Arm 2 — CPU-constrained (5,000 training rows)

Every family trains on **one** deterministic, fingerprinted, stratified
5,000-row subset drawn from the training partition alone (positive rate 0.4994,
membership hash `9cb0eb5dedf29de4…`). Validation and test stay at full size:
training is what is constrained, and scoring is cheap.

Budgets were set from measured per-epoch cost, so the FT-Transformer — about
eight times the per-epoch cost of the other networks — gets fewer trials rather
than a longer wall clock. Uneven by compute, not by favour.

| Metric | Logistic regression | XGBoost | MLP | FT-Transformer | Tabular ResNet |
| --- | --- | --- | --- | --- | --- |
| **ROC-AUC** | 0.81767 | 0.81630 | **0.82041** | 0.81868 | 0.81915 |
| PR-AUC | 0.78307 | 0.78258 | **0.78749** | 0.78684 | 0.78686 |
| Accuracy | 0.74260 | 0.74312 | 0.74335 | **0.74447** | 0.74237 |
| Precision | 0.71979 | 0.71994 | 0.73757 | 0.72770 | **0.74445** |
| Recall | 0.79359 | **0.79494** | 0.75468 | 0.78042 | 0.73731 |
| Specificity | 0.69171 | 0.69141 | 0.73204 | 0.70859 | **0.74742** |
| F1 | 0.75489 | **0.75558** | 0.74602 | 0.75314 | 0.74086 |
| Brier | 0.17304 | 0.17369 | **0.17168** | 0.17278 | 0.17238 |
| Log loss | 0.52002 | 0.52228 | **0.51659** | 0.51954 | 0.51872 |
| ECE | 0.01059 | 0.01262 | 0.01191 | **0.00794** | 0.01463 |
| Threshold | 0.46904 | 0.48364 | 0.50822 | 0.51278 | 0.52481 |

**The promotion baseline changed arms.** The policy names "the strongest
classical family under this protocol" rather than a fixed model, and at 5,000
rows that is **logistic regression** (0.81767) rather than XGBoost (0.81630).
That is the policy working as written, and it is the first hint of the
sample-efficiency result below.

| Comparison | ΔROC-AUC | 95% interval | Outcome |
| --- | --- | --- | --- |
| MLP − logistic regression | +0.00272 | [+0.00117, +0.00414] | **CLEAR IMPROVEMENT** |
| MLP − XGBoost | +0.00412 | [+0.00225, +0.00591] | **CLEAR IMPROVEMENT** |
| FT-Transformer − XGBoost | +0.00238 | [+0.00033, +0.00440] | **CLEAR IMPROVEMENT** |
| Tabular ResNet − XGBoost | +0.00288 | [+0.00103, +0.00469] | **CLEAR IMPROVEMENT** |
| FT-Transformer − logistic regression | +0.00098 | [−0.00118, +0.00306] | INCONCLUSIVE |
| Tabular ResNet − logistic regression | +0.00147 | [−0.00036, +0.00320] | INCONCLUSIVE |
| FT-Transformer − MLP | −0.00174 | [−0.00364, +0.00017] | INCONCLUSIVE |
| Tabular ResNet − ft_transformer | +0.00049 | [−0.00148, +0.00245] | INCONCLUSIVE |
| Tabular ResNet − MLP | −0.00125 | [−0.00235, −0.00021] | **CLEAR REGRESSION** |

### Promotion verdicts — baseline logistic regression

| Gate | MLP | FT-Transformer | Tabular ResNet |
| --- | --- | --- | --- |
| ΔROC-AUC above +0.005 | ✗ +0.00117 | ✗ −0.00118 | ✗ −0.00036 |
| ECE within +0.010 | ✓ +0.00132 | ✓ −0.00265 | ✓ +0.00405 |
| Recall within 0.020 | ✗ **−0.03892** | ✓ −0.01317 | ✗ **−0.05628** |
| Latency within 10× | ✓ 0.60× | ✓ 0.44× | ✓ 0.28× |
| **Verdict** | **REJECT** | **REJECT** | **REJECT** |

The FT-Transformer clears three of four gates here, failing only the one that
matters most: its discrimination advantage over logistic regression cannot be
distinguished from zero.

### The third challenger answered its question

The residual tower was added to test one hypothesis: that the MLP plateaus
because a shallow feed-forward stack is the wrong shape, not because the
features are exhausted. It has real depth (3 pre-normalised residual blocks,
200,321 parameters — the largest model in the study) and it finished at 0.81915,
*below* the 36,609-parameter MLP by a margin the bootstrap calls a **clear
regression**.

Depth was not the missing ingredient. Three architectures spanning
feed-forward, residual and attention-based designs land within 0.0017 ROC-AUC
of each other, and the best of them is 0.0027 above logistic regression.

## Sample efficiency

Which family extracts the most from the least data. Frozen conventional
configurations, no per-size search, every model scored on the full 13,376-row
test partition.

### ROC-AUC by training rows

| Family | 500 | 1,000 | 2,500 | 5,000 | 40,125 † |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | **0.80845** | **0.81542** | 0.81723 | 0.81807 | 0.81835 |
| XGBoost | 0.77362 | 0.79540 | 0.80891 | 0.81455 | **0.82229** |
| MLP | 0.80426 | 0.81517 | 0.81785 | **0.81989** | 0.82308 |
| FT-Transformer | 0.79216 | 0.80188 | 0.80710 | 0.81479 | 0.82302 |
| Tabular ResNet | 0.80333 | 0.81460 | **0.81859** | 0.81905 | — |

† the reference arm, with per-family search rather than a frozen configuration.
Shown for orientation; it is not a fifth point on the same curve.

### Mean ROC-AUC gain per doubling of training rows

| Family | Gain |
| --- | --- |
| XGBoost | **+0.01232** |
| FT-Transformer | +0.00681 |
| Tabular ResNet | +0.00473 |
| MLP | +0.00471 |
| Logistic regression | +0.00290 |

**The clearest result in the study, and it is not about deep learning.**
XGBoost is by far the most data-hungry family here: it is the *worst* model at
500 rows by a wide margin (0.77362, some 0.035 behind logistic regression), and
it is the only family still climbing steeply at 5,000. Give it the full 40,125
rows and it becomes the strongest classical model. Logistic regression is the
mirror image — the best model at 500 rows, and the first to flatten.

The deep models sit between the two, closer to logistic regression's profile
than to XGBoost's, and never separate meaningfully from the best classical model
at any budget on this curve.

For a practitioner the practical reading is: **on this problem, the choice of
family matters most when data is scarce, and matters least when it is
plentiful.** At 500 rows the spread between best and worst is 0.035 ROC-AUC. At
5,000 it is 0.005.

## Cost

Recorded as first-class evidence, because a metric gain has to be weighed
against what it costs to obtain and to serve.

| Family (constrained arm) | Search + fit | Parameters | Artifact | Median single-row |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.9 s | — | 2.0 KB | 7.58 ms |
| XGBoost | 14.5 s | 383 trees, depth 4 | 629 KB | 10.45 ms |
| Tabular ResNet | 49.3 s | 200,321 | 799 KB | 2.14 ms |
| MLP | 82.2 s | 36,609 | 152 KB | 4.52 ms |
| FT-Transformer | **485.2 s** | 38,017 | 157 KB | 3.37 ms |

The FT-Transformer costs roughly 540× the wall clock of the strongest model in
its own arm — logistic regression, which beat it — and about six times the MLP,
which beat it too. Measured per-epoch cost on the 5,000-row subset was 25.0 s
against 3.3 s for the MLP and 2.6 s for the residual tower.

**These timings are comparative, not absolute.** One machine, one process, wall
clock, no isolation from other load — and the constrained arm's *latency* column
in particular was measured while other work was running on the same machine, so
its absolute milliseconds are not comparable to the reference arm's. The
within-arm ratios, which is what the promotion policy reads, are unaffected.
The `train_seconds` figures in the sample-efficiency study carry one visible
artifact of the same kind: the MLP's 52.1 s at 500 rows against 5.4 s at 1,000
is process warm-up on the first network trained, not a property of the model.

## Error analysis (reference arm)

Restricted to what the dataset contains. The served contract holds no protected
attribute, so **no demographic fairness analysis is reported** rather than one
being invented for attributes that are not there.

| Model | False positives | False negatives | Confidently wrong | Uncertain band (0.4–0.6) |
| --- | --- | --- | --- | --- |
| Logistic regression | 2,137 | 1,314 | 93 | 2,546 rows, 53.6% correct |
| XGBoost | 2,046 | 1,350 | 121 | 2,062 rows, 53.9% correct |
| MLP | 1,706 | 1,675 | 175 | 2,311 rows, 54.0% correct |
| FT-Transformer | 1,638 | 1,812 | 119 | 2,476 rows, 52.1% correct |

**The errors are the same errors.** Across every model, false positives look
alike — general health around 3.2, BMI around 30.5, age band 9.7, high blood
pressure in 77–83% of cases — and so do false negatives: general health 2.4, BMI
27.4, age band 8.5, high blood pressure in 34–44%. Every model is confounded by
the same people: those carrying the classic risk markers without the diagnosis,
and those with the diagnosis but not the markers. Changing architecture does not
dissolve that overlap. This is the strongest evidence in the study that the
limit is the feature set.

**Between 15% and 19% of the test set sits in the 0.4–0.6 band**, where every
model is barely better than a coin flip. That is a product finding as much as a
modelling one: for roughly one visitor in six, the honest answer is that these
ten inputs do not determine an answer.

## Calibration

Candidates were scored **out of fold** inside validation, then the winner was
refitted on all of validation and applied to test. Judging a candidate on the
rows it was fitted on is not a fair comparison between a two-parameter sigmoid
and a non-parametric isotonic fit — measured during development, in-sample
isotonic scored an ECE of 1.6e-17 purely by memorising the rows it was graded on.

In the reference arm, isotonic won for three families; for the FT-Transformer no
candidate improved on the raw output, so **none was applied**, and it still
finished with the best test ECE of the four (0.00633). A transformer already
well calibrated is a real observation; the protocol declining to apply a
transform that would have made it worse is the machinery working, not an
omission.

## Two comparisons this study deliberately does not make

**Against the deployed production model.** The models in `model_artifacts/`
predate this repository's provenance system; their attestation records training
lineage as explicitly unknown — no run record, no dataset fingerprint, no seed.
Their published metric came from an unrecorded split by an unrecorded procedure.
Putting it in the tables above would compare things that differ in more ways
than the architecture, and the resemblance of the numbers would make that
comparison look sound when it is not. Track K therefore trained fresh research
instances of **all** families, classical ones included, under one protocol.

**Against any population prevalence.** See below.

## Limitations

**The base rate is engineered, and it is the most important caveat here.**

*Directly measured from the committed file:* 66,877 rows, 33,474 negative and
33,403 positive — a 49.95% positive rate, a class ratio of about 1.002:1, no
missing values, no duplicate rows.

*Inferred, and labelled as inference:* a near-exact 50/50 split is not what an
unselected screening population produces, so the file is consistent with a
deliberately balanced or resampled construction. The repository contains no
dataset card, provenance record or source citation for `cleaned_data.csv`, so
its actual construction is **not established here**. No external prevalence
figure is stated in this document, because none is evidenced in this repository.

*Consequence, which holds regardless of how the file was built:* every
probability in this study is conditional on a study base rate near 50%. A model
reporting "62%" means 62% under this dataset's prior, not a 62% chance of
diabetes in a general population. The calibration results measure internal
consistency on this dataset — a model perfectly calibrated here could still be
badly calibrated against a differently distributed population. Track K records
this and does **not** attempt to correct probabilities toward a population
prevalence, which would require a prespecified recalibration study against a
sourced base rate.

**Ten features.** Eleven further columns exist in the file (`Income`,
`CholCheck`, `Smoker`, `Stroke`, `Fruits`, `Veggies`, `HvyAlcoholConsump`,
`AnyHealthcare`, `NoDocbcCost`, `MentHlth`, `Sex`) and are excluded to match the
served contract. Whether they would lift the ceiling is a real question this
benchmark does not answer, and the error analysis suggests it is the most
promising one.

**One dataset, one split, one test set.** No cross-validation across splits and
no external validation. The intervals describe sampling variability within this
test partition, not generalisation to another population. A single split also
means the ~0.0008 reference-arm gap between XGBoost and the networks could
plausibly reverse on another one — that is what "inconclusive" means.

**Modest search budgets, and unequal ones in the constrained arm.** 20/30/20/15
trials in the reference arm; 8/8/6/4/6 in the constrained arm, uneven because
per-epoch cost is uneven. A far larger deep-learning search might find something
better. The flat learning curves argue it would move little, but this study
cannot rule it out and does not claim to.

**One reference run.** The 40,125-row arm was executed once and preserved rather
than repeated. Its numbers therefore carry no run-to-run variance estimate, only
the within-test-set bootstrap intervals.

**Single hardware profile**, with the timing caveats noted above.

## What would change this answer

Stated in advance of anyone asking, because a negative result should say what
would overturn it:

1. **More features.** The eleven excluded columns are the most likely source of
   headroom. Nothing here suggests architecture is the constraint; the identical
   error profiles suggest information is.
2. **A dataset with a real base rate and recorded provenance**, which would make
   the calibration results transferable instead of internal.
3. **Interaction-rich or higher-cardinality data.** FT-Transformers earn their
   keep where feature interactions are complex; ten mostly-ordinal columns on
   which logistic regression lands within 0.005 of everything is not that
   setting.
4. **A materially larger search budget**, if someone wants to rule out the
   possibility this study cannot.

## Reproducing this

```powershell
# Reference arm: every family on the full training partition. ~90 min, CPU.
python -m research.track_k.benchmark

# Constrained arm: one fixed 5,000-row training subset.
python -m research.track_k.benchmark --profile cpu_constrained

# Sample efficiency across the nested subset ladder.
python -m research.track_k.sample_efficiency

# The pipeline in seconds, tiny configurations. NOT research results.
python -m research.track_k.benchmark --smoke
```

Output lands in `research_artifacts/track_k/<run-id>/`, which is gitignored:
runs are reproducible from the committed protocol and seeds rather than from
committed weights. Each run writes

| File | Contents |
| --- | --- |
| `run_manifest.json` | protocol version, training profile, dataset and split fingerprints, per-model seeds and configurations, search records, calibration decisions, comparisons, promotion verdicts, resource telemetry, environment, git state, source hashes, artifact hashes |
| `split_manifest.json` | the frozen split's dataset hash and per-partition index hashes |
| `subset_manifest.json` | the training subset ladder, with membership hashes (constrained arm) |
| `<family>_metrics.json` | test and validation metrics, reliability bins, bootstrap intervals, error analysis |
| `<family>_test_proba.npy` | per-row test predictions, so any metric can be recomputed |
| `<family>_model.joblib` / `<family>_checkpoint.pt` | the trained research instance |
| `<family>_learning_curve.json` | per-epoch train loss, validation loss and validation ROC-AUC |

Seeds derive from the split seed 42: logistic regression 1042, XGBoost 2042,
MLP 3042, FT-Transformer 4042, tabular ResNet 5042. The subset seed is 20260829,
deliberately distinct so that changing which rows a subset holds cannot be
mistaken for changing the split.

### Correcting a finished run without retraining

`python -m research.track_k.recompute <run-dir> --reason "..."` re-derives
metrics, intervals and comparisons from a run's saved per-row predictions. It
never retrains, never rewrites the original manifest, and fails closed if an
artifact no longer matches its hash or the frozen split has moved.

This was used once. After the reference run completed, the bootstrap's average
precision was found to mishandle tied scores — isotonic calibration is a step
function, so it maps 13,376 predictions onto roughly a hundred distinct values
and almost every row ties with hundreds of others. Recomputation moved the
PR-AUC points by −0.00692, −0.00694 and −0.00877 for the three isotonic-calibrated
models and by −0.00001 for the uncalibrated FT-Transformer, and left **every
other statistic identical** — every ROC-AUC, recall, ECE, Brier and all five
paired comparisons. That is both a bound on the defect's scope and a proof that
the pipeline is deterministic.

### Provenance note

The constrained-arm manifest records `git.dirty: true`: documentation was being
edited while it ran. Every file under `research/track_k/` that the run executed
was verified to match commit `9f86d3d` byte for byte via the manifest's recorded
source hashes, so the dirty flag reflects uncommitted prose, not uncommitted
research code.

## The production boundary

Track K changed no production behaviour. `model_artifacts/`,
`provenance/legacy_artifact_attestation.json`, the committed metrics, the drift
baselines and the SHAP artifacts are byte-identical to what they were before
this work, and `/predict` serves exactly what it served before. That would have
been true had a challenger won: Track K decides whether a model has earned
promotion **consideration**, and wiring one into serving would be a separate
track with its own review.

The boundary is enforced, not merely asserted:

- every Track K module is parsed and checked to contain no evaluated string
  literal naming `model_artifacts`, the legacy attestation or `provenance/`;
- the deployed artifacts are hashed before the benchmark tests run and compared
  bit for bit afterwards;
- every manifest records `production_artifacts_touched: false`, and
  `verify_run_manifest` rejects a manifest claiming otherwise.
