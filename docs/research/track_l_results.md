# Track L results — thirty-one algorithms at 1,000 training rows

**`RESOURCE_CONSTRAINED_EXPLORATORY`.** One thousand training rows is 2.5% of
Track K's reference arm, configurations are frozen defaults rather than tuned,
and no model here is a promotion candidate. For any model Track K tested,
Track K's numbers remain the stronger evidence. See
[track_l_model_zoo.md](track_l_model_zoo.md) for the protocol.

Run `zoo-1000-3ae6a7d2a5` · 29 completed, 2 skipped, 0 failed · 13,376 test rows.

## The finding

**The five best models in a thirty-one-model zoo all have a linear decision
boundary.** The best deep architecture finishes below the worst of them.

| Rank | Model | Family | ROC-AUC |
| --- | --- | --- | --- |
| 1 | Logistic regression (L1) | linear | **0.81558** |
| 2 | Logistic regression (elastic net) | linear | 0.81550 |
| 3 | Logistic regression (L2) | linear | 0.81542 |
| 4 | Linear SVM | kernel | 0.81505 |
| 5 | Linear discriminant analysis | probabilistic | 0.81495 |
| 6 | Gated residual MLP | deep | 0.81259 |
| 7 | Deep & Cross Network | deep | 0.81196 |
| 8 | Tabular ResNet | deep | 0.81184 |

Five different fitting procedures — a lasso, an elastic net, a ridge, a hinge
margin and a generative Gaussian model — arrive within **0.0006 ROC-AUC** of
each other. That is not five models agreeing; that is one decision boundary
found five times.

## Three results worth more than the ranking

### 1. A model that cannot represent interactions keeps up

The **neural additive model** learns an arbitrary shape function per feature and
sums them. It is structurally incapable of representing any interaction between
two features. It scored **0.80887** — within 0.004 of the best deep architecture
and 0.007 of the best model in the zoo, from 1,771 parameters.

Track K left open whether its plateau was caused by an MLP failing to *find*
interactions or by interactions not being there. Two models here answer it from
opposite directions: the Deep & Cross Network computes bounded-degree
interactions explicitly and gains almost nothing (0.81196), while the additive
model forbids them entirely and loses almost nothing. **On these ten features,
feature interactions are not where the remaining signal lives.**

### 2. Track K's sample-efficiency result reproduces under a different harness

XGBoost placed **eighth from bottom at 0.78534** — 0.030 behind logistic
regression. Track K measured it 0.035 behind at 500 rows, best classical at
40,125. Track L is a different runner, different configuration, different
comparison set, and it reproduces the same relationship.

The mini study makes the mechanism visible:

| Model | 250 rows | 500 | 1,000 | ROC-AUC gain per doubling |
| --- | --- | --- | --- | --- |
| FT-Transformer | 0.76565 | 0.77539 | 0.79792 | **+0.01614** |
| XGBoost | 0.75642 | 0.76277 | 0.78534 | **+0.01446** |
| Logistic regression | 0.79218 | 0.80843 | 0.81542 | +0.01162 |
| Random forest | 0.79700 | 0.80219 | 0.80949 | +0.00624 |
| MLP | 0.79423 | 0.79734 | 0.80230 | +0.00403 |

XGBoost is worst at every budget *and* climbing fastest. It is not a weak
algorithm here; it is an algorithm being asked to work with a twentieth of the
data it needs. Reading its Track L position as "boosting is bad" would be
exactly the error the evidence class exists to prevent.

### 3. Twenty-nine algorithms make substantially the same mistakes

| Measurement | Value |
| --- | --- |
| Mean pairwise error Jaccard (406 pairs) | **0.6336** |
| Rows all 29 models got wrong | 448 |
| Rows all 29 models got right | 3,106 |
| Test set wrong by at least half the zoo | **25.74%** |
| Mean score rank correlation (27 models) † | **0.9008** |
| Within-family disagreement | 0.1244 |
| Between-family disagreement | 0.1412 |

† excluding `sgd_modified_huber`, which did not converge — see below.

Between-family disagreement exceeds within-family by **+0.0169**. A
nearest-centroid classifier, an RBF kernel, a boosted ensemble and a
transformer are barely more different from each other than two logistic
regressions are. Logistic L2 and elastic net rank the test set **identically**
(correlation 1.0000).

This is Track K's information-bottleneck hypothesis tested at seven times the
model count, and it survives: **thirty models are not thirty independent pieces
of evidence about this problem.** The binding constraint is the ten-feature
information set.

## Full results

| Model | Family | ROC-AUC | PR-AUC | Recall | Brier | ECE | Fit s | Params | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_l1 | linear | 0.81558 | 0.78504 | 0.8183 | 0.17432 | 0.02218 | 0.0 | — | completed |
| logistic_elasticnet | linear | 0.81550 | 0.78494 | 0.7848 | 0.17436 | 0.02289 | 0.0 | — | completed |
| logistic_l2 | linear | 0.81542 | 0.78483 | 0.7866 | 0.17440 | 0.02239 | 0.0 | — | completed |
| linear_svm | kernel | 0.81505 | 0.78442 | 0.7972 | 0.19581 | 0.13070 | 0.0 | — | completed |
| lda | probabilistic | 0.81495 | 0.78451 | 0.7910 | 0.17459 | 0.02183 | 0.0 | — | completed |
| gated_residual_mlp | deep | 0.81259 | 0.78137 | 0.7819 | 0.17531 | **0.01024** | 2.0 | 26,241 | completed |
| deep_cross | deep | 0.81196 | 0.78275 | 0.7568 | 0.17589 | 0.01702 | 2.1 | 5,159 | completed |
| tabular_resnet | deep | 0.81184 | 0.78450 | 0.7354 | 0.17645 | 0.02336 | 1.4 | 51,009 | completed |
| random_forest | tree | 0.80949 | 0.77591 | 0.7589 | 0.17722 | 0.02471 | 1.8 | — | completed |
| neural_additive | deep | 0.80887 | 0.77666 | 0.7780 | 0.17798 | 0.03052 | 16.0 | 1,771 | completed |
| wide_and_deep | deep | 0.80832 | 0.77470 | 0.7455 | 0.17699 | 0.01768 | 1.7 | 2,828 | completed |
| extra_trees | tree | 0.80585 | 0.77286 | 0.7999 | 0.17967 | 0.03136 | 0.8 | — | completed |
| adaboost | boosting | 0.80480 | 0.77604 | 0.7792 | 0.20521 | 0.14249 | 2.1 | — | completed |
| rbf_svm | kernel | 0.80399 | 0.76119 | 0.7882 | 0.18201 | 0.04941 | 0.1 | — | completed |
| mlp | deep | 0.80230 | 0.77009 | 0.7822 | 0.18244 | 0.04653 | 5.3 | 10,113 | completed |
| gradient_boosting | boosting | 0.79907 | 0.76516 | 0.7551 | 0.18242 | 0.02747 | 1.5 | — | completed |
| feature_cnn | deep | 0.79821 | 0.76702 | 0.7610 | 0.18462 | 0.04920 | 7.3 | 865 | completed |
| ft_transformer | deep | 0.79792 | 0.76314 | 0.7839 | 0.18386 | 0.03515 | 4.0 | 5,585 | completed |
| qda | probabilistic | 0.79780 | 0.75761 | 0.7851 | 0.19611 | 0.11588 | 0.0 | — | completed |
| feature_token_mixer | deep | 0.79602 | 0.75871 | 0.7155 | 0.18287 | 0.02096 | 4.2 | 3,501 | completed |
| gaussian_nb | probabilistic | 0.79474 | 0.75893 | 0.7872 | 0.22306 | 0.18495 | 0.0 | — | completed |
| hist_gradient_boosting | boosting | 0.79458 | 0.76134 | 0.7876 | 0.18420 | 0.02420 | 11.0 | — | completed |
| decision_tree | tree | 0.79033 | 0.75697 | 0.6797 | 0.18655 | 0.03570 | 0.0 | — | completed |
| knn | distance | 0.78903 | 0.73725 | 0.7435 | 0.18705 | 0.02592 | 0.0 | — | completed |
| tab_transformer | deep | 0.78596 | 0.75051 | 0.7913 | 0.18816 | 0.02818 | 2.8 | 14,899 | completed |
| xgboost | boosting | 0.78534 | 0.75365 | 0.7673 | 0.19138 | 0.05418 | 0.4 | — | completed |
| sgd_logistic | linear | 0.78102 | 0.74761 | 0.7812 | 0.26254 | 0.24241 | 0.0 | — | completed |
| **sgd_modified_huber** | linear | **0.51015** | 0.50602 | 0.0659 | 0.47967 | 0.47244 | 0.1 | — | completed |
| nearest_centroid | distance | **undefined** | undefined | 0.7376 | undefined | undefined | 0.0 | — | completed |
| lightgbm | boosting | — | — | — | — | — | — | — | **skipped** |
| catboost | boosting | — | — | — | — | — | — | — | **skipped** |

### Best in each family

| Family | Model | ROC-AUC |
| --- | --- | --- |
| Linear | logistic_l1 | 0.81558 |
| Kernel | linear_svm | 0.81505 |
| Probabilistic | lda | 0.81495 |
| Deep | gated_residual_mlp | 0.81259 |
| Tree | random_forest | 0.80949 |
| Boosting | adaboost | 0.80480 |
| Distance | knn | 0.78903 |

## Failures and undefined results

**`sgd_modified_huber` — 0.51015, essentially chance.** A real failure, kept in
the table. Diagnosis: early stopping halted it after **27 of 2,000 iterations**,
and it predicted only 796 positives out of 13,376 rows on a balanced test set.
The modified-Huber loss with `early_stopping=True` and `n_iter_no_change=10`
stopped on a small internal validation slice before the model converged.

A different early-stopping configuration would very likely fix it. Changing it
now, having seen the test result, is exactly what the protocol forbids — so the
configuration stays frozen and the failure is reported. What this row honestly
supports is "this configuration fails at this budget", not "modified-Huber SGD
is a poor loss".

**`nearest_centroid` — threshold-free metrics undefined.** It exposes no
ranking score of any kind. Its ROC-AUC is recorded as `None` rather than
computed from its 0/1 output, which would have produced a number that looks
like a measurement. Its recall of 0.7376 is real and is reported.

**`lightgbm`, `catboost` — skipped.** Registered but absent from both
lockfiles; the registry detected them, downgraded them to `OPTIONAL`, and the
run recorded them with the missing package named. `pip install lightgbm
catboost` includes them.

**No model hit the 300 s resource limit.** The whole zoo fits in **64.8 seconds**
of total training time; the slowest single model is the neural additive at 16.0 s.

## Calibration

The declarations made in the registry before the run predicted the outcome.

| Model | ECE | What its spec said in advance |
| --- | --- | --- |
| gated_residual_mlp | **0.01024** | native probabilistic |
| deep_cross | 0.01702 | native probabilistic |
| wide_and_deep | 0.01768 | native probabilistic |
| lda | 0.02183 | native probabilistic |
| ... | | |
| linear_svm | 0.13070 | *requires external calibration* — squashed margin |
| adaboost | 0.14249 | *"usually the worst calibrated in the zoo"* |
| gaussian_nb | 0.18495 | *"famously over-confident"* |
| sgd_logistic | 0.24241 | native probabilistic — but see failure note |

Gaussian naive Bayes and AdaBoost were flagged as poorly calibrated in their
registry rationales, written before any result existed, and finished second- and
third-worst among converged models. The linear SVM's 0.13070 is **not** a
calibration measurement: it has no native probability, and the number comes from
a logistic squash of its decision margin, which the manifest records.

The best-calibrated model in the zoo is a deep one, and the three best are all
deep — consistent with Track K, where the FT-Transformer was the best-calibrated
of its four.

## Serialization

**29 of 29 round trips passed, with a maximum absolute prediction difference of
exactly 0.0.** Every model was saved, reloaded into a fresh object and re-scored;
predictions matched bit-for-bit.

| | Model | Format | Size |
| --- | --- | --- | --- |
| Largest | extra_trees | joblib | 3,919,689 B |
| | random_forest | joblib | 3,847,385 B |
| Smallest | linear_svm | joblib | 1,885 B |
| | logistic_l2 | joblib | 2,001 B |

The best model in the zoo serializes to 2 KB. The two forests are roughly two
thousand times larger for less accuracy at this budget.

Neural models are stored as **state dicts, not pickled modules** — a state dict
cannot execute code on load, and the adapter reconstructs the architecture from
the registry before loading weights.

## The experimental control behaved as a control

The **feature CNN** — whose convolutional inductive bias has no justification
here, since the feature ordering is arbitrary — scored 0.79821, mid-pack. It is
neither embarrassed nor vindicated, which is roughly what a model with an
irrelevant structural prior should look like on a problem where structure does
not much matter. It remains **not a production candidate** and is labelled as an
inductive-bias control wherever it appears.

## Limitations

- **1,000 training rows.** The single most important caveat. Track K measured
  this dataset's families to be strongly sample-dependent; these rankings
  describe the constrained regime and do not extrapolate. XGBoost's position
  here is the clearest illustration.
- **No tuning.** Frozen sensible defaults. A poor result may be a poor
  configuration, as `sgd_modified_huber` demonstrates.
- **One run, no intervals.** Unlike Track K there is no paired bootstrap here.
  Differences of 0.001-0.005 ROC-AUC in the table above should be read as ties.
  The top five models are within 0.0006 of each other and are not distinguishable.
- **One split, one test set**, as in Track K.
- **The base rate is engineered.** The dataset is close to 50/50 positive, so
  every probability is conditional on that prior and is not a population disease
  probability.
- **Timings are comparative**, from one CPU under ordinary load.
- **The run's manifest records `git.dirty: true`.** Documentation and CI files
  were being edited while it ran. Every file under `research/model_zoo/` was
  verified identical to commit `f651f4c` for the duration, so the dirty flag
  reflects uncommitted prose, not uncommitted model code.

## The production boundary

`model_artifacts/` and `provenance/` are byte-identical to their state before
Track L began; the manifest records `production_artifacts_touched: false`.
`/predict` serves what it has always served. No model in this zoo is a promotion
candidate, and Track L does not supersede Track K.
