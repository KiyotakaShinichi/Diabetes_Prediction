# Track M — what twenty-nine models' explanations actually said

**Research artefact.** Model behaviour on a research dataset. Not a diagnostic
tool, not clinical guidance, not a basis for promoting any model. Every figure
below describes what a model's prediction depends on, which is an association
and not a cause. `model_artifacts/` is untouched by this track.

Protocol and the corrections that produced it:
[track_m_xai.md](track_m_xai.md).

---

## The run

One sweep, `xai-1000-dadfc3f4ed`, 26.5 minutes on one CPU.

| | |
|---|---|
| models attempted | 29 active, **0 failed to fit** |
| methods attempted | 9 |
| pairs attempted | 2,484 |
| explanations produced | **2,350** |
| pairs recorded unsupported | 134 |
| training rows | 1,000, Track K's fingerprinted subset |
| permutation / PD rows | 500 validation, never the fitting rows, never test |
| cases explained locally | 40 |

Nothing was dropped. Every unsupported pair carries the capability it lacked.

---

## Finding 1: the disagreement is between methods, not between model families

This is the result the track was built to find, and it points the opposite way
from the obvious expectation.

| comparison | pairs | mean Spearman | median | worst | top-1 agreement |
|---|---|---|---|---|---|
| **methods on one model** | 2,176 | **0.426** | 0.431 | −0.875 | 0.380 |
| models within a family | 6,392 | 0.722 | 0.854 | −0.782 | 0.415 |
| **models across families** | 13,286 | **0.770** | 0.876 | −0.636 | 0.404 |

A linear model and a transformer, explained the same way, agree substantially
better (0.770) than two methods applied to the *same* model (0.426). Swapping
the architecture perturbs the explanation less than swapping the lens.

Read with Track K and Track L, that is coherent rather than surprising: those
tracks found the families converging on the same predictions, and this one finds
them converging on similar attributions. What does not converge is the toolkit.

**The honest limit on this.** Both figures are full-ranking correlations, and
Track M's own measurements show that statistic is dominated by the
uninformative tail when attribution concentrates. The comparison between 0.426
and 0.770 is meaningful because both are computed the same way on the same
records; neither number should be read as "the methods agree" in absolute terms.
The top-1 column says why.

## Finding 2: which feature is "most important" is a coin flip

Top-1 agreement is **0.38 to 0.42** in every grouping. Two randomly chosen
explanations of these models name the same leading feature about two times in
five — and that holds whether the two explanations come from one model or from
two different families.

The sentence "the most important feature is X" is the one that escapes into a
slide deck, and on this dataset it is not reproducible across methods.

## Finding 3: a consensus exists, and its top three are a near-tie

| rank | feature | mean rank |
|---|---|---|
| 1 | `Age` | 3.45 |
| 2 | `GenHlth` | 3.53 |
| 3 | `BMI` | 3.77 |
| 4 | `Education` | 4.97 |
| 5 | `HighBP` | 5.09 |
| 6 | `HighChol` | 6.03 |
| 7 | `DiffWalk` | 6.48 |
| 8 | `HeartDiseaseorAttack` | 6.95 |
| 9 | `PhysHlth` | 7.04 |
| 10 | `PhysActivity` | 7.68 |

`Age`, `GenHlth` and `BMI` lead, separated by 0.08 and 0.24 of a rank. That is
not an ordering. It is a three-way tie at the top of a list whose bottom is
clearly separated, and the report prints the near-ties rather than presenting
first, second and third as a ranking.

## Finding 4: the fitted surfaces are essentially additive

Friedman's H-statistic on the log-odds scale, over each model's top five
consensus features:

| | |
|---|---|
| models measured | 28 |
| median of each model's strongest pair | **H = 0.023** |
| models whose strongest pair is below 0.1 | **86%** |
| most interactive model | `sgd_modified_huber`, H = 0.291 |
| every linear model | H = 0.000 exactly |

**This is the strongest evidence Track M produces for the bottleneck
conclusion.** If the richer families were beating the simpler ones by finding
joint structure, it would appear here, and it does not: gradient boosting reaches
0.086, XGBoost 0.074, and the deep models less. The ten served features are used
close to independently by almost everything in the zoo.

The linear models reporting exactly zero is the check that the measurement
works. On the probability scale they report 0.18 — their own sigmoid — which is
why the statistic is computed on log odds.

## Finding 5: stability is real but not universal

The top feature's survival under perturbation, in training standard deviations,
measured with occlusion profiles over 40 cases:

| strongest perturbation the leading feature survived | models |
|---|---|
| 1.0 deviations | 9 |
| 0.5 | 9 |
| 0.1 | 6 |
| did not survive 0.1 | **4** |

Eighteen of twenty-eight models keep their leading feature through half a
standard deviation of noise. Four lose it at a tenth, which for a binary flag is
a flip in about 5% of rows. For those four, "this model's most important
feature" is not a claim the data supports.

## Finding 6: three models' explanations fail their own faithfulness control

Each model's consensus ranking was scored against five shuffled rankings over
the same rows.

| | |
|---|---|
| beat the shuffled control | **25 of 28** |
| failed | 3 — `gaussian_nb`, `sgd_logistic`, `sgd_modified_huber` |
| median deletion gap | 0.044 |
| range | −0.036 to 0.121 |

A failure here means the ranking's *order* carried no information the shuffle
did not: deleting its "important" features first damaged the model no faster
than deleting features at random. Those three explanations should not be quoted.

The same three appear at the top of the interaction table and the bottom of the
stability one, which is consistent — an explanation of a model whose surface is
unusual is where every proxy in this package is weakest.

---

## Capability coverage, which bounds all of the above

| capability | active models |
|---|---|
| permutation importance, occlusion, PDP/ICE, interaction | 28 |
| gradients (all three) | 8 |
| native coefficients | 7 |
| native feature importance | 7 |
| tree SHAP | 5 |

The 134 unsupported pairs are the study's second result. `nearest_centroid`
supports nothing: it emits hard labels, so no perturbation has a score to move.
`ft_transformer` and `tab_transformer` support no gradient method, because they
reach their discrete features through embedding tables and an input gradient is
exactly 0.0 for nine of ten features — a number that would have read as "the
transformer ignores nine features".

A figure computed over five models is not a cross-family result, and tree SHAP's
five is why it appears nowhere in Findings 1 to 3.

---

## What this does not show

- **Not causation.** No claim that a feature causes diabetes, that changing one
  would change a person's risk, or any treatment recommendation.
- **Not correctness.** The methods agreeing across families does not make them
  right; they share assumptions. Faithfulness tests a ranking against the
  model's own behaviour, which is weaker than testing it against the truth.
- **Not a promotion.** Nothing here changes a threshold, an architecture, a
  feature set or a deployed artefact.
- **Bounded by the contract.** Ten features. A feature absent from the contract
  cannot appear in any attribution, however much the outcome depends on it.
- **Bounded by the budget.** 1,000 training rows, 500 evaluation rows, 40
  explained cases, frozen default configurations. Permutation importance at 500
  rows carries about 0.02 ROC-AUC points of sampling noise, which is why the
  near-ties in Finding 3 are reported as ties.
