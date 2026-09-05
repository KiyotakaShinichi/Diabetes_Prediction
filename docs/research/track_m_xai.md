# Track M — cross-family explainability, stability and faithfulness

**Research artefact.** Everything here describes model behaviour on a research
dataset. It is not a diagnostic tool, not clinical guidance, and not a basis for
promoting any model. Attribution describes what a model's prediction depends on,
which is an association and not a cause. The deployed artefacts in
`model_artifacts/` are untouched by this track.

---

## The question

Track K compared four model families under one frozen protocol and found them
within a few thousandths of ROC-AUC of each other, failing on the same patients.
Track L widened that to twenty-nine algorithms at a 1,000-row budget and found
the same thing. Both concluded that the limit is the ten-feature information set
rather than the model class.

That is a claim about *what the models know*. Track M asks the adjacent question
neither track could answer: **do they know it in the same way?** Twenty-nine
algorithms agreeing on predictions is compatible with two very different
worlds - one where they have all found the same structure, and one where they
have found different structure that happens to be equally predictive. The first
supports the bottleneck conclusion. The second would undermine it.

Answering that requires explanations to be comparable objects rather than plots,
which is what most of this track is.

---

## What an explanation is here

Every explanation is an `ExplanationRecord`: raw attribution values, a
normalised share, a rank, the method and its version, the baseline it measured
against, the seed, the runtime, and the hashes tying it to a model and a data
subset. Three decisions shape it.

**Attributions are not comparable in their raw units.** A logistic coefficient,
a permutation importance in ROC-AUC points, an integrated gradient in logit
space and an impurity decrease are four different physical quantities. Putting
them in one table without normalising produces a number that looks like a
comparison and is not one. Every record therefore carries all three forms, and
the cross-model analysis reads only the last two.

**Evidence is JSON.** A pickled explainer is a snapshot of a library version
that cannot be diffed, cannot be read in five years, and executes code on load.

**Attribution is not causation.** The vocabulary is association and model
dependence, and `tests/test_xai_language.py` enforces that on the generated
documents rather than trusting review.

---

## What each model can be asked

A capability is declared per model and validated against the constructed model.
Track L caught `hist_gradient_boosting` claiming a native feature importance
scikit-learn does not expose, and fixed the declaration rather than the test;
Track M inherits that discipline.

| capability | active models |
|---|---|
| native coefficients | 7 |
| native feature importance | 7 |
| permutation importance | 28 |
| occlusion | 28 |
| partial dependence / ICE | 28 |
| gradient, gradient x input, integrated gradients | 8 |
| tree SHAP | 5 |
| interaction analysis | 28 |

Two of those numbers are findings rather than bookkeeping.

**A hard-label model is excluded from every method.** `nearest_centroid` emits
0/1 and nothing else, so a permuted feature either flips a label or does
nothing; occlusion and partial dependence become step functions. Reporting a
number for it would put something in the table that looks like a measurement.

**Two of the ten deep models cannot be differentiated with respect to their
inputs.** "It is a neural network, so it has gradients" is true of the
parameters and false of the inputs. `ft_transformer` tokenises all nine of its
discrete features through an embedding table and `tab_transformer` nine of ten;
the derivative of a table index does not exist. Asking either for an input
gradient does not raise - it returns a well-formed vector with 0.0 in nine of
ten positions, which reads in a results table as a transformer that ignores nine
features. `tests/test_xai_deep.py` builds every torch model and counts the slots
that actually carry a derivative, so the declaration cannot drift from the
architectures.

---

## Worlds with a known answer

An explanation cannot be validated against the real data: nobody knows which
features the diabetes label actually depends on, which is why the models were
built. Agreement between methods on real rows shows they agree, not that either
is right.

Four synthetic worlds generate the label from a rule this repository wrote, so
"the right answer" becomes defensible. They are chosen to break things:

| world | the rule | what it exposes |
|---|---|---|
| `ONE_DOMINANT_FEATURE` | GenHlth drives the label | the easy case every method must pass |
| `ADDITIVE_TWO_FEATURE` | BMI and Age contribute equally | whether both drivers are found |
| `XOR_INTERACTION` | HighBP xor HighChol | separates marginal from joint methods |
| `PURE_NOISE` | labels independent of everything | the negative control |

They corrected several claims that were written before being measured. Those
corrections are the substance of what follows.

---

## Measured corrections

Each of these was written one way, measured, and rewritten. They are recorded
because a reader has no way to tell a checked claim from a plausible one.

### Partial dependence is blind to interaction; occlusion and permutation are not

The usual summary says single-feature methods cannot see interactions. Measured
on the XOR world against a random forest that learns the rule to 0.96 held-out
ROC-AUC:

| method | result |
|---|---|
| partial dependence | top two ranges are **inert columns**, at all five seeds tried |
| row-wise occlusion | both drivers in the top two, at all five seeds |
| permutation importance | both drivers in the top two, at 0.50 and 0.43 ROC-AUC points |
| coefficients (linear) | the model cannot fit the rule at all (0.55 held-out) |

Partial dependence fails because it averages over the population, and the
driver's effect is positive for half of it and negative for the other half.
Occlusion is a *local* intervention and sees the pair; what it cannot do is
attribute the effect to the pair, since it credits each member with the whole
joint swing separately. Permutation breaks the joint structure by shuffling one
member while the partner stays in place, which is exactly why it works here.

A flat partial-dependence curve is evidence about an average, never evidence
that the model ignores a feature.

### Training-row permutation importance measures memorisation

On `PURE_NOISE`, where the label is independent of every column:

| model | scored on its own fitting rows | scored on held-out rows |
|---|---|---|
| random forest | 0.116 – 0.145 | 0.022 – 0.065 |
| decision tree | 0.111 – 0.156 | 0.034 – 0.081 |
| logistic regression | 0.019 – 0.041 | 0.013 – 0.026 |

Ranges across five seeds, in ROC-AUC points, where the truth is zero. The
high-capacity models are being asked how much they need each feature to
reproduce answers they memorised; the linear model shows no gap because it has
no capacity to memorise with. Every Track M call site therefore scores
permutation importance on a partition the model was not fitted on.

### Full-ranking correlation is dominated by the uninformative tail

Three genuinely different explainers - a coefficient, an impurity decrease and a
permutation drop, over two different models - on a world with one known driver:

| reading | world with a driver | pure noise |
|---|---|---|
| top-1 agreement | **1.00 at every seed** | **0.00 at every seed** |
| mean Spearman | 0.20 – 0.54 | -0.01 – 0.41 |
| mean top-3 overlap | 0.44 – 0.67 | 0.22 – 0.44 |

When one feature takes almost all the attribution, nine of ten ranks are
ordering noise and Spearman is mostly a measurement of that noise. Top-3
overlap does not separate the two worlds either - at one seed both score an
identical 0.444 - because ranks two and three are noise on both sides. Give the
world a second real driver and top-3 starts working again.

**Read the top-1 column.** The report prints that caveat wherever it prints a
correlation.

### Faithfulness needs per-row shift, not the mean score

The obvious deletion curve - watch the average predicted probability move as
features are ablated - barely registers, because ablation pushes rows in
opposite directions toward the same central prediction and they cancel.

| statistic | driver deleted first | inert feature deleted first | ratio |
|---|---|---|---|
| change in mean score | 0.016 | 0.004 | 4x |
| mean absolute per-row shift | 0.310 | 0.010 | 31x |

The per-row shift carries 95% of the total available damage in the single
deletion of the driver. The mean is not inert, but it is an order of magnitude
less discriminating on a scale small enough for anything to swamp.

### Interaction is not scale-free

A logistic regression is additive in the logit by construction. Measured on the
probability its `decision_scores` return, it reports an H-statistic of **0.18**
for BMI against Age. The sigmoid produced that, not the model. Left
uncorrected, every probability-valued model would carry that floor while the
tree families - whose probabilities are vote averages rather than squashed
sums - would be compared against it on a different footing.

Probability-valued scores are therefore converted to log odds per row before
anything is averaged, selected by the model's own capability declaration rather
than by sniffing the value range. On that scale the logistic regression falls
below 0.05, and the XOR pair scores 1.00 and 0.99.

### Gradient x input has a mixed reference point

Track K's encoder standardises the five continuous and ordinal features and
leaves binary indicators on their raw 0/1 scale. Zero in that space therefore
means *the training mean* for GenHlth, BMI, Age, PhysHlth and Education, and the
literal value *"No"* for HighBP, HighChol, DiffWalk, HeartDiseaseorAttack and
PhysActivity.

Every patient without high blood pressure receives exactly zero attribution on
HighBP, at every model, however heavily the model relies on it. Half the zeros
this method produces on binary features are structural, not measured.

### Integrated gradients needs a budget set by the worst architecture

Completeness gap - how far the summed attribution falls short of the score
difference the axiom promises - in logits:

| steps | seven of eight architectures | `feature_token_mixer` |
|---|---|---|
| 8 | < 0.030 | 1.339 |
| 32 | < 0.007 | 0.105 |
| 128 | < 0.002 | 0.004 |

The token-mixing blocks make the path from baseline to input far more curved
than the others'. The budget is 128 steps, set by the hardest architecture, and
the per-row gap is recorded regardless: a step count chosen once cannot
guarantee an axiom holds for a model nobody has run yet.

---

## The protocol

| decision | value | why |
|---|---|---|
| training rows | 1,000, from Track K's fingerprinted ladder | the same rows Track L fitted, so these explanations describe the zoo's models |
| permutation / PD partition | 2,000 validation rows, sampled once | never the fitting rows, never test |
| explained cases | 40 validation rows, deterministic | enough for a stable profile at a single-CPU budget |
| baselines | median of the fitting rows | a baseline from evaluation rows would leak that distribution |
| perturbation scale | training standard deviations | a number means the same thing for BMI and PhysHlth |
| integrated-gradient steps | 128 midpoint | set from the measured completeness gap |
| interaction sweep | pairs among each model's consensus top five | a full 45-pair sweep costs minutes per model |
| agreement bands | 0.80 high, 0.50 moderate | frozen before any aggregate was computed |

Perturbations are coerced back into the served feature contract - rounded where
the contract says integer, clipped to the declared range. Unconstrained noise
produces a BMI of 91 and an Education level of 2.7, and the instability that
follows would measure the model leaving its training distribution rather than
the explanation being fragile.

---

## Running it

```bash
# See the resolved (model, method) matrix without running anything.
python -m research.xai.run --dry-run

# A representative slice.
python -m research.xai.run --train-rows 1000 \
    --models logistic_l2 random_forest mlp \
    --methods coefficients permutation_importance occlusion \
    --case-limit 20 --output-dir /tmp/xai

# The full sweep. Explicit research execution, not a push gate.
python -m research.xai.run
```

Output goes to `research_artifacts/xai/<run_id>/`: one JSON file of records per
model, `outcomes.json` with a row for every attempted pair, `analysis.json`, and
`run_manifest.json` **written last** - a run that dies halfway leaves records
and no manifest, which is unambiguous.

CI gates the capability contract and a five-case smoke run, not the full sweep.
A registry that no longer resolves is a broken repository; a sweep measured in
minutes is not a push gate.

---

## What this does not show

- **Not causation.** Nothing here supports a claim that a feature causes
  diabetes, that changing a feature would change a person's risk, or any
  treatment recommendation.
- **Not correctness.** Methods agreeing does not make them right; they share
  assumptions, and two methods built on the same assumption agree about the same
  mistake. Faithfulness tests rankings against the model's own behaviour, which
  is a weaker question than whether the ranking is true.
- **Not a promotion.** No model examined here is proposed for production, and
  no explanation result may be used to retune hyperparameters, change an
  architecture, change feature selection, or move a threshold.
- **Not clinical.** A research dataset at an exploratory training budget.
- **Bounded by the contract.** Ten features are all any explanation here can
  range over. A feature absent from the contract cannot appear in any
  attribution, however much the outcome depends on it.
