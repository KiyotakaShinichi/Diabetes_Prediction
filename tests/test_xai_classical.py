"""Coefficient, native-importance and model-agnostic explainers under test.

Two kinds of test live here and they are doing different jobs.

The first kind checks explainers against **ground truth**: a world whose label
was generated from a rule this repository wrote, where "GenHlth is the driver"
is a fact rather than an interpretation. Almost nothing else in explainability
research can be tested this way, and it is the only reason the numbers in Track
M's report can be called right or wrong rather than merely reported.

The second kind checks that a **capability declaration matches the
implementation**. Track L caught `hist_gradient_boosting` claiming a native
feature importance that scikit-learn's histogram implementation does not have,
and fixed the declaration. The same failure in the XAI layer would be worse:
the whole cross-family comparison rests on the claim that a gap in the results
table is a property of the model rather than an oversight in the harness.

Three assertions here contradict the usual summary of what these methods do.
Row-wise occlusion *does* recover interacting features; permutation importance
*does* recover them too; partial dependence is the one that genuinely fails. All
three were measured across five seeds before being written down, and the
docstrings on the explainers carry the numbers.
"""
import json
import warnings

import numpy as np
import pytest

from ml_core import feature_contract
from research.model_zoo.contracts import Framework, ResearchStatus
from research.model_zoo.registry import REGISTRY
from research.xai import worlds
from research.xai.capabilities import XaiCapability, profile_for
from research.xai.contracts import CapabilityError, RunStatus, Scope
from research.xai.explainers import classical
from research.xai.registry import METHODS
from research.xai.worlds import XaiWorld

FEATURES = feature_contract.FEATURE_NAMES
INDEX = {name: i for i, name in enumerate(FEATURES)}

#: The only model that declares a native feature importance this module cannot
#: serve. Its importance is a per-feature contribution inside a torch module,
#: read by the deep explainers rather than by ``feature_importances_``. Named
#: explicitly so that a second model joining it fails a test instead of quietly
#: widening a gap in the results table.
TORCH_NATIVE_IMPORTANCE = {"neural_additive"}


def _classical_model_ids():
    """Active, non-torch models: everything this module is meant to explain."""
    return [
        spec.model_id
        for spec in REGISTRY
        if spec.effective_status() is ResearchStatus.ACTIVE
        and spec.framework is not Framework.TORCH
    ]


def _fit(model_id, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REGISTRY.build(model_id).fit(X, y)


@pytest.fixture(scope="module")
def dominant():
    """One driver, an easy world, split into fit and explanation partitions."""
    dataset = worlds.make(XaiWorld.ONE_DOMINANT_FEATURE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    return dataset, X_train, y_train, X_eval, y_eval


@pytest.fixture(scope="module")
def additive():
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    return dataset, X_train, y_train, X_eval, y_eval


@pytest.fixture(scope="module")
def xor():
    dataset = worlds.make(XaiWorld.XOR_INTERACTION, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    return dataset, X_train, y_train, X_eval, y_eval


@pytest.fixture(scope="module")
def classical_fitted():
    """Every non-torch model, fitted once.

    The two declaration-versus-implementation tests each sweep the whole
    roster; fitting it twice doubled the file's runtime for no extra evidence.
    """
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=200, seed=1)
    return {
        model_id: _fit(model_id, dataset.X, dataset.y)
        for model_id in _classical_model_ids()
    }


@pytest.fixture(scope="module")
def dominant_models(dominant):
    _, X_train, y_train, _, _ = dominant
    return {
        model_id: _fit(model_id, X_train, y_train)
        for model_id in ("logistic_l2", "random_forest", "decision_tree")
    }


def _top(values, k=1):
    return {FEATURES[i] for i in np.argsort(-np.abs(np.asarray(values)))[:k]}


# ================================================================ coefficients

@pytest.mark.parametrize("model_id", ["logistic_l2", "logistic_l1", "lda", "linear_svm"])
def test_coefficients_find_the_one_feature_that_generated_the_label(
    model_id, dominant
):
    _, X_train, y_train, _, _ = dominant
    model = _fit(model_id, X_train, y_train)

    weights = classical.coefficient_attributions(model)

    assert len(weights) == len(FEATURES)
    assert _top(weights) == {"GenHlth"}, f"{model_id} ranked {_top(weights)} first"


def test_coefficients_are_signed_because_a_linear_direction_is_real_information(
    dominant
):
    """Magnitude alone would discard something the local reports depend on.

    GenHlth runs 1 (excellent) to 5 (poor) and the label rises with it, so the
    coefficient must be positive. A method that returned magnitudes would make
    "poor health raises the model's score" indistinguishable from its opposite.
    """
    _, X_train, y_train, _, _ = dominant
    model = _fit("logistic_l2", X_train, y_train)

    weights = classical.coefficient_attributions(model)

    assert weights[INDEX["GenHlth"]] > 0


def test_asking_a_forest_for_coefficients_names_the_capability_contract():
    """The failure has to be legible, because the runner records the message."""
    dataset = worlds.make(XaiWorld.PURE_NOISE, rows=120, seed=1)
    model = _fit("random_forest", dataset.X, dataset.y)

    with pytest.raises(CapabilityError, match="capability profile"):
        classical.coefficient_attributions(model)


def test_every_model_claiming_coefficients_actually_produces_them(classical_fitted):
    """Declaration versus implementation, in both directions.

    A model that claims coefficients and cannot produce them would appear in
    the results table as a numerical failure. A model that produces them
    without claiming them would be silently excluded from the linear-family
    comparison, and nothing would ever say so.
    """
    for model_id, model in classical_fitted.items():
        claims = profile_for(REGISTRY.get(model_id)).supports(
            XaiCapability.NATIVE_COEFFICIENTS
        )

        if claims:
            weights = classical.coefficient_attributions(model)
            assert len(weights) == len(FEATURES), (
                f"{model_id} claims coefficients but returned {len(weights)} of "
                f"{len(FEATURES)} values"
            )
            assert np.isfinite(weights).all(), f"{model_id} returned non-finite weights"
        else:
            with pytest.raises(CapabilityError):
                classical.coefficient_attributions(model)


# =========================================================== native importance

@pytest.mark.parametrize("model_id", ["random_forest", "decision_tree", "extra_trees"])
def test_native_importance_finds_the_generating_feature(model_id, dominant):
    _, X_train, y_train, _, _ = dominant
    model = _fit(model_id, X_train, y_train)

    values = classical.native_importance_attributions(model)

    assert len(values) == len(FEATURES)
    assert _top(values) == {"GenHlth"}


def test_a_linear_model_has_no_native_importance_to_report(dominant):
    """Coefficients are a different quantity and are not offered as a substitute.

    The sklearn adapter's own ``feature_importance`` does fall back to
    coefficients, which is reasonable for a per-model report. Here it would be
    a category error: the cross-family table would show an impurity decrease
    and a logistic weight in one column labelled "native importance".
    """
    _, X_train, y_train, _, _ = dominant
    model = _fit("logistic_l2", X_train, y_train)

    with pytest.raises(CapabilityError, match="feature_importances_"):
        classical.native_importance_attributions(model)


def test_histogram_boosting_still_genuinely_has_no_native_importance(classical_fitted):
    """Guards the Track L correction against a silent library change.

    This declaration was wrong once. If a future scikit-learn adds the
    attribute, the profile should gain the capability deliberately rather than
    have the harness start reporting a quantity nobody reviewed.
    """
    model = classical_fitted["hist_gradient_boosting"]

    assert not profile_for(REGISTRY.get("hist_gradient_boosting")).supports(
        XaiCapability.NATIVE_FEATURE_IMPORTANCE
    )
    with pytest.raises(CapabilityError):
        classical.native_importance_attributions(model)


def test_every_model_claiming_native_importance_actually_produces_it(classical_fitted):
    for model_id, model in classical_fitted.items():
        claims = profile_for(REGISTRY.get(model_id)).supports(
            XaiCapability.NATIVE_FEATURE_IMPORTANCE
        )

        if claims:
            values = classical.native_importance_attributions(model)
            assert len(values) == len(FEATURES), f"{model_id} returned the wrong shape"
            assert np.isfinite(values).all()
        else:
            with pytest.raises(CapabilityError):
                classical.native_importance_attributions(model)


def test_only_the_named_torch_model_claims_an_importance_this_module_cannot_serve():
    """Keeps the one known gap explicit instead of letting it widen quietly."""
    claiming = {
        spec.model_id
        for spec in REGISTRY
        if spec.effective_status() is ResearchStatus.ACTIVE
        and spec.framework is Framework.TORCH
        and profile_for(spec).supports(XaiCapability.NATIVE_FEATURE_IMPORTANCE)
    }
    assert claiming == TORCH_NATIVE_IMPORTANCE


# ================================================== local linear contributions

def test_local_contributions_sum_exactly_to_the_logit_they_explain(dominant):
    """The one attribution in the zoo that is exact rather than approximate.

    For a linear model the terms are the decomposition of the decision function,
    so they must reconstruct it to floating-point precision. Everything else in
    Track M is measured against a faithfulness proxy; this is the fixed point
    those proxies can be sanity-checked against.
    """
    _, X_train, y_train, _, _ = dominant
    model = _fit("logistic_l2", X_train, y_train)
    estimator = classical._inner_estimator(model)
    scaler = classical._scaler(model)
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])

    for position in range(25):
        row = X_train.iloc[[position]]
        contributions = classical.local_linear_contribution(model, row)
        logit = float(estimator.decision_function(scaler.transform(row))[0])

        assert contributions.sum() == pytest.approx(logit - intercept, abs=1e-9)


def test_local_contributions_are_reported_in_the_models_own_feature_space(dominant):
    """Standardised, because that is the space the coefficients live in.

    A row at the training mean contributes nothing under this decomposition -
    not because the feature is unimportant, but because the reference point is
    the mean. That is a real property of the method and the record says so.
    """
    _, X_train, y_train, _, _ = dominant
    model = _fit("logistic_l2", X_train, y_train)

    mean_row = X_train.mean().to_frame().T
    contributions = classical.local_linear_contribution(model, mean_row)

    assert np.allclose(contributions, 0.0, atol=1e-9)


# ======================================================= permutation importance

def test_permutation_importance_finds_the_single_driver(dominant, dominant_models):
    """Every family, one answer, and the driver takes almost all the magnitude."""
    _, _, _, X_eval, y_eval = dominant

    for model_id, model in dominant_models.items():
        drops = classical.permutation_importance(model, X_eval, y_eval, seed=7)
        share = abs(drops[INDEX["GenHlth"]]) / np.abs(drops).sum()

        assert _top(drops) == {"GenHlth"}, f"{model_id} ranked {_top(drops)} first"
        assert share > 0.90, f"{model_id} gave the driver only {share:.3f} of the total"


def test_permutation_importance_finds_both_drivers_of_an_additive_rule(additive):
    _, X_train, y_train, X_eval, y_eval = additive

    for model_id in ("logistic_l2", "random_forest", "decision_tree"):
        model = _fit(model_id, X_train, y_train)
        drops = classical.permutation_importance(model, X_eval, y_eval, seed=7)

        assert _top(drops, k=2) == {"BMI", "Age"}, (
            f"{model_id} put {_top(drops, k=2)} on top of an additive BMI+Age rule"
        )


def test_permutation_importance_recovers_features_that_only_matter_jointly(xor):
    """Contradicts the usual "permutation cannot see interactions" summary.

    Shuffling one member of an interacting pair while the partner stays in place
    destroys the joint structure, so the loss is large. It is partial dependence
    that fails on this world, not this method, and conflating the two would put
    a false limitation on the zoo's cross-family baseline.
    """
    _, X_train, y_train, X_eval, y_eval = xor
    model = _fit("random_forest", X_train, y_train)

    drops = classical.permutation_importance(model, X_eval, y_eval, seed=7)

    assert _top(drops, k=2) == {"HighBP", "HighChol"}
    assert min(drops[INDEX["HighBP"]], drops[INDEX["HighChol"]]) > 0.20


def test_permutation_importance_is_near_zero_where_there_is_nothing_to_find():
    """The negative control, scored on rows the model was not fitted to."""
    dataset = worlds.make(XaiWorld.PURE_NOISE, rows=600, seed=3)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=3)
    model = _fit("random_forest", X_train, y_train)

    drops = classical.permutation_importance(model, X_eval, y_eval, seed=7)

    assert np.abs(drops).max() < 0.15, (
        f"a model reported {np.abs(drops).max():.4f} ROC-AUC points of importance "
        "on labels independent of every feature"
    )


def test_permutation_importance_on_the_fitting_rows_measures_memorisation():
    """Why every Track M call site passes a held-out partition.

    On labels containing nothing, a random forest scored on the rows it was
    fitted to reports importances several times larger than the same forest
    scored on unseen rows - it is being asked how much it needs each feature to
    reproduce answers it memorised. Logistic regression shows no such gap
    because it has no capacity to memorise with, which is what makes this a
    property of the partition rather than of the dataset.
    """
    dataset = worlds.make(XaiWorld.PURE_NOISE, rows=600, seed=1)
    X_train, y_train, X_eval, y_eval = worlds.split(dataset, seed=1)

    forest = _fit("random_forest", X_train, y_train)
    on_fit = np.abs(classical.permutation_importance(forest, X_train, y_train, seed=7)).max()
    on_eval = np.abs(classical.permutation_importance(forest, X_eval, y_eval, seed=7)).max()

    assert on_fit > 2.0 * on_eval, (
        f"expected the fitting rows to inflate importance; got {on_fit:.4f} "
        f"against {on_eval:.4f}"
    )

    linear = _fit("logistic_l2", X_train, y_train)
    linear_fit = np.abs(classical.permutation_importance(linear, X_train, y_train, seed=7)).max()
    assert linear_fit < 0.06, (
        "a linear model has nothing to memorise, so its fitting-row importance "
        f"should stay near zero; got {linear_fit:.4f}"
    )


def test_permutation_importance_is_stochastic_exactly_as_it_declares(dominant, dominant_models):
    """The registry says STOCHASTIC, and the seed study reads that flag.

    A method wrongly declared stochastic would have a seed sweep run over it and
    report a variance of exactly zero; one wrongly declared deterministic would
    have real variance go unmeasured.
    """
    _, _, _, X_eval, y_eval = dominant
    model = dominant_models["random_forest"]

    assert METHODS.get("permutation_importance").determinism.value == "stochastic"

    repeated = classical.permutation_importance(model, X_eval, y_eval, seed=1)
    same_seed = classical.permutation_importance(model, X_eval, y_eval, seed=1)
    other_seed = classical.permutation_importance(model, X_eval, y_eval, seed=2)

    assert np.array_equal(repeated, same_seed)
    assert not np.array_equal(repeated, other_seed)


def test_permutation_importance_is_denominated_in_roc_auc_points(dominant, dominant_models):
    """Which is what makes it comparable across families at all.

    An inert feature can score slightly negative - shuffling it happened to
    help - and that is meaningful rather than a bug: it puts a scale on the
    method's own noise floor.
    """
    _, _, _, X_eval, y_eval = dominant
    drops = classical.permutation_importance(dominant_models["logistic_l2"], X_eval, y_eval, seed=7)

    assert drops.max() <= 1.0
    assert drops.min() >= -1.0
    assert 0.3 < drops[INDEX["GenHlth"]] < 1.0


def test_averaging_several_shuffles_is_not_the_same_as_one(dominant, dominant_models):
    _, _, _, X_eval, y_eval = dominant
    model = dominant_models["random_forest"]

    once = classical.permutation_importance(model, X_eval, y_eval, repeats=1, seed=4)
    averaged = classical.permutation_importance(model, X_eval, y_eval, repeats=5, seed=4)

    assert not np.array_equal(once, averaged)
    assert _top(once) == _top(averaged) == {"GenHlth"}


# ==================================================================== occlusion

def test_occlusion_finds_the_driver_when_read_across_a_case_sample(dominant, dominant_models):
    _, X_train, _, _, _ = dominant
    baseline = worlds.baseline_row(X_train)

    for model_id, model in dominant_models.items():
        totals = np.zeros(len(FEATURES))
        for position in range(40):
            totals += np.abs(
                classical.occlusion_attributions(model, X_train.iloc[[position]], baseline)
            )

        assert _top(totals) == {"GenHlth"}, f"{model_id} ranked {_top(totals)} first"


def test_occlusion_recovers_interacting_features_though_it_cannot_pair_them(xor):
    """A local intervention, not a population average - so XOR is visible.

    Each driver is credited with the whole joint swing separately, which is the
    real limitation: one effect reported twice. That is the interaction audit's
    problem to solve, and it is a different problem from not seeing the features
    at all.
    """
    _, X_train, y_train, _, _ = xor
    model = _fit("random_forest", X_train, y_train)
    baseline = worlds.baseline_row(X_train)

    totals = np.zeros(len(FEATURES))
    for position in range(60):
        totals += np.abs(
            classical.occlusion_attributions(model, X_train.iloc[[position]], baseline)
        )

    assert _top(totals, k=2) == {"HighBP", "HighChol"}


def test_a_feature_already_at_its_baseline_occludes_to_exactly_zero(dominant, dominant_models):
    """The method's real blind spot, asserted rather than described.

    Replacing a value with itself cannot move the score. For a binary feature
    with a median baseline this silences roughly half the rows, which is why no
    conclusion in Track M is drawn from a single row's occlusion vector.
    """
    _, X_train, _, _, _ = dominant
    baseline = worlds.baseline_row(X_train)
    model = dominant_models["random_forest"]

    row = X_train.iloc[[0]].copy()
    row["BMI"] = baseline["BMI"]

    deltas = classical.occlusion_attributions(model, row, baseline)

    assert deltas[INDEX["BMI"]] == 0.0


def test_the_batched_occlusion_is_exactly_the_per_row_one(dominant, dominant_models):
    """Same arithmetic, forty times faster - and "same" has to be exact.

    A one-row DataFrame carries about 87ms of scikit-learn dispatch overhead on
    these pipelines, so occluding forty rows one at a time spends almost all of
    its time outside the model; the stability sweep, which repeats that ten
    times over, cost six minutes per model before it was batched and costs
    seven seconds after. A speedup that changed the numbers would be worthless,
    so the equivalence is asserted rather than assumed.
    """
    _, X_train, _, _, _ = dominant
    baseline = worlds.baseline_row(X_train)
    frame = X_train.iloc[:12]

    for model_id, model in dominant_models.items():
        batched = classical.occlusion_matrix(model, frame, baseline)
        per_row = np.vstack([
            classical.occlusion_attributions(model, frame.iloc[[i]], baseline)
            for i in range(len(frame))
        ])

        assert batched.shape == (12, len(FEATURES))
        assert np.allclose(batched, per_row, atol=1e-12), f"{model_id} disagreed"


def test_occlusion_returns_one_value_per_feature(dominant, dominant_models):
    _, X_train, _, _, _ = dominant
    deltas = classical.occlusion_attributions(
        dominant_models["logistic_l2"], X_train.iloc[[3]], worlds.baseline_row(X_train)
    )
    assert deltas.shape == (len(FEATURES),)


# =========================================================== partial dependence

def test_partial_dependence_tracks_the_driver_and_ignores_inert_columns(
    dominant, dominant_models
):
    _, X_train, _, _, _ = dominant

    for model_id, model in dominant_models.items():
        driver = classical.partial_dependence(model, X_train, "GenHlth")
        inert = classical.partial_dependence(model, X_train, "Education")

        assert driver["range"] > 10 * inert["range"], (
            f"{model_id}: driver range {driver['range']:.4f} against inert "
            f"{inert['range']:.4f}"
        )


def test_partial_dependence_rises_with_the_feature_that_raises_the_label(
    dominant, dominant_models
):
    """Direction, not just magnitude. GenHlth runs from excellent to poor."""
    _, X_train, _, _, _ = dominant
    curve = classical.partial_dependence(dominant_models["logistic_l2"], X_train, "GenHlth")

    assert curve["average_score"][-1] > curve["average_score"][0]


def test_partial_dependence_is_blind_to_a_purely_interactive_rule(xor):
    """The measured failure, asserted so the report can state it as a result.

    The forest here reaches roughly 0.96 held-out ROC-AUC using both drivers, and
    partial dependence still puts two inert columns on top. Averaging over the
    population cancels an effect that is positive for half of it and negative
    for the other half. This held at every seed tried.
    """
    _, X_train, y_train, X_eval, y_eval = xor
    model = _fit("random_forest", X_train, y_train)

    ranges = {
        feature: classical.partial_dependence(model, X_train, feature)["range"]
        for feature in FEATURES
    }
    top_two = set(sorted(ranges, key=lambda f: -ranges[f])[:2])

    assert not (top_two & {"HighBP", "HighChol"}), (
        f"partial dependence ranked {top_two} first; if it has started detecting "
        "the interaction, the reported limitation needs revisiting"
    )

    drops = classical.permutation_importance(model, X_eval, y_eval, seed=7)
    assert _top(drops, k=2) == {"HighBP", "HighChol"}, (
        "the contrast is only meaningful while another method still finds them"
    )


def test_the_partial_dependence_grid_never_leaves_the_observed_range(
    dominant, dominant_models
):
    """No extrapolation: the curve stays where the model has actually seen data."""
    _, X_train, _, _, _ = dominant
    curve = classical.partial_dependence(dominant_models["random_forest"], X_train, "BMI")

    assert min(curve["grid"]) >= float(X_train["BMI"].min())
    assert max(curve["grid"]) <= float(X_train["BMI"].max())
    assert "no extrapolation" in curve["support"]


def test_the_partial_dependence_grid_honours_the_requested_resolution(
    dominant, dominant_models
):
    curve = classical.partial_dependence(
        dominant_models["logistic_l2"], dominant[1], "BMI", grid_points=7
    )
    assert len(curve["grid"]) == 7
    assert len(curve["average_score"]) == 7


def test_a_constant_feature_does_not_collapse_the_grid(dominant, dominant_models):
    """A degenerate column has no percentile spread and must not divide by it."""
    _, X_train, _, _, _ = dominant
    frozen = X_train.copy()
    frozen["BMI"] = 30.0

    curve = classical.partial_dependence(dominant_models["logistic_l2"], frozen, "BMI")

    assert len(curve["grid"]) == classical.PDP_GRID_POINTS
    assert curve["range"] == pytest.approx(0.0, abs=1e-12)


# ========================================================================= ICE

def test_the_partial_dependence_curve_is_the_mean_of_the_ice_curves(
    dominant, dominant_models
):
    """An identity, so it holds to floating-point precision or something is wrong.

    PDP is defined as the average of the per-row curves. Checking it here means
    a later disagreement between the two is a real finding about heterogeneity
    rather than two implementations of the same idea drifting apart.
    """
    _, X_train, _, _, _ = dominant
    model = dominant_models["logistic_l2"]

    curve = classical.partial_dependence(model, X_train, "GenHlth")
    ice = classical.individual_conditional_expectation(
        model, X_train, "GenHlth", rows=len(X_train)
    )

    assert ice["row_count"] == len(X_train)
    assert np.allclose(curve["grid"], ice["grid"])
    assert np.allclose(np.mean(np.asarray(ice["curves"]), axis=0), curve["average_score"])


def test_ice_samples_rows_reproducibly_and_never_asks_for_more_than_it_has(
    dominant, dominant_models
):
    _, X_train, _, _, _ = dominant
    model = dominant_models["random_forest"]

    first = classical.individual_conditional_expectation(model, X_train, "BMI", rows=15, seed=2)
    again = classical.individual_conditional_expectation(model, X_train, "BMI", rows=15, seed=2)
    everything = classical.individual_conditional_expectation(
        model, X_train, "BMI", rows=10_000, seed=2
    )

    assert first["row_count"] == 15
    assert np.allclose(first["curves"], again["curves"])
    assert everything["row_count"] == len(X_train)


def test_ice_reports_the_spread_the_average_curve_hides(dominant, dominant_models):
    _, X_train, _, _, _ = dominant
    ice = classical.individual_conditional_expectation(
        dominant_models["random_forest"], X_train, "GenHlth", rows=30, seed=1
    )

    assert ice["mean_span"] > 0.0
    assert ice["span_dispersion"] >= 0.0


# ===================================================================== records

def test_a_record_carries_raw_values_normalised_shares_and_a_ranking(dominant_models):
    """All three, because each answers a question the others cannot.

    Raw values keep the method's own units and sign; the normalised share is
    what makes a logistic coefficient comparable to an impurity decrease; the
    ranking is what the cross-family analysis actually reads.
    """
    weights = classical.coefficient_attributions(dominant_models["logistic_l2"])

    record = classical.build_record(
        "logistic_l2", "coefficients", "1.0.0", Scope.GLOBAL, FEATURES, weights,
        baseline_reference="not_applicable", runtime_seconds=0.01,
    )

    assert record.model_family == "linear"
    assert record.resource_status is RunStatus.SUCCESS
    assert record.raw_attributions == tuple(float(v) for v in weights)
    assert sum(record.normalized_attributions) == pytest.approx(1.0)
    assert record.top_feature == "GenHlth"
    assert record.rank_of("GenHlth") == 1
    assert record.attribution_for("GenHlth") > 0.5


def test_a_record_serialises_to_plain_json(dominant_models):
    """Evidence is JSON, not a pickled explainer."""
    values = classical.native_importance_attributions(dominant_models["random_forest"])

    record = classical.build_record(
        "random_forest", "native_importance", "1.0.0", Scope.GLOBAL, FEATURES, values,
        baseline_reference="not_applicable", sample_id=None, seed=42,
    )
    restored = json.loads(json.dumps(record.as_dict()))

    assert restored["model_id"] == "random_forest"
    assert restored["ranking"][0] == "GenHlth"
    assert restored["seed"] == 42


def test_the_same_explanation_gets_the_same_identifier_twice(dominant_models):
    values = classical.native_importance_attributions(dominant_models["decision_tree"])

    def build(**kwargs):
        return classical.build_record(
            "decision_tree", "native_importance", "1.0.0", Scope.GLOBAL, FEATURES,
            values, baseline_reference="not_applicable", **kwargs
        )

    assert build(sample_id=4).explanation_id == build(sample_id=4).explanation_id
    assert build(sample_id=4).explanation_id != build(sample_id=5).explanation_id


def test_timing_a_call_returns_its_result_and_a_real_duration():
    result, seconds = classical.timed(sum, [1, 2, 3])

    assert result == 6
    assert seconds >= 0.0


def test_a_hard_label_model_is_reported_as_having_no_ranking_score():
    """Which is why the model-agnostic methods exclude it rather than degrade."""
    assert classical.has_probability("logistic_l2")
    assert not classical.has_probability("nearest_centroid")
