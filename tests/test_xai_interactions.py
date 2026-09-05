"""Friedman's H-statistic, checked against a world whose interaction is known.

The exclusive-or world is what makes this testable. Neither driver has any
marginal association with the label, so every one-dimensional method in the
package is blind to the pair - partial dependence ranks two inert columns above
both drivers - while a random forest learns the rule to 0.96 held-out ROC-AUC.
If the H-statistic cannot find that pair, it cannot find anything, and the
interaction audit would be decoration.

The arithmetic is checked separately on hand-built surfaces, where exact
additivity and exact degeneracy both have known answers, because a surface
produced by a real model is never exactly either.

Everything here runs on a deliberately small grid and row sample. The full
forty-five-pair sweep at production resolution takes minutes per model; these
tests are about correctness, and correctness does not need resolution.
"""
import warnings

import numpy as np
import pytest

from research.model_zoo.registry import REGISTRY
from research.xai import interactions, worlds
from research.xai.worlds import XaiWorld

#: Small enough to keep the suite fast, large enough that an interaction has
#: somewhere to show up. The production defaults are declared in the module.
GRID = 6
ROWS = 60


def _fit(model_id, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REGISTRY.build(model_id).fit(X, y)


@pytest.fixture(scope="module")
def xor():
    dataset = worlds.make(XaiWorld.XOR_INTERACTION, rows=400, seed=1)
    X_train, y_train, _, _ = worlds.split(dataset, seed=1)
    return _fit("random_forest", X_train, y_train), X_train


@pytest.fixture(scope="module")
def additive():
    dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=400, seed=1)
    X_train, y_train, _, _ = worlds.split(dataset, seed=1)
    return _fit("logistic_l2", X_train, y_train), X_train


# ============================================================== the arithmetic

def test_an_exactly_additive_surface_has_no_interaction():
    """Zero by construction, so anything else is an implementation error."""
    curve_a = np.arange(5, dtype=float)
    curve_b = np.arange(5, dtype=float) * 3.0
    surface = {
        "joint": curve_a[:, None] + curve_b[None, :],
        "curve_a": curve_a,
        "curve_b": curve_b,
    }

    assert interactions.h_statistic(surface) == pytest.approx(0.0, abs=1e-12)


def test_a_purely_multiplicative_surface_is_almost_all_interaction():
    values = np.array([-1.0, 1.0])
    surface = {
        "joint": np.outer(values, values),
        "curve_a": np.zeros(2),
        "curve_b": np.zeros(2),
    }

    assert interactions.h_statistic(surface) == pytest.approx(1.0)


def test_a_flat_surface_reports_no_interaction_rather_than_a_failure():
    """A model whose score never moves has no variance to decompose.

    Returning zero says "no interaction". Returning NaN or dividing by zero
    would put a missing measurement into the table beside real ones.
    """
    surface = {"joint": np.zeros((4, 4)), "curve_a": np.zeros(4), "curve_b": np.zeros(4)}

    assert interactions.h_statistic(surface) == 0.0


def test_the_statistic_is_bounded_at_one():
    surface = {
        "joint": np.array([[5.0, -5.0], [-5.0, 5.0]]),
        "curve_a": np.array([10.0, -10.0]),
        "curve_b": np.array([10.0, -10.0]),
    }

    assert 0.0 <= interactions.h_statistic(surface) <= 1.0


# ============================================================== the surface

def test_a_pair_needs_two_distinct_features(xor):
    model, X_train = xor

    with pytest.raises(ValueError, match="two distinct features"):
        interactions.two_way_partial_dependence(model, X_train, "BMI", "BMI")


def test_the_surface_carries_both_marginal_curves_on_the_same_grid(xor):
    """Computed together so the three cannot drift onto different grids."""
    model, X_train = xor

    surface = interactions.two_way_partial_dependence(
        model, X_train, "HighBP", "BMI", grid_points=GRID, sample_rows=ROWS
    )

    assert np.asarray(surface["joint"]).shape == (
        len(surface["grid_a"]), len(surface["grid_b"])
    )
    assert len(surface["curve_a"]) == len(surface["grid_a"])
    assert len(surface["curve_b"]) == len(surface["grid_b"])
    assert surface["sample_rows"] == ROWS


def test_the_grid_never_leaves_the_observed_range(xor):
    model, X_train = xor

    surface = interactions.two_way_partial_dependence(
        model, X_train, "BMI", "Age", grid_points=GRID, sample_rows=ROWS
    )

    assert min(surface["grid_a"]) >= float(X_train["BMI"].min())
    assert max(surface["grid_a"]) <= float(X_train["BMI"].max())
    assert min(surface["grid_b"]) >= float(X_train["Age"].min())
    assert max(surface["grid_b"]) <= float(X_train["Age"].max())


def test_a_sample_smaller_than_the_request_is_used_whole(additive):
    model, X_train = additive

    surface = interactions.two_way_partial_dependence(
        model, X_train.iloc[:12], "BMI", "Age", grid_points=3, sample_rows=1000
    )

    assert surface["sample_rows"] == 12


# ================================================================ ground truth

def test_the_statistic_finds_the_pair_no_one_dimensional_method_can_see(xor):
    """The test the whole module exists to pass.

    Neither XOR driver has a marginal association with the label, so partial
    dependence, occlusion and coefficients are all blind to the pair. The forest
    uses both. If this does not separate them from an inert pair, the audit adds
    nothing that the one-dimensional methods did not already provide.
    """
    model, X_train = xor

    drivers = interactions.measure_pair(
        model, X_train, "HighBP", "HighChol", grid_points=GRID, sample_rows=ROWS
    )
    inert = interactions.measure_pair(
        model, X_train, "Education", "PhysActivity", grid_points=GRID, sample_rows=ROWS
    )

    assert drivers.h_statistic > inert.h_statistic, (
        f"the XOR pair scored {drivers.h_statistic:.4f} against an inert pair's "
        f"{inert.h_statistic:.4f}"
    )
    assert drivers.h_statistic > 0.2
    assert drivers.excess_range > 0.0


def test_the_xor_pair_leads_a_bounded_sweep(xor):
    """Ranked first among the pairs it competes with, not merely above one pair."""
    model, X_train = xor
    features = ["HighBP", "HighChol", "BMI", "Education", "PhysActivity"]

    ranked = interactions.rank_interactions(
        model, X_train, features=features, grid_points=GRID, sample_rows=ROWS
    )

    assert {ranked[0].feature_a, ranked[0].feature_b} == {"HighBP", "HighChol"}
    assert len(ranked) == 10


def test_an_additive_model_reports_essentially_no_interaction(additive):
    """A linear model cannot represent one, so a large H would be an artefact."""
    model, X_train = additive

    result = interactions.measure_pair(
        model, X_train, "BMI", "Age", grid_points=GRID, sample_rows=ROWS
    )

    assert result.h_statistic < 0.05, (
        f"a linear model reported an interaction of {result.h_statistic:.4f}; it "
        "has no functional form that could produce one"
    )


def test_measuring_on_the_probability_scale_would_manufacture_an_interaction(additive):
    """The correction this module makes, shown rather than asserted.

    Interaction is scale-dependent. A logistic regression is additive in the
    logit by construction, and on the probability its scores are reported in it
    looks meaningfully non-additive - the sigmoid did that, not the model. Every
    probability-valued model in the zoo would carry the same floor, and the tree
    families, whose probabilities are vote averages rather than squashed sums,
    would be compared against it on a different footing.
    """
    model, X_train = additive

    def surface(log_odds):
        sample = X_train.iloc[:ROWS]
        grid_a = np.linspace(*np.percentile(X_train["BMI"], [2.5, 97.5]), GRID)
        grid_b = np.linspace(*np.percentile(X_train["Age"], [2.5, 97.5]), GRID)
        joint = np.empty((GRID, GRID))
        probe = sample.copy()
        for i, a in enumerate(grid_a):
            probe["BMI"] = a
            for j, b in enumerate(grid_b):
                probe["Age"] = b
                joint[i, j] = float(
                    np.mean(interactions._scores(model, probe, log_odds=log_odds))
                )

        def curve(feature, grid):
            values = np.empty(GRID)
            probe = sample.copy()
            for index, point in enumerate(grid):
                probe[feature] = point
                values[index] = float(
                    np.mean(interactions._scores(model, probe, log_odds=log_odds))
                )
            return values

        return {
            "joint": joint,
            "curve_a": curve("BMI", grid_a),
            "curve_b": curve("Age", grid_b),
        }

    on_log_odds = interactions.h_statistic(surface(True))
    on_probability = interactions.h_statistic(surface(False))

    assert on_log_odds < 0.05
    assert on_probability > 4 * on_log_odds, (
        f"the probability scale reported {on_probability:.4f} and the log-odds "
        f"scale {on_log_odds:.4f}; the link correction is no longer doing work"
    )


def test_a_model_that_emits_a_decision_function_is_left_on_its_own_scale():
    """Only probabilities get the link. An SVM margin is already unbounded.

    Sniffing the value range instead of reading the declaration would transform
    a decision function that happened to land inside [0, 1], which is the kind
    of mistake that produces a plausible number and no error.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = worlds.make(XaiWorld.ADDITIVE_TWO_FEATURE, rows=120, seed=2)
        margin_model = REGISTRY.build("linear_svm").fit(dataset.X, dataset.y)
        probability_model = REGISTRY.build("logistic_l2").fit(dataset.X, dataset.y)

    assert not interactions.emits_probabilities(margin_model)
    assert interactions.emits_probabilities(probability_model)

    raw = margin_model.decision_scores(dataset.X.iloc[:5])
    assert np.allclose(interactions._scores(margin_model, dataset.X.iloc[:5]), raw)


# ============================================================ bounds and shape

def test_a_sweep_is_truncated_at_the_declared_pair_budget(additive):
    """Budgeted runs must measure the same subset every time, not a new one."""
    model, X_train = additive
    features = ["BMI", "Age", "GenHlth", "Education"]

    ranked = interactions.rank_interactions(
        model, X_train, features=features, max_pairs=3,
        grid_points=3, sample_rows=20,
    )
    again = interactions.rank_interactions(
        model, X_train, features=features, max_pairs=3,
        grid_points=3, sample_rows=20,
    )

    assert len(ranked) == 3
    assert [(r.feature_a, r.feature_b) for r in ranked] == [
        (r.feature_a, r.feature_b) for r in again
    ]


def test_results_are_ordered_strongest_first_with_a_deterministic_tie_break(additive):
    model, X_train = additive

    ranked = interactions.rank_interactions(
        model, X_train, features=["BMI", "Age", "GenHlth", "Education"],
        grid_points=3, sample_rows=20,
    )
    values = [r.h_statistic for r in ranked]

    assert values == sorted(values, reverse=True)
    assert all(np.isfinite(v) for v in values)


def test_a_result_serialises_to_plain_data(additive):
    model, X_train = additive
    payload = interactions.measure_pair(
        model, X_train, "BMI", "Age", grid_points=3, sample_rows=20
    ).as_dict()

    assert payload["feature_a"] == "BMI"
    assert payload["grid_points"] == 3
    assert payload["sample_rows"] == 20
    assert payload["excess_range"] == pytest.approx(
        payload["joint_range"] - payload["additive_range"]
    )


# ================================================================== summaries

def test_summarising_no_pairs_reports_nothing_rather_than_crashing():
    summary = interactions.summarise([])

    assert summary["pairs"] == 0
    assert summary["strongest"] is None
    assert summary["additive_share"] is None


def test_a_summary_names_the_strongest_pair_and_how_additive_the_rest_are():
    results = [
        interactions.InteractionResult("a", "b", 0.4, 1.0, 0.6, 8, 100),
        interactions.InteractionResult("c", "d", 0.01, 1.0, 0.99, 8, 100),
        interactions.InteractionResult("e", "f", 0.02, 1.0, 0.98, 8, 100),
    ]

    summary = interactions.summarise(results)

    assert summary["pairs"] == 3
    assert summary["strongest"]["features"] == ["a", "b"]
    assert summary["max_h"] == pytest.approx(0.4)
    assert summary["additive_share"] == pytest.approx(2 / 3)


def test_the_summary_describes_the_fitted_surface_and_not_biology():
    """A departure from additivity is a property of what the model learned.

    Whether it reflects anything about diabetes is not answerable from a partial
    dependence grid, and the wording must not imply otherwise.
    """
    results = [interactions.InteractionResult("BMI", "Age", 0.4, 1.0, 0.6, 8, 100)]
    text = str(interactions.summarise(results)).lower()

    for forbidden in ("causes", "interact biologically", "treatment", "synergy"):
        assert forbidden not in text
