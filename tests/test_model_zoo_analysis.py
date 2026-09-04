"""Agreement analysis and generated documentation.

The analysis answers Track K's open question at a wider scale: do thirty
algorithms fail on the same patients? A measurement that important has to be
checked against cases whose answer is known, because "the models all agree" is
also what a broken agreement metric says.

So every statistic here is tested against constructed inputs - identical
models, opposite models, independent models - where the right answer is
arithmetic rather than opinion.
"""
import json

import numpy as np
import pytest

from research.model_zoo import analysis, cards
from research.model_zoo.contracts import Family, ProbabilityBehavior
from research.model_zoo.registry import REGISTRY

pytest.importorskip("torch", reason="the zoo's deep models need PyTorch")


@pytest.fixture
def truth():
    rng = np.random.default_rng(3)
    return rng.integers(0, 2, 400)


# ======================================================== error overlap

def test_two_identical_models_share_every_error(truth):
    predictions = {"a": truth.copy(), "b": truth.copy()}
    predictions["a"][:40] = 1 - predictions["a"][:40]
    predictions["b"] = predictions["a"].copy()

    pair = analysis.error_overlap(truth, predictions)[0]

    assert pair.jaccard == pytest.approx(1.0)
    assert pair.disagreement_rate == pytest.approx(0.0)
    assert pair.left_errors_shared == pytest.approx(1.0)


def test_two_models_with_disjoint_errors_share_none(truth):
    a, b = truth.copy(), truth.copy()
    a[:40] = 1 - a[:40]
    b[40:80] = 1 - b[40:80]

    pair = analysis.error_overlap(truth, {"a": a, "b": b})[0]

    assert pair.jaccard == pytest.approx(0.0)
    assert pair.left_errors_shared == pytest.approx(0.0)


def test_a_perfect_model_has_no_errors_to_share(truth):
    """Dividing by an empty error set must give nan, not a crash or a zero."""
    wrong = truth.copy()
    wrong[:50] = 1 - wrong[:50]

    pair = analysis.error_overlap(truth, {"perfect": truth.copy(), "wrong": wrong})[0]

    assert np.isnan(pair.left_errors_shared)
    assert pair.right_errors_shared == pytest.approx(0.0)


def test_every_pair_is_reported_once(truth):
    predictions = {name: truth.copy() for name in ("a", "b", "c", "d")}

    pairs = analysis.error_overlap(truth, predictions)

    assert len(pairs) == 6
    assert len({frozenset((p.left, p.right)) for p in pairs}) == 6


# ==================================================== score correlation

def test_identical_scores_correlate_perfectly():
    scores = np.linspace(0, 1, 100)

    matrix = analysis.score_correlation({"a": scores, "b": scores.copy()})

    assert matrix["a"]["b"] == pytest.approx(1.0)
    assert matrix["a"]["a"] == pytest.approx(1.0)


def test_reversed_scores_correlate_negatively():
    scores = np.linspace(0, 1, 100)

    matrix = analysis.score_correlation({"up": scores, "down": scores[::-1].copy()})

    assert matrix["up"]["down"] == pytest.approx(-1.0)


def test_correlation_is_rank_based_not_scale_based():
    """The zoo mixes probabilities with squashed margins; only order matters."""
    probabilities = np.linspace(0.1, 0.9, 100)
    margins = np.log(probabilities / (1 - probabilities)) * 37.0

    matrix = analysis.score_correlation({"p": probabilities, "m": margins})

    assert matrix["p"]["m"] == pytest.approx(1.0)


# ==================================================== family diversity

def test_diversity_reports_within_and_between_family_disagreement(truth):
    rng = np.random.default_rng(7)
    predictions = {
        "logistic_l2": truth.copy(),
        "logistic_l1": truth.copy(),
        "random_forest": rng.integers(0, 2, len(truth)),
        "decision_tree": rng.integers(0, 2, len(truth)),
    }

    diversity = analysis.family_diversity(truth, predictions)

    assert diversity["mean_within_family_disagreement"] is not None
    assert diversity["mean_between_family_disagreement"] is not None
    assert "linear" in diversity["within_family_by_family"]


def test_diversity_reading_says_when_families_converge(truth):
    """All models agreeing must produce the convergence reading, not silence."""
    predictions = {
        # Two linear models, so a within-family pair exists to compare against.
        "logistic_l2": truth.copy(),
        "logistic_l1": truth.copy(),
        "random_forest": truth.copy(),
        "mlp": truth.copy(),
    }

    diversity = analysis.family_diversity(truth, predictions)

    assert "feature set rather than the model class" in diversity["interpretation"]


def test_diversity_reading_says_when_families_diverge(truth):
    """And the opposite reading when they genuinely disagree."""
    rng = np.random.default_rng(11)
    predictions = {
        "logistic_l2": truth.copy(),
        "logistic_l1": truth.copy(),
        "random_forest": rng.integers(0, 2, len(truth)),
        "decision_tree": rng.integers(0, 2, len(truth)),
    }

    diversity = analysis.family_diversity(truth, predictions)

    assert "different" in diversity["interpretation"]


def test_diversity_needs_more_than_one_family_to_compare(truth):
    diversity = analysis.family_diversity(
        truth, {"logistic_l2": truth.copy(), "logistic_l1": truth.copy()}
    )

    assert diversity["mean_between_family_disagreement"] is None
    assert "not enough models" in diversity["interpretation"]


def test_unregistered_models_are_ignored_rather_than_crashing(truth):
    """An old run may name a model the registry no longer holds."""
    diversity = analysis.family_diversity(
        truth, {"logistic_l2": truth.copy(), "some_retired_model": truth.copy()}
    )

    assert diversity is not None


# ======================================================== hardest rows

def test_rows_every_model_gets_wrong_are_counted(truth):
    a, b = truth.copy(), truth.copy()
    a[:30] = 1 - a[:30]
    b[:30] = 1 - b[:30]

    hardest = analysis.hardest_rows(truth, {"a": a, "b": b})

    assert hardest["rows_every_model_got_wrong"] == 30
    assert hardest["rows_every_model_got_right"] == len(truth) - 30
    assert hardest["model_count"] == 2


def test_the_hardest_rows_are_ranked_by_how_many_models_missed_them(truth):
    a, b, c = truth.copy(), truth.copy(), truth.copy()
    a[0] = 1 - a[0]
    b[0] = 1 - b[0]
    c[0] = 1 - c[0]
    a[1] = 1 - a[1]

    hardest = analysis.hardest_rows(truth, {"a": a, "b": b, "c": c}, top=2)

    assert hardest["hardest_row_indices"][0] == 0
    assert hardest["hardest_row_error_counts"][0] == 3


def test_the_summary_reports_overlap_and_correlation(truth):
    rng = np.random.default_rng(2)
    predictions = {
        "logistic_l2": truth.copy(),
        "logistic_l1": truth.copy(),
        "random_forest": rng.integers(0, 2, len(truth)),
    }
    scores = {name: rng.random(len(truth)) for name in predictions}

    text = analysis.summarise(truth, predictions, scores)

    assert "mean pairwise error Jaccard" in text
    assert "Score rank correlation" in text
    assert "least correlated pair" in text


def test_the_summary_survives_a_model_set_with_no_within_family_pair(truth):
    """One model per family leaves within-family disagreement undefined.

    That is a perfectly valid run - it is what --models logistic_l2 mlp gives -
    and formatting the undefined statistic used to raise a TypeError and take
    the whole summary down with it.
    """
    rng = np.random.default_rng(4)
    predictions = {"logistic_l2": truth.copy(), "mlp": rng.integers(0, 2, len(truth))}

    text = analysis.summarise(truth, predictions, {})

    assert "undefined (too few pairs)" in text


# ======================================================= capability matrix

def test_the_matrix_covers_every_registered_model():
    rows = cards.capability_matrix()

    assert {row["model_id"] for row in rows} == set(REGISTRY.ids())


def test_the_matrix_and_the_registry_cannot_disagree():
    """There is one roster; the matrix is a projection of it."""
    rows = {row["model_id"]: row for row in cards.capability_matrix()}

    for spec in REGISTRY:
        row = rows[spec.model_id]
        assert row["family"] == spec.family.value
        assert row["supports_predict_proba"] == spec.capabilities.supports_predict_proba
        assert row["status"] == spec.effective_status().value


def test_the_rendered_table_lists_every_model():
    table = cards.render_capability_table()

    for spec in REGISTRY:
        assert spec.display_name in table


# ============================================================ model cards

@pytest.fixture
def completed_manifest():
    return {
        "run_id": "zoo-test",
        "evidence_class": "RESOURCE_CONSTRAINED_EXPLORATORY",
        "train_rows": 1000,
        "results": [
            {
                "model_id": "logistic_l2",
                "family": "linear",
                "outcome": "completed",
                "metrics": {"roc_auc": 0.81, "pr_auc": 0.78, "recall": 0.75,
                            "brier_score": 0.18, "ece": 0.02},
                "training": {"fit_seconds": 0.1, "training_rows": 1000,
                             "parameter_count": None},
                "serialization": {"format": "joblib", "bytes_written": 2000,
                                  "round_trip_ok": True},
                "error": None,
            },
            {
                "model_id": "catboost",
                "family": "boosting",
                "outcome": "skipped",
                "metrics": {},
                "training": {},
                "serialization": {},
                "error": "optional dependency 'catboost' is not installed",
            },
        ],
    }


def test_a_card_carries_the_evidence_class(completed_manifest):
    card = cards.build_card(completed_manifest["results"][0], completed_manifest)

    assert card["evidence_class"] == "RESOURCE_CONSTRAINED_EXPLORATORY"


def test_a_card_states_why_the_model_is_in_the_zoo(completed_manifest):
    card = cards.build_card(completed_manifest["results"][0], completed_manifest)

    assert card["why_included"] == REGISTRY.get("logistic_l2").rationale
    assert card["assumptions"]


def test_a_card_for_a_skipped_model_says_so_rather_than_showing_blanks(completed_manifest):
    """A model that did not run must not get a card that looks like a result."""
    card = cards.build_card(completed_manifest["results"][1], completed_manifest)

    assert card["status"] == "skipped"
    assert "not installed" in card["error"]
    assert any("Did not complete" in item for item in card["limitations"])


def test_every_card_carries_the_training_budget_caveat(completed_manifest):
    for result in completed_manifest["results"]:
        card = cards.build_card(result, completed_manifest)
        joined = " ".join(card["limitations"])

        assert "1,000 rows" in joined
        assert "Track K" in joined


def test_a_card_explains_what_its_probabilities_mean(completed_manifest):
    for model_id in ("logistic_l2", "gaussian_nb", "linear_svm", "nearest_centroid"):
        spec = REGISTRY.get(model_id)
        result = dict(completed_manifest["results"][0], model_id=model_id,
                      family=spec.family.value)
        card = cards.build_card(result, completed_manifest)

        assert card["calibration"]["behavior"] == spec.probability_behavior.value
        assert len(card["calibration"]["note"]) > 30


def test_an_uncalibrated_model_is_flagged_in_its_limitations(completed_manifest):
    spec = REGISTRY.get("gaussian_nb")
    assert spec.probability_behavior is ProbabilityBehavior.NATIVE_UNCALIBRATED

    result = dict(completed_manifest["results"][0], model_id="gaussian_nb",
                  family=spec.family.value)
    card = cards.build_card(result, completed_manifest)

    assert any("uncalibrated" in item.lower() for item in card["limitations"])


def test_a_failed_round_trip_appears_in_the_limitations(completed_manifest):
    result = dict(
        completed_manifest["results"][0],
        serialization={"format": "joblib", "bytes_written": 1,
                       "round_trip_ok": False, "error": "state lost"},
    )
    card = cards.build_card(result, completed_manifest)

    assert any("serialization round trip" in item for item in card["limitations"])


def test_cards_render_to_markdown(completed_manifest):
    card = cards.build_card(completed_manifest["results"][0], completed_manifest)

    text = cards.render_card(card)

    assert "logistic_l2" in text
    assert "**Limitations**" in text
    assert "0.81000" in text


def test_generate_writes_the_matrix_and_every_card(tmp_path, completed_manifest):
    path = cards.generate(completed_manifest, tmp_path)

    assert path.is_file()
    matrix = json.loads((tmp_path / "capability_matrix.json").read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "model_cards.json").read_text(encoding="utf-8"))

    assert len(matrix) == len(REGISTRY)
    assert {card["model_id"] for card in written} == {"logistic_l2", "catboost"}
    assert (tmp_path / "capability_matrix.md").is_file()


def test_generated_cards_never_invent_a_metric(tmp_path, completed_manifest):
    """A card's numbers must come from the run, not from a default."""
    card = cards.build_card(completed_manifest["results"][1], completed_manifest)

    assert all(value is None for value in card["metrics"].values())


def test_every_family_has_a_stated_assumption():
    """A new family without one would produce a card that explains nothing."""
    for family in Family:
        result = {"model_id": "logistic_l2", "family": family.value,
                  "outcome": "completed", "metrics": {}, "training": {},
                  "serialization": {}, "error": None}
        manifest = {"run_id": "x", "evidence_class": "E", "train_rows": 1, "results": []}
        # build_card reads the spec's family, so assert the mapping directly.
        assert cards._assumptions(family, ProbabilityBehavior.NATIVE_PROBABILISTIC)
        assert result["family"] == family.value
        assert manifest["train_rows"] == 1


# ================================================ the document and the code

def test_the_protocol_document_names_every_family():
    """A family the documentation omits is a design nobody reviewed."""
    from conftest import REPO_ROOT

    text = (REPO_ROOT / "docs" / "research" / "track_l_model_zoo.md").read_text(
        encoding="utf-8"
    )

    for family in Family:
        assert family.value.capitalize() in text or family.value in text.lower(), (
            f"{family.value} is registered but absent from the protocol document"
        )


def test_the_documents_state_the_evidence_class():
    """Track L must never be readable as superseding Track K."""
    from conftest import REPO_ROOT

    docs = REPO_ROOT / "docs" / "research"
    for name in ("track_l_model_zoo.md", "track_l_results.md"):
        text = (docs / name).read_text(encoding="utf-8")
        assert "RESOURCE_CONSTRAINED_EXPLORATORY" in text, name
        assert "Track K" in text, name


def test_the_results_document_records_the_failed_model():
    """A failure that vanishes from the write-up is a flattering write-up."""
    from conftest import REPO_ROOT

    text = (REPO_ROOT / "docs" / "research" / "track_l_results.md").read_text(
        encoding="utf-8"
    )

    assert "sgd_modified_huber" in text
    assert "skipped" in text.lower()
    assert "undefined" in text.lower(), "the hard-label model's metrics must be named"


def test_the_experimental_control_is_labelled_in_the_documents():
    from conftest import REPO_ROOT

    text = (REPO_ROOT / "docs" / "research" / "track_l_model_zoo.md").read_text(
        encoding="utf-8"
    )

    assert "EXPERIMENTAL_INDUCTIVE_BIAS_BASELINE" in text
    assert "not" in text and "production candidate" in text
