"""The vocabulary boundary, enforced on generated text rather than on review.

Track M produces documents nobody reads line by line: a card per model, a card
per method, a report per run. That is exactly where a causal claim gets in.
"BMI is the most important feature" is a defensible statement about a model's
attribution and one word away from a statement about diabetes, and the drift
happens in generated prose long before it happens in a paper.

So the rules are asserted against the rendered output.

**Allowed**: model attribution, feature importance, association, prediction
dependence - claims about what a model's output depends on.

**Not allowed without evidence this project does not have**: that a feature
causes diabetes, that changing a feature would reduce anyone's risk, or anything
that reads as a treatment recommendation.

Two further rules that are easy to lose. Every card must carry limitations - a
card listing only capabilities is marketing, and the capability gaps are the
finding. And no agreement band may present itself as a confidence or a
probability, because nothing in this package estimates the chance an explanation
is correct.

The banned-phrase list is deliberately about *claims*, not about words. "Risk"
appears throughout this project's vocabulary and is fine; "reduce your risk" is
not. Matching whole phrases keeps the test pointed at the failure it exists to
catch instead of at a thesaurus.

The matching is also **negation-aware**, and it has to be. The strongest
disclaimer this project writes is "none of it supports a claim that a feature
causes diabetes", which contains the exact phrase the test forbids. A naive
substring check fails on the sentence that does the most to keep the boundary -
the same shape of bug Track K hit when a docstring mentioning a protected path
tripped a production-boundary test. So the text is split into sentences and a
sentence carrying a negation is not treated as making the claim inside it.

That relaxation could obviously be used to smuggle a claim through, so
`test_the_checker_catches_a_claim_that_is_actually_made` asserts the checker
still fires on an affirmative sentence, and
`test_the_checker_is_not_fooled_by_a_stray_negation_elsewhere` pins the
sentence-level granularity.
"""
import json
import re
import warnings

import pytest

from research.model_zoo.contracts import ResearchStatus
from research.model_zoo.registry import REGISTRY
from research.xai import cards, hard_region, report
from research.xai.contracts import AgreementLabel, DisagreementLabel
from research.xai.registry import METHODS

#: Claims this project cannot support from an attribution. Phrases, not words:
#: the failure being guarded against is a claim about the world, and a single
#: banned word would flag correct sentences and teach people to route around
#: the test.
FORBIDDEN_CLAIMS = (
    "causes diabetes",
    "cause diabetes",
    "causal effect on",
    "will reduce risk",
    "will reduce your risk",
    "reduce their risk",
    "lowers risk",
    "should be treated",
    "treatment recommendation",
    "we recommend treating",
    "prescribe",
    "diagnose",
    "diagnostic tool for",
    "clinically actionable",
    "proves that",
)

#: Words that would misrepresent a descriptive band as a calibrated quantity.
FORBIDDEN_CERTAINTY = ("confidence that", "probability that the explanation", "certainty")

#: Markers that turn a sentence into a disclaimer rather than a claim. Kept
#: short and explicit: a longer list would start excusing sentences that are
#: making the claim while merely containing a stray "no".
NEGATIONS = (
    " not ", " no ", "n't", "never", "nothing", "none ", "cannot", "without",
    "refuses", "unable",
)


def _asserted_sentences(text: str) -> list[str]:
    """Sentences that make a claim, dropping the ones that disclaim one.

    Sentence-level rather than document-level, so a disclaimer in one paragraph
    cannot license a claim in the next.
    """
    return [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text.lower())
        if not any(marker in f" {sentence} " for marker in NEGATIONS)
    ]


def _claims_found(text: str, phrases) -> list[str]:
    """Every forbidden phrase that appears in a sentence actually asserting it."""
    sentences = _asserted_sentences(text)
    return [
        phrase
        for phrase in phrases
        if any(phrase in sentence for sentence in sentences)
    ]


def _documents():
    """Every document Track M can generate, with a name for the failure message."""
    manifest = _synthetic_manifest()
    return {
        "capability table": cards.render_capability_table(),
        "method cards": cards.render_method_cards(),
        "model cards": cards.render_model_cards(),
        "all cards": cards.render_all(),
        "run report": report.render(manifest),
        "limits section": report.render_limits(),
    }


def _synthetic_manifest():
    """A manifest shaped like a real one, so the report renders every section.

    Built rather than run: this file is about wording, and a real run would make
    it slow and would make a language failure depend on a model fitting.
    """
    features = [
        "GenHlth", "BMI", "Age", "HighBP", "HighChol",
        "DiffWalk", "PhysHlth", "Education", "PhysActivity", "HeartDiseaseorAttack",
    ]
    return {
        "run_id": "xai-fixture",
        "evidence_class": "RESOURCE_CONSTRAINED_EXPLORATORY",
        "train_rows": 1000,
        "case_limit": 40,
        "models_requested": ["logistic_l2", "random_forest"],
        "methods_requested": ["coefficients", "permutation_importance"],
        "fit_failures": {"catboost": "ImportError: not installed"},
        "counts": {"success": 40, "unsupported": 12, "numerical_failure": 1},
        "provenance": {
            "train_rows": 1000,
            "evaluation_rows": 2000,
            "case_rows": 40,
            "baseline_source": "median of the fitting rows",
            "permutation_scored_on": "validation partition, never the fitting rows or test",
        },
        "analysis": {
            "records": 40,
            "zoo_consensus": features,
            "zoo_mean_ranks": {name: 1.0 + index * 0.2 for index, name in enumerate(features)},
            "within_model": {"pairs": 10, "mean_spearman": 0.6, "median_spearman": 0.7,
                             "min_spearman": 0.1, "top_1_agreement_rate": 0.8},
            "within_family": {"pairs": 4, "mean_spearman": 0.8, "median_spearman": 0.8,
                              "min_spearman": 0.5, "top_1_agreement_rate": 0.9},
            "between_families": {
                "pairs": 20, "mean_spearman": 0.5, "median_spearman": 0.55,
                "min_spearman": -0.2, "top_1_agreement_rate": 0.65,
                "worst_pair": {"left": "a/coefficients", "right": "b/native_importance",
                               "spearman": -0.2},
            },
        },
        "interactions": {
            "random_forest": {
                "features": features[:5], "pairs": 10,
                "strongest": {"features": ["BMI", "Age"], "h_statistic": 0.3,
                              "excess_range": 0.1},
                "mean_h": 0.1, "median_h": 0.08, "max_h": 0.3, "additive_share": 0.6,
            },
            "broken_model": {"error": "ValueError: deliberate", "features": features[:5]},
        },
    }


# ============================================================== the boundary

@pytest.mark.parametrize("name", list(_documents()))
def test_no_generated_document_makes_a_causal_or_clinical_claim(name):
    """The rule the whole track is bounded by, checked where prose is produced."""
    found = _claims_found(_documents()[name], FORBIDDEN_CLAIMS)

    assert not found, (
        f"the {name} asserts {found}; attribution describes model dependence "
        "and cannot support it"
    )


@pytest.mark.parametrize("name", list(_documents()))
def test_no_generated_document_dresses_a_band_as_a_probability(name):
    found = _claims_found(_documents()[name], FORBIDDEN_CERTAINTY)

    assert not found, (
        f"the {name} uses {found}; nothing here estimates the chance an "
        "explanation is correct"
    )


def test_the_checker_catches_a_claim_that_is_actually_made():
    """Test the test. A negation-aware checker that never fires is decoration."""
    assert _claims_found(
        "GenHlth causes diabetes in this population.", FORBIDDEN_CLAIMS
    ) == ["causes diabetes"]

    assert _claims_found(
        "Lowering BMI will reduce risk for these patients.", FORBIDDEN_CLAIMS
    ) == ["will reduce risk"]


def test_the_checker_is_not_fooled_by_a_stray_negation_elsewhere():
    """Granularity is per sentence, so a disclaimer cannot license a later claim."""
    text = (
        "This is not a clinical tool. GenHlth causes diabetes."
    )

    assert _claims_found(text, FORBIDDEN_CLAIMS) == ["causes diabetes"]


def test_the_checker_accepts_the_disclaimer_it_was_built_to_accept():
    """The sentence that does the most to keep the boundary must survive it."""
    disclaimer = (
        "None of it supports a claim that a feature causes diabetes, that "
        "changing a feature would change a person's risk, or any treatment "
        "recommendation."
    )

    assert _claims_found(disclaimer, FORBIDDEN_CLAIMS) == []


def test_the_allowed_vocabulary_is_actually_used():
    """A test that only forbids can be passed by saying nothing.

    The documents have to make the claims they are entitled to make, or the
    boundary is being kept by silence rather than by precision.
    """
    text = cards.render_all().lower() + report.render(_synthetic_manifest()).lower()

    for phrase in ("attribution", "association", "depends on", "importance"):
        assert phrase in text, f"the documents never use the allowed term {phrase!r}"


# ============================================================== card contents

@pytest.mark.parametrize("method_id", METHODS.ids())
def test_every_method_card_states_how_the_method_misleads(method_id):
    """A card listing no limitation is a marketing document."""
    card = cards.render_method_card(METHODS.get(method_id))

    assert "**Limitations.**" in card
    assert METHODS.get(method_id).failure_modes, f"{method_id} declares no failure mode"
    for mode in METHODS.get(method_id).failure_modes:
        assert mode.split(".")[0][:40] in card


@pytest.mark.parametrize(
    "model_id",
    [s.model_id for s in REGISTRY if s.effective_status() is ResearchStatus.ACTIVE],
)
def test_every_model_card_names_what_it_cannot_do_and_why(model_id):
    """A gap is a property of the model, so the card has to say which and why."""
    card = cards.render_model_card(model_id)

    assert "**Explanations unavailable.**" in card
    assert "**Why those gaps exist.**" in card
    assert "**Limitations.**" in card


def test_every_document_carries_the_research_scope_note():
    """The caveat most likely to be lost is the one on the page that gets copied."""
    for name in ("method cards", "model cards", "all cards", "run report"):
        assert cards.SCOPE_NOTE in _documents()[name], f"{name} lost the scope note"


def test_the_report_states_what_it_cannot_show():
    limits = report.render_limits()

    for expected in ("Not causation", "Not correctness", "Not a promotion", "Not clinical"):
        assert expected in limits


def test_the_report_puts_the_failures_before_the_agreement_figures():
    """Order is an argument. The denominator has to arrive before the number."""
    text = report.render(_synthetic_manifest())

    assert text.index("What was answerable") < text.index("Do the explanations agree?")


def test_the_report_refuses_to_print_a_correlation_without_its_caveat():
    text = report.render(_synthetic_manifest())

    assert "Read the top-1 column, not the Spearman column." in text


def test_the_report_names_a_near_tie_as_a_near_tie():
    """A consensus presents first and second as firmly however close they were."""
    text = report.render(_synthetic_manifest())

    assert "Ordered, but barely." in text


def test_a_model_that_would_not_fit_is_named_in_the_report():
    text = report.render(_synthetic_manifest())

    assert "catboost" in text
    assert "absent from every figure below" in text


def test_a_failed_interaction_sweep_is_shown_as_failed():
    text = report.render(_synthetic_manifest())

    assert "measurement failed" in text


def test_the_agreement_labels_are_adjectives_not_probabilities():
    for label in (*AgreementLabel, *DisagreementLabel):
        assert not re.search(r"\d", label.value)
        assert "confidence" not in label.value
        assert "probability" not in label.value


def test_the_hard_region_reading_offers_an_association_and_not_a_mechanism():
    """The most tempting place in the track to slip, so it is checked twice."""
    contrast = hard_region.RegionContrast(
        hard_rows=10, easy_rows=10,
        hard_profile=(0.5, 0.5), easy_profile=(0.5, 0.5),
        feature_names=("a", "b"),
        profile_agreement=1.0,
        hard_concentration=0.4, easy_concentration=0.4,
    )
    payload = hard_region.summarise(contrast)
    text = json.dumps(payload)

    assert "association" in text.lower()
    assert _claims_found(text, FORBIDDEN_CLAIMS) == []


def test_the_scope_note_itself_stays_inside_the_boundary():
    """The note is prepended to everything, so an error in it would be everywhere."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = cards.SCOPE_NOTE.lower()

    assert "not a diagnostic tool" in text
    assert "not clinical guidance" in text
    assert "association and not a cause" in text
