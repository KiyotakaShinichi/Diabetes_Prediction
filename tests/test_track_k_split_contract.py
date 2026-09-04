"""The Track K protocol is frozen, and the split it defines is one split.

A benchmark where two models are scored on different rows measures the split
rather than the models, and a protocol that can be edited after seeing results
is not a protocol. These tests pin both.

They are deliberately cheap: the split is derived from a committed CSV with no
model training involved, so the whole module runs in seconds and can guard every
commit.
"""
import json

import pytest

from conftest import REPO_ROOT
from ml_core import feature_contract
from research.track_k import protocol, split

PROTOCOL_DOC = REPO_ROOT / "docs" / "research" / "track_k_protocol.md"


@pytest.fixture(scope="module")
def frame():
    return split.load_dataset()


@pytest.fixture(scope="module")
def splits(frame):
    return split.build_split(frame)


# ============================================================ the protocol

def test_the_protocol_reuses_the_served_feature_contract():
    """Benchmarking different features would answer a different question."""
    assert protocol.FEATURE_NAMES == feature_contract.FEATURE_NAMES
    assert protocol.TARGET_COLUMN == feature_contract.TARGET_COLUMN


def test_the_primary_metric_is_one_the_evaluator_produces():
    assert protocol.PRIMARY_METRIC == "roc_auc"
    assert protocol.PRIMARY_METRIC not in protocol.SECONDARY_METRICS


def test_pr_auc_is_still_reported():
    """It was rejected as PRIMARY, not discarded - the choice stays auditable."""
    assert "pr_auc" in protocol.SECONDARY_METRICS


def test_every_family_has_a_distinct_derived_seed():
    seeds = [protocol.model_seed(family) for family in protocol.MODEL_FAMILIES]

    assert len(set(seeds)) == len(seeds)
    assert all(isinstance(seed, int) for seed in seeds)


def test_an_unknown_family_has_no_seed():
    with pytest.raises(ValueError, match="unknown model family"):
        protocol.model_seed("random_forest")


def test_the_families_partition_into_classical_and_deep():
    assert set(protocol.CLASSICAL_FAMILIES) | set(protocol.DEEP_FAMILIES) == set(
        protocol.MODEL_FAMILIES
    )
    assert not set(protocol.CLASSICAL_FAMILIES) & set(protocol.DEEP_FAMILIES)


def test_the_promotion_policy_demands_more_than_a_nonzero_delta():
    """An interval above zero alone would promote an arbitrarily small gain."""
    policy = protocol.PROMOTION_POLICY

    assert policy.min_primary_delta > 0
    assert policy.max_ece_regression > 0
    assert policy.max_recall_regression > 0
    assert policy.max_latency_multiple > 1


def test_only_three_verdicts_exist():
    """No "promising" escape hatch for an interval that contains zero."""
    assert set(protocol.VERDICTS) == {"PROMOTE", "REJECT", "INCONCLUSIVE"}
    assert set(protocol.COMPARISON_OUTCOMES) == {
        "CLEAR IMPROVEMENT", "CLEAR REGRESSION", "INCONCLUSIVE",
    }


# ============================================ the document and the code agree

def test_the_protocol_document_exists_and_states_its_version():
    text = PROTOCOL_DOC.read_text(encoding="utf-8")

    assert protocol.PROTOCOL_VERSION in text


def test_the_document_records_the_same_frozen_numbers():
    """Prose and constants are one contract; drift between them is a defect.

    Thousands separators are stripped before comparing: the document is written
    for a reader ("2,000 resamples") and the constant is written for a machine.
    """
    text = PROTOCOL_DOC.read_text(encoding="utf-8").replace(",", "")

    assert str(protocol.SPLIT_SEED) in text
    assert str(protocol.BOOTSTRAP_RESAMPLES) in text
    assert str(protocol.BOOTSTRAP_SEED) in text
    assert f"{protocol.PROMOTION_POLICY.min_primary_delta}" in text
    assert f"{protocol.PROMOTION_POLICY.max_ece_regression}" in text
    assert f"{protocol.PROMOTION_POLICY.max_recall_regression}" in text


def test_the_document_justifies_the_primary_metric_choice():
    """The brief forbids choosing ROC-AUC reflexively; the reasoning is recorded."""
    text = PROTOCOL_DOC.read_text(encoding="utf-8").lower()

    assert "pr-auc" in text
    assert "balanced" in text


def test_the_document_records_the_base_rate_limitation():
    """The engineered 50/50 balance is the study's largest caveat."""
    text = PROTOCOL_DOC.read_text(encoding="utf-8").lower()

    assert "base rate" in text or "base-rate" in text
    assert "prevalence" in text


# ================================================================ the split

def test_the_split_partitions_every_row_exactly_once(frame, splits):
    train, val, test = (set(part.index) for part in (splits.X_train, splits.X_val, splits.X_test))

    assert train | val | test == set(frame.index)
    assert len(train) + len(val) + len(test) == len(frame)


@pytest.mark.parametrize(("left", "right"), [("X_train", "X_val"), ("X_train", "X_test"), ("X_val", "X_test")])
def test_no_rows_overlap_between_partitions(splits, left, right):
    assert not set(getattr(splits, left).index) & set(getattr(splits, right).index)


def test_the_split_is_deterministic(frame):
    first = split.build_split(frame)
    second = split.build_split(frame)

    assert list(first.X_test.index) == list(second.X_test.index)
    assert list(first.X_train.index) == list(second.X_train.index)


def test_the_split_is_stratified(frame, splits):
    overall = float(frame[protocol.TARGET_COLUMN].mean())

    for part in (splits.y_train, splits.y_val, splits.y_test):
        assert float(part.mean()) == pytest.approx(overall, abs=0.01)


def test_the_split_uses_the_contract_feature_order(splits):
    assert splits.feature_names == feature_contract.FEATURE_NAMES
    assert list(splits.X_train.columns) == list(feature_contract.FEATURE_NAMES)


def test_the_proportions_match_the_protocol(frame, splits):
    total = len(frame)

    assert len(splits.X_test) == pytest.approx(total * protocol.TEST_SIZE, rel=0.01)
    expected_val = total * (1 - protocol.TEST_SIZE) * protocol.VALIDATION_SIZE_OF_REMAINDER
    assert len(splits.X_val) == pytest.approx(expected_val, rel=0.01)


# ========================================================= the fingerprint

def test_the_fingerprint_is_stable_across_runs(frame):
    first = split.fingerprint_split(split.build_split(frame))
    second = split.fingerprint_split(split.build_split(frame))

    assert first.combined_sha256 == second.combined_sha256


def test_the_fingerprint_distinguishes_different_row_membership(frame, splits):
    baseline = split.fingerprint_split(splits)
    shuffled = splits.X_test.index[::-1]

    assert split.hash_indices(shuffled) != baseline.test_indices_sha256


def test_the_manifest_records_identity_without_leaking_rows(splits, tmp_path):
    manifest = split.build_split_manifest(splits)
    written = split.write_split_manifest(splits, tmp_path / "split.json")
    text = written.read_text(encoding="utf-8")

    assert manifest["dataset"]["rows"] == 66877
    assert manifest["sizes"] == splits.sizes
    # Identity only: no index list, no feature value.
    assert "train_indices_sha256" in text
    assert '"indices"' not in text
    payload = json.loads(text)
    assert set(payload) == {
        "protocol_version", "dataset", "features", "split", "sizes", "class_balance"
    }


def test_the_manifest_records_the_class_balance(splits):
    balance = split.build_split_manifest(splits)["class_balance"]

    for part in ("train", "val", "test"):
        assert balance[part]["positive_rate"] == pytest.approx(0.4995, abs=0.01)


# ==================================================== fail-closed behaviour

def test_an_unchanged_split_verifies_clean(splits):
    assert split.verify_split(splits, split.build_split_manifest(splits)) == []


def test_a_changed_dataset_fingerprint_fails_closed(splits):
    manifest = split.build_split_manifest(splits)
    manifest["split"]["dataset_sha256"] = "0" * 64

    problems = split.verify_split(splits, manifest)

    assert any("dataset_sha256" in problem for problem in problems)


def test_changed_row_membership_fails_closed(splits):
    manifest = split.build_split_manifest(splits)
    manifest["split"]["test_indices_sha256"] = "0" * 64

    problems = split.verify_split(splits, manifest)

    assert any("test_indices_sha256" in problem for problem in problems)


def test_a_protocol_version_change_fails_closed(splits):
    manifest = split.build_split_manifest(splits)
    manifest["protocol_version"] = "0.0.1"

    problems = split.verify_split(splits, manifest)

    assert any("protocol_version" in problem for problem in problems)


def test_loading_against_a_drifted_manifest_raises(splits):
    manifest = split.build_split_manifest(splits)
    manifest["split"]["train_indices_sha256"] = "0" * 64

    with pytest.raises(split.SplitIntegrityError, match="no longer reproduces"):
        split.load_frozen_split(manifest)


def test_loading_against_a_matching_manifest_succeeds(splits):
    manifest = split.build_split_manifest(splits)

    reloaded = split.load_frozen_split(manifest)

    assert list(reloaded.X_test.index) == list(splits.X_test.index)


# ================================================ research/production boundary

@pytest.mark.parametrize(
    "path", ["research/track_k/protocol.py", "research/track_k/split.py"]
)
def test_research_code_never_touches_the_production_artifact_namespace(path):
    """Research outputs must be impossible to confuse with deployed models."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")

    assert "model_artifacts" not in text, f"{path} references the production artifact directory"
    assert "legacy_artifact_attestation" not in text
