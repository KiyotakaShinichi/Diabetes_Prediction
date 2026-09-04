"""The deterministic training subsets the resource-constrained arm trains on.

The properties under test are the ones that make a sample-efficiency curve
interpretable: the subsets are nested, stratified, drawn only from train,
identical on every run, and fingerprinted so they cannot silently drift.

A subset that quietly changed between runs would turn every comparison in the
constrained arm into a comparison of two different datasets, which is exactly
the failure this module exists to prevent.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from ml_core import training as core_training
from research.track_k import protocol, split, subsets


@pytest.fixture(scope="module")
def splits() -> core_training.TrainingSplits:
    return split.build_split(split.load_dataset())


@pytest.fixture(scope="module")
def ladder(splits) -> dict[int, pd.Index]:
    return subsets.build_nested_subsets(splits)


# ==================================================== the nesting property

def test_every_subset_is_contained_in_the_next_larger_one(ladder):
    """More data at each step, not different data."""
    sizes = sorted(ladder)

    for smaller, larger in itertools.pairwise(sizes):
        assert ladder[smaller].isin(ladder[larger]).all(), (
            f"the {smaller}-row subset is not contained in the {larger}-row subset"
        )


def test_each_subset_has_the_size_it_was_asked_for(ladder):
    for size, index in ladder.items():
        assert len(index) == size


def test_the_ladder_covers_the_declared_sizes(ladder):
    assert sorted(ladder) == sorted(protocol.SAMPLE_EFFICIENCY_SIZES)


# ======================================================= stratification

def test_every_subset_preserves_the_parent_positive_rate(splits, ladder):
    """A drifted base rate would move calibration for the wrong reason."""
    parent = float(splits.y_train.mean())

    for size, index in ladder.items():
        rate = float(splits.y_train.loc[index].mean())
        assert rate == pytest.approx(parent, abs=0.01), f"{size}-row subset drifted to {rate}"


def test_the_smallest_subset_is_still_balanced(splits, ladder):
    """500 rows is where unstratified sampling would visibly drift."""
    target = splits.y_train.loc[ladder[500]]

    assert int(target.sum()) == 250


# ================================================= partition containment

def test_subsets_are_drawn_only_from_the_training_partition(splits, ladder):
    for size, index in ladder.items():
        assert index.isin(splits.X_train.index).all(), f"{size}-row subset escaped train"


def test_no_subset_row_comes_from_validation_or_test(splits, ladder):
    """The selection partition and the held-out partition stay untouched."""
    for size, index in ladder.items():
        assert not index.isin(splits.X_val.index).any(), f"{size}-row subset touched validation"
        assert not index.isin(splits.X_test.index).any(), f"{size}-row subset touched test"


# ========================================================= determinism

def test_building_the_ladder_twice_gives_the_same_rows(splits):
    first = subsets.build_nested_subsets(splits)
    second = subsets.build_nested_subsets(splits)

    for size in first:
        assert first[size].equals(second[size])


def test_a_different_seed_selects_different_rows(splits, ladder):
    """Otherwise the seed would be decorative."""
    other = subsets.build_nested_subsets(splits, seed=protocol.SUBSET_SEED + 1)

    assert not other[5000].equals(ladder[5000])
    assert len(other[5000]) == len(ladder[5000])


def test_membership_not_draw_order_defines_a_subset(ladder):
    """Sorted indices, so two runs selecting the same rows fingerprint alike."""
    for index in ladder.values():
        assert list(index) == sorted(index)


# ============================================================ narrowing

def test_taking_a_subset_narrows_train_and_nothing_else(splits, ladder):
    narrowed = subsets.take(splits, ladder[5000])

    assert len(narrowed.X_train) == 5000
    assert len(narrowed.y_train) == 5000
    assert narrowed.X_val.equals(splits.X_val)
    assert narrowed.X_test.equals(splits.X_test)
    assert narrowed.y_test.equals(splits.y_test)


def test_the_narrowed_split_keeps_feature_order(splits, ladder):
    narrowed = subsets.take(splits, ladder[1000])

    assert list(narrowed.X_train.columns) == list(splits.X_train.columns)
    assert narrowed.feature_names == splits.feature_names


def test_features_and_target_stay_aligned_after_narrowing(splits, ladder):
    narrowed = subsets.take(splits, ladder[2500])

    assert narrowed.X_train.index.equals(narrowed.y_train.index)
    assert narrowed.y_train.equals(splits.y_train.loc[ladder[2500]])


def test_narrowing_to_rows_outside_train_is_refused(splits, ladder):
    """A subset containing a test row must fail loudly, not train on it."""
    contaminated = ladder[500].append(splits.X_test.index[:1])

    with pytest.raises(subsets.SubsetIntegrityError, match="not in the train partition"):
        subsets.take(splits, contaminated)


def test_a_subset_larger_than_the_train_partition_is_refused(splits):
    with pytest.raises(ValueError, match="exceeds"):
        subsets.build_nested_subsets(splits, sizes=(len(splits.X_train) + 1,))


# ========================================================= fingerprints

def test_a_fingerprint_records_the_partition_it_came_from(splits, ladder):
    fingerprint = subsets.fingerprint_subset(
        ladder[5000], splits, requested_size=5000
    ).as_dict()

    assert fingerprint["parent_partition"] == "train"
    assert fingerprint["parent_rows"] == len(splits.X_train)
    assert fingerprint["size"] == 5000
    assert len(fingerprint["indices_sha256"]) == 64
    assert len(fingerprint["dataset_sha256"]) == 64


def test_fingerprints_differ_between_sizes(splits, ladder):
    digests = {
        size: subsets.fingerprint_subset(index, splits, requested_size=size).indices_sha256
        for size, index in ladder.items()
    }

    assert len(set(digests.values())) == len(digests)


def test_a_manifest_describes_every_size(splits):
    manifest = subsets.build_subset_manifest(splits)

    assert manifest["drawn_from"] == "train"
    assert manifest["nested"] is True
    assert sorted(int(k) for k in manifest["subsets"]) == sorted(
        protocol.SAMPLE_EFFICIENCY_SIZES
    )


# ========================================================== fail closed

def test_an_unchanged_ladder_verifies(splits):
    manifest = subsets.build_subset_manifest(splits)

    assert subsets.verify_subsets(splits, manifest) == []


def test_a_tampered_membership_hash_is_detected(splits):
    manifest = subsets.build_subset_manifest(splits)
    manifest["subsets"]["5000"]["indices_sha256"] = "0" * 64

    problems = subsets.verify_subsets(splits, manifest)

    assert any("5000 indices_sha256" in problem for problem in problems)


def test_a_changed_parent_partition_is_detected(splits):
    """If the split moved, the recorded subset describes rows that are gone."""
    manifest = subsets.build_subset_manifest(splits)
    manifest["subsets"]["500"]["parent_indices_sha256"] = "0" * 64

    problems = subsets.verify_subsets(splits, manifest)

    assert any("parent_indices_sha256" in problem for problem in problems)


def test_a_missing_size_is_detected(splits):
    manifest = subsets.build_subset_manifest(splits)
    del manifest["subsets"]["1000"]

    problems = subsets.verify_subsets(splits, manifest)

    assert any("1000: not recorded" in problem for problem in problems)


def test_loading_a_drifted_ladder_raises_rather_than_returning_it(splits):
    manifest = subsets.build_subset_manifest(splits)
    manifest["subsets"]["2500"]["dataset_sha256"] = "0" * 64

    with pytest.raises(subsets.SubsetIntegrityError):
        subsets.load_verified_subsets(splits, manifest)


def test_loading_an_intact_ladder_returns_the_same_rows(splits, ladder):
    manifest = subsets.build_subset_manifest(splits)

    loaded = subsets.load_verified_subsets(splits, manifest)

    for size in ladder:
        assert loaded[size].equals(ladder[size])


def test_a_ladder_that_cannot_be_rebuilt_reports_that_rather_than_crashing(splits):
    manifest = subsets.build_subset_manifest(splits)
    manifest["sizes"] = [len(splits.X_train) + 1]

    problems = subsets.verify_subsets(splits, manifest)

    assert any("cannot be rebuilt" in problem for problem in problems)


# ================================================== class quota arithmetic

@pytest.mark.parametrize("size", [2, 10, 501, 1000, 4999])
def test_a_quota_always_sums_to_the_requested_size(size):
    positives, negatives = subsets._class_quota(size, 0.4995)

    assert positives + negatives == size
    assert positives >= 1 and negatives >= 1


@pytest.mark.parametrize("rate", [0.001, 0.5, 0.999])
def test_a_quota_never_produces_a_single_class_subset(rate):
    """Even an extreme parent rate must leave both classes represented."""
    positives, negatives = subsets._class_quota(100, rate)

    assert positives >= 1
    assert negatives >= 1


def test_the_stratified_ordering_uses_every_training_row(splits):
    orders = subsets._stratified_order(splits.y_train, seed=protocol.SUBSET_SEED)

    total = len(orders[0]) + len(orders[1])
    assert total == len(splits.y_train)
    assert len(np.intersect1d(orders[0], orders[1])) == 0
