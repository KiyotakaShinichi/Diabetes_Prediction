"""The extraction changed no training semantics.

Two independent lines of defence.

Golden values: tests/fixtures/pipeline_equivalence_baseline.json was captured by
running the PRE-refactor implementations (commit a42f430, where each pipeline
owned its own copy of the seven orchestration helpers) against the deterministic
fixture built below. Every value here is what the duplicated code produced. When
the shared implementations were substituted, all of it compared bit-identical on
the capture machine; the tolerances below exist only to absorb cross-platform
floating-point noise in CI, not to paper over a semantic change.

Cross-pipeline identity: because both pipelines now dispatch to one
implementation, feeding the same model and the same splits through the logistic
binding and the boosted binding must produce byte-equal results. That assertion
is exact - it runs in one process on one platform, so there is no noise to
absorb - and it fails the moment the two configurations start diverging in
anything but their spec.

The fixture deliberately stores no filesystem paths, only artifact basenames:
the repository forbids committed absolute paths.
"""
import contextlib
import io
import json

import numpy as np
import pytest

import boostedtrees_ab as xgb_pipeline
import logisticregression_only as lr_pipeline
from conftest import REPO_ROOT
from tests.test_training_primitives import make_dataset

BASELINE = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "pipeline_equivalence_baseline.json").read_text(
        encoding="utf-8"
    )
)

#: The fixture the pre-refactor capture ran against. Changing either value
#: invalidates every golden below.
FIXTURE_ROWS = 240
FIXTURE_SEED = 23

LR_PARAMS = {"C": 1.0, "solver": "lbfgs"}
XGB_PARAMS = {"n_estimators": 15, "max_depth": 2, "learning_rate": 0.3}

#: Loose enough for a different BLAS, far tighter than any semantic change.
TOL = {"rtol": 1e-6, "atol": 1e-9}


@pytest.fixture(scope="module")
def dataset_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("equivalence") / "fixture.csv"
    make_dataset(rows=FIXTURE_ROWS, seed=FIXTURE_SEED).to_csv(path, index=False)
    return path


def run_pipeline_stages(module, dataset_csv, fit_name, params, calibrate_name):
    """Everything the golden capture recorded, in one deterministic pass."""
    with contextlib.redirect_stdout(io.StringIO()):
        splits = module.prepare_training_data(dataset_csv, verbose=False)
        model = getattr(module, fit_name)(splits, params)
        threshold, val_metrics = module.select_threshold(model, splits)
        test_proba, _, test_metrics = module.evaluate_on_test(model, splits, threshold)
        calibrated, proba_final, _, calibrated_metrics, before, after = getattr(
            module, calibrate_name
        )(model, splits, threshold, test_proba)

    return {
        "splits": splits,
        "model": model,
        "calibrated": calibrated,
        "split_index": {
            name: list(map(int, getattr(splits, name).index))
            for name in ("X_train", "X_val", "X_test")
        },
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_proba_head": [float(x) for x in test_proba[:8]],
        "cal_proba_head": [float(x) for x in proba_final[:8]],
        "brier_before": float(before),
        "brier_after": float(after),
        "calibrated_metrics": calibrated_metrics,
    }


@pytest.fixture(scope="module")
def logistic_run(dataset_csv):
    return run_pipeline_stages(
        lr_pipeline, dataset_csv, "fit_final_pipeline", LR_PARAMS, "calibrate_pipeline"
    )


@pytest.fixture(scope="module")
def boosted_run(dataset_csv):
    return run_pipeline_stages(
        xgb_pipeline, dataset_csv, "fit_final_model", XGB_PARAMS, "calibrate_model"
    )


def assert_metrics_match(actual, expected, label):
    assert set(actual) == set(expected), f"{label}: metric keys changed"
    for key, value in expected.items():
        np.testing.assert_allclose(
            actual[key], value, **TOL, err_msg=f"{label}.{key} moved"
        )


# ================================================= goldens from the old code

@pytest.mark.parametrize("tag", ["lr", "xgb"])
def test_split_identity_is_unchanged(tag, logistic_run, boosted_run):
    """Row-for-row: not a tolerance, an exact set of indices."""
    run = logistic_run if tag == "lr" else boosted_run

    assert run["split_index"] == BASELINE[tag]["split_index"]


@pytest.mark.parametrize("tag", ["lr", "xgb"])
def test_selected_threshold_is_unchanged(tag, logistic_run, boosted_run):
    run = logistic_run if tag == "lr" else boosted_run

    np.testing.assert_allclose(run["threshold"], BASELINE[tag]["threshold"], **TOL)


@pytest.mark.parametrize("tag", ["lr", "xgb"])
@pytest.mark.parametrize("stage", ["val_metrics", "test_metrics", "calibrated_metrics"])
def test_metric_dictionaries_are_unchanged(tag, stage, logistic_run, boosted_run):
    run = logistic_run if tag == "lr" else boosted_run

    assert_metrics_match(run[stage], BASELINE[tag][stage], f"{tag}.{stage}")


@pytest.mark.parametrize("tag", ["lr", "xgb"])
@pytest.mark.parametrize("head", ["test_proba_head", "cal_proba_head"])
def test_probabilities_are_unchanged(tag, head, logistic_run, boosted_run):
    run = logistic_run if tag == "lr" else boosted_run

    np.testing.assert_allclose(run[head], BASELINE[tag][head], **TOL)


@pytest.mark.parametrize("tag", ["lr", "xgb"])
def test_calibration_improvement_is_unchanged(tag, logistic_run, boosted_run):
    run = logistic_run if tag == "lr" else boosted_run

    np.testing.assert_allclose(run["brier_before"], BASELINE[tag]["brier_before"], **TOL)
    np.testing.assert_allclose(run["brier_after"], BASELINE[tag]["brier_after"], **TOL)


@pytest.mark.parametrize(
    ("tag", "module"),
    [pytest.param("lr", lr_pipeline, id="logistic_regression"),
     pytest.param("xgb", xgb_pipeline, id="boosted_trees")],
)
def test_artifact_basenames_are_unchanged(tag, module, tmp_path):
    """Filenames moved into PIPELINE_SPEC; the names themselves did not move."""
    resolved = {role: path.name for role, path in module.artifact_paths(tmp_path).items()}

    assert resolved == BASELINE[tag]["artifact_paths"]


@pytest.mark.parametrize(
    ("module", "trials"),
    [pytest.param(lr_pipeline, 100, id="logistic_regression"),
     pytest.param(xgb_pipeline, 50, id="boosted_trees")],
)
def test_parsed_defaults_still_point_at_the_production_configuration(module, trials):
    """Asserted against the module constants, never against a committed path."""
    args = module.parse_args([])

    assert args.data_path == module.DATA_PATH
    assert args.artifacts_dir == module.ARTIFACTS_DIR
    assert args.optuna_trials == trials == module.OPTUNA_TRIALS


@pytest.mark.parametrize(
    ("module", "flag", "attribute"),
    [pytest.param(lr_pipeline, "--data-path", "data_path", id="lr_data"),
     pytest.param(lr_pipeline, "--artifacts-dir", "artifacts_dir", id="lr_artifacts"),
     pytest.param(xgb_pipeline, "--data-path", "data_path", id="xgb_data"),
     pytest.param(xgb_pipeline, "--artifacts-dir", "artifacts_dir", id="xgb_artifacts")],
)
def test_overrides_still_win_over_the_defaults(module, flag, attribute, tmp_path):
    args = module.parse_args([flag, str(tmp_path / "elsewhere")])

    assert getattr(args, attribute) == tmp_path / "elsewhere"


# ============================================ one implementation, two bindings

def test_shared_stages_agree_across_both_bindings(logistic_run):
    """Exact, not approximate: same process, same object, one implementation.

    The logistic model is deliberately pushed through the boosted pipeline's
    stage bindings. Identical output is only possible while both resolve to the
    same function.
    """
    splits, model = logistic_run["splits"], logistic_run["model"]

    lr_threshold, lr_val = lr_pipeline.select_threshold(model, splits)
    xgb_threshold, xgb_val = xgb_pipeline.select_threshold(model, splits)

    assert lr_threshold == xgb_threshold
    assert lr_val == xgb_val

    lr_proba, lr_pred, lr_metrics = lr_pipeline.evaluate_on_test(model, splits, lr_threshold)
    xgb_proba, xgb_pred, xgb_metrics = xgb_pipeline.evaluate_on_test(
        model, splits, xgb_threshold
    )

    assert np.array_equal(lr_proba, xgb_proba)
    assert np.array_equal(lr_pred, xgb_pred)
    assert lr_metrics == xgb_metrics


def test_both_pipelines_prepare_identical_splits(dataset_csv):
    """Same seed, same sizes, same contract - so the same rows, exactly."""
    with contextlib.redirect_stdout(io.StringIO()):
        from_lr = lr_pipeline.prepare_training_data(dataset_csv, verbose=False)
        from_xgb = xgb_pipeline.prepare_training_data(dataset_csv, verbose=False)

    for name in ("X_train", "X_val", "X_test"):
        assert list(getattr(from_lr, name).index) == list(getattr(from_xgb, name).index)
    assert from_lr.feature_names == from_xgb.feature_names
    assert from_lr.sizes == from_xgb.sizes


def test_calibration_agrees_across_both_bindings(logistic_run):
    """Same method and cv on both sides, so identical calibrated probabilities."""
    splits, model = logistic_run["splits"], logistic_run["model"]
    threshold = logistic_run["threshold"]
    test_proba, _, _ = lr_pipeline.evaluate_on_test(model, splits, threshold)

    with contextlib.redirect_stdout(io.StringIO()):
        _, lr_proba, lr_pred, lr_metrics, lr_before, lr_after = lr_pipeline.calibrate_pipeline(
            model, splits, threshold, test_proba
        )
        _, xgb_proba, xgb_pred, xgb_metrics, xgb_before, xgb_after = xgb_pipeline.calibrate_model(
            model, splits, threshold, test_proba
        )

    assert np.array_equal(lr_proba, xgb_proba)
    assert np.array_equal(lr_pred, xgb_pred)
    assert lr_metrics == xgb_metrics
    assert (lr_before, lr_after) == (xgb_before, xgb_after)
