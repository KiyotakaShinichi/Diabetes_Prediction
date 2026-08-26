"""One implementation owner per shared training concern.

Track G4 decomposed both pipelines into stages, but decomposed them twice: seven
orchestration helpers ended up as near-identical top-level definitions in
logisticregression_only.py and boostedtrees_ab.py. A fix in one would silently
not reach the other.

These tests pin the repair. They are deliberately NOT raw-LOC or whole-file
similarity thresholds - those fail on unrelated edits and say nothing about who
owns what. Instead they assert ownership two ways:

* by identity, at runtime: both pipelines must resolve a shared concern to the
  SAME underlying function object in ml_core.pipeline;
* by AST, in the source: neither pipeline file may re-declare one of those
  names as a top-level def.

The mirror image matters just as much. Model-specific behaviour - estimator
construction, search spaces, drift schemas, SHAP, writers - must stay locally
owned and must NOT be shared, because the two variants genuinely differ there.
"""
import ast
import functools
import inspect
from pathlib import Path

import pytest

import boostedtrees_ab as xgb_pipeline
import logisticregression_only as lr_pipeline
from conftest import REPO_ROOT
from ml_core import pipeline as shared_pipeline

#: Shared concern -> attribute name on each pipeline module. The boosted
#: pipeline calls its calibration stage calibrate_model; the name differs, the
#: implementation must not.
SHARED_CONCERNS = {
    "argument parsing": ("parse_args", "parse_args"),
    "artifact path resolution": ("artifact_paths", "artifact_paths"),
    "data preparation": ("prepare_training_data", "prepare_training_data"),
    "threshold selection": ("select_threshold", "select_threshold"),
    "test evaluation": ("evaluate_on_test", "evaluate_on_test"),
    "calibration": ("calibrate_pipeline", "calibrate_model"),
    "provenance emission": ("emit_provenance", "emit_provenance"),
}

#: Behaviour that must remain per-variant, as (attribute, module) pairs.
LOCAL_TO_LOGISTIC = (
    "build_pipeline", "fit_final_pipeline", "cross_validate_folds",
    "optimize_hyperparameters", "build_shap_explainer", "report_coefficients",
    "build_logistic_drift_baseline", "write_training_outputs", "main",
)
LOCAL_TO_BOOSTED = (
    "build_model", "fit_final_model", "optimize_hyperparameters",
    "build_shap_explainer", "report_gain_importance",
    "build_boosted_drift_baseline", "write_training_outputs", "main",
)

PIPELINE_SOURCES = {
    lr_pipeline: REPO_ROOT / "logisticregression_only.py",
    xgb_pipeline: REPO_ROOT / "boostedtrees_ab.py",
}


def implementation_of(obj):
    """The function a module attribute ultimately dispatches to.

    Unwraps functools.partial, which is how each pipeline binds its own spec and
    defaults to a shared implementation without redeclaring it.
    """
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.unwrap(obj)


def top_level_defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# =============================================== shared concerns have one owner

@pytest.mark.parametrize(
    ("concern", "lr_name", "xgb_name"),
    [pytest.param(c, a, b, id=c.replace(" ", "_")) for c, (a, b) in SHARED_CONCERNS.items()],
)
def test_both_pipelines_dispatch_to_the_same_implementation(concern, lr_name, xgb_name):
    lr_impl = implementation_of(getattr(lr_pipeline, lr_name))
    xgb_impl = implementation_of(getattr(xgb_pipeline, xgb_name))

    assert lr_impl is xgb_impl, f"{concern} has two implementations"
    assert lr_impl.__module__ == shared_pipeline.__name__


@pytest.mark.parametrize(
    ("concern", "lr_name", "xgb_name"),
    [pytest.param(c, a, b, id=c.replace(" ", "_")) for c, (a, b) in SHARED_CONCERNS.items()],
)
def test_neither_pipeline_redeclares_a_shared_concern(concern, lr_name, xgb_name):
    """A local def would shadow the import and silently reintroduce the fork."""
    assert lr_name not in top_level_defs(PIPELINE_SOURCES[lr_pipeline]), concern
    assert xgb_name not in top_level_defs(PIPELINE_SOURCES[xgb_pipeline]), concern


def test_every_shared_implementation_is_reachable_from_the_shared_module():
    owners = {
        implementation_of(getattr(lr_pipeline, lr_name)).__name__
        for lr_name, _ in SHARED_CONCERNS.values()
    }

    for name in owners:
        assert hasattr(shared_pipeline, name), f"{name} is not exported by ml_core.pipeline"


def test_each_pipeline_binds_its_own_spec():
    """One implementation, two configurations - not one implementation, one config."""
    assert lr_pipeline.PIPELINE_SPEC is not xgb_pipeline.PIPELINE_SPEC
    assert lr_pipeline.PIPELINE_SPEC.variant == "A"
    assert xgb_pipeline.PIPELINE_SPEC.variant == "B"
    assert lr_pipeline.PIPELINE_SPEC.scaler == "StandardScaler"
    assert xgb_pipeline.PIPELINE_SPEC.scaler is None
    assert set(lr_pipeline.PIPELINE_SPEC.filenames.values()).isdisjoint(
        xgb_pipeline.PIPELINE_SPEC.filenames.values()
    )


# ============================================ model-specific code stays local

@pytest.mark.parametrize(
    ("module", "names"),
    [pytest.param(lr_pipeline, LOCAL_TO_LOGISTIC, id="logistic_regression"),
     pytest.param(xgb_pipeline, LOCAL_TO_BOOSTED, id="boosted_trees")],
)
def test_model_specific_functions_remain_locally_owned(module, names):
    declared = top_level_defs(PIPELINE_SOURCES[module])

    for name in names:
        assert name in declared, f"{name} moved out of {module.__name__}"
        assert getattr(module, name).__module__ == module.__name__
        assert not hasattr(shared_pipeline, name), f"{name} leaked into ml_core.pipeline"


@pytest.mark.parametrize(
    "name", sorted(set(LOCAL_TO_LOGISTIC) & set(LOCAL_TO_BOOSTED))
)
def test_same_named_model_specific_functions_are_still_distinct(name):
    """optimize_hyperparameters and friends share a name, not a search space."""
    assert getattr(lr_pipeline, name) is not getattr(xgb_pipeline, name)


# ================================================ the shared module stays generic

def test_shared_module_is_model_agnostic():
    """ml_core.pipeline must not learn about a specific estimator family."""
    source = (REPO_ROOT / "ml_core" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for forbidden in ("xgboost", "shap", "optuna", "sklearn.linear_model"):
        assert not any(m == forbidden or m.startswith(forbidden + ".") for m in imported), forbidden


def test_shared_module_does_not_import_the_pipelines_back():
    """No cycle: the pipelines depend on ml_core, never the reverse."""
    source = (REPO_ROOT / "ml_core" / "pipeline.py").read_text(encoding="utf-8")

    for name in ("logisticregression_only", "boostedtrees_ab"):
        assert name not in source
