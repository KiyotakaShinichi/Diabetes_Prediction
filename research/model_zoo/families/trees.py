"""Axis-aligned partitions: one tree, and two ways of averaging many.

The three models here share a hypothesis class and differ only in how they
reduce variance, which makes them a controlled experiment rather than three
entries on a list. A single decision tree is the high-variance base case. A
random forest averages bootstrapped trees that each see a random feature subset
at every split. Extra trees goes further and randomises the split *thresholds*
too, trading fit quality per tree for decorrelation between them.

At 1,000 rows and ten features, variance is the binding constraint - which is
exactly the regime where the difference between these three is legible.

None of them are scaled. Trees split on thresholds within a single feature at a
time, so any monotone rescaling produces an identical tree; inserting a scaler
would add a fitted object to serialize and change nothing else.
"""
from __future__ import annotations

from typing import Any

from research.model_zoo.contracts import (
    Capabilities,
    Family,
    Framework,
    Preprocessing,
    ProbabilityBehavior,
    ResourceClass,
)
from research.model_zoo.preprocessing import wrap
from research.model_zoo.registry import ModelSpec, register


def _adapter(model_id: str, estimator: Any, spec: ModelSpec) -> Any:
    from research.model_zoo.adapters.sklearn_adapter import SklearnAdapter

    return SklearnAdapter(
        model_id,
        wrap(estimator, spec.preprocessing),
        capabilities=spec.capabilities,
        probability_behavior=spec.probability_behavior,
    )


def build_decision_tree(
    *, spec: ModelSpec, max_depth: int = 6, min_samples_leaf: int = 20
) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    # Depth 6 and 20-sample leaves at 1,000 rows: an unbounded tree memorises
    # the training set here and its "probabilities" become 0 and 1.
    return _adapter(
        spec.model_id,
        DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=spec.seed
        ),
        spec,
    )


def build_random_forest(
    *, spec: ModelSpec, n_estimators: int = 300, max_depth: int = 12,
    min_samples_leaf: int = 5,
) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return _adapter(
        spec.model_id,
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            random_state=spec.seed,
        ),
        spec,
    )


def build_extra_trees(
    *, spec: ModelSpec, n_estimators: int = 300, max_depth: int = 12,
    min_samples_leaf: int = 5,
) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return _adapter(
        spec.model_id,
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            random_state=spec.seed,
        ),
        spec,
    )


_TREE_CAPABILITIES = Capabilities(
    supports_predict_proba=True,
    supports_calibration=True,
    supports_feature_importance=True,
    supports_serialization=True,
    requires_scaling=False,
)

register(ModelSpec(
    model_id="decision_tree",
    display_name="Decision Tree (depth 6)",
    family=Family.TREE,
    framework=Framework.SKLEARN,
    build=build_decision_tree,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_TREE_CAPABILITIES,
    resource_class=ResourceClass.LIGHT,
    default_config={"max_depth": 6, "min_samples_leaf": 20},
    rationale=(
        "The high-variance base case, and the only model in the zoo a person "
        "can read end to end. Its leaf frequencies are coarse probabilities."
    ),
))

register(ModelSpec(
    model_id="random_forest",
    display_name="Random Forest (300 trees)",
    family=Family.TREE,
    framework=Framework.SKLEARN,
    build=build_random_forest,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_TREE_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    rationale=(
        "Bagging plus feature subsampling - the standard answer to a single "
        "tree's variance, and a strong default on tabular data."
    ),
))

register(ModelSpec(
    model_id="extra_trees",
    display_name="Extremely Randomised Trees (300)",
    family=Family.TREE,
    framework=Framework.SKLEARN,
    build=build_extra_trees,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_TREE_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    rationale=(
        "Randomises split thresholds as well as features, buying more "
        "decorrelation at the cost of weaker individual trees. Against the "
        "forest it isolates how much the split search itself is worth."
    ),
))
