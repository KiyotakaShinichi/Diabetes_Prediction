"""Sequential ensembles: four core implementations and two optional ones.

Track K's most durable finding was about this family. XGBoost was the *worst*
model in the study at 500 training rows - 0.035 ROC-AUC behind logistic
regression - the fastest improving as data grew, and the strongest classical
model once given all 40,125 rows. Boosting is the zoo's data-hungry family, and
Track L runs at 1,000 rows, which is squarely in the region where Track K found
it still climbing.

So the expectation, recorded before the run: boosting should underperform its
reputation here. If it does, that is Track K's sample-efficiency result
reproducing under a different harness and a wider comparison set, which is
worth more than a number in isolation.

Four implementations of nominally the same idea are included because they are
not the same idea in practice: AdaBoost reweights samples with an exponential
loss, sklearn's GradientBoosting fits stagewise regression trees, its
histogram variant bins features first, and XGBoost adds a second-order
objective with explicit regularisation.

LightGBM and CatBoost are registered as OPTIONAL. They install cleanly on CPU
but are not in the core lockfile, so the zoo detects them and records them as
skipped-with-a-reason when absent. `import research.model_zoo.registry` works
either way.
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


def build_adaboost(
    *, spec: ModelSpec, n_estimators: int = 200, learning_rate: float = 0.5
) -> Any:
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Depth-2 stumps rather than depth-1: with ten features and a target that
    # is not linearly separable, a true stump underfits badly.
    return _adapter(
        spec.model_id,
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=spec.seed),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=spec.seed,
        ),
        spec,
    )


def build_gradient_boosting(
    *, spec: ModelSpec, n_estimators: int = 200, learning_rate: float = 0.05,
    max_depth: int = 3,
) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return _adapter(
        spec.model_id,
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            random_state=spec.seed,
        ),
        spec,
    )


def build_hist_gradient_boosting(
    *, spec: ModelSpec, max_iter: int = 300, learning_rate: float = 0.05,
    max_depth: int | None = 6,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    # early_stopping with an internal validation fraction: the split comes from
    # the training rows this model was handed, never from the zoo's validation
    # or test partitions.
    return _adapter(
        spec.model_id,
        HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=spec.seed,
        ),
        spec,
    )


def build_xgboost(
    *, spec: ModelSpec, n_estimators: int = 300, learning_rate: float = 0.05,
    max_depth: int = 4,
) -> Any:
    from xgboost import XGBClassifier

    return _adapter(
        spec.model_id,
        XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=spec.seed,
        ),
        spec,
    )


def build_lightgbm(
    *, spec: ModelSpec, n_estimators: int = 300, learning_rate: float = 0.05,
    num_leaves: int = 15,
) -> Any:
    from lightgbm import LGBMClassifier

    # num_leaves 15 rather than the default 31: at 1,000 rows the default grows
    # leaves holding a handful of samples each.
    return _adapter(
        spec.model_id,
        LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
            verbose=-1,
            random_state=spec.seed,
        ),
        spec,
    )


def build_catboost(
    *, spec: ModelSpec, iterations: int = 300, learning_rate: float = 0.05,
    depth: int = 4,
) -> Any:
    from catboost import CatBoostClassifier

    return _adapter(
        spec.model_id,
        CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=spec.seed,
        ),
        spec,
    )


_BOOSTING_CAPABILITIES = Capabilities(
    supports_predict_proba=True,
    supports_calibration=True,
    supports_feature_importance=True,
    supports_serialization=True,
    requires_scaling=False,
)

register(ModelSpec(
    model_id="adaboost",
    display_name="AdaBoost (200 x depth-2)",
    family=Family.BOOSTING,
    framework=Framework.SKLEARN,
    build=build_adaboost,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 200, "learning_rate": 0.5},
    rationale=(
        "Exponential-loss reweighting, the oldest boosting idea here. Its "
        "probabilities come from a normalised vote and are usually the worst "
        "calibrated in the zoo."
    ),
))

register(ModelSpec(
    model_id="gradient_boosting",
    display_name="Gradient Boosting (200 trees)",
    family=Family.BOOSTING,
    framework=Framework.SKLEARN,
    build=build_gradient_boosting,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    rationale="Stagewise gradient descent in function space; the textbook form.",
))

#: The one boosting model with no native importance. sklearn's histogram
#: implementation deliberately does not expose ``feature_importances_``:
#: binning makes the impurity-decrease statistic the other implementations
#: report ill-defined, and permutation importance is the recommended route
#: instead. Declaring it False is the honest reading, and the registry test
#: catches the alternative.
_HIST_BOOSTING_CAPABILITIES = Capabilities(
    supports_predict_proba=True,
    supports_calibration=True,
    supports_feature_importance=False,
    supports_serialization=True,
    requires_scaling=False,
)

register(ModelSpec(
    model_id="hist_gradient_boosting",
    display_name="Histogram Gradient Boosting",
    family=Family.BOOSTING,
    framework=Framework.SKLEARN,
    build=build_hist_gradient_boosting,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_HIST_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"max_iter": 300, "learning_rate": 0.05, "max_depth": 6},
    rationale=(
        "Bins features before splitting, the trick that made boosting fast. "
        "Its early stopping splits the training rows it was given, nothing else. "
        "The only boosting model here with no native feature importance."
    ),
))

register(ModelSpec(
    model_id="xgboost",
    display_name="XGBoost (300 trees)",
    family=Family.BOOSTING,
    framework=Framework.XGBOOST,
    build=build_xgboost,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
    rationale=(
        "The bridge to Track K, which found it strongest at 40,125 rows and "
        "weakest at 500. At 1,000 it should still be climbing."
    ),
))

register(ModelSpec(
    model_id="lightgbm",
    display_name="LightGBM (300 trees)",
    family=Family.BOOSTING,
    framework=Framework.LIGHTGBM,
    build=build_lightgbm,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15},
    optional_dependency="lightgbm",
    rationale=(
        "Leaf-wise growth rather than level-wise. Optional: it is not in the "
        "core lockfile, and the zoo records it as skipped when absent."
    ),
))

register(ModelSpec(
    model_id="catboost",
    display_name="CatBoost (300 iterations)",
    family=Family.BOOSTING,
    framework=Framework.CATBOOST,
    build=build_catboost,
    preprocessing=Preprocessing.RAW_NUMERIC,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=_BOOSTING_CAPABILITIES,
    resource_class=ResourceClass.MODERATE,
    default_config={"iterations": 300, "learning_rate": 0.05, "depth": 4},
    optional_dependency="catboost",
    rationale=(
        "Ordered boosting and symmetric trees - a genuinely different bias "
        "correction. Optional for the same reason as LightGBM."
    ),
))
