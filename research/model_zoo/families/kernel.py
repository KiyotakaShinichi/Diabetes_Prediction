"""Distance and kernel methods: three models, two probability problems.

These are the algorithms most sensitive to the zoo's 1,000-row budget, and that
is why they are interesting here. A support vector machine's fit is quadratic in
the sample count, so 1,000 rows is comfortable where 40,000 would not be - this
family is one of the few places where the constrained budget is a feature rather
than a limitation.

Both SVMs are deliberately built **without** `probability=True`. Platt scaling
inside `SVC` runs an internal five-fold cross-validation, which multiplies the
fit cost by five and - more importantly - produces probabilities from a
different fit than the one being scored, so `predict` and `predict_proba` can
disagree about the same row. The zoo instead takes the signed distance as a
ranking score and, where a probability is genuinely wanted, calibrates it
externally on validation rows. That is slower to explain and much easier to
defend.
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


def build_knn(*, spec: ModelSpec, n_neighbors: int = 25, weights: str = "distance") -> Any:
    from sklearn.neighbors import KNeighborsClassifier

    # 25 neighbours at 1,000 training rows is 2.5% of the sample per prediction.
    # Fewer would chase noise on a dataset where every model plateaus near 0.82;
    # distance weighting keeps the far ones from dominating.
    return _adapter(
        spec.model_id,
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=1),
        spec,
    )


def build_linear_svm(*, spec: ModelSpec, C: float = 1.0, max_iter: int = 5000) -> Any:
    from sklearn.svm import LinearSVC

    return _adapter(
        spec.model_id,
        LinearSVC(C=C, max_iter=max_iter, dual="auto", random_state=spec.seed),
        spec,
    )


def build_rbf_svm(*, spec: ModelSpec, C: float = 1.0, gamma: str = "scale") -> Any:
    from sklearn.svm import SVC

    # probability=False on purpose - see the module docstring.
    return _adapter(
        spec.model_id,
        SVC(C=C, kernel="rbf", gamma=gamma, probability=False, random_state=spec.seed),
        spec,
    )


_MARGIN_ONLY = Capabilities(
    supports_predict_proba=False,
    supports_calibration=True,
    supports_feature_importance=False,
    supports_serialization=True,
    requires_scaling=True,
)

register(ModelSpec(
    model_id="knn",
    display_name="K-Nearest Neighbours (k=25)",
    family=Family.DISTANCE,
    framework=Framework.SKLEARN,
    build=build_knn,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=Capabilities(
        supports_predict_proba=True,
        supports_calibration=True,
        supports_feature_importance=False,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    default_config={"n_neighbors": 25, "weights": "distance"},
    rationale=(
        "A purely local model with no global structure at all. If the ten "
        "features carry the signal, neighbourhood voting should find much of "
        "it; its probabilities are vote shares, not likelihoods."
    ),
))

register(ModelSpec(
    model_id="linear_svm",
    display_name="Linear SVM",
    family=Family.KERNEL,
    framework=Framework.SKLEARN,
    build=build_linear_svm,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.REQUIRES_EXTERNAL_CALIBRATION,
    capabilities=Capabilities(
        supports_predict_proba=False,
        supports_calibration=True,
        supports_feature_importance=True,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    default_config={"C": 1.0},
    rationale=(
        "The same linear boundary as logistic regression fitted to a margin "
        "instead of a likelihood. Ranks without any probability of its own."
    ),
))

register(ModelSpec(
    model_id="rbf_svm",
    display_name="RBF SVM",
    family=Family.KERNEL,
    framework=Framework.SKLEARN,
    build=build_rbf_svm,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.REQUIRES_EXTERNAL_CALIBRATION,
    capabilities=_MARGIN_ONLY,
    resource_class=ResourceClass.MODERATE,
    default_config={"C": 1.0, "gamma": "scale"},
    rationale=(
        "A non-linear boundary from a completely different mechanism than a "
        "tree or a network. If curvature helps at all, this is where it shows."
    ),
))
