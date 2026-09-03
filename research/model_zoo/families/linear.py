"""Linear and probabilistic models: nine algorithms, four probability stories.

Track K found logistic regression startlingly competitive - best of every
family at 500 training rows, and within 0.005 ROC-AUC of everything at 40,000.
This family exists to ask *which part* of that is doing the work. Is it the
linear decision boundary, the L2 penalty, the maximum-likelihood fit, or simply
that ten mostly-ordinal features do not support anything more elaborate?

Nine models separate those. Three logistic regressions differing only in
penalty isolate regularisation. Two discriminant analyses swap a shared
covariance matrix for per-class ones, isolating the linear/quadratic boundary.
Gaussian naive Bayes drops feature correlation entirely. Two SGD variants
change the loss while holding the boundary linear. Nearest centroid strips the
model down to two class means and no probability at all.

That last one earns its place precisely because it is the weakest: it is the
zoo's floor, and a floor is what tells you whether the ceiling means anything.
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

_ADAPTER_IMPORT = "research.model_zoo.adapters.sklearn_adapter"


def _adapter(model_id: str, estimator: Any, spec: ModelSpec) -> Any:
    from research.model_zoo.adapters.sklearn_adapter import SklearnAdapter

    return SklearnAdapter(
        model_id,
        wrap(estimator, spec.preprocessing),
        capabilities=spec.capabilities,
        probability_behavior=spec.probability_behavior,
    )


# ------------------------------------------------------------ logistic family

# scikit-learn 1.8 deprecated ``penalty=`` in favour of ``l1_ratio``, and will
# remove it in 1.10: l1_ratio=0 is ridge, 1 is lasso, and anything between is
# elastic net. The lock pins 1.8.0, so the three models below use the new
# spelling - it expresses the same three penalties, and it keeps the zoo's runs
# free of deprecation noise that would otherwise scroll past every result.
_RIDGE = 0.0
_LASSO = 1.0


def build_logistic_l2(*, spec: ModelSpec, C: float = 1.0, max_iter: int = 2000) -> Any:
    from sklearn.linear_model import LogisticRegression

    return _adapter(
        spec.model_id,
        LogisticRegression(
            l1_ratio=_RIDGE, C=C, solver="lbfgs", max_iter=max_iter, random_state=spec.seed
        ),
        spec,
    )


def build_logistic_l1(*, spec: ModelSpec, C: float = 1.0, max_iter: int = 2000) -> Any:
    from sklearn.linear_model import LogisticRegression

    # liblinear rather than saga: at 1,000 rows and ten features it converges
    # in a fraction of the time and reaches the same optimum.
    return _adapter(
        spec.model_id,
        LogisticRegression(
            l1_ratio=_LASSO, C=C, solver="liblinear", max_iter=max_iter,
            random_state=spec.seed,
        ),
        spec,
    )


def build_logistic_elasticnet(
    *, spec: ModelSpec, C: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 4000
) -> Any:
    from sklearn.linear_model import LogisticRegression

    return _adapter(
        spec.model_id,
        LogisticRegression(
            l1_ratio=l1_ratio,
            C=C,
            solver="saga",
            max_iter=max_iter,
            random_state=spec.seed,
        ),
        spec,
    )


# ------------------------------------------------------- discriminant analysis

def build_lda(*, spec: ModelSpec, solver: str = "svd") -> Any:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return _adapter(spec.model_id, LinearDiscriminantAnalysis(solver=solver), spec)


def build_qda(*, spec: ModelSpec, reg_param: float = 0.1) -> Any:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # A little regularisation is not optional here: several features are binary,
    # so a per-class covariance matrix is near-singular without it and the fit
    # emits collinearity warnings that are really a numerical failure.
    return _adapter(
        spec.model_id, QuadraticDiscriminantAnalysis(reg_param=reg_param), spec
    )


def build_gaussian_nb(*, spec: ModelSpec, var_smoothing: float = 1e-9) -> Any:
    from sklearn.naive_bayes import GaussianNB

    return _adapter(spec.model_id, GaussianNB(var_smoothing=var_smoothing), spec)


# ------------------------------------------------------------- SGD variants

def build_sgd_logistic(
    *, spec: ModelSpec, alpha: float = 1e-4, max_iter: int = 2000
) -> Any:
    from sklearn.linear_model import SGDClassifier

    return _adapter(
        spec.model_id,
        SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol=1e-4,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=spec.seed,
        ),
        spec,
    )


def build_sgd_modified_huber(
    *, spec: ModelSpec, alpha: float = 1e-4, max_iter: int = 2000
) -> Any:
    from sklearn.linear_model import SGDClassifier

    # modified_huber is the one non-log SGD loss that still exposes
    # predict_proba. Its probabilities come from a clipped quadratic rather
    # than a likelihood fit, so the spec calls them NATIVE_UNCALIBRATED.
    return _adapter(
        spec.model_id,
        SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            max_iter=max_iter,
            tol=1e-4,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=spec.seed,
        ),
        spec,
    )


def build_nearest_centroid(*, spec: ModelSpec, shrink_threshold: float | None = None) -> Any:
    from sklearn.neighbors import NearestCentroid

    return _adapter(
        spec.model_id, NearestCentroid(shrink_threshold=shrink_threshold), spec
    )


# =========================================================== registrations

_PROBABILISTIC_LINEAR = Capabilities(
    supports_predict_proba=True,
    supports_calibration=True,
    supports_feature_importance=True,
    supports_serialization=True,
    requires_scaling=True,
)

register(ModelSpec(
    model_id="logistic_l2",
    display_name="Logistic Regression (L2)",
    family=Family.LINEAR,
    framework=Framework.SKLEARN,
    build=build_logistic_l2,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=_PROBABILISTIC_LINEAR,
    resource_class=ResourceClass.LIGHT,
    default_config={"C": 1.0},
    rationale=(
        "The reference point. Track K found it within 0.005 ROC-AUC of every "
        "other family and the single best model at 500 training rows."
    ),
))

register(ModelSpec(
    model_id="logistic_l1",
    display_name="Logistic Regression (L1)",
    family=Family.LINEAR,
    framework=Framework.SKLEARN,
    build=build_logistic_l1,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=_PROBABILISTIC_LINEAR,
    resource_class=ResourceClass.LIGHT,
    default_config={"C": 1.0},
    rationale=(
        "Sparsity. With ten features an L1 penalty can zero some outright, "
        "which tests whether the information set is smaller than it looks."
    ),
))

register(ModelSpec(
    model_id="logistic_elasticnet",
    display_name="Logistic Regression (Elastic Net)",
    family=Family.LINEAR,
    framework=Framework.SKLEARN,
    build=build_logistic_elasticnet,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=_PROBABILISTIC_LINEAR,
    resource_class=ResourceClass.LIGHT,
    default_config={"C": 1.0, "l1_ratio": 0.5},
    rationale="Between the two penalties, to see whether either extreme matters.",
))

register(ModelSpec(
    model_id="lda",
    display_name="Linear Discriminant Analysis",
    family=Family.PROBABILISTIC,
    framework=Framework.SKLEARN,
    build=build_lda,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=Capabilities(
        supports_predict_proba=True,
        supports_calibration=True,
        supports_feature_importance=True,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    rationale=(
        "A linear boundary reached by generative assumptions rather than by "
        "maximising conditional likelihood - the same shape, a different fit."
    ),
))

register(ModelSpec(
    model_id="qda",
    display_name="Quadratic Discriminant Analysis",
    family=Family.PROBABILISTIC,
    framework=Framework.SKLEARN,
    build=build_qda,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=Capabilities(
        supports_predict_proba=True,
        supports_calibration=True,
        supports_feature_importance=False,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    default_config={"reg_param": 0.1},
    rationale=(
        "Per-class covariance, so the boundary curves. Tests whether the "
        "linear models are leaving anything on the table."
    ),
))

register(ModelSpec(
    model_id="gaussian_nb",
    display_name="Gaussian Naive Bayes",
    family=Family.PROBABILISTIC,
    framework=Framework.SKLEARN,
    build=build_gaussian_nb,
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
    rationale=(
        "Assumes the ten features are conditionally independent, which they "
        "are not. Its probabilities are famously over-confident, so it is the "
        "zoo's clearest case of good ranking with bad calibration."
    ),
))

register(ModelSpec(
    model_id="sgd_logistic",
    display_name="SGD Logistic",
    family=Family.LINEAR,
    framework=Framework.SKLEARN,
    build=build_sgd_logistic,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_PROBABILISTIC,
    capabilities=_PROBABILISTIC_LINEAR,
    resource_class=ResourceClass.LIGHT,
    rationale=(
        "The same objective as logistic_l2 reached by stochastic descent, so "
        "any gap between them is optimisation rather than model class."
    ),
))

register(ModelSpec(
    model_id="sgd_modified_huber",
    display_name="SGD Modified Huber",
    family=Family.LINEAR,
    framework=Framework.SKLEARN,
    build=build_sgd_modified_huber,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.NATIVE_UNCALIBRATED,
    capabilities=Capabilities(
        supports_predict_proba=True,
        supports_calibration=True,
        supports_feature_importance=True,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    rationale=(
        "A smoothed hinge loss: a linear boundary fitted to classify rather "
        "than to estimate probability, with probabilities available anyway."
    ),
))

register(ModelSpec(
    model_id="nearest_centroid",
    display_name="Nearest Centroid",
    family=Family.DISTANCE,
    framework=Framework.SKLEARN,
    build=build_nearest_centroid,
    preprocessing=Preprocessing.STANDARDIZED,
    probability_behavior=ProbabilityBehavior.HARD_LABELS_ONLY,
    capabilities=Capabilities(
        supports_predict_proba=False,
        supports_calibration=False,
        supports_feature_importance=False,
        supports_serialization=True,
        requires_scaling=True,
    ),
    resource_class=ResourceClass.LIGHT,
    rationale=(
        "Two class means and nothing else. It has no probability and no "
        "decision function, so its threshold-free metrics are reported as "
        "undefined rather than computed from its hard labels - the zoo's floor, "
        "and the test that the harness handles an honest absence."
    ),
))
