"""One adapter for every scikit-learn-compatible estimator in the zoo.

Twenty of the zoo's models reach the benchmark through this class. What varies
between them is not the fitting - ``estimator.fit(X, y)`` is the same call for a
decision tree and a support vector machine - but what a fitted estimator can
then be *asked*, and that is exactly what the declared capabilities describe.

The hard part is `decision_scores`. Threshold-free metrics need a ranking, and
scikit-learn offers three different ways to get one depending on the model:
``predict_proba`` for the probabilistic estimators, ``decision_function`` for
the margin-based ones, and nothing at all for a nearest-centroid classifier.
Resolving that here, once, against a declared contract, is what keeps the
benchmark runner free of per-model special cases - and what makes an SVM
comparable to a logistic regression on ROC-AUC without anyone pretending the
SVM's signed distance is a probability.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_zoo.contracts import (
    Capabilities,
    CapabilityError,
    ProbabilityBehavior,
    SerializationRecord,
    TrainingRecord,
)


class SklearnAdapter:
    """Wraps a scikit-learn estimator or pipeline in the zoo's contract."""

    def __init__(
        self,
        model_id: str,
        estimator: Any,
        *,
        capabilities: Capabilities,
        probability_behavior: ProbabilityBehavior,
    ) -> None:
        self.model_id = model_id
        self.estimator = estimator
        self.capabilities = capabilities
        self.probability_behavior = probability_behavior
        self._training: TrainingRecord | None = None

    # ------------------------------------------------------------- fitting

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnAdapter:
        started = time.perf_counter()
        self.estimator.fit(X, y)
        self._training = TrainingRecord(
            fit_seconds=time.perf_counter() - started,
            training_rows=len(X),
            parameter_count=None,
        )
        return self

    @property
    def training_record(self) -> TrainingRecord | None:
        return self._training

    # ----------------------------------------------------------- inference

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability, if this model honestly has one."""
        if not self.capabilities.supports_predict_proba:
            raise CapabilityError(
                f"{self.model_id} declares no probability output "
                f"({self.probability_behavior.value}); ask for decision_scores instead"
            )
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]

    def decision_scores(self, X: pd.DataFrame) -> np.ndarray:
        """A monotone ranking score, by whichever route this model provides one.

        Order matters: probabilities are preferred when they exist, because a
        calibrated score is also a ranking score. A decision function is used
        only when there is no probability, and a model with neither raises
        rather than returning its hard labels dressed as scores - 0/1 labels
        would give a degenerate ROC curve that looks like a measurement.
        """
        if self.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
            raise CapabilityError(
                f"{self.model_id} produces hard labels only; threshold-free "
                "metrics are undefined for it and are reported as such"
            )
        if hasattr(self.estimator, "predict_proba") and self.capabilities.supports_predict_proba:
            return np.asarray(self.estimator.predict_proba(X))[:, 1]
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(X)).ravel()
        raise CapabilityError(f"{self.model_id} exposes no ranking score")

    # ------------------------------------------------------------ contents

    def feature_importance(self) -> dict[str, float] | None:
        """Native importance where the model has one, else None.

        Coefficients and impurity importances are not the same quantity and are
        not compared across families; each is reported beside its own model.
        """
        if not self.capabilities.supports_feature_importance:
            return None
        model = self._inner()
        values = getattr(model, "feature_importances_", None)
        if values is None:
            coefficients = getattr(model, "coef_", None)
            if coefficients is None:
                return None
            values = np.asarray(coefficients).ravel()
        names = getattr(self, "_feature_names", None) or [
            f"f{i}" for i in range(len(np.asarray(values)))
        ]
        return {name: float(value) for name, value in zip(names, np.asarray(values), strict=False)}

    def _inner(self) -> Any:
        """The estimator itself, past any preprocessing pipeline."""
        steps = getattr(self.estimator, "named_steps", None)
        if steps is not None and "model" in steps:
            return steps["model"]
        return self.estimator

    # ------------------------------------------------------- serialization

    def serialize(self, path: Path) -> SerializationRecord:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        return SerializationRecord(
            format="joblib",
            bytes_written=path.stat().st_size,
            round_trip_ok=True,
        )

    @staticmethod
    def deserialize(path: Path) -> Any:
        import joblib

        return joblib.load(Path(path))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter": "sklearn",
            "estimator": type(self._inner()).__name__,
            "probability_behavior": self.probability_behavior.value,
            "capabilities": self.capabilities.as_dict(),
            "training": self._training.as_dict() if self._training else None,
        }
