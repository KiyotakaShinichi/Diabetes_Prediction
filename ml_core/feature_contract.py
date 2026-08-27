"""The canonical feature contract for the SERVED diabetes-risk models.

One authoritative, ordered description of the features the served models
consume, used by both maintained training pipelines, the FastAPI request schema,
the Streamlit UI, drift checking and provenance manifests. Before this module
the same list, labels, target column and bounds were maintained in five places
that merely happened to agree.

SCOPE. This describes the SERVED model schema only - the ten BRFSS short-name
columns in ``cleaned_data.csv`` with target ``Diabetes_binary``. The archived
single-model experiments were written against a different historical dataset
(``cleaned_data_upd.csv``, target ``DiabetesStatus``, long-form names such as
``GeneralHealth`` and ``HasHighBP``). Those are a genuinely different schema and
are deliberately NOT remapped here; see experiment_config.py.

ORDER IS PART OF THE CONTRACT. The committed bundles record this exact order in
``feature_columns`` and in ``pipeline.feature_names_in_``, so reordering
FEATURE_SPECS silently invalidates every committed artifact. The test suite
asserts the order against the artifacts themselves.

Pure and import-safe: no I/O, no model loading, no global state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

#: Target column of the served training dataset.
TARGET_COLUMN = "Diabetes_binary"

FeatureKind = Literal["binary", "ordinal", "continuous"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One served feature. Immutable: the contract is not adjustable at runtime.

    ``minimum``/``maximum`` are INCLUSIVE and are exactly the bounds the API
    enforces via Pydantic ``ge``/``le``. ``description`` is the API-facing text;
    ``display_label`` is the human-facing text used by the UI and stored in the
    model bundles.
    """

    name: str
    display_label: str
    description: str
    kind: FeatureKind
    dtype: type
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum {self.minimum} exceeds maximum {self.maximum}")
        if self.kind == "binary" and (self.minimum, self.maximum) != (0, 1):
            raise ValueError(f"{self.name}: a binary feature must be bounded 0..1")

    @property
    def allowed_values(self) -> tuple[int, ...] | None:
        """The discrete domain for binary and ordinal features, else None."""
        if self.kind == "continuous":
            return None
        return tuple(range(int(self.minimum), int(self.maximum) + 1))


#: THE contract. Order is significant and matches every committed artifact.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="GenHlth",
        display_label="General Health (1=Excellent to 5=Poor)",
        description="General health (1=Excellent to 5=Poor)",
        kind="ordinal", dtype=int, minimum=1, maximum=5,
    ),
    FeatureSpec(
        name="HighBP",
        display_label="Has High Blood Pressure",
        description="High blood pressure (0=No, 1=Yes)",
        kind="binary", dtype=int, minimum=0, maximum=1,
    ),
    FeatureSpec(
        name="BMI",
        display_label="Body Mass Index",
        description="Body Mass Index",
        kind="continuous", dtype=float, minimum=10, maximum=80,
    ),
    FeatureSpec(
        name="HighChol",
        display_label="Has High Cholesterol",
        description="High cholesterol (0=No, 1=Yes)",
        kind="binary", dtype=int, minimum=0, maximum=1,
    ),
    FeatureSpec(
        name="Age",
        display_label="Age Category (1=18-24 to 13=80+)",
        description="Age category (1=18-24 to 13=80+)",
        kind="ordinal", dtype=int, minimum=1, maximum=13,
    ),
    FeatureSpec(
        name="DiffWalk",
        display_label="Has Walking Difficulty",
        description="Difficulty walking (0=No, 1=Yes)",
        kind="binary", dtype=int, minimum=0, maximum=1,
    ),
    FeatureSpec(
        name="HeartDiseaseorAttack",
        display_label="Has Heart Disease or Had Heart Attack",
        description="Heart disease/MI history",
        kind="binary", dtype=int, minimum=0, maximum=1,
    ),
    FeatureSpec(
        name="PhysHlth",
        display_label="Poor Physical Health Days (last 30 days)",
        description="Poor physical health days (0-30)",
        kind="ordinal", dtype=int, minimum=0, maximum=30,
    ),
    FeatureSpec(
        name="Education",
        display_label="Education Level (1-6)",
        description="Education level (1-6)",
        kind="ordinal", dtype=int, minimum=1, maximum=6,
    ),
    FeatureSpec(
        name="PhysActivity",
        display_label="Is Physically Active",
        description="Physical activity (0=No, 1=Yes)",
        kind="binary", dtype=int, minimum=0, maximum=1,
    ),
)

if len({spec.name for spec in FEATURE_SPECS}) != len(FEATURE_SPECS):
    raise ValueError("duplicate feature name in FEATURE_SPECS")

#: Canonical ordered feature names. Everything downstream derives from this.
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

#: Human-readable labels, derived - not separately maintained.
FEATURE_LABELS: dict[str, str] = {spec.name: spec.display_label for spec in FEATURE_SPECS}

#: name -> spec, for lookups that should not scan the tuple.
FEATURE_INDEX: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}

FEATURE_COUNT = len(FEATURE_SPECS)


def feature_list() -> list[str]:
    """Canonical names as a fresh mutable list, for callers that need one."""
    return list(FEATURE_NAMES)


def spec_for(name: str) -> FeatureSpec:
    try:
        return FEATURE_INDEX[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a served feature; expected one of {FEATURE_NAMES}") from None


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex a DataFrame into canonical feature order.

    Serving must never rely on dict insertion order, JSON key order or Pydantic
    field ordering. Callers pass their payload through here so the model always
    receives columns in the order it was trained on.
    """
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise KeyError(f"missing served features: {missing}")
    return frame[list(FEATURE_NAMES)]
