"""The frozen Track K benchmark protocol.

Everything a result depends on is declared here, in code, and committed BEFORE
any model is evaluated on the held-out test set. That ordering is the whole
point: metrics chosen after seeing which model wins are not evidence, and
thresholds invented to fit a result are not a policy.

The companion prose is docs/research/track_k_protocol.md. This module is the
machine-readable half, so a test can assert the two agree and a benchmark run
can record which protocol version produced it.

Nothing here trains, loads or evaluates anything - it is constants and the
small pure functions that derive from them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_core import feature_contract

#: Bump when any frozen decision below changes. Recorded in every artifact, so a
#: result can never be silently compared against one produced under other rules.
PROTOCOL_VERSION: Final[str] = "1.0.0"

#: The single dataset every model sees. Committed to the repository.
DATASET_FILENAME: Final[str] = "cleaned_data.csv"

#: The served feature contract, reused rather than redefined. Track K
#: deliberately benchmarks the SAME ten features production serves, so a result
#: transfers to the deployed problem instead of describing a different one.
FEATURE_NAMES: Final[tuple[str, ...]] = feature_contract.FEATURE_NAMES
TARGET_COLUMN: Final[str] = feature_contract.TARGET_COLUMN

#: Split proportions. Matched to the existing production convention - hold out
#: 20% as test, then 25% of the remainder as validation - so Track K numbers sit
#: on the same footing as the pipelines already in the repository.
TEST_SIZE: Final[float] = 0.2
VALIDATION_SIZE_OF_REMAINDER: Final[float] = 0.25

#: One seed governs the split for every model family. Model-specific seeds are
#: derived from it so a run is reproducible end to end without one global
#: mutable random state.
SPLIT_SEED: Final[int] = 42

#: Stratified, because even a balanced dataset can drift by chance in a 20%
#: holdout, and every model must see identical rows.
STRATIFY: Final[bool] = True

#: ---------------------------------------------------------------- metrics
#:
#: PRIMARY METRIC: ROC-AUC. The reasoning, recorded before any result exists:
#:
#: The brief warns against reaching for ROC-AUC reflexively, and against PR-AUC
#: unless imbalance justifies it. The dataset decides this. cleaned_data.csv is
#: 66,877 rows at 49.95% positive - a 1.002:1 ratio. PR-AUC earns its place when
#: positives are rare, because a precision-recall curve stays informative where
#: ROC's false-positive axis is dominated by an enormous negative class. At
#: parity that advantage does not exist: the PR baseline is 0.5, and PR-AUC
#: carries no information ROC-AUC lacks while being harder to compare against
#: the figures this repository already publishes.
#:
#: ROC-AUC is therefore primary as a threshold-free measure of ranking quality.
#: PR-AUC is still computed and reported as a secondary metric, so the choice
#: can be re-examined rather than taken on trust.
#:
#: Discrimination alone is not the whole decision, which is why the promotion
#: policy below adds calibration and recall guardrails: this product shows a
#: visitor a probability, so a model that ranks well but reports badly
#: calibrated numbers is not an improvement.
PRIMARY_METRIC: Final[str] = "roc_auc"

SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "brier_score",
    "log_loss",
)

CALIBRATION_METRICS: Final[tuple[str, ...]] = ("ece", "calibration_slope", "calibration_intercept")

#: Equal-width bins for expected calibration error. Ten is conventional and
#: keeps roughly 1,300 test rows per bin at this dataset size, which is enough
#: for a bin mean to mean something.
ECE_BINS: Final[int] = 10

#: ------------------------------------------------------------ uncertainty
#:
#: Every model predicts the SAME test rows, so comparisons are paired: the
#: bootstrap resamples row indices once per replicate and applies that identical
#: index set to every model. Pairing removes the between-model variance that
#: comes purely from which rows landed in the test split, which is exactly the
#: variance an unpaired comparison would mistake for a difference between models.
BOOTSTRAP_RESAMPLES: Final[int] = 2000
BOOTSTRAP_ALPHA: Final[float] = 0.05
BOOTSTRAP_SEED: Final[int] = 20260828

#: -------------------------------------------------------- promotion policy
#:
#: Frozen before the final evaluation. A challenger must clear every gate; the
#: verdict is otherwise INCONCLUSIVE or REJECT. Nothing here is a claim that a
#: promoted model would be deployed - Track K decides only whether a challenger
#: has earned promotion CONSIDERATION, and production serving is untouched.


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Thresholds a challenger must clear to be worth considering.

    Every value is justified in docs/research/track_k_protocol.md. They are
    deliberately modest: the point is to detect an improvement large enough to
    be worth added serving complexity, not to find the largest number.
    """

    #: The champion a challenger is measured against: the strongest classical
    #: baseline trained under this same protocol, decided by the primary metric.
    #: Named rather than hardcoded so the policy does not assume which one wins.
    baseline: str = "best_classical"

    #: A paired bootstrap interval for (challenger - baseline) on the primary
    #: metric must lie entirely above this value. Zero would promote a
    #: difference of any size; 0.005 ROC-AUC is the smallest gap worth the
    #: operational cost of serving a neural network in place of a linear model
    #: or a gradient-boosted ensemble.
    min_primary_delta: float = 0.005

    #: Calibration may not get materially worse. A visitor is shown a
    #: percentage, so a better-ranking but worse-calibrated model is not an
    #: improvement to this product.
    max_ece_regression: float = 0.01

    #: Screening context: a challenger may not lose meaningful sensitivity at
    #: the operating threshold, even for a discrimination gain.
    max_recall_regression: float = 0.02

    #: Serving cost ceiling, as a multiple of the baseline's median single-row
    #: CPU latency. A large multiple would need a correspondingly large
    #: accuracy gain, which min_primary_delta alone does not express.
    max_latency_multiple: float = 10.0


PROMOTION_POLICY: Final[PromotionPolicy] = PromotionPolicy()

#: Verdicts a comparison may produce. No fourth option exists, so a result
#: cannot be described as "promising" when the interval includes zero.
VERDICTS: Final[tuple[str, ...]] = ("PROMOTE", "REJECT", "INCONCLUSIVE")

#: Interpretations a paired comparison may carry.
COMPARISON_OUTCOMES: Final[tuple[str, ...]] = (
    "CLEAR IMPROVEMENT",
    "CLEAR REGRESSION",
    "INCONCLUSIVE",
)

#: The four families under study. Order is presentation order in the report.
MODEL_FAMILIES: Final[tuple[str, ...]] = (
    "logistic_regression",
    "xgboost",
    "mlp",
    "ft_transformer",
)

#: Classical families, i.e. the baselines a deep challenger must beat.
CLASSICAL_FAMILIES: Final[tuple[str, ...]] = ("logistic_regression", "xgboost")

#: Deep-learning challengers.
DEEP_FAMILIES: Final[tuple[str, ...]] = ("mlp", "ft_transformer")


def model_seed(family: str, *, base: int = SPLIT_SEED) -> int:
    """A stable per-family seed derived from the split seed.

    Deriving rather than hardcoding means one documented number reproduces the
    entire benchmark, and two families cannot accidentally share a seed and
    appear to agree for the wrong reason.
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family: {family!r}")
    return base + 1000 * (MODEL_FAMILIES.index(family) + 1)


def is_deep(family: str) -> bool:
    return family in DEEP_FAMILIES
