"""Capability matrix and model cards, generated - never hand-written.

There is exactly one roster of models in this repository: the registry. The
matrix and the cards are projections of it plus a run's measured results, which
is the point. A hand-maintained table beside a registry of thirty models is a
table that is wrong within a month, and worse, wrong in a flattering direction:
the entries that rot first are the caveats.

So the generator refuses to invent anything. A model that failed gets a card
saying it failed and why. A model with no probabilities gets a card saying its
threshold-free metrics are undefined, not a card with blanks where the good
numbers would go. Limitations come from the same structured record as the
metrics, so a favourable summary cannot be written by omission.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.model_zoo.contracts import Family, ProbabilityBehavior, RunOutcome
from research.model_zoo.registry import REGISTRY


def capability_matrix() -> list[dict[str, Any]]:
    """Machine-readable capabilities for every registered model."""
    rows = []
    for spec in REGISTRY:
        capabilities = spec.capabilities
        rows.append({
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "family": spec.family.value,
            "framework": spec.framework.value,
            "preprocessing": spec.preprocessing.value,
            "probability_behavior": spec.probability_behavior.value,
            "supports_predict_proba": capabilities.supports_predict_proba,
            "supports_calibration": capabilities.supports_calibration,
            "supports_feature_importance": capabilities.supports_feature_importance,
            "supports_serialization": capabilities.supports_serialization,
            "requires_scaling": capabilities.requires_scaling,
            "deterministic": capabilities.deterministic,
            "resource_class": spec.resource_class.value,
            "optional_dependency": spec.optional_dependency,
            "status": spec.effective_status().value,
        })
    return rows


def render_capability_table(rows: list[dict[str, Any]] | None = None) -> str:
    """The same data as a markdown table, from the same source."""
    rows = rows if rows is not None else capability_matrix()
    header = (
        "| Model | Family | Framework | Probability | Proba | Calib | Import | "
        "Serial | Scaling | Cost | Status |"
    )
    lines = [header, "| " + " | ".join(["---"] * 11) + " |"]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['family']} | {row['framework']} "
            f"| {row['probability_behavior'].replace('_', ' ')} "
            f"| {_tick(row['supports_predict_proba'])} "
            f"| {_tick(row['supports_calibration'])} "
            f"| {_tick(row['supports_feature_importance'])} "
            f"| {_tick(row['supports_serialization'])} "
            f"| {_tick(row['requires_scaling'])} "
            f"| {row['resource_class']} | {row['status']} |"
        )
    return "\n".join(lines)


def _tick(value: bool) -> str:
    return "yes" if value else "no"


def build_card(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One model's card, assembled from its spec and its measured result."""
    model_id = result["model_id"]
    spec = REGISTRY.get(model_id)
    metrics = result.get("metrics") or {}
    training = result.get("training") or {}
    outcome = RunOutcome(result["outcome"])

    return {
        "model_id": model_id,
        "display_name": spec.display_name,
        "family": spec.family.value,
        "framework": spec.framework.value,
        "evidence_class": manifest["evidence_class"],
        "status": outcome.value,
        "why_included": spec.rationale,
        "assumptions": _assumptions(spec.family, spec.probability_behavior),
        "preprocessing": spec.preprocessing.value,
        "configuration": dict(spec.default_config),
        "seed": spec.seed,
        "resources": {
            "training_rows": training.get("training_rows"),
            "fit_seconds": training.get("fit_seconds"),
            "parameter_count": training.get("parameter_count"),
            "epochs_run": training.get("epochs_run"),
            "resource_class": spec.resource_class.value,
        },
        "metrics": {
            name: metrics.get(name)
            for name in ("roc_auc", "pr_auc", "recall", "precision", "specificity",
                         "f1", "brier_score", "log_loss", "ece")
        },
        "calibration": {
            "behavior": spec.probability_behavior.value,
            "note": _calibration_note(spec.probability_behavior),
        },
        "serialization": result.get("serialization") or {},
        "limitations": _limitations(spec, result, manifest),
        "error": result.get("error"),
    }


def _assumptions(family: Family, behavior: ProbabilityBehavior) -> list[str]:
    """What this family assumes about the data, stated plainly."""
    by_family = {
        Family.LINEAR: "A decision boundary that is linear in the standardised features.",
        Family.PROBABILISTIC: (
            "A generative story for each class - shared or per-class Gaussians, "
            "or conditional independence between features."
        ),
        Family.KERNEL: "Separability in a kernel-induced space; a margin, not a likelihood.",
        Family.DISTANCE: "That nearby rows in scaled feature space share a label.",
        Family.TREE: "Axis-aligned splits; scale-invariant, interaction-friendly.",
        Family.BOOSTING: (
            "Additive stagewise correction of residuals; typically needs more "
            "data than a linear model to pay off."
        ),
        Family.DEEP: "A smooth function learnable by gradient descent from limited rows.",
    }
    assumptions = [by_family[family]]
    if behavior is ProbabilityBehavior.NATIVE_UNCALIBRATED:
        assumptions.append(
            "Its probability output is not fitted to be calibrated; read the "
            "ranking metrics before the Brier score."
        )
    if behavior is ProbabilityBehavior.REQUIRES_EXTERNAL_CALIBRATION:
        assumptions.append(
            "It has no native probability at all; the reported threshold metrics "
            "come from a squashed decision margin, not a probability estimate."
        )
    return assumptions


def _calibration_note(behavior: ProbabilityBehavior) -> str:
    return {
        ProbabilityBehavior.NATIVE_PROBABILISTIC: (
            "Fitted by maximum likelihood; Brier and ECE are meaningful."
        ),
        ProbabilityBehavior.NATIVE_UNCALIBRATED: (
            "Produces a distribution but was not fitted to calibrate it. Its "
            "Brier score and ECE describe the raw output, not a calibrated one."
        ),
        ProbabilityBehavior.REQUIRES_EXTERNAL_CALIBRATION: (
            "No native probability. Calibration would have to be fitted on "
            "validation rows; Track L reports the uncalibrated ranking instead."
        ),
        ProbabilityBehavior.HARD_LABELS_ONLY: (
            "No score of any kind. Threshold-free metrics are undefined and are "
            "reported as such rather than computed from labels."
        ),
    }[behavior]


def _limitations(spec: Any, result: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Every caveat that applies to this card, drawn from structured data."""
    limitations = [
        f"Trained on {manifest['train_rows']:,} rows - {manifest['evidence_class']}. "
        "Track K measured this dataset's families to be strongly sample-dependent, "
        "so this ranking describes the constrained regime only.",
        "Configuration is a frozen sensible default, not a tuned one.",
    ]
    if spec.probability_behavior is ProbabilityBehavior.HARD_LABELS_ONLY:
        limitations.append("Threshold-free metrics are undefined for this model.")
    if spec.probability_behavior is ProbabilityBehavior.NATIVE_UNCALIBRATED:
        limitations.append("Probabilities are uncalibrated; ECE reflects that.")
    serialization = result.get("serialization") or {}
    if serialization and not serialization.get("round_trip_ok", True):
        limitations.append(
            f"Failed its serialization round trip: {serialization.get('error')}"
        )
    if result["outcome"] != RunOutcome.COMPLETED.value:
        limitations.append(f"Did not complete: {result.get('error')}")
    return limitations


def render_card(card: dict[str, Any]) -> str:
    """One card as readable markdown."""
    lines = [
        f"### {card['display_name']} (`{card['model_id']}`)",
        "",
        f"**Family** {card['family']} · **Framework** {card['framework']} · "
        f"**Status** {card['status']}",
        "",
        f"*{card['why_included']}*",
        "",
        "**Assumptions**",
    ]
    lines += [f"- {item}" for item in card["assumptions"]]

    metrics = {k: v for k, v in card["metrics"].items() if v is not None}
    if metrics:
        lines += ["", "**Measured**", "", "| Metric | Value |", "| --- | --- |"]
        lines += [f"| {name} | {value:.5f} |" for name, value in metrics.items()]

    resources = card["resources"]
    lines += [
        "",
        f"**Cost** {resources['fit_seconds']}s on {resources['training_rows']} rows"
        + (
            f", {resources['parameter_count']:,} parameters"
            if resources.get("parameter_count")
            else ""
        ),
        "",
        f"**Calibration** {card['calibration']['note']}",
        "",
        "**Limitations**",
    ]
    lines += [f"- {item}" for item in card["limitations"]]
    return "\n".join(lines)


def generate(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write the capability matrix and every card for a completed run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = capability_matrix()
    (out_dir / "capability_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8"
    )
    (out_dir / "capability_matrix.md").write_text(
        render_capability_table(matrix), encoding="utf-8"
    )

    cards = [build_card(result, manifest) for result in manifest["results"]]
    (out_dir / "model_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")

    rendered = [
        f"# Track L model cards — run {manifest['run_id']}",
        "",
        f"**{manifest['evidence_class']}** · {manifest['train_rows']:,} training rows",
        "",
    ]
    rendered += [render_card(card) + "\n" for card in cards]
    path = out_dir / "model_cards.md"
    path.write_text("\n".join(rendered), encoding="utf-8")
    return path
