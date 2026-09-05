"""Run every applicable explanation method over the Track L model zoo.

The combinatorics are the whole problem. Twenty-nine active models times nine
methods is two hundred and sixty-one pairs, most of which are invalid - you
cannot take a gradient through a random forest and an RBF kernel has no
coefficients. Resolving that pairwise inside the loop would be a wall of
conditionals that silently rots as models are added, so each method declares the
capability it needs and each model declares what it has. This module only joins
the two and records what happened.

**Every attempted pair produces a row, including the ones that fail.** A results
file containing only the combinations that worked is a flattering file, and the
central claim of the capability contract is that a gap in the table is a
property of the model rather than an oversight in the harness. `UNSUPPORTED`
names the missing capability; `OPTIONAL_DEPENDENCY_MISSING` names the absent
library; `NUMERICAL_FAILURE` carries the exception text.

Three partition decisions, each of which could leak if made carelessly.

**Models are fitted on the same 1,000 TRAIN rows Track L used**, drawn through
Track K's fingerprinted subset ladder rather than a second sampler. The zoo's
explanations therefore describe the zoo's models, not near-copies of them.

**Permutation importance is scored on the VALIDATION partition**, never on the
fitting rows and never on test. Scored on the rows it was fitted to, a random
forest reports importances of about 0.12 ROC-AUC points on labels containing
nothing at all - it is being asked how much it needs each feature to reproduce
answers it memorised. Validation is the partition that exists for questions like
this, and Track L already used it for calibration, so nothing new is spent.

**Baselines and perturbation scales come from the TRAIN rows only.** A baseline
built from the evaluation distribution would smuggle it into the explanation of
a model being judged on it, and would look like nothing worse than unusually
well-behaved perturbations.

The manifest is written last. A run that dies halfway leaves records and no
manifest, which is unambiguous; a manifest written first would leave a run that
claims results it does not have.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.model_zoo.contracts import Framework, ResearchStatus
from research.model_zoo.registry import REGISTRY, ModelSpec
from research.track_k import artifacts as track_k_artifacts
from research.track_k import split, subsets
from research.xai import agreement, faithfulness, interactions, stability
from research.xai.capabilities import XaiCapability, profile_for
from research.xai.contracts import (
    ExplanationRecord,
    MethodOutcome,
    RunStatus,
    Scope,
    XaiError,
    hash_payload,
)
from research.xai.explainers import classical, deep
from research.xai.registry import METHODS, MethodSpec

#: Same budget as Track L, drawn through the same fingerprinted ladder, so the
#: explanations describe the zoo's models rather than near-copies of them.
DEFAULT_TRAIN_ROWS: int = 1000

#: Rows explained by the local methods. Forty is enough for a profile to be
#: stable while keeping integrated gradients - 128 forward-and-backward passes
#: per row per model - inside a single-CPU budget.
DEFAULT_CASE_LIMIT: int = 40

#: Validation rows the global methods are scored on.
#:
#: Sized from measurement. The whole validation partition is 13,376 rows and
#: partial dependence rescores it once per grid point per feature, so a single
#: model would cost about 2.7 million row-scores; a random forest at 2,000 rows
#: still took ten minutes, which puts a twenty-eight model sweep past four hours
#: for an estimate that is already stable.
#:
#: Five hundred rows are sampled deterministically instead, drawn once and
#: shared by every model, so a difference between two models is a difference
#: between the models rather than between their evaluation rows. The cost of
#: that choice is real and worth naming: permutation importance on 500 rows
#: carries roughly 0.02 ROC-AUC points of sampling noise, which is ample to
#: separate a leading feature from an inert one and not enough to order two
#: features that are genuinely close. The report flags near-ties for exactly
#: this reason rather than presenting the ordering as firm.
DEFAULT_EVAL_ROWS: int = 500

#: Grid resolution for the partial-dependence attribution. Ten points rather
#: than the module's twenty: the runner reads only the curve's *range*, and a
#: range measured over ten points across the observed support differs from one
#: measured over twenty by far less than the differences being compared, at half
#: the cost. The full-resolution curve is still what `partial_dependence`
#: returns when a shape is wanted rather than a scalar.
RUN_PD_GRID_POINTS: int = 10

#: Wall-clock ceiling per (model, method) pair. A method that exceeds it is
#: recorded as RESOURCE_LIMIT rather than left running; "too slow at this
#: budget" is a finding about the method, not a reason to have no result.
DEFAULT_METHOD_BUDGET_SECONDS: float = 240.0

#: Feature pairs measured for interaction per model. Ten features give
#: forty-five pairs and a full sweep costs minutes per model, so the runner
#: scopes it to the pairs among the consensus top five.
DEFAULT_INTERACTION_FEATURES: int = 5

#: Perturbation strengths the stability sweep uses, and replicates per strength.
#: Narrower than `stability.STABILITY_MAGNITUDES`: three points spanning
#: "a rounding error" to "a different patient" is enough to see whether the
#: leading feature holds, and the sweep is repeated for every model.
RUN_STABILITY_MAGNITUDES: tuple[float, ...] = (0.1, 0.5, 1.0)
RUN_STABILITY_REPEATS: int = 3

#: Rows the stability and faithfulness sweeps run on. Both re-score the sample
#: many times over, so they read the case sample rather than the full
#: evaluation partition.
RUN_SWEEP_ROWS: int = 40

XAI_ROOT = track_k_artifacts.PROJECT_ROOT / "research_artifacts" / "xai"

#: Stamped on every record. Track M inherits Track L's evidence class: broad,
#: cheap, exploratory, and not a basis for promoting anything.
EVIDENCE_CLASS = "RESOURCE_CONSTRAINED_EXPLORATORY"


@dataclass
class RunContext:
    """Everything a method needs, assembled once and shared by every model."""

    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_eval: pd.DataFrame
    y_eval: pd.Series
    cases: pd.DataFrame
    baseline: pd.Series
    scale: stability.PerturbationScale
    feature_names: tuple[str, ...]
    subset_manifest: dict[str, Any] = field(default_factory=dict)


# ============================================================ method execution

def _global_attributions(
    method: MethodSpec, model: Any, context: RunContext
) -> np.ndarray:
    """The one attribution vector a global method produces for a model."""
    if method.method_id == "coefficients":
        return classical.coefficient_attributions(model)
    if method.method_id == "native_importance":
        if getattr(model, "model", None) is not None:
            # The neural additive model's own per-feature terms, averaged into a
            # global profile. Exact by construction rather than approximated.
            terms = np.abs(deep.additive_contributions(model, context.cases))
            mean_terms: np.ndarray = terms.mean(axis=0)
            return mean_terms
        return classical.native_importance_attributions(model)
    if method.method_id == "permutation_importance":
        return classical.permutation_importance(
            model, context.X_eval, context.y_eval, seed=0
        )
    if method.method_id == "partial_dependence":
        # A PD curve is a shape, not a number. Its range across the observed
        # support is the comparable scalar: how much the model's average score
        # moves when this feature is swept. Flat curve, no attribution.
        return np.array(
            [
                classical.partial_dependence(
                    model, context.X_eval, feature, grid_points=RUN_PD_GRID_POINTS
                )["range"]
                for feature in context.feature_names
            ],
            dtype=float,
        )
    raise XaiError(f"{method.method_id} is not a global method this runner knows")


def _local_attributions(
    method: MethodSpec, model: Any, context: RunContext
) -> np.ndarray:
    """One attribution row per case, for a local method."""
    if method.method_id == "occlusion":
        return classical.occlusion_matrix(model, context.cases, context.baseline)
    if method.method_id == "gradient":
        return deep.input_gradient(model, context.cases)
    if method.method_id == "gradient_x_input":
        return deep.gradient_x_input(model, context.cases)
    if method.method_id == "integrated_gradients":
        return deep.integrated_gradients(model, context.cases, context.baseline)
    if method.method_id == "tree_shap":
        return _tree_shap(model, context)
    raise XaiError(f"{method.method_id} is not a local method this runner knows")


def _tree_shap(model: Any, context: RunContext) -> np.ndarray:
    """Exact Shapley values for a tree ensemble, if the optional library is here.

    Isolated in its own function because it is the one method whose absence is
    routine. The import failure is allowed to propagate; the caller turns it
    into OPTIONAL_DEPENDENCY_MISSING so the pair stays in the table as skipped
    rather than disappearing from it.
    """
    import shap  # type: ignore[import-untyped]

    estimator = classical._inner_estimator(model)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(context.cases, check_additivity=False)

    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        # Some versions return (rows, features, classes); take the positive one.
        array = array[..., -1]
    return array


def explain_pair(
    spec: ModelSpec,
    model: Any,
    method: MethodSpec,
    context: RunContext,
    *,
    budget: float = DEFAULT_METHOD_BUDGET_SECONDS,
) -> list[MethodOutcome]:
    """Run one method against one model, recording whatever happens.

    Returns a list because a local method produces one outcome per explained
    case while a global method produces exactly one. Both shapes go into the
    same results file, distinguished by the record's scope and sample_id.
    """
    profile = profile_for(spec)
    if not profile.supports(method.required_capability):
        return [MethodOutcome(
            model_id=spec.model_id,
            method=method.method_id,
            status=RunStatus.UNSUPPORTED,
            error=f"model does not provide {method.required_capability.value}",
        )]

    if not method.is_available():
        return [MethodOutcome(
            model_id=spec.model_id,
            method=method.method_id,
            status=RunStatus.OPTIONAL_DEPENDENCY_MISSING,
            error=f"optional dependency {method.optional_dependency!r} is not installed",
        )]

    started = time.perf_counter()
    try:
        if method.scope is Scope.GLOBAL:
            attributions = np.asarray(
                _global_attributions(method, model, context), dtype=float
            )
            rows = attributions[None, :]
        else:
            rows = np.atleast_2d(
                np.asarray(_local_attributions(method, model, context), dtype=float)
            )
    except ImportError as error:
        return [MethodOutcome(
            model_id=spec.model_id, method=method.method_id,
            status=RunStatus.OPTIONAL_DEPENDENCY_MISSING, error=str(error),
            runtime_seconds=time.perf_counter() - started,
        )]
    except Exception as error:  # noqa: BLE001 - a failed method is a recorded result
        return [MethodOutcome(
            model_id=spec.model_id, method=method.method_id,
            status=RunStatus.NUMERICAL_FAILURE,
            error=f"{type(error).__name__}: {error}",
            runtime_seconds=time.perf_counter() - started,
        )]

    elapsed = time.perf_counter() - started
    if elapsed > budget:
        return [MethodOutcome(
            model_id=spec.model_id, method=method.method_id,
            status=RunStatus.RESOURCE_LIMIT,
            error=f"took {elapsed:.1f}s against a {budget:.0f}s budget",
            runtime_seconds=elapsed,
        )]

    per_row = elapsed / max(len(rows), 1)
    outcomes: list[MethodOutcome] = []
    for position, values in enumerate(rows):
        record = classical.build_record(
            spec.model_id,
            method.method_id,
            method.version,
            method.scope,
            context.feature_names,
            values,
            baseline_reference=method.baseline_strategy.value,
            sample_id=None if method.scope is Scope.GLOBAL else position,
            seed=0 if method.determinism.value == "stochastic" else None,
            runtime_seconds=per_row,
        )
        outcomes.append(MethodOutcome(
            model_id=spec.model_id,
            method=method.method_id,
            status=RunStatus.SUCCESS,
            record=record,
            runtime_seconds=per_row,
        ))
    return outcomes


# ==================================================================== planning

def plan(
    model_ids: list[str] | None = None, method_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """The (model, method) matrix, resolved but not executed.

    What `--dry-run` prints. Being able to see which pairs are supported before
    spending an hour on them is the difference between a budget and a hope.
    """
    specs = [REGISTRY.get(mid) for mid in model_ids] if model_ids else [
        s for s in REGISTRY if s.effective_status() is ResearchStatus.ACTIVE
    ]
    methods = [METHODS.get(mid) for mid in method_ids] if method_ids else list(METHODS)

    matrix = []
    for spec in specs:
        profile = profile_for(spec)
        for method in methods:
            supported = profile.supports(method.required_capability)
            matrix.append({
                "model_id": spec.model_id,
                "family": spec.family.value,
                "method": method.method_id,
                "scope": method.scope.value,
                "supported": supported,
                "available": method.is_available(),
                "runtime_class": method.runtime_class,
                "reason": None if supported else (
                    f"model does not provide {method.required_capability.value}"
                ),
            })
    return matrix


# ===================================================================== the run

def _build_context(
    train_rows: int,
    case_limit: int,
    *,
    eval_rows: int = DEFAULT_EVAL_ROWS,
    seed: int = 0,
) -> tuple[RunContext, dict[str, Any]]:
    """Assemble the partitions once, verifying the split before anything else."""
    frame = split.load_dataset()
    splits = split.build_split(frame)
    drift = split.verify_split(splits, split.build_split_manifest(splits))
    if drift:
        raise split.SplitIntegrityError("; ".join(drift))

    manifest = subsets.build_subset_manifest(splits, sizes=(train_rows,))
    ladder = subsets.load_verified_subsets(splits, manifest)
    narrowed = subsets.take(splits, ladder[train_rows])

    X_fit, y_fit = narrowed.X_train, narrowed.y_train

    # One evaluation sample, drawn once and shared by every model, so a
    # difference between two models is a difference between the models.
    rng = np.random.default_rng(seed)
    full_val = narrowed.X_val
    take = min(eval_rows, len(full_val))
    evaluation = np.sort(rng.choice(len(full_val), size=take, replace=False))
    X_eval = full_val.iloc[evaluation].reset_index(drop=True)
    y_eval = narrowed.y_val.iloc[evaluation].reset_index(drop=True)

    chosen = np.sort(rng.choice(len(X_eval), size=min(case_limit, len(X_eval)), replace=False))
    cases = X_eval.iloc[chosen].reset_index(drop=True)

    context = RunContext(
        X_fit=X_fit,
        y_fit=y_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        cases=cases,
        baseline=X_fit.median(axis=0),
        scale=stability.fit_scale(X_fit),
        feature_names=tuple(str(column) for column in X_fit.columns),
        subset_manifest=manifest,
    )
    provenance = {
        "train_rows": len(X_fit),
        "evaluation_rows": len(X_eval),
        "case_rows": len(cases),
        "case_indices": [int(i) for i in chosen],
        "baseline_source": "median of the fitting rows",
        "permutation_scored_on": "validation partition, never the fitting rows or test",
        "subset_fingerprints": manifest,
    }
    return context, provenance


def _fit_model(spec: ModelSpec, context: RunContext, overrides: dict[str, Any]) -> Any:
    config = dict(overrides) if spec.framework is Framework.TORCH else {}
    return REGISTRY.build(spec.model_id, **config).fit(context.X_fit, context.y_fit)


def run(
    *,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    model_ids: list[str] | None = None,
    method_ids: list[str] | None = None,
    case_limit: int = DEFAULT_CASE_LIMIT,
    eval_rows: int = DEFAULT_EVAL_ROWS,
    output_root: Path | None = None,
    method_budget: float = DEFAULT_METHOD_BUDGET_SECONDS,
    overrides: dict[str, Any] | None = None,
    with_interactions: bool = True,
    with_sweeps: bool = True,
) -> dict[str, Any]:
    """Explain the zoo, write the evidence, and return the manifest."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    run_id = f"xai-{train_rows}-{uuid.uuid4().hex[:10]}"
    root = Path(output_root) if output_root is not None else XAI_ROOT
    out_dir = root / run_id
    (out_dir / "records").mkdir(parents=True, exist_ok=True)

    context, provenance = _build_context(train_rows, case_limit, eval_rows=eval_rows)

    specs = [REGISTRY.get(mid) for mid in model_ids] if model_ids else [
        s for s in REGISTRY if s.effective_status() is ResearchStatus.ACTIVE
    ]
    methods = [METHODS.get(mid) for mid in method_ids] if method_ids else list(METHODS)

    outcomes: list[MethodOutcome] = []
    records: list[ExplanationRecord] = []
    fit_failures: dict[str, str] = {}
    interaction_summaries: dict[str, Any] = {}
    stability_summaries: dict[str, Any] = {}
    faithfulness_summaries: dict[str, Any] = {}

    for spec in specs:
        try:
            model = _fit_model(spec, context, overrides or {})
        except Exception as error:  # noqa: BLE001 - a model that will not fit is a result
            fit_failures[spec.model_id] = f"{type(error).__name__}: {error}"
            continue

        model_records: list[ExplanationRecord] = []
        for method in methods:
            for outcome in explain_pair(spec, model, method, context, budget=method_budget):
                outcomes.append(outcome)
                if outcome.record is not None:
                    records.append(outcome.record)
                    model_records.append(outcome.record)

        if with_interactions and model_records:
            interaction_summaries[spec.model_id] = _interactions_for(
                model, context, model_records
            )
        if with_sweeps and model_records:
            faithfulness_summaries[spec.model_id] = _faithfulness_for(
                model, context, model_records
            )
            if profile_for(spec).supports(XaiCapability.OCCLUSION_COMPATIBLE):
                stability_summaries[spec.model_id] = _stability_for(model, context)

        _write_json(
            out_dir / "records" / f"{spec.model_id}.json",
            [record.as_dict() for record in model_records],
        )

    analysis = _analyse(records)
    _write_json(out_dir / "outcomes.json", [o.as_dict() for o in outcomes])
    _write_json(out_dir / "analysis.json", analysis)
    if interaction_summaries:
        _write_json(out_dir / "interactions.json", interaction_summaries)
    if stability_summaries:
        _write_json(out_dir / "stability.json", stability_summaries)
    if faithfulness_summaries:
        _write_json(out_dir / "faithfulness.json", faithfulness_summaries)

    manifest = {
        "run_id": run_id,
        "evidence_class": EVIDENCE_CLASS,
        "started_at": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "train_rows": train_rows,
        "case_limit": case_limit,
        "eval_rows": len(context.X_eval),
        "models_requested": [s.model_id for s in specs],
        "methods_requested": [m.method_id for m in methods],
        "fit_failures": fit_failures,
        "provenance": provenance,
        "counts": _counts(outcomes),
        "analysis": analysis,
        "interactions": interaction_summaries,
        "stability": stability_summaries,
        "faithfulness": faithfulness_summaries,
        "records_hash": hash_payload([r.as_dict() for r in records]),
    }
    # Written last, deliberately: a run that dies halfway leaves records and no
    # manifest, which is unambiguous. The reverse would claim results it lacks.
    _write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def _occlusion_profile(model: Any, frame: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
    """Mean absolute occlusion over a frame: a global profile from a local method.

    Used as the stability probe because it is the one method available on every
    model with a ranking score, which is what makes a stability number
    comparable across families rather than a property of whichever method
    happened to apply.
    """
    stacked = classical.occlusion_matrix(model, frame, baseline)
    profile: np.ndarray = np.abs(stacked).mean(axis=0)
    return profile


def _stability_for(model: Any, context: RunContext) -> dict[str, Any]:
    """How far the explanation moves when the data is perturbed.

    The model is not refitted. This is the narrower of the two possible
    questions - whether a fixed model's explanation is robust to the rows it is
    explained on - and not whether a model refitted on perturbed data would find
    the same structure, which is a question about training variance.
    """
    sample = context.cases.iloc[:RUN_SWEEP_ROWS]
    try:
        points = stability.stability_curve(
            lambda frame: _occlusion_profile(model, frame, context.baseline),
            sample,
            context.scale,
            magnitudes=RUN_STABILITY_MAGNITUDES,
            repeats=RUN_STABILITY_REPEATS,
            seed=0,
        )
    except Exception as error:  # noqa: BLE001 - a failed sweep is a recorded result
        return {"error": f"{type(error).__name__}: {error}"}
    return {"probe": "occlusion profile", **stability.summarise(points)}


def _faithfulness_for(
    model: Any, context: RunContext, records: list[ExplanationRecord]
) -> dict[str, Any]:
    """Whether this model's consensus ranking predicts what the model uses.

    Scored against shuffled rankings over the same rows. On a model dominated by
    one feature almost any ranking eventually produces a dramatic curve, so only
    the gap between the ranking and the shuffle says the order carried
    information.
    """
    sample = context.cases.iloc[:RUN_SWEEP_ROWS]
    ranking = agreement.consensus_ranking(records)
    try:
        result = faithfulness.evaluate(
            model, sample, ranking, context.baseline, seed=0
        )
    except Exception as error:  # noqa: BLE001 - a failed sweep is a recorded result
        return {"error": f"{type(error).__name__}: {error}", "ranking": list(ranking)}
    return {"ranking": list(ranking), **result.as_dict()}


def _interactions_for(
    model: Any, context: RunContext, records: list[ExplanationRecord]
) -> dict[str, Any]:
    """Interaction sweep over the pairs among this model's own top features.

    Scoped rather than exhaustive. A full forty-five-pair sweep costs minutes
    per model, and the pairs worth measuring are the ones among features the
    model actually leans on - an interaction between two features it ignores is
    not a finding about this model.
    """
    consensus = agreement.consensus_ranking(records)
    features = list(consensus[:DEFAULT_INTERACTION_FEATURES])
    try:
        ranked = interactions.rank_interactions(model, context.X_eval, features=features)
    except Exception as error:  # noqa: BLE001 - a failed sweep is a recorded result
        return {"error": f"{type(error).__name__}: {error}", "features": features}
    return {"features": features, **interactions.summarise(ranked)}


def _analyse(records: list[ExplanationRecord]) -> dict[str, Any]:
    """Every agreement grouping the records support, and nothing they do not."""
    if not records:
        return {"records": 0}

    return {
        "records": len(records),
        "within_model": agreement.summarise(agreement.within_model(records)),
        "within_family": agreement.summarise(agreement.within_family(records)),
        "between_families": agreement.summarise(agreement.between_families(records)),
        "zoo_consensus": list(agreement.consensus_ranking(records)),
        "zoo_mean_ranks": agreement.mean_ranks(records),
        "family_consensus": {
            family: list(order)
            for family, order in agreement.family_consensus(records).items()
        },
    }


def _counts(outcomes: list[MethodOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status.value] = counts.get(outcome.status.value, 0) + 1
    return counts


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def summarise(manifest: dict[str, Any]) -> str:
    """A short human summary, failures included rather than filtered out."""
    counts = manifest.get("counts", {})
    lines = [
        f"run {manifest['run_id']} ({manifest['evidence_class']})",
        f"  {manifest['train_rows']} training rows, {manifest['case_limit']} explained cases",
        f"  models: {len(manifest['models_requested'])}, "
        f"methods: {len(manifest['methods_requested'])}",
        "  outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    ]
    if manifest.get("fit_failures"):
        lines.append(f"  models that would not fit: {sorted(manifest['fit_failures'])}")

    analysis = manifest.get("analysis", {})
    if analysis.get("records"):
        lines.append(f"  zoo consensus top three: {analysis['zoo_consensus'][:3]}")
        between = analysis.get("between_families", {})
        lines.append(
            "  cross-family top-1 agreement: "
            f"{_optional(between.get('top_1_agreement_rate'))}"
        )
    return "\n".join(lines)


def _optional(value: float | None) -> str:
    """Format a statistic that may be undefined for this set of records."""
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain the Track L model zoo.")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--case-limit", type=int, default=DEFAULT_CASE_LIMIT)
    parser.add_argument(
        "--eval-rows", type=int, default=DEFAULT_EVAL_ROWS,
        help="Validation rows the global methods are scored on.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method-budget", type=float, default=DEFAULT_METHOD_BUDGET_SECONDS
    )
    parser.add_argument(
        "--no-interactions", action="store_true",
        help="Skip the interaction sweep, which dominates the run's cost.",
    )
    parser.add_argument(
        "--no-sweeps", action="store_true",
        help="Skip the stability and faithfulness sweeps.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the (model, method) matrix without running anything.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        matrix = plan(args.models, args.methods)
        supported = sum(1 for row in matrix if row["supported"] and row["available"])
        print(f"{len(matrix)} pairs, {supported} runnable")
        for row in matrix:
            mark = "run " if row["supported"] and row["available"] else "skip"
            print(f"  {mark} {row['model_id']:24s} {row['method']:22s} {row['reason'] or ''}")
        return 0

    manifest = run(
        train_rows=args.train_rows,
        model_ids=args.models,
        method_ids=args.methods,
        case_limit=args.case_limit,
        eval_rows=args.eval_rows,
        output_root=args.output_dir,
        method_budget=args.method_budget,
        with_interactions=not args.no_interactions,
        with_sweeps=not args.no_sweeps,
    )
    print(summarise(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
