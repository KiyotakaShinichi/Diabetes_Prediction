"""Turn a run manifest into prose a person can check.

The hard part of reporting this track is not the numbers; it is refusing to
round them into a story. Three habits are built in rather than left to whoever
writes the summary.

**The failures are reported first.** Outcome counts lead, before any agreement
figure, because a cross-family agreement statistic computed over three models
means something different from one computed over twenty-eight and the reader
needs the denominator before the number.

**Every agreement figure is qualified by the reading that produced it.** On this
data, full-ranking correlation is dominated by the uninformative tail: three
explainers over a world with one true driver agree on the top feature at every
seed while correlating at 0.20 to 0.54, a band that overlaps what the same
methods score on pure noise. The report says so wherever it prints a
correlation, so the sentence "the methods agree, Spearman 0.5" cannot be written
without its contradiction beside it.

**A near-tie is shown as a near-tie.** A consensus ranking presents first and
second as firmly whether they are separated by six ranks or by five hundredths
of one, so the mean ranks are printed alongside and any gap under a quarter of a
rank is called out.

The vocabulary boundary applies to everything generated here and is enforced by
`tests/test_xai_language.py` against the rendered text, not against the source.
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any

from research.xai.cards import SCOPE_NOTE, coverage_counts

#: Mean-rank separation below which two features are reported as tied. A quarter
#: of a rank means fewer than one explainer in four ordered them differently,
#: which is not a distinction worth printing as an ordering.
TIE_MARGIN: float = 0.25


def _optional(value: Any, places: int = 3) -> str:
    """Format a statistic that may be undefined for this set of records."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def render_provenance(manifest: dict[str, Any]) -> str:
    """Which rows produced what, stated before any result is."""
    provenance = manifest.get("provenance", {})
    lines = [
        "## How this run was set up",
        "",
        f"- Run id: `{manifest.get('run_id', 'unknown')}`",
        f"- Evidence class: **{manifest.get('evidence_class', 'unknown')}**",
        f"- Models attempted: {len(manifest.get('models_requested', []))}",
        f"- Methods attempted: {len(manifest.get('methods_requested', []))}",
        f"- Training rows: {_optional(provenance.get('train_rows'))}",
        f"- Rows used to score permutation importance: "
        f"{_optional(provenance.get('evaluation_rows'))}",
        f"- Cases explained by the local methods: {_optional(provenance.get('case_rows'))}",
        f"- Baseline: {provenance.get('baseline_source', 'unknown')}",
        f"- Permutation partition: {provenance.get('permutation_scored_on', 'unknown')}",
    ]
    if manifest.get("fit_failures"):
        lines += [
            "",
            "Models that would not fit at this budget, and are therefore absent "
            "from every figure below:",
            *[f"- `{model}`: {reason}" for model, reason in
              sorted(manifest["fit_failures"].items())],
        ]
    return "\n".join(lines)


def render_outcomes(manifest: dict[str, Any]) -> str:
    """Outcome counts, reported before any agreement figure.

    A gap in the table is a property of a model, so the shape of the gaps is a
    result in its own right and belongs above the numbers computed from what
    remains.
    """
    counts = manifest.get("counts", {})
    total = sum(counts.values())
    lines = [
        "## What was answerable",
        "",
        f"{total} (model, method) pairs were attempted.",
        "",
        "| outcome | pairs |",
        "|---|---|",
    ]
    lines += [f"| {name} | {count} |" for name, count in sorted(counts.items())]
    lines += [
        "",
        "An `unsupported` pair is not a failure of the harness. It is a model "
        "that genuinely cannot provide that explanation - a kernel machine has "
        "no coefficients in feature space, a hard-label classifier has no score "
        "for a perturbation to move - and the count is one of the findings.",
        "",
        "### Capability coverage across the active roster",
        "",
    ]
    lines += [
        f"- `{name}`: {count} models"
        for name, count in coverage_counts().items()
    ]
    lines += [
        "",
        "A method available on five models is not a cross-family comparison, "
        "however good the method is. The counts above bound what any figure "
        "computed from them can support.",
    ]
    return "\n".join(lines)


def render_consensus(analysis: dict[str, Any]) -> str:
    """The consensus ranking, with the mean ranks that show how close it was."""
    consensus = analysis.get("zoo_consensus") or []
    mean_ranks = analysis.get("zoo_mean_ranks") or {}
    if not consensus:
        return "## Consensus\n\nNo explanation records were produced.\n"

    lines = [
        "## What the zoo attributes to, on average",
        "",
        "| rank | feature | mean rank across explanations |",
        "|---|---|---|",
    ]
    for position, feature in enumerate(consensus, start=1):
        lines.append(f"| {position} | `{feature}` | {_optional(mean_ranks.get(feature), 2)} |")

    ties = _near_ties(consensus, mean_ranks)
    lines += ["", ""]
    if ties:
        lines.append(
            "**Ordered, but barely.** These neighbouring pairs are separated by "
            f"less than {TIE_MARGIN} of a rank, which is not a distinction worth "
            "reading as an ordering: "
            + ", ".join(f"`{a}`/`{b}` ({gap:.2f})" for a, b, gap in ties)
            + "."
        )
    else:
        lines.append(
            f"No two neighbouring features sit within {TIE_MARGIN} of a rank of "
            "each other, so the ordering above is not resting on a near-tie."
        )
    lines += [
        "",
        "A consensus is a description of what these models attribute to. It is "
        "not evidence that these features cause the outcome, and it carries no "
        "implication that acting on them would change anyone's risk.",
    ]
    return "\n".join(lines)


def _near_ties(
    consensus: list[str], mean_ranks: dict[str, float]
) -> list[tuple[str, str, float]]:
    ties = []
    for left, right in pairwise(consensus):
        if left in mean_ranks and right in mean_ranks:
            gap = abs(float(mean_ranks[right]) - float(mean_ranks[left]))
            if gap < TIE_MARGIN:
                ties.append((left, right, gap))
    return ties


def render_agreement(analysis: dict[str, Any]) -> str:
    """The three groupings, each printed with the caveat that reads them."""
    lines = [
        "## Do the explanations agree?",
        "",
        "| grouping | pairs | mean Spearman | median | worst | top-1 agreement |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (
        ("within_model", "methods on one model"),
        ("within_family", "models within a family"),
        ("between_families", "models across families"),
    ):
        summary = analysis.get(key) or {}
        lines.append(
            f"| {label} | {summary.get('pairs', 0)} | "
            f"{_optional(summary.get('mean_spearman'))} | "
            f"{_optional(summary.get('median_spearman'))} | "
            f"{_optional(summary.get('min_spearman'))} | "
            f"{_optional(summary.get('top_1_agreement_rate'))} |"
        )

    lines += [
        "",
        "**Read the top-1 column, not the Spearman column.** When attribution "
        "concentrates on one or two features, most of the ranking is ordering "
        "noise and a full-ranking correlation mostly measures that noise. On a "
        "world with a known single driver, three explainers agreed on the top "
        "feature at every seed while correlating at 0.20 to 0.54 - a band that "
        "overlaps what the same three score on labels containing no signal at "
        "all. The correlation columns describe the tail; the top-1 column "
        "describes the finding.",
    ]

    worst = (analysis.get("between_families") or {}).get("worst_pair")
    if worst:
        lines += [
            "",
            f"The least-agreeing cross-family pair was `{worst['left']}` against "
            f"`{worst['right']}` at Spearman {_optional(worst['spearman'])}. A "
            "mean can be high while a specific pair disagrees completely, so the "
            "worst pair is named rather than averaged away.",
        ]
    return "\n".join(lines)


def render_interactions(manifest: dict[str, Any]) -> str:
    """Whether the models found joint structure, per model."""
    summaries = manifest.get("interactions") or {}
    if not summaries:
        return (
            "## Interactions\n\nThe interaction sweep did not run for this "
            "configuration.\n"
        )

    lines = [
        "## Did the models use features jointly?",
        "",
        "| model | strongest pair | H | median H | pairs essentially additive |",
        "|---|---|---|---|---|",
    ]
    for model_id, summary in sorted(summaries.items()):
        if "error" in summary:
            lines.append(f"| `{model_id}` | measurement failed | - | - | {summary['error']} |")
            continue
        strongest = summary.get("strongest") or {}
        features = strongest.get("features", ["-", "-"])
        lines.append(
            f"| `{model_id}` | `{features[0]}` x `{features[1]}` | "
            f"{_optional(strongest.get('h_statistic'))} | "
            f"{_optional(summary.get('median_h'))} | "
            f"{_optional(summary.get('additive_share'))} |"
        )

    lines += [
        "",
        "Measured on the log-odds scale. On the probability scale a logistic "
        "regression - additive by construction - reports an H of 0.18, which is "
        "the sigmoid rather than the model, and would give every "
        "probability-valued model a floor of apparent interaction.",
        "",
        "A departure from additivity describes the model's fitted surface. It is "
        "not evidence that two clinical factors act on each other.",
    ]
    return "\n".join(lines)


def render_limits() -> str:
    """What this run cannot show, stated as plainly as what it can."""
    return "\n".join([
        "## What this does not show",
        "",
        "- **Not causation.** Every figure here describes what a model's output "
        "depends on. None of it supports a claim that a feature causes diabetes, "
        "that changing a feature would change a person's risk, or any treatment "
        "recommendation.",
        "- **Not correctness.** Methods agreeing does not make them right. They "
        "share assumptions, and two methods built on the same assumption agree "
        "about the same mistake. The faithfulness measurements test rankings "
        "against the model's own behaviour, which is a different and weaker "
        "question than whether the ranking is true.",
        "- **Not a promotion.** No model examined here is proposed for "
        "production. The deployed artefacts are untouched by this track.",
        "- **Not clinical.** This is a research dataset at an exploratory "
        "training budget. Nothing here extends the project's non-diagnostic, "
        "non-clinical scope.",
        "- **Bounded by the feature set.** Ten features are all any of these "
        "explanations can range over. A feature absent from the contract cannot "
        "appear in any attribution, however much the outcome depends on it.",
    ])


def render(manifest: dict[str, Any]) -> str:
    """The full report for one run."""
    analysis = manifest.get("analysis", {})
    return "\n\n".join([
        "# Track M: cross-family explanation report",
        SCOPE_NOTE,
        render_provenance(manifest),
        render_outcomes(manifest),
        render_consensus(analysis),
        render_agreement(analysis),
        render_interactions(manifest),
        render_limits(),
    ]) + "\n"
