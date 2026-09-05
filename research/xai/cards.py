"""Cards: what each method measures, and what each model can be asked.

A card here is not a summary of results. It is the standing description of an
instrument - what it measures, against what reference, and how it misleads -
written once and rendered wherever a number from that instrument appears.

Two card types, because the two questions are different. A **method card** says
what a technique measures and where it breaks; it is the same for every model.
A **model card** says which techniques that model can be asked for at all, and
names the reason for each gap; it is the same for every method.

Every card ends with limitations, and that is enforced rather than encouraged.
`tests/test_xai_language.py` asserts that no rendered card is free of them,
because a card listing only capabilities is marketing, and the whole point of
the capability contract is that the gaps are findings.

The vocabulary boundary is enforced in the same place. These cards describe
model attributions, feature importance, association and prediction dependence.
They do not say a feature causes diabetes, that changing it would reduce risk,
or anything that reads as a treatment recommendation - and since this text is
generated rather than reviewed line by line, the test checks the output.
"""
from __future__ import annotations

from typing import Any

from research.model_zoo.contracts import ResearchStatus
from research.model_zoo.registry import REGISTRY
from research.xai.capabilities import XaiCapability, capability_matrix, profile_for
from research.xai.registry import METHODS, MethodSpec

#: Prefixed to every rendered document. Short enough to survive being copied
#: into a slide, which is where a caveat is most often lost.
SCOPE_NOTE = (
    "Research artefact. Describes model behaviour on a research dataset; not a "
    "diagnostic tool, not clinical guidance, and not a basis for promoting any "
    "model. Attribution describes what a model's prediction depends on, which "
    "is an association and not a cause."
)


def render_method_card(method: MethodSpec) -> str:
    """One method's standing description, limitations included by construction."""
    lines = [
        f"### {method.display_name} (`{method.method_id}` v{method.version})",
        "",
        f"- **Scope**: {method.scope.value}",
        f"- **Determinism**: {method.determinism.value}",
        f"- **Reference point**: {method.baseline_strategy.value}",
        f"- **Requires**: {method.required_capability.value}",
        f"- **Cost class**: {method.runtime_class}",
    ]
    if method.optional_dependency:
        lines.append(
            f"- **Optional dependency**: `{method.optional_dependency}` "
            f"({'installed' if method.is_available() else 'absent - pairs are recorded as skipped'})"
        )
    lines += [
        "",
        f"**Measures.** {method.measures}",
        "",
        "**Limitations.**",
    ]
    lines += [f"- {mode}" for mode in method.failure_modes]
    if method.causal_reading_invalid:
        lines += [
            "",
            "**Not a causal statement.** This measures the model's dependence on "
            "an input, not an effect of that input on the outcome.",
        ]
    return "\n".join(lines)


def render_method_cards() -> str:
    """Every registered method, in registration order."""
    body = "\n\n".join(render_method_card(method) for method in METHODS)
    return f"## Method cards\n\n{SCOPE_NOTE}\n\n{body}\n"


def render_model_card(model_id: str) -> str:
    """One model's explainability profile: what it offers, and why not the rest."""
    spec = REGISTRY.get(model_id)
    profile = profile_for(spec)

    supported = sorted(c.value for c in profile.capabilities)
    missing = sorted(c.value for c in XaiCapability if c not in profile.capabilities)

    lines = [
        f"### {spec.display_name} (`{spec.model_id}`)",
        "",
        f"- **Family**: {spec.family.value}",
        f"- **Framework**: {spec.framework.value}",
        f"- **Status**: {spec.effective_status().value}",
        f"- **Probability behaviour**: {spec.probability_behavior.value}",
        f"- **Preprocessing**: {spec.preprocessing.value}",
        "",
        "**Explanations available.**",
    ]
    lines += [f"- {name}" for name in supported] or ["- none"]
    lines += ["", "**Explanations unavailable.**"]
    lines += [f"- {name}" for name in missing] or ["- none"]

    lines += ["", "**Why those gaps exist.**"]
    if profile.exclusions:
        lines += [f"- {reason}" for reason in profile.exclusions]
    else:
        lines += ["- no capability is excluded for this model"]

    lines += [
        "",
        "**Limitations.** An unavailable explanation is a property of this model, "
        "not a gap in the study. Everything reported for it describes association "
        "and prediction dependence on a research dataset, at an exploratory "
        "training budget, and supports no clinical or causal claim.",
    ]
    return "\n".join(lines)


def render_model_cards(model_ids: list[str] | None = None) -> str:
    """Cards for the active roster, or for a named subset."""
    ids = model_ids or [
        spec.model_id
        for spec in REGISTRY
        if spec.effective_status() is ResearchStatus.ACTIVE
    ]
    body = "\n\n".join(render_model_card(model_id) for model_id in ids)
    return f"## Model explainability cards\n\n{SCOPE_NOTE}\n\n{body}\n"


def render_capability_table(rows: list[dict[str, Any]] | None = None) -> str:
    """The whole matrix as one Markdown table.

    Reading a matrix is how a person notices that a family is systematically
    unexplainable, which a per-model card cannot show however carefully each one
    is written.
    """
    matrix = rows if rows is not None else capability_matrix()
    capabilities = [c.value for c in XaiCapability]

    header = "| model | family | " + " | ".join(
        c.replace("_compatible", "").replace("_", " ") for c in capabilities
    ) + " |"
    divider = "|" + "---|" * (len(capabilities) + 2)

    body = []
    for row in matrix:
        cells = ["yes" if row[c] else "-" for c in capabilities]
        body.append(f"| `{row['model_id']}` | {row['family']} | " + " | ".join(cells) + " |")

    return "\n".join([header, divider, *body])


def coverage_counts() -> dict[str, int]:
    """How many active models can be asked for each capability.

    The number that turns the matrix into a finding: a method available on three
    models is not a cross-family comparison however good the method is.
    """
    counts: dict[str, int] = {}
    for capability in XaiCapability:
        counts[capability.value] = sum(
            1
            for spec in REGISTRY
            if spec.effective_status() is ResearchStatus.ACTIVE
            and profile_for(spec).supports(capability)
        )
    return counts


def render_all() -> str:
    """The complete card set, matrix first."""
    counts = coverage_counts()
    active = sum(
        1 for spec in REGISTRY if spec.effective_status() is ResearchStatus.ACTIVE
    )
    summary = "\n".join(
        f"- `{name}`: {count} of {active} active models" for name, count in counts.items()
    )
    return "\n\n".join([
        "# Track M explainability cards",
        SCOPE_NOTE,
        "## Capability matrix",
        render_capability_table(),
        "## Capability coverage",
        summary,
        render_method_cards(),
        render_model_cards(),
    ])
