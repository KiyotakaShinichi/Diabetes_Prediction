"""The model registry: one typed entry per algorithm, and a factory.

Every model in the zoo is described by a `ModelSpec` and built by a callable
the spec carries. Nothing anywhere else in the package knows the list of
models: the benchmark runner, the capability matrix, the card generator and
the tests all read this registry, so adding an algorithm is one entry plus its
proving tests rather than an edit in six places.

Two rules keep the registry honest:

**A spec cannot be optimistic.** `tests/test_model_zoo_registry.py` builds
every ACTIVE model and asserts that what it actually does matches what its spec
claims - probabilities, importances, serialization, determinism. A model whose
declaration drifts from its behaviour fails, rather than quietly emitting a
column of numbers that mean something other than the header says.

**An optional dependency cannot break the core.** Specs whose framework is not
installed resolve to `OPTIONAL` and are recorded as skipped by the benchmark.
`import research.model_zoo.registry` must succeed with only the core lockfile
installed, which is why the builders import their libraries inside the function
rather than at module scope.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from research.model_zoo.contracts import (
    Capabilities,
    Family,
    Framework,
    Preprocessing,
    ProbabilityBehavior,
    ResearchStatus,
    ResourceClass,
)

#: Derived from the split seed the whole repository already uses, so a zoo run
#: is reproducible from one documented number.
ZOO_SEED: int = 42


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to build, run, describe and judge one algorithm."""

    model_id: str
    display_name: str
    family: Family
    framework: Framework
    build: Callable[..., Any]
    preprocessing: Preprocessing
    probability_behavior: ProbabilityBehavior
    capabilities: Capabilities
    resource_class: ResourceClass
    default_config: Mapping[str, Any] = field(default_factory=dict)
    seed: int = ZOO_SEED
    optional_dependency: str | None = None
    research_status: ResearchStatus = ResearchStatus.ACTIVE
    #: Why this algorithm is in the zoo, and what it is expected to show. One
    #: sentence, and it appears verbatim on the model's generated card.
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family.value,
            "framework": self.framework.value,
            "preprocessing": self.preprocessing.value,
            "probability_behavior": self.probability_behavior.value,
            "capabilities": self.capabilities.as_dict(),
            "resource_class": self.resource_class.value,
            "default_config": dict(self.default_config),
            "seed": self.seed,
            "optional_dependency": self.optional_dependency,
            "research_status": self.effective_status().value,
            "rationale": self.rationale,
        }

    def is_available(self) -> bool:
        """Whether this model's framework can actually be imported here."""
        if self.optional_dependency is None:
            return True
        return importlib.util.find_spec(self.optional_dependency) is not None

    def effective_status(self) -> ResearchStatus:
        """Declared status, downgraded when an optional dependency is absent.

        A model whose library is not installed is OPTIONAL rather than ACTIVE.
        The distinction is what lets the benchmark record it as skipped-with-a-
        reason instead of crashing, and what lets the core install stay small.
        """
        if self.research_status is ResearchStatus.ACTIVE and not self.is_available():
            return ResearchStatus.OPTIONAL
        return self.research_status


class ModelRegistry:
    """An ordered, immutable collection of specs, addressable by id."""

    def __init__(self, specs: list[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ModelSpec) -> ModelSpec:
        if spec.model_id in self._specs:
            raise ValueError(f"duplicate model_id: {spec.model_id!r}")
        self._specs[spec.model_id] = spec
        return spec

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs.values())

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._specs:
            raise KeyError(
                f"unknown model_id {model_id!r}; registered: {sorted(self._specs)}"
            )
        return self._specs[model_id]

    def ids(self) -> list[str]:
        return list(self._specs)

    def by_family(self, family: Family) -> list[ModelSpec]:
        return [spec for spec in self if spec.family is family]

    def by_status(self, status: ResearchStatus) -> list[ModelSpec]:
        return [spec for spec in self if spec.effective_status() is status]

    def active(self) -> list[ModelSpec]:
        """Everything the benchmark will actually attempt to run."""
        return self.by_status(ResearchStatus.ACTIVE)

    def build(self, model_id: str, **overrides: Any) -> Any:
        """Instantiate one model from its spec.

        Config overrides exist for the tests, which build every model on tiny
        synthetic data. The benchmark itself always uses the frozen defaults.
        """
        spec = self.get(model_id)
        config = {**spec.default_config, **overrides}
        return spec.build(spec=spec, **config)


#: The single registry the whole package reads. Populated by the family modules
#: at import time, in family order, so the results table has a stable shape.
REGISTRY = ModelRegistry()


def register(spec: ModelSpec) -> ModelSpec:
    """Add a spec to the shared registry."""
    return REGISTRY.register(spec)


def _load_families() -> None:
    """Import the family modules for their registration side effects.

    Imported here rather than at the top of the file because each family module
    imports this one for ``register``; deferring the import to a function call
    at the end of module execution breaks the cycle without any import-time
    trickery in the family modules themselves.
    """
    from research.model_zoo.families import (  # noqa: F401
        boosting,
        deep,
        kernel,
        linear,
        trees,
    )


_load_families()
