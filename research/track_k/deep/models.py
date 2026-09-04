"""The three deep challengers: an MLP, an FT-Transformer and a residual tower.

Both are written here rather than pulled from a tabular-DL library, so the
architecture under test is one this repository owns and can explain. Both take
the same call signature - ``(numeric, levels)`` - so the training loop, the
evaluator and the checkpoint format do not care which is running.

Sizing is driven by the problem, not by fashion. Ten features and 40,125
training rows do not support a large network; an over-parameterised model would
mostly measure regularisation. Both models are deliberately small, and their
parameter counts are recorded in the benchmark so a metric gain can be weighed
against the cost of serving it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from research.track_k.deep.preprocessing import OrdinalVocabulary


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Feed-forward challenger. Every field is recorded into provenance."""

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    activation: str = "relu"
    #: Normalising activations keeps training stable at this depth and costs
    #: little; disabled for the very small configurations used by the smoke test.
    batch_norm: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "mlp",
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
        }


@dataclass(frozen=True, slots=True)
class FTTransformerConfig:
    """FT-Transformer challenger.

    ``d_token`` is the width every feature is projected to before attention.
    Kept small: with ten features, a wide token dimension adds parameters
    without adding anything for them to represent.
    """

    d_token: int = 32
    n_blocks: int = 3
    n_heads: int = 4
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    #: Feed-forward expansion inside each block, as a multiple of d_token.
    ffn_factor: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "ft_transformer",
            "d_token": self.d_token,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
            "residual_dropout": self.residual_dropout,
            "ffn_factor": self.ffn_factor,
        }


@dataclass(frozen=True, slots=True)
class TabularResNetConfig:
    """Residual challenger, in the shape Gorishniy et al. use as a tabular baseline.

    The question it answers is narrow and worth answering: the plain MLP
    plateaus within one epoch, and there are two possible reasons. Either the
    features are exhausted, or a shallow feed-forward stack is the wrong shape
    for what remains. A residual tower is the standard way to give a tabular
    network real depth without the optimisation problems depth normally brings,
    so if depth is the missing ingredient this is the model that should find it.

    Deliberately NOT a fourth flavour of the same idea: it differs from the MLP
    by having skip connections and pre-normalised blocks, and from the
    FT-Transformer by having no attention and no per-feature tokens.
    """

    d_hidden: int = 64
    #: Inner width of each block, as a multiple of d_hidden. The block widens,
    #: activates, narrows, and adds - the standard bottleneck shape.
    d_expansion: float = 2.0
    n_blocks: int = 3
    dropout: float = 0.1
    residual_dropout: float = 0.0
    activation: str = "relu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "tabular_resnet",
            "d_hidden": self.d_hidden,
            "d_expansion": self.d_expansion,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
            "residual_dropout": self.residual_dropout,
            "activation": self.activation,
        }


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"unsupported activation: {name!r}")
    return activations[name]()


class TabularMLP(nn.Module):
    """A plain feed-forward network over the standardised feature vector.

    It consumes ``numeric`` and ignores ``levels``: the MLP's modelling
    assumption is that the ordinal codes carry usable numeric order once
    standardised. That assumption is exactly what the FT-Transformer below
    declines to make, which is what makes the pair informative.
    """

    def __init__(self, n_features: int, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = n_features
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(width, hidden))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels  # the MLP treats ordinal codes as numeric; see the docstring
        logits: torch.Tensor = self.network(numeric)
        return logits.reshape(-1)


class FeatureTokenizer(nn.Module):
    """Turn each feature into a token, the step the architecture is named for.

    Two treatments, chosen from the served contract rather than guessed:

    * a CONTINUOUS feature (BMI) gets a learned per-feature weight and bias, so
      its token is a linear projection of one scalar;
    * a DISCRETE feature (the binary and ordinal ones) gets an embedding table
      with one row per legal level, so level 3 of GenHlth is not assumed to sit
      exactly one unit from level 2. This is the substantive difference from the
      MLP and the reason an ordinal-heavy tabular problem is worth putting a
      transformer on at all.

    A CLS token is prepended; the classification head reads its final state.
    """

    def __init__(self, vocabulary: OrdinalVocabulary, d_token: int) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.d_token = d_token
        self.n_features = len(vocabulary.feature_names)

        self.continuous_positions = tuple(
            index for index, size in enumerate(vocabulary.cardinalities) if size == 0
        )
        self.discrete_positions = tuple(
            index for index, size in enumerate(vocabulary.cardinalities) if size > 0
        )

        if self.continuous_positions:
            self.continuous_weight = nn.Parameter(
                torch.empty(len(self.continuous_positions), d_token)
            )
            self.continuous_bias = nn.Parameter(
                torch.empty(len(self.continuous_positions), d_token)
            )
        else:  # pragma: no cover - the contract always has BMI
            self.register_parameter("continuous_weight", None)
            self.register_parameter("continuous_bias", None)

        # One flat table for every discrete level, indexed by per-feature offset.
        self.level_embeddings = nn.Embedding(max(vocabulary.total_tokens, 1), d_token)
        self.register_buffer(
            "level_offsets",
            torch.tensor(list(vocabulary.offsets), dtype=torch.long),
            persistent=True,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Uniform initialisation scaled by token width, as in the FT-Transformer paper."""
        bound = 1.0 / math.sqrt(self.d_token)
        if self.continuous_weight is not None:
            nn.init.uniform_(self.continuous_weight, -bound, bound)
            nn.init.uniform_(self.continuous_bias, -bound, bound)
        nn.init.uniform_(self.level_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.cls_token, -bound, bound)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        batch = numeric.shape[0]
        tokens = numeric.new_zeros((batch, self.n_features, self.d_token))

        if self.continuous_positions:
            positions = list(self.continuous_positions)
            weight = self.continuous_weight
            bias = self.continuous_bias
            assert weight is not None and bias is not None  # noqa: S101 - see __init__
            values = numeric[:, positions].unsqueeze(-1)
            tokens[:, positions, :] = values * weight + bias

        if self.discrete_positions:
            positions = list(self.discrete_positions)
            # register_buffer is typed as Tensor | Module, so the concrete type
            # has to be stated before indexing it.
            offset_table = cast(torch.Tensor, self.level_offsets)
            offsets = offset_table[positions].reshape(1, -1)
            flat = levels[:, positions] + offsets
            tokens[:, positions, :] = self.level_embeddings(flat)

        cls = self.cls_token.expand(batch, 1, self.d_token)
        return torch.cat([cls, tokens], dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: attention, then a position-wise feed-forward.

    Pre-norm rather than post-norm because it trains stably at small depth
    without a warmup schedule, which keeps the training loop simple.
    """

    def __init__(self, config: FTTransformerConfig) -> None:
        super().__init__()
        hidden = int(config.d_token * config.ffn_factor)
        self.norm_attention = nn.LayerNorm(config.d_token)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_token,
            num_heads=config.n_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.norm_ffn = nn.LayerNorm(config.d_token)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_token, hidden),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(hidden, config.d_token),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(tokens)
        attended, _weights = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + self.residual_dropout(attended)
        tokens = tokens + self.residual_dropout(self.ffn(self.norm_ffn(tokens)))
        return tokens


class FTTransformer(nn.Module):
    """FT-Transformer for this ten-feature contract.

    Simplifications from the published architecture, stated rather than hidden:

    * no feature-wise attention biases beyond standard multi-head attention;
    * the CLS token is read directly instead of a learned pooling module;
    * a single flat embedding table serves every discrete feature via offsets,
      which is equivalent to per-feature tables and cheaper to checkpoint.

    None of these changes the essential idea being tested: give each feature its
    own token and let attention learn the interactions, rather than flattening
    everything into one vector as the MLP does.
    """

    def __init__(self, vocabulary: OrdinalVocabulary, config: FTTransformerConfig) -> None:
        super().__init__()
        if config.d_token % config.n_heads != 0:
            raise ValueError(
                f"d_token={config.d_token} must divide evenly by n_heads={config.n_heads}"
            )
        self.config = config
        self.tokenizer = FeatureTokenizer(vocabulary, config.d_token)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_blocks)
        )
        self.head_norm = nn.LayerNorm(config.d_token)
        self.head = nn.Linear(config.d_token, 1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(numeric, levels)
        for block in self.blocks:
            tokens = block(tokens)
        cls = tokens[:, 0, :]
        logits: torch.Tensor = self.head(self.head_norm(cls))
        return logits.reshape(-1)


class ResidualBlock(nn.Module):
    """Pre-norm bottleneck block: normalise, widen, activate, narrow, add.

    Pre-normalisation rather than post- because it is what makes a residual
    tower trainable at depth without a warmup schedule, which this study has no
    budget to tune.
    """

    def __init__(self, d_hidden: int, config: TabularResNetConfig) -> None:
        super().__init__()
        d_inner = max(1, round(d_hidden * config.d_expansion))
        self.norm = nn.BatchNorm1d(d_hidden)
        self.widen = nn.Linear(d_hidden, d_inner)
        self.activation = _activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        self.narrow = nn.Linear(d_inner, d_hidden)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        residual = self.dropout(self.activation(self.widen(residual)))
        residual = self.residual_dropout(self.narrow(residual))
        summed: torch.Tensor = x + residual
        return summed


class TabularResNet(nn.Module):
    """A residual tower over the standardised feature vector.

    Like the MLP it consumes ``numeric`` and ignores ``levels``, so the only
    difference between the two is the shape of the network. That is the point:
    holding the input representation fixed makes the comparison a comparison of
    architecture rather than of preprocessing.
    """

    def __init__(self, n_features: int, config: TabularResNetConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(n_features, config.d_hidden)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.d_hidden, config) for _ in range(config.n_blocks)
        )
        self.head_norm = nn.BatchNorm1d(config.d_hidden)
        self.head = nn.Linear(config.d_hidden, 1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels  # ordinal codes are treated as numeric; see TabularMLP
        x = self.project(numeric)
        for block in self.blocks:
            x = block(x)
        logits: torch.Tensor = self.head(self.head_norm(x))
        return logits.reshape(-1)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, reported beside every benchmark result."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
