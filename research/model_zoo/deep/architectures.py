"""Seven tabular architectures, each testing a different structural hypothesis.

Track K established that an MLP, a residual tower and an FT-Transformer land
within 0.0017 ROC-AUC of one another on this problem, and concluded the ceiling
is in the ten features rather than in the model. That conclusion is stronger if
it survives architectures built on *different* assumptions, so each model here
was chosen for the specific hypothesis it encodes, not to lengthen a list:

* **TabTransformer** - attention over the categorical features only, with the
  continuous ones bypassing it. FT-Transformer tokenises everything; this asks
  whether attention is worth having on the discrete features alone.
* **Deep & Cross Network** - explicit bounded-degree feature crossing in a
  dedicated branch. If the missing signal is multiplicative interactions, a
  network that computes them directly should find them where an MLP has to
  approximate them.
* **Wide & Deep** - a linear term added to a deep one. Track K found logistic
  regression within 0.005 of everything; this keeps that linear term and asks
  what depth adds on top of it.
* **Gated Residual MLP** - gated linear units decide how much of each block to
  apply. If the right answer is "mostly the linear path", a gate can learn to
  say so, which an ordinary residual block cannot.
* **Feature-token Mixer** - alternating token-mixing and channel-mixing MLPs.
  All-to-all feature interaction with no attention, isolating whether the
  attention mechanism or merely the token representation was doing the work.
* **Neural Additive Model** - one subnetwork per feature, summed. Deliberately
  cannot represent interactions at all. If it matches the others, interactions
  are not where the remaining signal lives, which would be a direct answer to
  Track K's open question.
* **Feature CNN** - an experimental control, documented as such below.

All seven take Track K's ``(numeric, levels)`` call signature and train through
Track K's loop, so they are comparable to its results rather than merely
adjacent to them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from research.track_k.deep.preprocessing import OrdinalVocabulary


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Shared knobs. Each architecture reads the subset it needs."""

    d_hidden: int = 64
    n_blocks: int = 2
    n_heads: int = 4
    d_token: int = 16
    dropout: float = 0.1
    expansion: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "d_hidden": self.d_hidden,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "d_token": self.d_token,
            "dropout": self.dropout,
            "expansion": self.expansion,
        }


def _categorical_mask(vocabulary: OrdinalVocabulary) -> list[int]:
    """Indices of features the contract gives a discrete level set."""
    return [i for i, size in enumerate(vocabulary.cardinalities) if size > 0]


class TabTransformer(nn.Module):
    """Attention over categorical embeddings; continuous features bypass it.

    The distinction from Track K's FT-Transformer is the whole point. There,
    every feature - continuous ones included - becomes a token and attends to
    every other. Here only the contract's discrete features are embedded and
    attended over; the continuous ones are normalised and concatenated to the
    attention output just before the head.

    On this dataset that means eight of ten features enter attention and BMI
    plus PhysHlth skip it, which is the original TabTransformer's design and a
    genuinely different inductive bias from FT-Transformer's.
    """

    def __init__(self, vocabulary: OrdinalVocabulary, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.categorical = _categorical_mask(vocabulary)
        self.offsets = torch.tensor(
            [vocabulary.offsets[i] for i in self.categorical], dtype=torch.long
        )
        self.n_continuous = len(vocabulary.cardinalities) - len(self.categorical)

        self.embedding = nn.Embedding(max(vocabulary.total_tokens, 1), config.d_token)
        encoder = nn.TransformerEncoderLayer(
            d_model=config.d_token,
            nhead=config.n_heads,
            dim_feedforward=int(config.d_token * config.expansion),
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=config.n_blocks)
        self.continuous_norm = (
            nn.LayerNorm(self.n_continuous) if self.n_continuous else nn.Identity()
        )
        head_width = config.d_token * max(len(self.categorical), 1) + self.n_continuous
        self.head = nn.Sequential(
            nn.Linear(head_width, config.d_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, 1),
        )

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets.to(levels.device)
        tokens = self.embedding(levels[:, self.categorical] + offsets)
        attended = self.encoder(tokens).flatten(start_dim=1)

        continuous_index = [
            i for i in range(numeric.shape[1]) if i not in set(self.categorical)
        ]
        if continuous_index:
            continuous = self.continuous_norm(numeric[:, continuous_index])
            attended = torch.cat([attended, continuous], dim=1)

        logits: torch.Tensor = self.head(attended)
        return logits.reshape(-1)


class CrossLayer(nn.Module):
    """One explicit feature cross: ``x0 * (W x + b) + x``.

    Each layer raises the polynomial degree of the interactions by one, so a
    two-layer stack represents every three-way product of the input features -
    exactly and cheaply, where an MLP must approximate them.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.weight = nn.Linear(n_features, n_features)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        crossed: torch.Tensor = x0 * self.weight(x) + x
        return crossed


class DeepCrossNetwork(nn.Module):
    """A cross branch and a deep branch, concatenated at the head.

    The cross branch computes bounded-degree interactions explicitly; the deep
    branch is an ordinary MLP. If Track K's plateau is caused by an MLP failing
    to *find* interactions rather than by interactions not existing, this
    architecture is the one positioned to show it.
    """

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.crosses = nn.ModuleList(
            CrossLayer(n_features) for _ in range(config.n_blocks)
        )
        self.deep = nn.Sequential(
            nn.Linear(n_features, config.d_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(n_features + config.d_hidden, 1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        crossed = numeric
        for layer in self.crosses:
            crossed = layer(numeric, crossed)
        joined = torch.cat([crossed, self.deep(numeric)], dim=1)
        logits: torch.Tensor = self.head(joined)
        return logits.reshape(-1)


class WideAndDeep(nn.Module):
    """A linear term plus a deep term, summed in logit space.

    Chosen because of Track K's central result: logistic regression came within
    0.005 ROC-AUC of every deep model. Keeping an explicit linear path means the
    deep branch only has to learn the residual, and if that residual is close to
    nothing, this model can still reach the linear baseline rather than having
    to rediscover it.
    """

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.wide = nn.Linear(n_features, 1)
        self.deep = nn.Sequential(
            nn.Linear(n_features, config.d_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.d_hidden // 2, 1),
        )

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        logits: torch.Tensor = self.wide(numeric) + self.deep(numeric)
        return logits.reshape(-1)


class GatedResidualBlock(nn.Module):
    """A residual block whose contribution is gated by a learned sigmoid.

    The gate is the reason this is not simply Track K's residual tower again.
    A GLU can learn to close - to pass the input through nearly untouched - so
    if the best available answer really is close to linear, this architecture
    can represent that decision explicitly instead of having to cancel the
    block out through its weights.
    """

    def __init__(self, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.project = nn.Linear(d_hidden, d_hidden)
        self.gate = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        activated = torch.nn.functional.elu(self.project(hidden))
        gated: torch.Tensor = x + self.dropout(activated * torch.sigmoid(self.gate(hidden)))
        return gated


class GatedResidualMLP(nn.Module):
    """A stack of gated residual blocks over the standardised feature vector."""

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(n_features, config.d_hidden)
        self.blocks = nn.ModuleList(
            GatedResidualBlock(config.d_hidden, config.dropout)
            for _ in range(config.n_blocks)
        )
        self.head_norm = nn.LayerNorm(config.d_hidden)
        self.head = nn.Linear(config.d_hidden, 1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        x = self.project(numeric)
        for block in self.blocks:
            x = block(x)
        logits: torch.Tensor = self.head(self.head_norm(x))
        return logits.reshape(-1)


class MixerBlock(nn.Module):
    """One MLP across features, then one across channels.

    Token mixing gives every feature access to every other without attention's
    quadratic score matrix - the comparison that isolates whether attention's
    *content-dependent* weighting mattered, or merely the fact that features
    could interact at all.
    """

    def __init__(self, n_tokens: int, d_token: int, expansion: float, dropout: float) -> None:
        super().__init__()
        hidden_tokens = max(2, int(n_tokens * expansion))
        hidden_channels = max(2, int(d_token * expansion))
        self.token_norm = nn.LayerNorm(d_token)
        self.token_mix = nn.Sequential(
            nn.Linear(n_tokens, hidden_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_tokens, n_tokens),
        )
        self.channel_norm = nn.LayerNorm(d_token)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_token, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, d_token),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, tokens, channels)
        mixed = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mix(mixed).transpose(1, 2)
        channel_mixed: torch.Tensor = x + self.channel_mix(self.channel_norm(x))
        return channel_mixed


class FeatureTokenMixer(nn.Module):
    """MLP-Mixer over per-feature tokens."""

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        # One learned projection per feature: a scalar becomes a token.
        self.tokenize = nn.Parameter(torch.empty(n_features, config.d_token))
        self.token_bias = nn.Parameter(torch.zeros(n_features, config.d_token))
        nn.init.normal_(self.tokenize, std=1.0 / math.sqrt(config.d_token))
        self.blocks = nn.ModuleList(
            MixerBlock(n_features, config.d_token, config.expansion, config.dropout)
            for _ in range(config.n_blocks)
        )
        self.norm = nn.LayerNorm(config.d_token)
        self.head = nn.Linear(config.d_token, 1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        tokens = numeric.unsqueeze(-1) * self.tokenize + self.token_bias
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        logits: torch.Tensor = self.head(pooled)
        return logits.reshape(-1)


class NeuralAdditiveModel(nn.Module):
    """One small subnetwork per feature, summed. No interactions, by design.

    This is the zoo's most informative constraint. A neural additive model can
    learn an arbitrary non-linear shape function for each feature but cannot
    represent any interaction between two of them. If it matches the
    unconstrained networks, then interactions are not where the remaining
    signal lives - which is a direct, falsifiable answer to the question Track K
    left open, and it is also fully interpretable: each feature's contribution
    can be plotted.
    """

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        width = max(8, config.d_hidden // 4)
        self.shape_functions = nn.ModuleList(
            nn.Sequential(
                nn.Linear(1, width),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(width, width // 2),
                nn.ReLU(),
                nn.Linear(width // 2, 1),
            )
            for _ in range(n_features)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def feature_contributions(self, numeric: torch.Tensor) -> torch.Tensor:
        """Each feature's additive contribution to the logit, per row."""
        parts = [
            self.shape_functions[i](numeric[:, i : i + 1])
            for i in range(self.n_features)
        ]
        return torch.cat(parts, dim=1)

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        logits: torch.Tensor = self.feature_contributions(numeric).sum(dim=1) + self.bias
        return logits.reshape(-1)


class FeatureCNN(nn.Module):
    """EXPERIMENTAL INDUCTIVE-BIAS CONTROL. Not a serious candidate.

    A 1D convolution slides a shared kernel across adjacent positions, which
    encodes the assumption that neighbouring positions are related. For an
    image or a time series that assumption is true. Here the "positions" are the
    ten served features in the order the contract happens to list them -
    ``GenHlth``, ``HighBP``, ``BMI``, ``HighChol`` and so on - and that order is
    arbitrary. Permuting the columns would change this model's predictions while
    changing nothing about the problem.

    It is included as a **negative control on inductive bias**: a model whose
    structural assumption is known to be unjustified, kept in the table so the
    comparison has a case where the architecture demonstrably does not match the
    data. It must not be read as a production candidate, and the benchmark
    labels it accordingly.
    """

    def __init__(self, n_features: int, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        channels = max(4, config.d_token)
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, numeric: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        del levels
        pooled = self.network(numeric.unsqueeze(1))
        logits: torch.Tensor = self.head(pooled)
        return logits.reshape(-1)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, reported beside every deep result."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
