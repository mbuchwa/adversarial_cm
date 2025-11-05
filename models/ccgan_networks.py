"""Neural networks used by the LightningCcGAN module.

This module re-implements the original MLP based generator and discriminator
from the public CcGAN release while adding support for multi-dimensional
continuous conditioning vectors.  When the conditioning dimensionality is set
to one the behaviour is identical to the reference implementation: labels are
interpreted as angles on the unit circle and embedded via their sine and cosine
components before being concatenated with the latent code / real samples.

For higher dimensional conditioning vectors we apply the same sinusoidal
embedding to each component independently and concatenate the resulting pairs.
This allows the LightningCcGAN module to operate on arbitrary continuous label
vectors without changing the training dynamics described in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn


def _make_mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int,
    num_hidden_layers: int,
    bias: bool,
) -> nn.Sequential:
    """Construct the sequential MLP used by both generator and discriminator."""

    layers: List[nn.Module] = []
    in_features = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_features, hidden_dim, bias=bias))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim, bias=bias))
    return nn.Sequential(*layers)


class ContinuousLabelEmbedding(nn.Module):
    """Embed continuous labels via their sine and cosine representations."""

    def __init__(self, label_dim: int, radius: float = 1.0) -> None:
        super().__init__()
        if label_dim < 1:
            raise ValueError("label_dim must be at least 1")
        self.label_dim = int(label_dim)
        self.radius = float(radius)

    def forward(self, labels: Tensor) -> Tensor:
        labels = labels.view(labels.size(0), self.label_dim)
        angles = labels * (2 * math.pi)
        sin_components = torch.sin(angles)
        cos_components = torch.cos(angles)
        embedded = torch.cat(
            (self.radius * sin_components, self.radius * cos_components), dim=1
        )
        return embedded

    @property
    def output_dim(self) -> int:
        return 2 * self.label_dim


@dataclass
class ContinuousGeneratorConfig:
    latent_dim: int = 2
    output_dim: int = 2
    hidden_dim: int = 100
    num_hidden_layers: int = 6
    label_dim: int = 1
    radius: float = 1.0


class ContinuousConditionalGenerator(nn.Module):
    """MLP generator matching the original CcGAN architecture."""

    def __init__(self, config: ContinuousGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = ContinuousLabelEmbedding(config.label_dim, config.radius)
        input_dim = config.latent_dim + self.embedding.output_dim
        self.net = _make_mlp(
            input_dim,
            config.output_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            bias=False,
        )

    def forward(self, latent: Tensor, labels: Tensor) -> Tensor:
        latent = latent.view(latent.size(0), self.config.latent_dim)
        embedded_labels = self.embedding(labels)
        inputs = torch.cat((latent, embedded_labels), dim=1)
        return self.net(inputs)


@dataclass
class ContinuousDiscriminatorConfig:
    input_dim: int = 2
    hidden_dim: int = 100
    num_hidden_layers: int = 5
    label_dim: int = 1
    radius: float = 1.0


class ContinuousConditionalDiscriminator(nn.Module):
    """MLP discriminator mirroring the original CcGAN implementation."""

    def __init__(self, config: ContinuousDiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = ContinuousLabelEmbedding(config.label_dim, config.radius)
        input_dim = config.input_dim + self.embedding.output_dim

        layers: List[nn.Module] = []
        in_features = input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_features, config.hidden_dim, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_features = config.hidden_dim
        layers.append(nn.Linear(in_features, 1, bias=False))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, samples: Tensor, labels: Tensor) -> Tensor:
        samples = samples.view(samples.size(0), self.config.input_dim)
        embedded_labels = self.embedding(labels)
        inputs = torch.cat((samples, embedded_labels), dim=1)
        return self.net(inputs).view(-1, 1)

