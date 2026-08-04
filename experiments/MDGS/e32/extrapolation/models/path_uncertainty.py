"""Latent diffusion-path bottleneck diagnostics for E31."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PathUncertaintyResult:
    """Two-view path stability signals from the latent diffusion trajectory."""

    per_witness_embedding_a: torch.Tensor
    per_witness_embedding_b: torch.Tensor
    example_embedding_a: torch.Tensor
    example_embedding_b: torch.Tensor
    per_witness_path_disagreement: torch.Tensor
    path_disagreement: torch.Tensor
    path_similarity: torch.Tensor
    path_confidence: torch.Tensor
    contrastive_loss: torch.Tensor


class LatentPathEncoder(nn.Module):
    """Compact bottleneck encoder over one reverse-diffusion trajectory per witness."""

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.point = nn.Sequential(
            nn.LayerNorm(self.latent_dim * 2),
            nn.Linear(self.latent_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.embedding_dim),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory shaped [steps, samples, batch, latent] into [samples, batch, emb]."""

        if trajectory.ndim != 4:
            raise ValueError("trajectory must have shape (steps, samples, batch, latent_dim)")
        steps, num_samples, batch_size, latent_dim = trajectory.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        if steps <= 0 or num_samples <= 0 or batch_size <= 0:
            raise ValueError("trajectory must include at least one step, sample, and batch item")

        path = trajectory.permute(2, 1, 0, 3).contiguous()
        if steps > 1:
            deltas = path[:, :, 1:, :] - path[:, :, :-1, :]
            deltas = torch.cat([torch.zeros_like(deltas[:, :, :1, :]), deltas], dim=2)
        else:
            deltas = torch.zeros_like(path)

        features = torch.cat([path, deltas], dim=-1)
        encoded = self.point(features.reshape(batch_size * num_samples * steps, latent_dim * 2))
        encoded = encoded.reshape(batch_size, num_samples, steps, self.hidden_dim)
        temporal_mean = encoded.mean(dim=2)
        temporal_std = encoded.std(dim=2, unbiased=False)
        terminal = encoded[:, :, -1, :]
        summary = torch.cat([temporal_mean, temporal_std, terminal], dim=-1)
        embedding = self.head(summary)
        return embedding.permute(1, 0, 2).contiguous()


class LatentPathUncertainty(nn.Module):
    """Compute cooperative two-view path-stability diagnostics."""

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        contrastive_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = LatentPathEncoder(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.contrastive_temperature = max(float(contrastive_temperature), 1e-6)

    def _contrastive_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError("contrastive views must have matching shapes")
        batch_size = left.shape[0]
        if batch_size <= 1:
            return left.new_zeros(())
        left_n = F.normalize(left.float(), dim=-1, eps=1e-8)
        right_n = F.normalize(right.float(), dim=-1, eps=1e-8)
        logits = left_n @ right_n.T / self.contrastive_temperature
        labels = torch.arange(batch_size, device=left.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    def forward(
        self,
        trajectory_a: torch.Tensor,
        trajectory_b: torch.Tensor,
    ) -> PathUncertaintyResult:
        embedding_a = self.encoder(trajectory_a)
        embedding_b = self.encoder(trajectory_b)
        example_a = embedding_a.mean(dim=0)
        example_b = embedding_b.mean(dim=0)

        per_witness_similarity = F.cosine_similarity(
            embedding_a.float(),
            embedding_b.float(),
            dim=-1,
            eps=1e-8,
        ).transpose(0, 1)
        per_witness_path_disagreement = 1.0 - per_witness_similarity
        path_similarity = F.cosine_similarity(
            example_a.float(),
            example_b.float(),
            dim=-1,
            eps=1e-8,
        )
        path_disagreement = 1.0 - path_similarity
        path_confidence = ((path_similarity + 1.0) * 0.5).clamp(0.0, 1.0)

        return PathUncertaintyResult(
            per_witness_embedding_a=embedding_a,
            per_witness_embedding_b=embedding_b,
            example_embedding_a=example_a,
            example_embedding_b=example_b,
            per_witness_path_disagreement=per_witness_path_disagreement,
            path_disagreement=path_disagreement,
            path_similarity=path_similarity,
            path_confidence=path_confidence,
            contrastive_loss=self._contrastive_loss(example_a, example_b),
        )
