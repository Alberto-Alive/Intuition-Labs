"""Point-reader modules for E31 witness samples."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_witness_samples(witness_samples: torch.Tensor) -> tuple[int, int, int]:
    if witness_samples.ndim != 3:
        raise ValueError(
            "witness_samples must have shape (num_samples, batch_size, latent_dim)"
        )
    return witness_samples.shape  # type: ignore[return-value]


class IdentityPointReader(nn.Module):
    """Bypass the point reader and preserve witness latents unchanged."""

    def forward(self, witness_samples: torch.Tensor) -> torch.Tensor:
        _validate_witness_samples(witness_samples)
        return witness_samples


class LinearScalarMarginReader(nn.Module):
    """Project each raw witness directly to a scalar margin."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.projection = nn.Linear(self.latent_dim, 1)

    def forward(self, witness_samples: torch.Tensor) -> torch.Tensor:
        num_samples, batch_size, latent_dim = _validate_witness_samples(witness_samples)
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        flat = witness_samples.reshape(num_samples * batch_size, latent_dim)
        projected = self.projection(flat)
        return projected.reshape(num_samples, batch_size, 1)


class LinearSharedProjPointReader(nn.Module):
    """Apply a single affine map to each witness independently."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.projection = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, witness_samples: torch.Tensor) -> torch.Tensor:
        num_samples, batch_size, latent_dim = _validate_witness_samples(witness_samples)
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        flat = witness_samples.reshape(num_samples * batch_size, latent_dim)
        projected = self.projection(flat)
        return projected.reshape(num_samples, batch_size, self.latent_dim)


# Backward-compatible alias for older checkpoints and imports.
LinearPointReader = LinearSharedProjPointReader


def _normalize_mode(mode: str) -> str:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "linear":
        return "linear_shared_proj"
    if normalized_mode == "shallow_residual":
        return "linear_shared_proj"
    return normalized_mode


def build_point_reader(
    mode: str,
    *,
    latent_dim: int,
    hidden_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct a point reader from the named E31 family."""

    del hidden_dim, dropout
    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "identity":
        return IdentityPointReader()
    if normalized_mode == "linear_scalar_margin":
        return LinearScalarMarginReader(latent_dim=latent_dim)
    if normalized_mode == "linear_shared_proj":
        return LinearSharedProjPointReader(latent_dim=latent_dim)
    raise ValueError(
        "point_reader_mode must be one of {'identity', 'linear_scalar_margin', 'linear_shared_proj'}"
    )
