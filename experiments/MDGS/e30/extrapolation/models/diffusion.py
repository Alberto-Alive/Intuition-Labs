"""Conditional latent diffusion modules for E30."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import TRACE_INPUT_DIM


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class DiffusionTrainingState:
    """Outputs from the diffusion-denoising training path."""

    loss: torch.Tensor
    clean_latent: torch.Tensor | None
    timesteps: torch.Tensor | None
    noised_latent: torch.Tensor | None
    predicted_noise: torch.Tensor | None


class TraceTargetEncoder(nn.Module):
    """Maps trace inputs into the diffusion latent space used only for denoising supervision."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        trace_input_dim: int = TRACE_INPUT_DIM,
    ) -> None:
        super().__init__()
        self.net = _make_mlp(trace_input_dim, hidden_dim, latent_dim, dropout)

    def forward(self, trace_inputs: torch.Tensor) -> torch.Tensor:
        return self.net(trace_inputs)


class ConditionalLatentDenoiser(nn.Module):
    """Predict epsilon from a noised latent, an anchor latent, and a timestep."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_steps: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.timestep_embed = nn.Embedding(num_steps + 1, 32)
        self.net = _make_mlp(latent_dim * 2 + 32, hidden_dim, latent_dim, dropout)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t: torch.Tensor, anchor: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_features = self.timestep_embed(timesteps.clamp(min=0, max=self.num_steps))
        return self.net(torch.cat([z_t, anchor, time_features], dim=-1))


class ConditionalLatentDiffusion(nn.Module):
    """DDPM-style latent diffusion conditioned only on the encoder anchor."""

    def __init__(
        self,
        latent_dim: int,
        num_steps: int,
        hidden_dim: int,
        target_hidden_dim: int,
        beta_start: float,
        beta_end: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.target_encoder = TraceTargetEncoder(
            latent_dim=latent_dim,
            hidden_dim=target_hidden_dim,
            dropout=dropout,
        )
        self.denoiser = ConditionalLatentDenoiser(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def training_loss(
        self,
        anchor: torch.Tensor,
        trace_inputs: torch.Tensor | None,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> DiffusionTrainingState:
        if trace_inputs is None:
            zero = anchor.new_zeros(())
            return DiffusionTrainingState(
                loss=zero,
                clean_latent=None,
                timesteps=None,
                noised_latent=None,
                predicted_noise=None,
            )

        clean_latent = self.target_encoder(trace_inputs)
        batch_size = clean_latent.size(0)

        if timesteps is None:
            timesteps = torch.randint(
                low=1,
                high=self.num_steps + 1,
                size=(batch_size,),
                device=clean_latent.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_latent)

        alpha_bar_t = self.alpha_bars[timesteps - 1].unsqueeze(-1)
        noised_latent = torch.sqrt(alpha_bar_t) * clean_latent + torch.sqrt(1.0 - alpha_bar_t) * noise
        predicted_noise = self.denoiser(noised_latent, anchor, timesteps)
        loss = F.mse_loss(predicted_noise, noise, reduction="mean")
        return DiffusionTrainingState(
            loss=loss,
            clean_latent=clean_latent,
            timesteps=timesteps,
            noised_latent=noised_latent,
            predicted_noise=predicted_noise,
        )

    def sample(
        self,
        anchor: torch.Tensor,
        *,
        num_samples: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic reverse denoising from K initial noise draws."""

        batch_size, latent_dim = anchor.shape
        expected_shape = (num_samples, batch_size, latent_dim)
        if noise is None:
            noise = torch.randn(expected_shape, device=anchor.device, dtype=anchor.dtype)
        elif tuple(noise.shape) != expected_shape:
            raise ValueError(f"Expected sampling noise shape {expected_shape}, got {tuple(noise.shape)}")

        z_t = noise.reshape(num_samples * batch_size, latent_dim)
        anchor_expanded = anchor.unsqueeze(0).expand(num_samples, -1, -1).reshape(num_samples * batch_size, latent_dim)

        for step in range(self.num_steps, 0, -1):
            timestep = torch.full(
                (num_samples * batch_size,),
                fill_value=step,
                device=anchor.device,
                dtype=torch.long,
            )
            beta_t = self.betas[step - 1]
            alpha_t = self.alphas[step - 1]
            alpha_bar_t = self.alpha_bars[step - 1]
            predicted_noise = self.denoiser(z_t, anchor_expanded, timestep)
            mean = (
                z_t
                - (beta_t / torch.sqrt((1.0 - alpha_bar_t).clamp_min(1e-6))) * predicted_noise
            ) / torch.sqrt(alpha_t.clamp_min(1e-6))
            z_t = mean

        return z_t.reshape(num_samples, batch_size, latent_dim)

    @property
    def trace_input_dim(self) -> int:
        return TRACE_INPUT_DIM
