"""
Disturbance Profile Uncertainty
================================

A compact PyTorch scaffold for structured perturbation profiling:

    input -> encoder -> latent anchor z
          -> disturbance probes -> perturbed latent witnesses
          -> response measurements -> uncertainty profile
          -> risk / abstention head

The design goal is to keep class direction separate from uncertainty:
    - class_logits come from the base classifier on the clean latent anchor
    - uncertainty_profile comes from how predictions react to disturbances
    - risk_logit is learned from the profile and can drive abstention/routing

This file is model-agnostic. Plug in any base model that implements:
    encode(x) -> z
    classify_latent(z) -> logits

A small MLP example is included for tabular/vector inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim).clamp_min(eps)


def _normalize(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return x / _safe_norm(x, dim=dim, keepdim=True, eps=eps)


def entropy_from_probs(p: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def top2_margin(logits: Tensor) -> Tensor:
    """Top-1 minus top-2 logit margin. Shape: [...]."""
    top2 = logits.topk(k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def binary_or_top_margin(logits: Tensor) -> Tensor:
    """
    Signed-ish confidence margin.
    - For binary logits with shape [..., 1], returns raw logit.
    - For binary/multiclass logits with shape [..., C>=2], returns top1-top2 margin.
    """
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return top2_margin(logits)


def pairwise_direction_diversity(directions: Tensor) -> Tensor:
    """
    Penalize learned probe directions that collapse to the same direction.

    directions: [B, K, D]
    returns scalar mean absolute off-diagonal cosine similarity.
    """
    if directions.shape[1] <= 1:
        return directions.new_tensor(0.0)
    d = _normalize(directions, dim=-1)
    sim = torch.einsum("bkd,bld->bkl", d, d)
    k = sim.shape[-1]
    eye = torch.eye(k, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye)
    return off_diag.abs().mean()


UNCERTAINTY_AXIS_NAMES = [
    "consensus_disorder",
    "boundary_fragility",
    "perturbation_instability",
    "method_conflict",
    "support_novelty",
    "evidence_coverage",
    "calibration_residual",
    "risk_head_uncertainty",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    axis: str
    direction: float
    normalizer: str = "robust"
    missing_policy: str = "zero"
    source: str = "profile"


def _feature_source(name: str) -> str:
    return name.split("/", 1)[0] if "/" in name else "profile"


def _infer_feature_spec(name: str) -> FeatureSpec:
    axis = "perturbation_instability"
    direction = 1.0
    source = _feature_source(name)

    if name.startswith("consensus/"):
        axis = "consensus_disorder"
        direction = -1.0 if "agreement_mean" in name else 1.0
    elif name.startswith("boundary/") or name == "clean/margin":
        axis = "boundary_fragility"
        direction = -1.0 if name.endswith("abs_margin") or name == "clean/margin" else 1.0
    elif name == "clean/entropy":
        axis = "boundary_fragility"
        direction = 1.0
    elif name.startswith("method/"):
        axis = "method_conflict"
        direction = -1.0 if name.endswith("confidence") else 1.0
    elif name.startswith("support/"):
        axis = "support_novelty"
        direction = -1.0 if name.endswith("local_density") or name.endswith("prototype_margin") else 1.0
        if name.endswith("memory_available"):
            direction = 0.0
    elif name.startswith("evidence/"):
        axis = "evidence_coverage"
        if name.endswith("positive_fraction") or name.endswith("negative_fraction"):
            direction = 0.0
        elif "balance_abs" in name or "effective" in name:
            direction = -1.0
        else:
            direction = 1.0
    elif name.startswith("calibration/"):
        axis = "calibration_residual"
        direction = -1.0 if name.endswith("top1_confidence") or name.endswith("probability_margin") else 1.0
    elif name.startswith("risk_head/") or name.startswith("probe_family/"):
        axis = "risk_head_uncertainty"
        direction = 1.0
    elif name.endswith("/flip_rate") or name.endswith("/delta_prob_mean") or name.endswith("/js_to_clean"):
        axis = "perturbation_instability"
        direction = 1.0
    elif name.endswith("/entropy_mean") or name.endswith("/entropy_var"):
        axis = "perturbation_instability"
        direction = 1.0
    elif name.endswith("/margin_drop") or name.endswith("/margin_var"):
        axis = "boundary_fragility"
        direction = 1.0
    elif name.endswith("/margin_mean") or name.endswith("/boundary_flip_distance"):
        axis = "boundary_fragility"
        direction = -1.0
    elif name.endswith("/boundary_grad_norm") or name.endswith("/boundary_flip_rate"):
        axis = "boundary_fragility"
        direction = 1.0
    elif name.endswith("/latent_dist") or name.endswith("/noise_norm") or name.endswith("/masked_fraction"):
        axis = "perturbation_instability"
        direction = 0.0

    return FeatureSpec(name=name, axis=axis, direction=direction, source=source)


def _build_feature_specs(names: Iterable[str]) -> list[FeatureSpec]:
    return [_infer_feature_spec(name) for name in names]


def _axis_scores_from_profile(
    normalized_profile: Tensor,
    specs: list[FeatureSpec],
    *,
    axis_names: list[str] | None = None,
) -> Tensor:
    names = axis_names or UNCERTAINTY_AXIS_NAMES
    scores = []
    for axis in names:
        indices = [
            idx
            for idx, spec in enumerate(specs)
            if spec.axis == axis and abs(float(spec.direction)) > 0.0
        ]
        if not indices:
            scores.append(normalized_profile.new_zeros(normalized_profile.shape[0]))
            continue
        direction = normalized_profile.new_tensor([specs[idx].direction for idx in indices])
        axis_values = normalized_profile[:, indices] * direction[None, :]
        scores.append(axis_values.mean(dim=1))
    return torch.stack(scores, dim=1)


def _feature_specs_as_dicts(specs: list[FeatureSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "axis": spec.axis,
            "direction": float(spec.direction),
            "normalizer": spec.normalizer,
            "missing_policy": spec.missing_policy,
            "source": spec.source,
        }
        for spec in specs
    ]


# -----------------------------------------------------------------------------
# Base latent classifier interface and example implementation
# -----------------------------------------------------------------------------


class LatentClassifier(nn.Module):
    """
    Interface expected by DisturbanceProfileModel.

    Your existing model can be adapted by adding:
        encode(x) -> latent tensor [B, D]
        classify_latent(z) -> logits [B, C]
    """

    def encode(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def classify_latent(self, z: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        z = self.encode(x)
        logits = self.classify_latent(z)
        return {"z": z, "logits": logits}


class MLPLatentClassifier(LatentClassifier):
    """Simple tabular/vector baseline. Replace with your real encoder/classifier."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d_in = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d_in = hidden_dim
        layers += [nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim)]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(latent_dim, num_classes)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def classify_latent(self, z: Tensor) -> Tensor:
        return self.head(z)


# -----------------------------------------------------------------------------
# Disturbance probes
# -----------------------------------------------------------------------------


@dataclass
class ProbeOutput:
    name: str
    z_perturbed: Tensor  # [B, K, D]
    aux: Dict[str, Tensor]


class DisturbanceProbe(nn.Module):
    """
    Base class for a disturbance probe.

    A probe receives the clean latent anchor z and returns K perturbed versions.
    The probe itself should not decide the class. It only creates stress tests.
    """

    name: str = "base"

    def forward(
        self,
        z: Tensor,
        model: LatentClassifier,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def regularization(self) -> Dict[str, Tensor]:
        return {}


class GaussianNoiseProbe(DisturbanceProbe):
    """Generic local robustness probe: z + sigma * N(0, I)."""

    name = "gaussian"

    def __init__(self, k: int = 8, sigma: float = 0.05) -> None:
        super().__init__()
        self.k = k
        self.sigma = sigma

    def forward(
        self,
        z: Tensor,
        model: LatentClassifier,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeOutput:
        del model, context
        noise = torch.randn(z.shape[0], self.k, z.shape[1], device=z.device, dtype=z.dtype)
        z_pert = z[:, None, :] + self.sigma * noise
        return ProbeOutput(self.name, z_pert, {"noise_norm": _safe_norm(self.sigma * noise, dim=-1).mean(dim=1)})


class FeatureMaskProbe(DisturbanceProbe):
    """
    Feature reliance probe: randomly masks latent dimensions.

    If the prediction changes a lot when a small set of latent dimensions is hidden,
    the decision may depend on fragile/narrow evidence.
    """

    name = "feature_mask"

    def __init__(self, k: int = 8, mask_prob: float = 0.15, rescale: bool = True) -> None:
        super().__init__()
        self.k = k
        self.mask_prob = mask_prob
        self.rescale = rescale

    def forward(
        self,
        z: Tensor,
        model: LatentClassifier,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeOutput:
        del model, context
        keep = (torch.rand(z.shape[0], self.k, z.shape[1], device=z.device, dtype=z.dtype) > self.mask_prob).to(z.dtype)
        if self.rescale:
            keep = keep / max(1.0 - self.mask_prob, 1e-6)
        z_pert = z[:, None, :] * keep
        masked_fraction = (keep == 0).to(z.dtype).mean(dim=-1).mean(dim=1)
        return ProbeOutput(self.name, z_pert, {"masked_fraction": masked_fraction})


class BoundaryProbe(DisturbanceProbe):
    """
    Boundary fragility probe.

    It estimates the latent direction that reduces the clean top1-top2 margin,
    then steps along that direction. If a small step flips the prediction, the
    sample is close to a decision boundary.

    By default, the gradient direction is detached. This avoids expensive and
    sometimes unstable higher-order gradients during normal training.
    """

    name = "boundary"

    def __init__(self, k: int = 4, epsilon: float = 0.10, detach_direction: bool = True) -> None:
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.detach_direction = detach_direction

    def forward(
        self,
        z: Tensor,
        model: LatentClassifier,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeOutput:
        with torch.enable_grad():
            z_work = z.detach().requires_grad_(True) if self.detach_direction else z.requires_grad_(True)
            classify_kwargs = dict(context or {})
            clean_logits = classify_kwargs.pop("clean_logits", None)
            if clean_logits is None:
                logits = model.classify_latent(z_work, **classify_kwargs)
                clean_logits = logits.detach()
            else:
                logits = clean_logits if z_work is z else model.classify_latent(z_work, **classify_kwargs)
            margin = binary_or_top_margin(logits).sum()
            grad = torch.autograd.grad(
                margin,
                z_work,
                create_graph=not self.detach_direction,
                retain_graph=True,
                only_inputs=True,
            )[0]
            # Move opposite the confidence-margin gradient to seek a flip.
            direction = -_normalize(grad, dim=-1)
            if self.detach_direction:
                direction = direction.detach()

        # Multiple step strengths give a small fragility curve.
        scales = torch.linspace(1.0 / self.k, 1.0, self.k, device=z.device, dtype=z.dtype)
        z_pert = z[:, None, :] + self.epsilon * scales[None, :, None] * direction[:, None, :]
        flat_perturbed = z_pert.reshape(z.shape[0] * self.k, z.shape[1])
        pert_logits = model.classify_latent(flat_perturbed, **classify_kwargs)
        pert_logits = pert_logits.reshape(z.shape[0], self.k, -1)
        clean_pred = clean_logits.argmax(dim=-1)
        pert_pred = pert_logits.argmax(dim=-1)
        flip_mask = pert_pred != clean_pred[:, None]
        first_flip_index = torch.full(
            (z.shape[0],),
            fill_value=self.k - 1,
            device=z.device,
            dtype=torch.long,
        )
        if self.k > 0:
            any_flip = flip_mask.any(dim=-1)
            if bool(any_flip.any()):
                first_flip_index = torch.where(
                    any_flip,
                    flip_mask.float().argmax(dim=-1),
                    first_flip_index,
                )
        boundary_flip_distance = scales[first_flip_index]
        boundary_flip_rate = flip_mask.to(z.dtype).mean(dim=-1)
        return ProbeOutput(
            self.name,
            z_pert,
            {
                "boundary_grad_norm": _safe_norm(grad.detach(), dim=-1),
                "boundary_flip_distance": boundary_flip_distance,
                "boundary_flip_rate": boundary_flip_rate,
                "perturbed_logits": pert_logits,
            },
        )


class LearnedDirectionProbe(DisturbanceProbe):
    """
    Optional learned probe.

    The probe learns K bounded directions as functions of z. Use it after fixed
    probes already show useful signal; otherwise it may become opaque or learn
    shortcuts. Regularize with small norm + diversity.
    """

    name = "learned"

    def __init__(
        self,
        latent_dim: int,
        k: int = 4,
        epsilon: float = 0.05,
        hidden_dim: int = 128,
        diversity_weight: float = 0.01,
        norm_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.norm_weight = norm_weight
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, k * latent_dim),
        )
        self._last_delta: Optional[Tensor] = None

    def forward(
        self,
        z: Tensor,
        model: LatentClassifier,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProbeOutput:
        del model, context
        raw = self.net(z).view(z.shape[0], self.k, self.latent_dim)
        delta = self.epsilon * torch.tanh(raw)
        self._last_delta = delta
        z_pert = z[:, None, :] + delta
        return ProbeOutput(self.name, z_pert, {"learned_delta_norm": _safe_norm(delta, dim=-1).mean(dim=1)})

    def regularization(self) -> Dict[str, Tensor]:
        if self._last_delta is None:
            return {}
        delta = self._last_delta
        return {
            "learned_probe_norm_reg": self.norm_weight * _safe_norm(delta, dim=-1).mean(),
            "learned_probe_diversity_reg": self.diversity_weight * pairwise_direction_diversity(delta),
        }


# -----------------------------------------------------------------------------
# Measuring probe responses into profile features
# -----------------------------------------------------------------------------


@dataclass
class ProfileMeasurements:
    profile: Tensor  # [B, F]
    names: List[str]
    per_probe: Dict[str, Tensor]


class ProfileMeasurer(nn.Module):
    """
    Converts perturbed logits into named uncertainty-profile features.

    Features per probe:
      - delta_prob_mean: average probability movement vs clean prediction
      - js_to_clean: Jensen-Shannon style divergence to clean distribution
      - entropy_mean: average entropy under perturbation
      - entropy_var: variance of entropy under perturbation
      - flip_rate: fraction of perturbed predictions whose argmax changes
      - margin_mean: average confidence margin under perturbation
      - margin_var: margin variance under perturbation
      - margin_drop: clean margin minus perturbed margin mean
      - latent_dist: average latent movement from anchor
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        *,
        clean_z: Tensor,
        clean_logits: Tensor,
        probe_outputs: Iterable[ProbeOutput],
        model: LatentClassifier,
        classify_kwargs: Optional[Dict[str, Any]] = None,
        detach_base: bool = False,
    ) -> ProfileMeasurements:
        if detach_base:
            # Keep the profile branch observational when requested.
            clean_z = clean_z.detach()
            clean_logits = clean_logits.detach()
        clean_probs = F.softmax(clean_logits, dim=-1) if clean_logits.shape[-1] > 1 else torch.sigmoid(clean_logits)
        clean_pred = clean_logits.argmax(dim=-1) if clean_logits.shape[-1] > 1 else (clean_logits.squeeze(-1) > 0).long()
        clean_margin = binary_or_top_margin(clean_logits)
        classify_kwargs = dict(classify_kwargs or {})

        features: List[Tensor] = []
        names: List[str] = []
        per_probe: Dict[str, Tensor] = {}

        for out in probe_outputs:
            b, k, d = out.z_perturbed.shape
            flat_z = out.z_perturbed.reshape(b * k, d)
            flat_logits = out.aux.get("perturbed_logits")
            if flat_logits is None:
                flat_logits = model.classify_latent(flat_z, **classify_kwargs)
            if detach_base:
                flat_logits = flat_logits.detach()
            pert_logits = flat_logits.reshape(b, k, -1)

            if clean_logits.shape[-1] > 1:
                pert_probs = F.softmax(pert_logits, dim=-1)
                pert_pred = pert_logits.argmax(dim=-1)
                # Jensen-Shannon-like divergence to clean distribution.
                p = clean_probs[:, None, :].expand_as(pert_probs).clamp_min(self.eps)
                q = pert_probs.clamp_min(self.eps)
                m = 0.5 * (p + q)
                kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
                kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
                js = 0.5 * (kl_pm + kl_qm)
                entropy = entropy_from_probs(pert_probs, dim=-1, eps=self.eps)
                delta_prob = (pert_probs - clean_probs[:, None, :]).abs().mean(dim=-1)
            else:
                pert_probs = torch.sigmoid(pert_logits)
                pert_pred = (pert_logits.squeeze(-1) > 0).long()
                p = clean_probs[:, None, :].expand_as(pert_probs).clamp(self.eps, 1 - self.eps)
                q = pert_probs.clamp(self.eps, 1 - self.eps)
                # Binary entropy and symmetric-ish probability movement.
                entropy = -(q * q.log() + (1 - q) * (1 - q).log()).squeeze(-1)
                delta_prob = (q - p).abs().squeeze(-1)
                js = delta_prob.pow(2)  # lightweight proxy for binary case

            pert_margin = binary_or_top_margin(pert_logits)
            latent_dist = _safe_norm(out.z_perturbed - clean_z[:, None, :], dim=-1)
            flip_rate = (pert_pred != clean_pred[:, None]).to(clean_z.dtype).mean(dim=1)

            probe_features = {
                f"{out.name}/delta_prob_mean": delta_prob.mean(dim=1),
                f"{out.name}/js_to_clean": js.mean(dim=1),
                f"{out.name}/entropy_mean": entropy.mean(dim=1),
                f"{out.name}/entropy_var": entropy.var(dim=1, unbiased=False),
                f"{out.name}/flip_rate": flip_rate,
                f"{out.name}/margin_mean": pert_margin.mean(dim=1),
                f"{out.name}/margin_var": pert_margin.var(dim=1, unbiased=False),
                f"{out.name}/margin_drop": clean_margin - pert_margin.mean(dim=1),
                f"{out.name}/latent_dist": latent_dist.mean(dim=1),
            }

            for aux_name, aux_value in out.aux.items():
                if aux_name == "perturbed_logits":
                    continue
                if not torch.is_tensor(aux_value):
                    continue
                aux_tensor = aux_value.to(device=clean_z.device, dtype=clean_z.dtype)
                if detach_base:
                    aux_tensor = aux_tensor.detach()
                if aux_tensor.ndim == 0:
                    aux_tensor = aux_tensor.expand(clean_z.shape[0])
                elif aux_tensor.shape[0] != clean_z.shape[0]:
                    continue
                probe_features[f"{out.name}/{aux_name}"] = aux_tensor

            for key, value in probe_features.items():
                features.append(value[:, None])
                names.append(key)
            per_probe[out.name] = torch.cat([v[:, None] for v in probe_features.values()], dim=1)

        if not features:
            raise ValueError("At least one disturbance probe is required.")

        profile = torch.cat(features, dim=1)
        return ProfileMeasurements(profile=profile, names=names, per_probe=per_probe)


@dataclass
class DisturbanceProfileResult:
    """Outputs from the latent disturbance-profile head."""

    profile: torch.Tensor
    feature_names: List[str]
    normalized_profile: torch.Tensor
    feature_specs: List[FeatureSpec]
    axis_scores: torch.Tensor
    axis_names: List[str]
    risk_profile: torch.Tensor
    risk_profile_names: List[str]
    per_probe: Dict[str, Tensor]
    risk_logit: torch.Tensor
    risk_prob: torch.Tensor
    calibrated_risk: torch.Tensor
    commit: torch.Tensor
    clean_margin: torch.Tensor
    regularization_loss: torch.Tensor
    regularization_terms: Dict[str, torch.Tensor]


class LatentSupportMemory(nn.Module):
    """Train-split latent memory used only for support/novelty profile features."""

    feature_names = [
        "support/knn_distance_mean",
        "support/knn_distance_min",
        "support/local_density",
        "support/local_label_entropy",
        "support/local_error_rate",
        "support/prototype_distance_min",
        "support/prototype_margin",
        "support/prototype_distance_predicted",
        "support/memory_available",
    ]

    def __init__(
        self,
        *,
        latent_dim: int,
        max_items: int = 4096,
        k: int = 16,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_items = int(max_items)
        self.k = int(k)
        self.num_classes = int(num_classes)
        self.register_buffer(
            "support_anchors",
            torch.empty(0, self.latent_dim),
            persistent=False,
        )
        self.register_buffer("support_labels", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("support_incorrect", torch.empty(0), persistent=False)
        self.register_buffer(
            "support_prototypes",
            torch.empty(self.num_classes, self.latent_dim),
            persistent=False,
        )
        self.register_buffer(
            "support_prototype_mask",
            torch.zeros(self.num_classes, dtype=torch.bool),
            persistent=False,
        )

    @torch.no_grad()
    def fit(
        self,
        anchors: Tensor,
        labels: Tensor,
        incorrect: Optional[Tensor] = None,
        *,
        max_items: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        if anchors.ndim != 2 or anchors.shape[-1] != self.latent_dim:
            raise ValueError(
                f"anchors must have shape [N, {self.latent_dim}], got {tuple(anchors.shape)}"
            )
        labels = labels.detach().reshape(-1).long().to(device=anchors.device)
        if labels.shape[0] != anchors.shape[0]:
            raise ValueError("labels must have the same leading dimension as anchors")
        if incorrect is None:
            incorrect = torch.zeros(labels.shape[0], device=anchors.device)
        incorrect = incorrect.detach().reshape(-1).to(device=anchors.device, dtype=anchors.dtype)
        if incorrect.shape[0] != anchors.shape[0]:
            raise ValueError("incorrect must have the same leading dimension as anchors")

        limit = max(0, int(max_items if max_items is not None else self.max_items))
        anchors = anchors.detach().to(dtype=torch.float32)
        if limit == 0:
            anchors = anchors[:0]
            labels = labels[:0]
            incorrect = incorrect[:0]
        elif anchors.shape[0] > limit:
            anchors = anchors[:limit]
            labels = labels[:limit]
            incorrect = incorrect[:limit]

        inferred_classes = int(labels.max().item()) + 1 if labels.numel() else self.num_classes
        class_count = max(int(num_classes or self.num_classes), inferred_classes, 1)
        prototypes = anchors.new_zeros(class_count, self.latent_dim)
        prototype_mask = torch.zeros(class_count, device=anchors.device, dtype=torch.bool)
        for class_idx in range(class_count):
            mask = labels == class_idx
            if bool(mask.any()):
                prototypes[class_idx] = anchors[mask].mean(dim=0)
                prototype_mask[class_idx] = True

        self.num_classes = class_count
        self.support_anchors = anchors
        self.support_labels = labels
        self.support_incorrect = incorrect.to(dtype=torch.float32)
        self.support_prototypes = prototypes
        self.support_prototype_mask = prototype_mask

    def forward(self, clean_z: Tensor, clean_logits: Tensor, *, k: Optional[int] = None) -> tuple[Tensor, List[str]]:
        batch_size = clean_z.shape[0]
        names = list(self.feature_names)
        if self.support_anchors.numel() == 0:
            return clean_z.new_zeros(batch_size, len(names)), names

        anchors = self.support_anchors.to(device=clean_z.device, dtype=torch.float32)
        labels = self.support_labels.to(device=clean_z.device)
        incorrect = self.support_incorrect.to(device=clean_z.device, dtype=torch.float32)
        query = clean_z.to(dtype=torch.float32)
        distances = torch.cdist(query, anchors)
        k_eff = min(max(1, int(k if k is not None else self.k)), int(anchors.shape[0]))
        knn_distance, knn_index = distances.topk(k=k_eff, dim=-1, largest=False)

        knn_labels = labels[knn_index]
        knn_incorrect = incorrect[knn_index]
        knn_distance_mean = knn_distance.mean(dim=-1)
        knn_distance_min = knn_distance.min(dim=-1).values
        local_density = 1.0 / (1.0 + knn_distance_mean)

        class_count = max(int(self.num_classes), int(clean_logits.shape[-1]), 1)
        label_counts = query.new_zeros(batch_size, class_count)
        valid_labels = knn_labels.clamp(0, class_count - 1)
        label_counts.scatter_add_(
            1,
            valid_labels,
            torch.ones_like(valid_labels, dtype=query.dtype),
        )
        label_probs = label_counts / float(k_eff)
        local_label_entropy = entropy_from_probs(label_probs, dim=-1)
        if class_count > 1:
            local_label_entropy = local_label_entropy / math.log(float(class_count))
        local_error_rate = knn_incorrect.mean(dim=-1).to(dtype=query.dtype)

        prototypes = self.support_prototypes.to(device=clean_z.device, dtype=torch.float32)
        prototype_mask = self.support_prototype_mask.to(device=clean_z.device)
        if prototypes.shape[0] < class_count:
            pad = query.new_zeros(class_count - prototypes.shape[0], self.latent_dim)
            prototypes = torch.cat([prototypes, pad], dim=0)
            prototype_mask = torch.cat(
                [
                    prototype_mask,
                    torch.zeros(class_count - prototype_mask.shape[0], device=clean_z.device, dtype=torch.bool),
                ],
                dim=0,
            )
        prototype_distances = torch.cdist(query, prototypes[:class_count])
        finite_fill = prototype_distances.new_full(prototype_distances.shape, float("inf"))
        masked_distances = torch.where(prototype_mask[:class_count][None, :], prototype_distances, finite_fill)
        any_prototype = bool(prototype_mask[:class_count].any())
        if any_prototype:
            prototype_distance_min = masked_distances.min(dim=-1).values
            sorted_distances = masked_distances.sort(dim=-1).values
            first = sorted_distances[:, 0]
            second = sorted_distances[:, 1] if class_count > 1 else first
            prototype_margin = torch.where(torch.isfinite(second), second - first, query.new_zeros(batch_size))
            predicted = clean_logits.argmax(dim=-1).clamp(0, class_count - 1)
            predicted_distance = prototype_distances.gather(1, predicted[:, None]).squeeze(1)
            predicted_valid = prototype_mask[:class_count][predicted]
            prototype_distance_predicted = torch.where(
                predicted_valid,
                predicted_distance,
                prototype_distance_min,
            )
            prototype_distance_min = torch.where(
                torch.isfinite(prototype_distance_min),
                prototype_distance_min,
                query.new_zeros(batch_size),
            )
        else:
            prototype_distance_min = query.new_zeros(batch_size)
            prototype_margin = query.new_zeros(batch_size)
            prototype_distance_predicted = query.new_zeros(batch_size)

        memory_available = query.new_ones(batch_size)
        features = torch.stack(
            [
                knn_distance_mean,
                knn_distance_min,
                local_density,
                local_label_entropy,
                local_error_rate,
                prototype_distance_min,
                prototype_margin,
                prototype_distance_predicted,
                memory_available,
            ],
            dim=1,
        )
        return features.to(device=clean_z.device, dtype=clean_z.dtype), names

    def summary(self) -> dict[str, Any]:
        return {
            "support_memory_size": int(self.support_anchors.shape[0]),
            "support_memory_max_items": int(self.max_items),
            "support_k": int(self.k),
            "support_num_classes": int(self.num_classes),
            "support_prototype_classes": int(self.support_prototype_mask.sum().item()),
            "support_feature_names": list(self.feature_names),
        }


class LatentDisturbanceProfile(nn.Module):
    """Probe bank + profile measurer + risk head attached after the latent anchor."""

    def __init__(
        self,
        *,
        latent_dim: int,
        probes: Iterable[DisturbanceProbe],
        aggregator: Optional[ProfileAggregator] = None,
        config: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.probes = nn.ModuleList(list(probes))
        self.measurer = ProfileMeasurer()
        self.aggregator = aggregator if aggregator is not None else ProfileAggregator()
        self.config = config if config is not None else DisturbanceProfileConfig()
        self.learned_probe_active = bool(getattr(self.config, "enable_learned_probe", False))
        self.axis_names = list(UNCERTAINTY_AXIS_NAMES)
        self.support_memory = LatentSupportMemory(
            latent_dim=self.latent_dim,
            max_items=int(getattr(self.config, "profile_support_max_items", 4096)),
            k=int(getattr(self.config, "profile_support_k", 16)),
            num_classes=3,
        )
        self.register_buffer("profile_center", torch.empty(0), persistent=False)
        self.register_buffer("profile_scale", torch.empty(0), persistent=False)
        self.register_buffer("profile_normalizer_count", torch.zeros((), dtype=torch.long), persistent=False)

    def set_requires_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            try:
                parameter.requires_grad_(bool(enabled))
            except ValueError:
                # Lazy heads are initialized on first profile forward.
                continue

    def set_learned_probe_active(self, enabled: bool) -> None:
        self.learned_probe_active = bool(enabled)

    def _active_probes(self) -> list[DisturbanceProbe]:
        active: list[DisturbanceProbe] = []
        for probe in self.probes:
            if getattr(probe, "name", "") == "learned" and not self.learned_probe_active:
                continue
            active.append(probe)
        return active

    @torch.no_grad()
    def fit_support_memory(
        self,
        anchors: Tensor,
        labels: Tensor,
        incorrect: Optional[Tensor] = None,
    ) -> None:
        self.support_memory.fit(
            anchors,
            labels,
            incorrect,
            max_items=int(getattr(self.config, "profile_support_max_items", 4096)),
            num_classes=3,
        )

    @torch.no_grad()
    def fit_feature_normalizer(self, raw_profile: Tensor) -> None:
        if raw_profile.ndim != 2 or raw_profile.shape[0] == 0:
            self.profile_center = raw_profile.new_empty(0)
            self.profile_scale = raw_profile.new_empty(0)
            self.profile_normalizer_count = raw_profile.new_zeros((), dtype=torch.long)
            return
        profile = raw_profile.detach().to(dtype=torch.float32)
        center = profile.median(dim=0).values
        q25 = profile.quantile(0.25, dim=0)
        q75 = profile.quantile(0.75, dim=0)
        robust_scale = (q75 - q25) / 1.349
        std_scale = profile.std(dim=0, unbiased=False)
        scale = torch.where(robust_scale > 1e-6, robust_scale, std_scale)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        self.profile_center = center.to(device=raw_profile.device)
        self.profile_scale = scale.to(device=raw_profile.device)
        self.profile_normalizer_count = torch.tensor(
            int(profile.shape[0]),
            device=raw_profile.device,
            dtype=torch.long,
        )

    def _normalize_profile(self, profile: Tensor) -> Tensor:
        if not bool(getattr(self.config, "enable_profile_normalization", True)):
            return profile
        if self.profile_center.numel() != profile.shape[1] or self.profile_scale.numel() != profile.shape[1]:
            return profile
        center = self.profile_center.to(device=profile.device, dtype=profile.dtype)
        scale = self.profile_scale.to(device=profile.device, dtype=profile.dtype).clamp_min(1e-6)
        normalized = (profile - center[None, :]) / scale[None, :]
        clip = float(getattr(self.config, "profile_normalization_clip", 5.0))
        if clip > 0:
            normalized = normalized.clamp(-clip, clip)
        return torch.nan_to_num(normalized, nan=0.0, posinf=clip if clip > 0 else 0.0, neginf=-clip if clip > 0 else 0.0)

    def profile_state_summary(self) -> dict[str, Any]:
        summary = self.support_memory.summary()
        summary.update(
            {
                "axis_names": list(self.axis_names),
                "enable_support_novelty": bool(getattr(self.config, "enable_support_novelty", True)),
                "enable_profile_normalization": bool(getattr(self.config, "enable_profile_normalization", True)),
                "profile_normalizer_count": int(self.profile_normalizer_count.item()),
                "profile_normalizer_dim": int(self.profile_center.numel()),
                "profile_risk_input_mode": str(getattr(self.config, "profile_risk_input_mode", "axis_scores")),
            }
        )
        return summary

    def _risk_profile(
        self,
        *,
        raw_profile: Tensor,
        normalized_profile: Tensor,
        axis_scores: Tensor,
        feature_names: list[str],
    ) -> tuple[Tensor, list[str]]:
        mode = str(getattr(self.config, "profile_risk_input_mode", "axis_scores")).strip().lower()
        if not bool(getattr(self.config, "enable_uncertainty_axis_profile", True)) and mode == "axis_scores":
            mode = "normalized_profile"
        if mode == "axis_scores":
            return axis_scores, list(self.axis_names)
        if mode == "normalized_profile":
            return normalized_profile, [f"normalized/{name}" for name in feature_names]
        if mode == "raw_profile":
            return raw_profile, list(feature_names)
        if mode == "normalized_profile_plus_axes":
            return torch.cat([normalized_profile, axis_scores], dim=1), [
                *[f"normalized/{name}" for name in feature_names],
                *[f"axis/{name}" for name in self.axis_names],
            ]
        raise ValueError(
            "profile_risk_input_mode must be one of "
            "{'axis_scores', 'normalized_profile', 'raw_profile', 'normalized_profile_plus_axes'}"
        )

    def forward(
        self,
        *,
        model: LatentClassifier,
        clean_z: Tensor,
        clean_logits: Tensor,
        classify_kwargs: Optional[Dict[str, Any]] = None,
        context_features: Optional[Dict[str, Tensor]] = None,
    ) -> DisturbanceProfileResult:
        classify_kwargs = dict(classify_kwargs or {})
        classify_kwargs.setdefault("deterministic", True)
        detach_profile = bool(getattr(self.config, "detach_profile_from_base", False))
        profile_z = clean_z.detach() if detach_profile else clean_z
        profile_logits = clean_logits.detach() if detach_profile else clean_logits
        context = dict(classify_kwargs)
        context["clean_logits"] = profile_logits
        probe_outputs = [
            probe(profile_z, model, context=context)
            for probe in self._active_probes()
        ]
        measurements = self.measurer(
            clean_z=profile_z,
            clean_logits=profile_logits,
            probe_outputs=probe_outputs,
            model=model,
            classify_kwargs=classify_kwargs,
            detach_base=detach_profile,
        )

        profile = measurements.profile
        extra_features: list[Tensor] = []
        extra_names: list[str] = []
        if getattr(self.config, "use_clean_margin_feature", True):
            extra_features.append(binary_or_top_margin(profile_logits)[:, None])
            extra_names.append("clean/margin")
        if getattr(self.config, "use_clean_entropy_feature", True):
            if profile_logits.shape[-1] > 1:
                p = F.softmax(profile_logits, dim=-1)
                ent = entropy_from_probs(p, dim=-1)
            else:
                p = torch.sigmoid(profile_logits).clamp(1e-8, 1 - 1e-8)
                ent = -(p * p.log() + (1 - p) * (1 - p).log()).squeeze(-1)
            extra_features.append(ent[:, None])
            extra_names.append("clean/entropy")
        if bool(getattr(self.config, "use_calibration_features", True)):
            if profile_logits.shape[-1] > 1:
                probabilities = F.softmax(profile_logits, dim=-1)
                top2_probability = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1).values
                top1_confidence = top2_probability[:, 0]
                if probabilities.shape[-1] > 1:
                    probability_margin = top2_probability[:, 0] - top2_probability[:, 1]
                    entropy_norm = entropy_from_probs(probabilities, dim=-1) / math.log(
                        float(probabilities.shape[-1])
                    )
                else:
                    probability_margin = top1_confidence
                    entropy_norm = torch.zeros_like(top1_confidence)
                margin_confidence = torch.sigmoid(binary_or_top_margin(profile_logits))
            else:
                probability = torch.sigmoid(profile_logits).clamp(1e-8, 1 - 1e-8).squeeze(-1)
                top1_confidence = torch.maximum(probability, 1.0 - probability)
                probability_margin = (2.0 * top1_confidence - 1.0).clamp_min(0.0)
                entropy_norm = -(probability * probability.log() + (1 - probability) * (1 - probability).log())
                entropy_norm = entropy_norm / math.log(2.0)
                margin_confidence = top1_confidence
            extra_features.extend(
                [
                    top1_confidence[:, None],
                    entropy_norm[:, None],
                    (top1_confidence - margin_confidence).abs()[:, None],
                    probability_margin[:, None],
                ]
            )
            extra_names.extend(
                [
                    "calibration/top1_confidence",
                    "calibration/entropy_norm",
                    "calibration/margin_confidence_gap",
                    "calibration/probability_margin",
                ]
            )
        for name, value in (context_features or {}).items():
            if not torch.is_tensor(value):
                continue
            feature = value.to(device=profile_z.device, dtype=profile_z.dtype)
            if detach_profile:
                feature = feature.detach()
            if feature.ndim == 0:
                feature = feature.expand(profile_z.shape[0])
            elif feature.ndim > 1:
                feature = feature.reshape(feature.shape[0], -1)
                if feature.shape[1] != 1:
                    continue
                feature = feature.squeeze(1)
            if feature.shape[0] != profile_z.shape[0]:
                continue
            extra_features.append(feature[:, None])
            extra_names.append(str(name))
        if bool(getattr(self.config, "enable_support_novelty", True)):
            support_features, support_names = self.support_memory(
                profile_z,
                profile_logits,
                k=int(getattr(self.config, "profile_support_k", 16)),
            )
            extra_features.append(support_features)
            extra_names.extend(support_names)
        if extra_features:
            profile = torch.cat([profile] + extra_features, dim=1)

        feature_names = measurements.names + extra_names
        probe_family_features: list[Tensor] = []
        probe_family_names: list[str] = []
        for suffix, family_name in (
            ("/flip_rate", "probe_family/flip_rate_spread"),
            ("/js_to_clean", "probe_family/js_to_clean_spread"),
            ("/margin_drop", "probe_family/margin_drop_spread"),
        ):
            indices = [idx for idx, name in enumerate(feature_names) if name.endswith(suffix)]
            if len(indices) >= 2:
                probe_family_features.append(profile[:, indices].std(dim=1, unbiased=False)[:, None])
                probe_family_names.append(family_name)
        if probe_family_features:
            profile = torch.cat([profile] + probe_family_features, dim=1)
            feature_names.extend(probe_family_names)
        feature_specs = _build_feature_specs(feature_names)
        normalized_profile = self._normalize_profile(profile)
        axis_scores = _axis_scores_from_profile(
            normalized_profile,
            feature_specs,
            axis_names=self.axis_names,
        )
        risk_profile, risk_profile_names = self._risk_profile(
            raw_profile=profile,
            normalized_profile=normalized_profile,
            axis_scores=axis_scores,
            feature_names=feature_names,
        )

        risk_logit = self.aggregator(risk_profile)
        risk_prob = torch.sigmoid(risk_logit)
        temperature = max(float(getattr(self.config, "profile_risk_temperature", 1.0)), 1e-6)
        calibrated_risk = torch.sigmoid(risk_logit / temperature)
        clean_margin = binary_or_top_margin(clean_logits)
        threshold = float(getattr(self.config, "commit_risk_threshold", 0.5))
        min_margin = float(getattr(self.config, "min_clean_margin", 0.0))
        if bool(getattr(self.config, "use_disturbance_risk_in_commitment_gate", True)):
            commit = (calibrated_risk < threshold) & (clean_margin.abs() >= min_margin)
        else:
            commit = (risk_prob < threshold) & (clean_margin.abs() >= min_margin)

        regularization_terms: Dict[str, torch.Tensor] = {}
        for probe in self._active_probes():
            regularization_terms.update(probe.regularization())
        if regularization_terms:
            regularization_loss = torch.stack([value for value in regularization_terms.values()]).sum()
        else:
            regularization_loss = clean_logits.new_zeros(())

        return DisturbanceProfileResult(
            profile=profile,
            feature_names=feature_names,
            normalized_profile=normalized_profile,
            feature_specs=feature_specs,
            axis_scores=axis_scores,
            axis_names=list(self.axis_names),
            risk_profile=risk_profile,
            risk_profile_names=risk_profile_names,
            per_probe=measurements.per_probe,
            risk_logit=risk_logit,
            risk_prob=risk_prob,
            calibrated_risk=calibrated_risk,
            commit=commit,
            clean_margin=clean_margin,
            regularization_loss=regularization_loss,
            regularization_terms=regularization_terms,
        )


# -----------------------------------------------------------------------------
# Profile aggregator and full model wrapper
# -----------------------------------------------------------------------------


class ProfileAggregator(nn.Module):
    """
    Learns profile -> risk_logit.

    risk_logit > 0 means likely unsafe / should abstain.
    risk_logit < 0 means likely safe to commit.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, profile: Tensor) -> Tensor:
        return self.net(profile).squeeze(-1)


@dataclass
class DisturbanceProfileConfig:
    detach_profile_from_base: bool = False
    use_clean_entropy_feature: bool = True
    use_clean_margin_feature: bool = True
    enable_uncertainty_axis_profile: bool = True
    profile_risk_input_mode: str = "axis_scores"
    enable_support_novelty: bool = True
    profile_support_k: int = 16
    profile_support_max_items: int = 4096
    enable_profile_normalization: bool = True
    profile_normalization_clip: float = 5.0
    use_calibration_features: bool = True
    use_disturbance_risk_in_commitment_gate: bool = True
    lambda_profile_risk: float = 0.25
    profile_risk_temperature: float = 1.0
    commit_risk_threshold: float = 0.10
    min_clean_margin: float = 0.0


class DisturbanceProfileModel(nn.Module):
    """
    Main wrapper.

    Outputs:
      clean_logits: base task prediction
      profile: structured uncertainty features
      risk_logit / risk_prob: learned risk estimate
      commit: boolean decision based on risk and clean margin
    """

    def __init__(
        self,
        base_model: LatentClassifier,
        probes: List[DisturbanceProbe],
        aggregator: Optional[ProfileAggregator] = None,
        config: Optional[DisturbanceProfileConfig] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.probes = nn.ModuleList(probes)
        self.measurer = ProfileMeasurer()
        self.aggregator = aggregator if aggregator is not None else ProfileAggregator()
        self.config = config if config is not None else DisturbanceProfileConfig()
        self.profile_names: List[str] = []

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        base = self.base_model(x)
        z = base["z"]
        clean_logits = base["logits"]

        probe_z = z.detach() if self.config.detach_profile_from_base else z
        probe_outputs = [probe(probe_z, self.base_model) for probe in self.probes]
        measured = self.measurer(
            clean_z=probe_z,
            clean_logits=clean_logits.detach() if self.config.detach_profile_from_base else clean_logits,
            probe_outputs=probe_outputs,
            model=self.base_model,
            detach_base=bool(self.config.detach_profile_from_base),
        )

        profile = measured.profile
        extra_names: List[str] = []
        extra_features: List[Tensor] = []
        if self.config.use_clean_margin_feature:
            extra_features.append(binary_or_top_margin(clean_logits)[:, None])
            extra_names.append("clean/margin")
        if self.config.use_clean_entropy_feature:
            if clean_logits.shape[-1] > 1:
                p = F.softmax(clean_logits, dim=-1)
                ent = entropy_from_probs(p, dim=-1)
            else:
                p = torch.sigmoid(clean_logits).clamp(1e-8, 1 - 1e-8)
                ent = -(p * p.log() + (1 - p) * (1 - p).log()).squeeze(-1)
            extra_features.append(ent[:, None])
            extra_names.append("clean/entropy")

        if extra_features:
            profile = torch.cat([profile] + extra_features, dim=1)
            names = measured.names + extra_names
        else:
            names = measured.names
        self.profile_names = names

        risk_logit = self.aggregator(profile)
        risk_prob = torch.sigmoid(risk_logit)
        temperature = max(float(getattr(self.config, "profile_risk_temperature", 1.0)), 1e-6)
        calibrated_risk = torch.sigmoid(risk_logit / temperature)
        margin = binary_or_top_margin(clean_logits)
        if self.config.use_disturbance_risk_in_commitment_gate:
            commit = (calibrated_risk < self.config.commit_risk_threshold) & (
                margin.abs() >= self.config.min_clean_margin
            )
        else:
            commit = (risk_prob < self.config.commit_risk_threshold) & (
                margin.abs() >= self.config.min_clean_margin
            )

        reg_terms: Dict[str, Tensor] = {}
        for probe in self.probes:
            reg_terms.update(probe.regularization())

        return {
            "clean_logits": clean_logits,
            "z": z,
            "profile": profile,
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "calibrated_risk": calibrated_risk,
            "commit": commit,
            "clean_margin": margin,
            **reg_terms,
        }

    @torch.no_grad()
    def predict_with_abstention(self, x: Tensor, abstain_id: int = -1) -> Dict[str, Tensor]:
        out = self.forward(x)
        logits = out["clean_logits"]
        if logits.shape[-1] == 1:
            pred = (logits.squeeze(-1) > 0).long()
        else:
            pred = logits.argmax(dim=-1)
        pred_abstain = pred.clone()
        pred_abstain[~out["commit"]] = abstain_id
        return {**out, "pred": pred, "pred_abstain": pred_abstain}


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


@dataclass
class LossConfig:
    task_weight: float = 1.0
    risk_weight: float = 1.0
    profile_stability_weight: float = 0.0
    uncertain_label: Optional[int] = None
    ignore_uncertain_for_task: bool = False
    risk_target_mode: str = "incorrect_only"  # incorrect | incorrect_only | uncertain_label | incorrect_or_uncertain


class DisturbanceProfileLoss(nn.Module):
    """
    Combines task classification loss with risk/abstention supervision.

    Risk target modes:
      - incorrect: risk=1 when clean prediction is wrong
      - incorrect_only: same as incorrect
      - uncertain_label: risk=1 when y == uncertain_label
      - incorrect_or_uncertain: union of both

    For best results:
      Stage 1: train base task model normally.
      Stage 2: freeze/partially freeze base; train risk head on disturbance profiles.
      Stage 3: fine-tune end-to-end with small risk/profile weights.
    """

    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else LossConfig()

    def forward(self, out: Dict[str, Tensor], y: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits = out["clean_logits"]
        cfg = self.config

        # Task CE, optionally ignoring an explicit uncertain/abstain label.
        if logits.shape[-1] == 1:
            # Binary labels expected as {0,1}. If uncertain labels exist, ignore them below.
            y_float = y.float()
            if cfg.uncertain_label is not None and cfg.ignore_uncertain_for_task:
                mask = y != cfg.uncertain_label
                if mask.any():
                    task_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1)[mask], y_float[mask])
                else:
                    task_loss = logits.new_tensor(0.0)
            else:
                task_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), y_float)
            pred = (logits.squeeze(-1) > 0).long()
        else:
            if cfg.uncertain_label is not None and cfg.ignore_uncertain_for_task:
                task_loss = F.cross_entropy(logits, y, ignore_index=cfg.uncertain_label)
            else:
                task_loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=-1)

        incorrect = (pred.detach() != y).float()
        uncertain = torch.zeros_like(incorrect)
        if cfg.uncertain_label is not None:
            uncertain = (y == cfg.uncertain_label).float()

        if cfg.risk_target_mode in {"incorrect", "incorrect_only"}:
            risk_target = incorrect
        elif cfg.risk_target_mode == "uncertain_label":
            risk_target = uncertain
        elif cfg.risk_target_mode == "incorrect_or_uncertain":
            risk_target = torch.maximum(incorrect, uncertain)
        else:
            raise ValueError(f"Unknown risk_target_mode: {cfg.risk_target_mode}")

        risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], risk_target)

        reg_loss = logits.new_tensor(0.0)
        for key, value in out.items():
            if key.endswith("_reg"):
                reg_loss = reg_loss + value

        # Optional gentle pressure: stable/low-risk examples should have lower average profile magnitude.
        # Usually keep this at 0 initially.
        profile_stability_loss = logits.new_tensor(0.0)
        if cfg.profile_stability_weight > 0:
            low_risk = (risk_target < 0.5).float()
            profile_mag = out["profile"].pow(2).mean(dim=1)
            if low_risk.sum() > 0:
                profile_stability_loss = (profile_mag * low_risk).sum() / low_risk.sum().clamp_min(1.0)

        total = (
            cfg.task_weight * task_loss
            + cfg.risk_weight * risk_loss
            + cfg.profile_stability_weight * profile_stability_loss
            + reg_loss
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_task": task_loss.detach(),
            "loss_risk": risk_loss.detach(),
            "loss_reg": reg_loss.detach(),
            "risk_target_rate": risk_target.mean().detach(),
            "risk_prob_mean": out["risk_prob"].mean().detach(),
            "commit_rate": out["commit"].float().mean().detach(),
        }
        return total, metrics


# -----------------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------------


def build_default_probes(
    latent_dim: int,
    *,
    gaussian_k: int = 8,
    mask_k: int = 8,
    boundary_k: int = 4,
    include_learned: bool = False,
    enable_boundary_probe: bool = True,
    enable_feature_mask_probe: bool = True,
    gaussian_sigma: float = 0.05,
    mask_prob: float = 0.15,
    boundary_epsilon: float = 0.10,
    learned_hidden_dim: int = 128,
    learned_diversity_weight: float = 0.01,
    learned_norm_weight: float = 0.01,
) -> List[DisturbanceProbe]:
    probes: List[DisturbanceProbe] = [
        GaussianNoiseProbe(k=gaussian_k, sigma=gaussian_sigma),
    ]
    if enable_feature_mask_probe:
        probes.append(FeatureMaskProbe(k=mask_k, mask_prob=mask_prob))
    if enable_boundary_probe:
        probes.append(BoundaryProbe(k=boundary_k, epsilon=boundary_epsilon, detach_direction=True))
    if include_learned:
        probes.append(
            LearnedDirectionProbe(
                latent_dim=latent_dim,
                k=4,
                epsilon=0.05,
                hidden_dim=learned_hidden_dim,
                diversity_weight=learned_diversity_weight,
                norm_weight=learned_norm_weight,
            )
        )
    return probes


def freeze_base_model(model: DisturbanceProfileModel, freeze: bool = True) -> None:
    for p in model.base_model.parameters():
        p.requires_grad_(not freeze)
