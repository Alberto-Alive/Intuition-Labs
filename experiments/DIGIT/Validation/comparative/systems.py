"""Mode-aware system factory for comparative attacks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..data.base import PrivateDataset
from ..data.ground_truth import GTConfig
from ..models.digit import ValidationConfig, ValidationDIGITModel
from ..models.dp_baseline import GenericDPLaplaceBaseline
from ..models.raw_baseline import RawBaseline
from .stability_gate import StabilityGateConfig, StabilityGatedDigitWrapper
from ..runners.train import load_checkpoint, train_all_seeds
from .specs import AttackMode, SystemSpec

logger = logging.getLogger(__name__)

DEFAULT_DP_EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
DIGIT_VARIANTS: Dict[str, Dict[str, int]] = {
    "a2_s2_c2_r2": {
        "num_answer_classes": 2,
        "num_support_classes": 2,
        "num_confidence_classes": 2,
        "num_risk_classes": 2,
    },
    "a3_s2_c2_r2": {
        "num_answer_classes": 3,
        "num_support_classes": 2,
        "num_confidence_classes": 2,
        "num_risk_classes": 2,
    },
    "a4_s3_c2_r2": {
        "num_answer_classes": 4,
        "num_support_classes": 3,
        "num_confidence_classes": 2,
        "num_risk_classes": 2,
    },
    "a6_s4_c3_r3": {
        "num_answer_classes": 6,
        "num_support_classes": 4,
        "num_confidence_classes": 3,
        "num_risk_classes": 3,
    },
}


def variant_name_for_mode(mode: AttackMode, setting: str = "default") -> str:
    return f"{mode.value}__{setting}"


def system_dir_name(system_spec: SystemSpec) -> str:
    return system_spec.system_key


def disambiguate_digit_variants(system_specs: List[SystemSpec]) -> bool:
    for family in ("digit", "digit_stability_gate", "digit_specificity_gate"):
        variants = {spec.digit_variant for spec in system_specs if spec.family == family}
        if len(variants) > 1:
            return True
    return False


def comparative_system_key(
    system_spec: SystemSpec,
    *,
    disambiguate_digit: bool = False,
) -> str:
    if system_spec.family == "digit" and disambiguate_digit:
        return f"digit_{system_spec.digit_variant}"
    if system_spec.family == "digit_stability_gate" and disambiguate_digit:
        return f"digit_stability_gate_{system_spec.digit_variant}"
    if system_spec.family == "digit_specificity_gate" and disambiguate_digit:
        return f"digit_specificity_gate_{system_spec.digit_variant}"
    return system_dir_name(system_spec)


def gt_config_for_mode(mode: AttackMode) -> GTConfig:
    return GTConfig(min_group_size=mode.min_group_size)


def make_digit_config(
    private_data: PrivateDataset,
    mode: AttackMode,
    digit_variant: str = "a6_s4_c3_r3",
) -> ValidationConfig:
    dims = DIGIT_VARIANTS[digit_variant]
    return ValidationConfig(
        field_vocab_sizes=private_data.info.field_vocab_sizes,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        d_ff=256,
        min_group_size=mode.min_group_size,
        num_answer_classes=dims["num_answer_classes"],
        num_support_classes=dims["num_support_classes"],
        num_confidence_classes=dims["num_confidence_classes"],
        num_risk_classes=dims["num_risk_classes"],
    )


def _checkpoint_path(ckpt_dir: Path, dataset_name: str, digit_variant: str, mode: AttackMode, seed: int) -> Path:
    return ckpt_dir / dataset_name / digit_variant / mode.value / f"seed_{seed}" / "best_model.pt"


def ensure_digit_checkpoints(
    *,
    dataset_name: str,
    train_data: PrivateDataset,
    val_data: PrivateDataset,
    mode: AttackMode,
    digit_variant: str,
    seeds: List[int],
    ckpt_dir: Path,
    device: torch.device,
    force_train: bool = False,
) -> Dict[int, Path]:
    cfg = make_digit_config(train_data, mode=mode, digit_variant=digit_variant)
    out_dir = ckpt_dir / dataset_name / digit_variant / mode.value
    missing = [s for s in seeds if not _checkpoint_path(ckpt_dir, dataset_name, digit_variant, mode, s).exists()]
    if missing and not force_train:
        logger.warning(
            "Missing comparative DIGIT checkpoints for %s %s %s seeds %s. Pass --train to build them.",
            dataset_name,
            digit_variant,
            mode.value,
            missing,
        )
    if missing and force_train:
        logger.info("Training DIGIT %s %s for seeds %s", digit_variant, mode.value, missing)
        train_all_seeds(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            out_dir=out_dir,
            seeds=missing,
            device=device,
            gt_cfg=gt_config_for_mode(mode),
        )
    return {s: _checkpoint_path(ckpt_dir, dataset_name, digit_variant, mode, s) for s in seeds}


def build_system(
    *,
    system_spec: SystemSpec,
    private_data: PrivateDataset,
    mode: AttackMode,
    seed: int,
    ckpt_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Return (system, system_factory, config_extra)."""
    gt_cfg = gt_config_for_mode(mode)

    if system_spec.family == "raw":
        system = RawBaseline(private_data, min_group_size=mode.min_group_size, cfg=gt_cfg)
        system_factory = lambda ds: RawBaseline(ds, min_group_size=mode.min_group_size, cfg=gt_cfg)
        return system, system_factory, {
            "system_type": "raw",
            "mode": mode.value,
            "min_group_size": mode.min_group_size,
        }

    if system_spec.family == "dp_laplace":
        eps = float(system_spec.epsilon or 1.0)
        system = GenericDPLaplaceBaseline(
            private_data,
            epsilon=eps,
            min_group_size=mode.min_group_size,
            cfg=gt_cfg,
            seed=seed,
        )
        system_factory = lambda ds, e=eps, rs=seed: GenericDPLaplaceBaseline(
            ds,
            epsilon=e,
            min_group_size=mode.min_group_size,
            cfg=gt_cfg,
            seed=rs,
        )
        return system, system_factory, {
            "system_type": "dp_laplace",
            "mode": mode.value,
            "min_group_size": mode.min_group_size,
            "epsilon": eps,
            "dp_seed": seed,
        }

    if system_spec.family == "digit":
        if ckpt_dir is None or dataset_name is None:
            raise ValueError("DIGIT system construction requires checkpoint dir and dataset name.")
        cfg = make_digit_config(private_data, mode=mode, digit_variant=system_spec.digit_variant)
        ckpt_path = _checkpoint_path(ckpt_dir, dataset_name, system_spec.digit_variant, mode, seed)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DIGIT checkpoint not found: {ckpt_path}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        system = load_checkpoint(cfg, ckpt_path, device)
        return system, None, {
            "system_type": "digit",
            "mode": mode.value,
            "min_group_size": mode.min_group_size,
            "digit_variant": system_spec.digit_variant,
            "checkpoint": str(ckpt_path),
        }

    if system_spec.family == "digit_stability_gate":
        if ckpt_dir is None or dataset_name is None:
            raise ValueError("DIGIT system construction requires checkpoint dir and dataset name.")
        cfg = make_digit_config(private_data, mode=mode, digit_variant=system_spec.digit_variant)
        ckpt_path = _checkpoint_path(ckpt_dir, dataset_name, system_spec.digit_variant, mode, seed)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DIGIT checkpoint not found: {ckpt_path}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        digit_model = load_checkpoint(cfg, ckpt_path, device)
        system = StabilityGatedDigitWrapper(
            digit_model,
            private_data,
            gate_cfg=StabilityGateConfig(max_drop_neighbors=8),
        )
        return system, None, {
            "system_type": "digit_stability_gate",
            "mode": mode.value,
            "min_group_size": mode.min_group_size,
            "digit_variant": system_spec.digit_variant,
            "checkpoint": str(ckpt_path),
            "stability_gate": "drop1_public_consensus",
            "stability_max_drop_neighbors": 8,
        }

    if system_spec.family == "digit_specificity_gate":
        if ckpt_dir is None or dataset_name is None:
            raise ValueError("DIGIT system construction requires checkpoint dir and dataset name.")
        cfg = make_digit_config(private_data, mode=mode, digit_variant=system_spec.digit_variant)
        ckpt_path = _checkpoint_path(ckpt_dir, dataset_name, system_spec.digit_variant, mode, seed)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DIGIT checkpoint not found: {ckpt_path}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        digit_model = load_checkpoint(cfg, ckpt_path, device)
        gate_cfg = StabilityGateConfig(
            max_drop_neighbors=8,
            max_public_specificity=gt_cfg.specificity_policy_cutoff,
            min_public_support_count=gt_cfg.confidence_thresholds[-1],
        )
        system = StabilityGatedDigitWrapper(
            digit_model,
            private_data,
            gate_cfg=gate_cfg,
        )
        return system, None, {
            "system_type": "digit_specificity_gate",
            "mode": mode.value,
            "min_group_size": mode.min_group_size,
            "digit_variant": system_spec.digit_variant,
            "checkpoint": str(ckpt_path),
            "stability_gate": "drop1_public_consensus",
            "stability_max_drop_neighbors": gate_cfg.max_drop_neighbors,
            "specificity_gate_cutoff": gate_cfg.max_public_specificity,
            "support_count_gate_min": gate_cfg.min_public_support_count,
        }

    raise ValueError(f"Unsupported system family: {system_spec.family}")
