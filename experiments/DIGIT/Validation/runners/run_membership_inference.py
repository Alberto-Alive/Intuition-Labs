"""Membership Inference run — Phase 1 validation entry point.

Runs MIA against all six systems (raw, digit, dp_laplace at ε ∈ {0.1,0.5,1.0,3.0})
across all five seeds for each of the four benchmark datasets.

Usage
-----
    python3 -m Validation.runners.run_membership_inference \
        --datasets nist_genomics mimic_iv_demo tcga \
        --seeds 0 1 2 3 4 \
        --checkpoint-dir checkpoints/ \
        --results-dir Validation/results/ \
        --train  # train models if no checkpoint exists

Dataset-specific notes
-----------------------
- nist_genomics : downloads 1000 Genomes data automatically
- mimic_iv_demo : requires PhysioNet files in data_cache/mimic_iv_demo/
- tcga          : downloads TCGA-BRCA clinical data from GDC API automatically

Output layout (per system per seed)
-------------------------------------
Validation/results/phase1_privacy/{dataset}/membership_inference/{system}/default/seed_{seed}/
    config.json
    metrics.json
    events.jsonl
    predictions.parquet
    summary.md
    stdout.log
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Ensure PoC package is importable ─────────────────────────────────────────
# parents[2] = experiments/DIGIT  (runner is at Validation/runners/run_*.py)
_digit_root = Path(__file__).resolve().parents[2]
_poc_path   = str(_digit_root / "PoC")
if _poc_path not in sys.path:
    sys.path.insert(0, _poc_path)

from Validation.data.base import PrivateDataset
from Validation.data.nist_genomics import load_nist_genomics
from Validation.data.tcga import load_tcga
from Validation.models.digit import ValidationConfig, ValidationDIGITModel
from Validation.models.dp_baseline import GenericDPLaplaceBaseline
from Validation.models.raw_baseline import RawBaseline
from Validation.attacks.membership_inference import (
    run_mia_full, run_mia_all_seeds, ShadowMIAConfig, SEEDS,
)
from Validation.harness.output_writer import write_mia_run, RESULTS_ROOT
from Validation.runners.train import train_all_seeds, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHASE = "phase1_privacy"
TASK  = "membership_inference"

# All ε values and the primary comparison point
DP_EPSILONS = [0.1, 0.5, 1.0, 3.0]
DP_PRIMARY  = 1.0

# System keys in results directory
def _dp_key(eps: float) -> str:
    return f"dp_laplace_e{eps}".replace(".", "p")   # e.g. dp_laplace_e1p0


SYSTEM_KEYS = ["raw", "digit"] + [_dp_key(e) for e in DP_EPSILONS]


# ── Dataset registry ──────────────────────────────────────────────────────────
def _load_dataset(
    name: str,
    data_cache: str,
    split_seed: int = 0,
) -> Tuple[PrivateDataset, PrivateDataset, PrivateDataset]:
    """Return (train, val, test) PrivateDataset for the named benchmark."""
    if name == "nist_genomics":
        return load_nist_genomics(
            cache_dir=str(Path(data_cache) / "nist_genomics"),
            split_seed=split_seed,
        )
    if name == "mimic_iv_demo":
        from Validation.data.mimic_iv_demo import load_mimic_iv_demo
        return load_mimic_iv_demo(
            data_dir=str(Path(data_cache) / "mimic_iv_demo"),
            split_seed=split_seed,
        )
    if name == "tcga":
        return load_tcga(
            cache_dir=str(Path(data_cache) / "tcga"),
            split_seed=split_seed,
        )
    raise ValueError(f"Unknown dataset: {name!r}")


def _make_digit_config(train_ds: PrivateDataset) -> ValidationConfig:
    return ValidationConfig(
        field_vocab_sizes=train_ds.info.field_vocab_sizes,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        d_ff=256,
    )


# ── Checkpoint handling ───────────────────────────────────────────────────────
def _checkpoint_path(ckpt_dir: Path, dataset: str, seed: int) -> Path:
    return ckpt_dir / dataset / f"seed_{seed}" / "best_model.pt"


def _ensure_checkpoints(
    dataset_name: str,
    train_ds: PrivateDataset,
    val_ds: PrivateDataset,
    seeds: List[int],
    ckpt_dir: Path,
    device: torch.device,
    force_train: bool = False,
) -> Dict[int, Path]:
    cfg = _make_digit_config(train_ds)
    out_dir = ckpt_dir / dataset_name
    missing = [s for s in seeds if not _checkpoint_path(ckpt_dir, dataset_name, s).exists()]

    if missing and not force_train:
        logger.warning(
            f"Missing checkpoints for {dataset_name} seeds {missing}. "
            f"Pass --train to train them automatically."
        )
    if missing:
        logger.info(f"Training {dataset_name} for seeds {missing}...")
        train_all_seeds(
            cfg=cfg,
            train_data=train_ds,
            val_data=val_ds,
            out_dir=out_dir,
            seeds=missing,
            device=device,
        )

    return {s: _checkpoint_path(ckpt_dir, dataset_name, s) for s in seeds}


# ── Shadow MIA default config (used for all systems) ─────────────────────────
_SHADOW_CFG = ShadowMIAConfig(
    n_shadow=32,
    n_queries_per_record=30,
    shadow_frac=0.5,
    batch_size=128,
)


# ── Per-seed MIA run ──────────────────────────────────────────────────────────
def _run_seed(
    *,
    system_key: str,
    system: Any,
    member_data: PrivateDataset,
    nonmember_data: PrivateDataset,
    dataset_name: str,
    seed: int,
    seed_mode: str,
    num_samples: int,
    results_root: Path,
    config_extra: Optional[Dict] = None,
    system_factory: Optional[Any] = None,
    shadow_cfg: Optional[ShadowMIAConfig] = None,
) -> Dict:
    # Capture log output for stdout.log
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        result = run_mia_full(
            system, member_data, nonmember_data, seed,
            num_samples=num_samples,
            shadow_cfg=shadow_cfg or _SHADOW_CFG,
            system_factory=system_factory,
        )
    finally:
        root_logger.removeHandler(handler)

    # Build predictions records — include both sub-attacks
    direct = result.get("direct", {})
    shadow = result.get("shadow", {})
    predictions = [
        {"seed": seed, "system": system_key, "dataset": dataset_name,
         "attack": result.get("attack", "direct"),
         "auroc": result["auroc"], "advantage": result["advantage"],
         "direct_auroc": direct.get("auroc", float("nan")),
         "shadow_auroc": shadow.get("auroc", float("nan"))}
    ]
    events = [
        {"step": 1000, "auroc": result["auroc"], "advantage": result["advantage"],
         "direct_auroc": direct.get("auroc", float("nan")),
         "shadow_auroc": shadow.get("auroc", float("nan")),
         "stronger_attack": result.get("attack", "direct")}
    ]

    write_mia_run(
        phase=PHASE,
        dataset=dataset_name,
        system_key=system_key,
        variant="default",
        seed=seed,
        seed_mode=seed_mode,
        mia_result=result,
        predictions=predictions,
        events=events,
        config_extra=config_extra,
        results_root=results_root,
        stdout_log=buf.getvalue(),
    )
    return result


# ── Main per-dataset runner ───────────────────────────────────────────────────
def run_dataset(
    dataset_name: str,
    seeds: List[int],
    seed_mode: str,
    ckpt_dir: Path,
    data_cache: str,
    results_root: Path,
    device: torch.device,
    num_samples: int = 1000,
    force_train: bool = False,
    split_seed: int = 0,
) -> Dict[str, Dict]:
    logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

    # Load data
    train_ds, val_ds, test_ds = _load_dataset(dataset_name, data_cache, split_seed)
    logger.info(
        f"Splits: train={train_ds.num_records} val={val_ds.num_records} "
        f"test={test_ds.num_records} | "
        f"positive rate (train): {train_ds.info.positive_rate:.3f}"
    )

    # Ensure trained DIGIT checkpoints
    ckpts = _ensure_checkpoints(
        dataset_name, train_ds, val_ds, seeds, ckpt_dir, device, force_train
    )
    cfg = _make_digit_config(train_ds)

    # Member = train split, non-member = test split
    member_data    = train_ds
    nonmember_data = test_ds

    all_results: Dict[str, Dict] = {}

    for system_key in SYSTEM_KEYS:
        logger.info(f"\n  System: {system_key}")

        seed_results = []
        for s in seeds:
            # Build the system and its shadow factory
            if system_key == "raw":
                system = RawBaseline(member_data)
                system_factory = lambda ds: RawBaseline(ds)
                extra = {"system_type": "raw"}
            elif system_key == "digit":
                ckpt_path = ckpts.get(s)
                if ckpt_path is None or not ckpt_path.exists():
                    logger.warning(f"No checkpoint for seed {s}; skipping.")
                    continue
                system = load_checkpoint(cfg, ckpt_path, device)
                # DIGIT accepts private_data at inference; no factory needed
                system_factory = None
                extra = {"checkpoint": str(ckpt_path)}
            elif system_key.startswith("dp_laplace_"):
                eps_str = system_key.replace("dp_laplace_e", "").replace("p", ".")
                eps = float(eps_str)
                system = GenericDPLaplaceBaseline(member_data, epsilon=eps, seed=s)
                # Capture eps and s in closure for shadow folds
                _eps, _s = eps, s
                system_factory = lambda ds, e=_eps, rs=_s: GenericDPLaplaceBaseline(
                    ds, epsilon=e, seed=rs
                )
                extra = {"epsilon": eps, "dp_seed": s}
            else:
                raise ValueError(f"Unknown system key: {system_key}")

            result = _run_seed(
                system_key=system_key,
                system=system,
                member_data=member_data,
                nonmember_data=nonmember_data,
                dataset_name=dataset_name,
                seed=s,
                seed_mode=seed_mode,
                num_samples=num_samples,
                results_root=results_root,
                config_extra=extra,
                system_factory=system_factory,
                shadow_cfg=_SHADOW_CFG,
            )
            seed_results.append(result)
            direct_auroc = result.get("direct", {}).get("auroc", float("nan"))
            shadow_auroc = result.get("shadow", {}).get("auroc", float("nan"))
            logger.info(
                f"    seed={s} AUROC={result['auroc']:.3f} [{result.get('attack','?')}] "
                f"direct={direct_auroc:.3f} shadow={shadow_auroc:.3f} "
                f"Adv={result['advantage']:.3f} "
                f"TPR@1%={result['tpr_at_1pct_fpr']:.3f} "
                f"TPR@5%={result['tpr_at_5pct_fpr']:.3f}"
            )

        # Aggregate across seeds
        if seed_results:
            metrics = ["auroc", "advantage", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"]
            agg = {m + "_mean": float(np.mean([r[m] for r in seed_results])) for m in metrics}
            agg.update({m + "_std": float(np.std([r[m] for r in seed_results])) for m in metrics})
            all_results[system_key] = {"per_seed": seed_results, **agg}

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="DIGIT Phase 1 — Membership Inference Run")
    p.add_argument(
        "--datasets", nargs="+",
        default=["nist_genomics", "tcga"],
        choices=["nist_genomics", "mimic_iv_demo", "tcga"],
        help="Datasets to evaluate (default: nist_genomics tcga)",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--reduced-seeds", action="store_true",
                   help="Use seeds 0,1,2 only (marks runs as seed_mode=reduced)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--data-cache",     type=str, default="data_cache")
    p.add_argument("--results-dir",    type=str, default=str(RESULTS_ROOT))
    p.add_argument("--num-samples",    type=int, default=1000,
                   help="Attack samples per seed (total, balanced)")
    p.add_argument("--train", action="store_true",
                   help="Train DIGIT models for missing checkpoints")
    p.add_argument("--split-seed",     type=int, default=0)
    p.add_argument("--device",         type=str, default="")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.reduced_seeds:
        seeds = [0, 1, 2]
        seed_mode = "reduced"
        logger.warning("Running in reduced-seed mode (0,1,2). "
                       "Results must not be used as primary evidence.")
    else:
        seeds = args.seeds
        seed_mode = "full"

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    ckpt_dir     = Path(args.checkpoint_dir)
    results_root = Path(args.results_dir)
    summary_all  = {}

    for ds_name in args.datasets:
        try:
            result = run_dataset(
                dataset_name=ds_name,
                seeds=seeds,
                seed_mode=seed_mode,
                ckpt_dir=ckpt_dir,
                data_cache=args.data_cache,
                results_root=results_root,
                device=device,
                num_samples=args.num_samples,
                force_train=args.train,
                split_seed=args.split_seed,
            )
            summary_all[ds_name] = result
        except Exception as exc:
            logger.error(f"Dataset {ds_name} failed: {exc}", exc_info=True)
            summary_all[ds_name] = {"error": str(exc)}

    # Print comparison table
    _print_summary(summary_all)

    # Save top-level summary JSON
    summary_path = results_root / PHASE / "mia_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_all, indent=2, default=str))
    logger.info(f"\nSummary → {summary_path}")


def _print_summary(summary: Dict) -> None:
    print("\n" + "=" * 95)
    print("MEMBERSHIP INFERENCE — SUMMARY  (best of direct + shadow attacks)")
    print("=" * 95)
    header = (
        f"{'Dataset':<20} {'System':<25} {'AUROC':>7} {'Direct':>7} "
        f"{'Shadow':>7} {'Adv':>7} {'TPR@1%':>8} {'TPR@5%':>8}"
    )
    print(header)
    print("-" * 95)
    for ds, ds_res in summary.items():
        if "error" in ds_res:
            print(f"{ds:<20}  ERROR: {ds_res['error']}")
            continue
        for sys_key, sys_res in ds_res.items():
            auroc  = sys_res.get("auroc_mean", float("nan"))
            adv    = sys_res.get("advantage_mean", float("nan"))
            t1     = sys_res.get("tpr_at_1pct_fpr_mean", float("nan"))
            t5     = sys_res.get("tpr_at_5pct_fpr_mean", float("nan"))
            mark   = " ←" if sys_key in ("digit", "dp_laplace_e1p0") else ""
            # Extract direct / shadow sub-results from per-seed records
            per_seed = sys_res.get("per_seed", [])
            d_vals = [r.get("direct", {}).get("auroc", float("nan")) for r in per_seed]
            s_vals = [r.get("shadow", {}).get("auroc", float("nan")) for r in per_seed]
            d_mean = float(np.mean([v for v in d_vals if not math.isnan(v)])) if d_vals else float("nan")
            s_mean = float(np.mean([v for v in s_vals if not math.isnan(v)])) if s_vals else float("nan")
            print(
                f"{ds:<20} {sys_key:<25} {auroc:>7.3f} {d_mean:>7.3f} "
                f"{s_mean:>7.3f} {adv:>7.3f} {t1:>8.3f} {t5:>8.3f}{mark}"
            )
    print("=" * 95)
    print("  AUROC = best(direct, shadow) | Direct = direct-query attack | Shadow = shadow-dataset attack")
    print("  ← marks primary comparison systems (DIGIT and DP-Laplace ε=1.0)")


if __name__ == "__main__":
    main()
