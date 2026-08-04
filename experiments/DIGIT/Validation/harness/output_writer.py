"""Harness output writer — writes the six required files per run.

File layout (harness.md v1.0):
  Validation/results/{phase}/{dataset}/{task}/{system}/{variant}/seed_{seed}/
    config.json
    metrics.json
    events.jsonl
    predictions.parquet   (or predictions.csv)
    summary.md
    stdout.log

Writers are idempotent: re-running overwrites existing files.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Attempt parquet; fall back to CSV
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ── Path builder ──────────────────────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def seed_dir(
    phase: str,
    dataset: str,
    task: str,
    system: str,
    variant: str,
    seed: int,
    results_root: Optional[Path] = None,
) -> Path:
    root = results_root or RESULTS_ROOT
    return root / phase / dataset / task / system / variant / f"seed_{seed}"


# ── Config writer ─────────────────────────────────────────────────────────────
def write_config(
    out_dir: Path,
    *,
    system_key: str,
    dataset: str,
    task: str,
    phase: str,
    variant: str,
    seed: int,
    seed_mode: str = "full",
    harness_version: str = "1.0",
    digit_commit: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "harness_version": harness_version,
        "system_key": system_key,
        "dataset": dataset,
        "task": task,
        "phase": phase,
        "variant": variant,
        "seed": seed,
        "seed_mode": seed_mode,
        "digit_commit": digit_commit or _git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    _write_text(out_dir / "config.json", json.dumps(cfg, indent=2))


# ── Metrics writer ─────────────────────────────────────────────────────────────
def write_metrics(out_dir: Path, metrics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(out_dir / "metrics.json", json.dumps(metrics, indent=2, default=_json_default))


# ── Events writer (JSONL) ─────────────────────────────────────────────────────
def write_events(out_dir: Path, events: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, default=_json_default) for e in events]
    _write_text(out_dir / "events.jsonl", "\n".join(lines) + "\n")


# ── Predictions writer ────────────────────────────────────────────────────────
def write_predictions(
    out_dir: Path,
    records: List[Dict],
    prefer_parquet: bool = True,
) -> str:
    """Write predictions to parquet or CSV.  Returns the filename used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if _HAS_PANDAS and prefer_parquet:
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            path = out_dir / "predictions.parquet"
            df.to_parquet(path, index=False)
            return "predictions.parquet"
        except Exception as exc:
            logger.warning(f"Parquet write failed ({exc}); falling back to CSV.")

    if _HAS_PANDAS:
        import pandas as pd
        df = pd.DataFrame(records)
        path = out_dir / "predictions.csv"
        df.to_csv(path, index=False)
        return "predictions.csv"

    # Last resort: JSON
    _write_text(
        out_dir / "predictions.json",
        json.dumps(records, indent=2, default=_json_default),
    )
    return "predictions.json"


# ── Summary writer ────────────────────────────────────────────────────────────
def write_summary(
    out_dir: Path,
    *,
    system_key: str,
    dataset: str,
    task: str,
    seed: int,
    metrics: Dict[str, Any],
    notable: str = "",
    deviations: str = "",
    seed_mode: str = "full",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Run Summary",
        f"",
        f"| Field    | Value |",
        f"|----------|-------|",
        f"| system   | `{system_key}` |",
        f"| dataset  | `{dataset}` |",
        f"| task     | `{task}` |",
        f"| seed     | `{seed}` |",
        f"| seed_mode | `{seed_mode}` |",
        f"| generated | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |",
        f"",
        f"## Top-line metrics",
        f"",
    ]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {_fmt(v)}")

    if notable:
        lines += ["", "## Notable observations", "", notable]
    if deviations:
        lines += ["", "## Deviations from harness", "", deviations]

    _write_text(out_dir / "summary.md", "\n".join(lines) + "\n")


def write_seed_run(
    *,
    phase: str,
    dataset: str,
    task: str,
    system_key: str,
    variant: str,
    seed: int,
    seed_mode: str,
    metrics: Dict[str, Any],
    predictions: List[Dict],
    events: List[Dict],
    config_extra: Optional[Dict] = None,
    results_root: Optional[Path] = None,
    stdout_log: str = "",
    notable: str = "",
    deviations: str = "",
) -> Path:
    """Write all six harness files for one seed run.

    Returns the output directory path.
    """
    out = seed_dir(phase, dataset, task, system_key, variant, seed, results_root)
    out.mkdir(parents=True, exist_ok=True)

    write_config(
        out,
        system_key=system_key,
        dataset=dataset,
        task=task,
        phase=phase,
        variant=variant,
        seed=seed,
        seed_mode=seed_mode,
        extra=config_extra,
    )

    write_metrics(out, metrics)
    write_events(out, events)
    write_predictions(out, predictions)
    write_summary(
        out,
        system_key=system_key,
        dataset=dataset,
        task=task,
        seed=seed,
        metrics=metrics,
        seed_mode=seed_mode,
        notable=notable,
        deviations=deviations,
    )

    if stdout_log:
        _write_text(out / "stdout.log", stdout_log)

    logger.info(f"Wrote harness output → {out}")
    return out


# ── Full MIA run writer ───────────────────────────────────────────────────────
def write_mia_run(
    *,
    phase: str,
    dataset: str,
    system_key: str,
    variant: str,
    seed: int,
    seed_mode: str,
    mia_result: Dict[str, Any],
    predictions: List[Dict],
    events: List[Dict],
    config_extra: Optional[Dict] = None,
    results_root: Optional[Path] = None,
    stdout_log: str = "",
) -> Path:
    metrics = {
        "auroc": mia_result.get("auroc", 0.5),
        "advantage": mia_result.get("advantage", 0.0),
        "tpr_at_1pct_fpr": mia_result.get("tpr_at_1pct_fpr", 0.0),
        "tpr_at_5pct_fpr": mia_result.get("tpr_at_5pct_fpr", 0.0),
    }
    return write_seed_run(
        phase=phase,
        dataset=dataset,
        task="membership_inference",
        system_key=system_key,
        variant=variant,
        seed=seed,
        seed_mode=seed_mode,
        metrics=metrics,
        predictions=predictions,
        events=events,
        config_extra=config_extra,
        results_root=results_root,
        stdout_log=stdout_log,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _git_head() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
