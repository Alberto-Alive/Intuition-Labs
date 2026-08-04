from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

from src.evaluation.report import write_report


TARGET_METHODS = {
    "activation_pca_mlp",
    "hidden_state_only_probe",
    "telemetry_only_label_probe",
    "output_only_oracle_probe",
    "text_only_coordinator",
    "capacity_matched_text_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare small CPU and CUDA validation runs.")
    parser.add_argument("--cpu", default="results/validation_parity_cpu.json")
    parser.add_argument("--cuda", default="results/validation_parity_cuda.json")
    parser.add_argument("--main", default="results/synthetic_cuda_results.json")
    parser.add_argument("--out", default="results/cpu_cuda_parity.json")
    parser.add_argument("--cpu-hidden", default=None)
    parser.add_argument("--cuda-hidden", default=None)
    args = parser.parse_args()

    cpu = json.loads(Path(args.cpu).read_text(encoding="utf-8"))
    cuda = json.loads(Path(args.cuda).read_text(encoding="utf-8"))
    rows = _compare(cpu, cuda)
    hidden_rows = _compare_hidden_states(
        _hidden_path(args.cpu_hidden, cpu),
        _hidden_path(args.cuda_hidden, cuda),
    )
    artifact = {
        "cpu_path": args.cpu,
        "cuda_path": args.cuda,
        "rows": rows,
        "hidden_state_rows": hidden_rows,
        "diagnosis": _diagnose(rows, hidden_rows),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    main_results = json.loads(Path(args.main).read_text(encoding="utf-8"))
    main_results.setdefault("validation", {})
    main_results["validation"]["cpu_cuda_parity"] = rows
    main_results["validation"]["cpu_cuda_hidden_state_parity"] = hidden_rows
    main_results["validation"]["cpu_cuda_parity_diagnosis"] = artifact["diagnosis"]
    Path(args.main).write_text(json.dumps(main_results, indent=2, sort_keys=True), encoding="utf-8")
    write_report(main_results, main_results["metadata"]["config"]["report_path"])


def _compare(cpu: Dict[str, object], cuda: Dict[str, object]) -> List[Dict[str, object]]:
    cpu_values = _test_means(cpu)
    cuda_values = _test_means(cuda)
    rows = []
    for key in sorted(set(cpu_values) & set(cuda_values)):
        benchmark, method = key
        cpu_acc = cpu_values[key]
        cuda_acc = cuda_values[key]
        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "cpu_accuracy": cpu_acc,
                "cuda_accuracy": cuda_acc,
                "abs_delta": abs(cpu_acc - cuda_acc),
            }
        )
    return rows


def _test_means(results: Dict[str, object]) -> Dict[Tuple[str, str], float]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in results["metrics"]:
        if row["split"] != "test" or row["condition"] != "none" or row["method"] not in TARGET_METHODS:
            continue
        key = (str(row["benchmark"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["accuracy"]))
    return {key: mean(values) for key, values in grouped.items()}


def _hidden_path(arg_path: str | None, results: Dict[str, object]) -> str | None:
    if arg_path:
        return arg_path
    metadata = results.get("metadata", {})
    if isinstance(metadata, dict):
        config = metadata.get("config", {})
        if isinstance(config, dict) and config.get("hidden_state_npz_path"):
            return str(config["hidden_state_npz_path"])
    return None


def _compare_hidden_states(cpu_path: str | None, cuda_path: str | None) -> List[Dict[str, object]]:
    if not cpu_path or not cuda_path:
        return []
    cpu_file = Path(cpu_path)
    cuda_file = Path(cuda_path)
    if not cpu_file.exists() or not cuda_file.exists():
        return []
    cpu = np.load(cpu_file, allow_pickle=True)
    cuda = np.load(cuda_file, allow_pickle=True)
    rows = []
    for key in sorted(set(cpu.files) & set(cuda.files)):
        if not key.endswith("_hidden_states"):
            continue
        lhs = cpu[key]
        rhs = cuda[key]
        if lhs.shape != rhs.shape:
            rows.append(
                {
                    "array": key,
                    "shape_match": False,
                    "cpu_shape": list(lhs.shape),
                    "cuda_shape": list(rhs.shape),
                }
            )
            continue
        diff = lhs.astype(np.float64) - rhs.astype(np.float64)
        rows.append(
            {
                "array": key,
                "shape_match": True,
                "shape": list(lhs.shape),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "cpu_checksum": hashlib.sha256(lhs.tobytes()).hexdigest(),
                "cuda_checksum": hashlib.sha256(rhs.tobytes()).hexdigest(),
                "exact_checksum_match": bool(hashlib.sha256(lhs.tobytes()).digest() == hashlib.sha256(rhs.tobytes()).digest()),
                "allclose_1e_4": bool(np.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)),
                "allclose_1e_3": bool(np.allclose(lhs, rhs, rtol=1e-3, atol=1e-3)),
            }
        )
    return rows


def _diagnose(rows: List[Dict[str, object]], hidden_rows: List[Dict[str, object]]) -> Dict[str, object]:
    hidden_delta = max((float(row.get("max_abs_diff", 0.0)) for row in hidden_rows), default=0.0)
    hidden_only_delta = max(
        (
            float(row["abs_delta"])
            for row in rows
            if row.get("method") in {"hidden_state_only_probe", "telemetry_only_label_probe"}
        ),
        default=0.0,
    )
    largest_row = max(rows, key=lambda row: float(row.get("abs_delta", 0.0)), default={})
    largest_accuracy_delta = float(largest_row.get("abs_delta", 0.0)) if largest_row else 0.0
    largest_delta_method = str(largest_row.get("method", "")) if largest_row else ""
    n_test = None
    for row in hidden_rows:
        shape = row.get("shape")
        if isinstance(shape, list) and len(shape) >= 1 and "test" in str(row.get("array", "")):
            n_test = int(shape[0])
            break
    if hidden_delta > 1e-3 and hidden_only_delta > 0:
        likely_source = "pre_fit_hidden_state_extraction_drift_plus_probe_sample_sensitivity"
    elif hidden_delta > 1e-3 and hidden_only_delta == 0:
        likely_source = "prior_hidden_only_delta_resolved_on_larger_sample_despite_pre_fit_hidden_drift"
    elif hidden_only_delta > 0:
        likely_source = "probe_training_or_small_sample_instability"
    else:
        likely_source = "no_accuracy_delta_observed"
    return {
        "hidden_state_max_abs_diff": hidden_delta,
        "hidden_state_only_abs_delta": hidden_only_delta,
        "largest_accuracy_delta": largest_accuracy_delta,
        "largest_delta_method": largest_delta_method,
        "test_examples_per_seed": n_test,
        "probe_initialization_seed_policy": "shared deterministic seed formulas in coordinator metadata/config",
        "likely_source": likely_source,
    }


if __name__ == "__main__":
    main()
