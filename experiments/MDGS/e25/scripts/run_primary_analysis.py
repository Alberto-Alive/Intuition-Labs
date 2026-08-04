"""DIGIT Extrapolation E25 — Primary ROC-AUC Analysis.

This script performs the full E25 experiment suite:
1. Training and LTH procedure for 5 seeds (Deliverable 1).
2. Primary Analysis (Deliverable 2):
    - Compute ROC-AUC for Familiarity Exposure (E_jk) vs. survival.
    - Compute ROC-AUC for Weight Magnitude (|W_jk|) vs. survival.
3. Comparative reporting and JSON export.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.DIGIT.Extrapolation.e25.extrapolation.experiment import run_deliverable1


def _bar(length: int = 80) -> str:
    return "=" * length


def _section(title: str) -> None:
    print()
    print(_bar())
    print(f"  {title}")
    print(_bar())


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC using sklearn. Masks are flattened to vectors."""
    y_true = labels.flatten()
    y_score = scores.flatten()
    
    # If all labels are the same (no variance), ROC-AUC is undefined.
    if len(np.unique(y_true)) < 2:
        return 0.5
        
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    device = "cuda" if len(sys.argv) <= 1 or sys.argv[1] != "cpu" else "cpu"
    
    print(_bar())
    print("  DIGIT Extrapolation E25 — Primary Analysis (Deliverable 2)")
    print("  Comparing Familiarity Exposure vs. Weight Magnitude as Pruning Predictors")
    print(_bar())
    print(f"  Device: {device}")
    print()

    # Step 1: Run Deliverable 1
    # This executes training, pruning, rewinding, and retraining for all 5 seeds.
    all_artifacts = run_deliverable1(seeds=(0, 1, 2, 3, 4), device=device)

    # Step 2: Extract Checkpoints and Layers
    checkpoints = ["epoch_5", "epoch_10", "epoch_20", "final"]
    layers = ["layer1", "layer2"]
    
    # Structure for storing all results
    # results[seed][layer][checkpoint] = {"familiarity_auc": X, "magnitude_auc": Y}
    results_by_seed = []

    for art in all_artifacts:
        seed_data = {"seed": art.seed, "layers": {}}
        
        # Binary survival mask (mask_jk) — this is our Ground Truth for ROC-AUC
        mask1 = art.pruning_mask.mask1  # (32, 32)
        mask2 = art.pruning_mask.mask2  # (32, 32)
        
        for layer in layers:
            layer_data = {}
            target_mask = mask1 if layer == "layer1" else mask2
            
            # Extract scores and weights at each checkpoint
            exposure_dict = art.exposure_layer1 if layer == "layer1" else art.exposure_layer2
            weight_dict = art.weights_layer1 if layer == "layer1" else art.weights_layer2
            
            for cp in checkpoints:
                # FAMILIARITY PREDICTOR: E_jk
                e_jk = exposure_dict.get(cp)
                f_auc = compute_roc_auc(e_jk, target_mask) if e_jk is not None else 0.5
                
                # MAGNITUDE PREDICTOR: |W_jk|
                w_jk = weight_dict.get(cp)
                m_auc = compute_roc_auc(np.abs(w_jk), target_mask) if w_jk is not None else 0.5
                
                layer_data[cp] = {
                    "familiarity_auc": f_auc,
                    "magnitude_auc": m_auc,
                    "difference": f_auc - m_auc
                }
            
            seed_data["layers"][layer] = layer_data
            
        results_by_seed.append(seed_data)

    # -----------------------------------------------------------------------
    # Step 3: Aggregation and Reporting
    # -----------------------------------------------------------------------
    _section("ROC-AUC Comparison: Familiarity vs. Magnitude")
    
    header = f"{'Seed':<5} {'Layer':<8} {'Time':<10} {'Fam-AUC':<10} {'Mag-AUC':<10} {'Diff':<8}"
    print(header)
    print("-" * len(header))

    # To store stats for final aggregation
    # stats[layer][checkpoint] = {"fam": [], "mag": []}
    aggregate_stats = {l: {cp: {"fam": [], "mag": []} for cp in checkpoints} for l in layers}

    for seed_res in results_by_seed:
        s_id = seed_res["seed"]
        for layer in layers:
            for cp in checkpoints:
                f_auc = seed_res["layers"][layer][cp]["familiarity_auc"]
                m_auc = seed_res["layers"][layer][cp]["magnitude_auc"]
                diff = seed_res["layers"][layer][cp]["difference"]
                
                print(f"{s_id:<5} {layer:<8} {cp:<10} {f_auc:<10.4f} {m_auc:<10.4f} {diff:<+8.4f}")
                
                aggregate_stats[layer][cp]["fam"].append(f_auc)
                aggregate_stats[layer][cp]["mag"].append(m_auc)
            print("-" * len(header))

    _section("Aggregated Results (Mean ± SD)")
    
    agg_header = f"{'Layer':<8} {'Time':<10} {'Fam AUC (μ±σ)':<20} {'Mag AUC (μ±σ)':<20} {'Better?'}"
    print(agg_header)
    print("-" * len(agg_header))

    for layer in layers:
        for cp in checkpoints:
            fam_vals = np.array(aggregate_stats[layer][cp]["fam"])
            mag_vals = np.array(aggregate_stats[layer][cp]["mag"])
            
            f_mean, f_std = fam_vals.mean(), fam_vals.std()
            m_mean, m_std = mag_vals.mean(), mag_vals.std()
            
            f_str = f"{f_mean:.3f} ± {f_std:.3f}"
            m_str = f"{m_mean:.3f} ± {m_std:.3f}"
            
            better = "FAMILIARITY" if f_mean > m_mean else "MAGNITUDE"
            
            print(f"{layer:<8} {cp:<10} {f_str:<20} {m_str:<20} {better}")
        print("-" * len(agg_header))

    # -----------------------------------------------------------------------
    # Step 4: JSON Export
    # -----------------------------------------------------------------------
    output_dir = Path("experiments/DIGIT/Extrapolation/e25/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e25_analysis.json"
    
    export_payload = {
        "metadata": {
            "experiment": "E25",
            "device": device,
            "sparsity": 0.5,
            "seeds": [0, 1, 2, 3, 4]
        },
        "per_seed": results_by_seed,
        "aggregated": {
            layer: {
                cp: {
                    "fam_mean": float(np.mean(aggregate_stats[layer][cp]["fam"])),
                    "fam_std": float(np.std(aggregate_stats[layer][cp]["fam"])),
                    "mag_mean": float(np.mean(aggregate_stats[layer][cp]["mag"])),
                    "mag_std": float(np.std(aggregate_stats[layer][cp]["mag"]))
                } for cp in checkpoints
            } for layer in layers
        }
    }
    
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    
    print(f"\n  Detailed results saved to: {json_path}")

    # -----------------------------------------------------------------------
    # Final Conclusion
    # -----------------------------------------------------------------------
    _section("CONCLUSION")
    
    # Claim 1: High correlation at final epoch (AUC > 0.6)
    final_fam_aucs = [np.mean(aggregate_stats[l]["final"]["fam"]) for l in layers]
    claim_1_pass = all(auc > 0.6 for auc in final_fam_aucs)
    
    # Claim 2: Early prediction (AUC at epoch 5/10 approaches final AUC)
    # We check if epoch 20 AUC is within 10% of final AUC
    e20_fam_aucs = [np.mean(aggregate_stats[l]["epoch_20"]["fam"]) for l in layers]
    claim_2_pass = all(e20 / final >= 0.9 for e20, final in zip(e20_fam_aucs, final_fam_aucs))

    if claim_1_pass and claim_2_pass:
        status = "PASS"
        evidence = f"Familiarity density correlates strongly with pruning survival (AUC={np.mean(final_fam_aucs):.3f}) and early familiarity predicts final survival before training completes."
    elif not claim_1_pass:
        status = "FAIL"
        evidence = f"Familiarity density does not show meaningful correlation with pruning survival (Final AUC={np.mean(final_fam_aucs):.3f} which is <= 0.6)."
    else:
        status = "INDETERMINATE"
        evidence = f"Correlation found at final epoch, but early prediction signal is weak or inconsistent."

    print(f"  {status}")
    print(f"  {evidence}")
    print(_bar())


if __name__ == "__main__":
    main()
