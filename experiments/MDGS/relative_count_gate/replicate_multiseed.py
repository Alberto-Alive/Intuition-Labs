"""Multi-seed replication of the relative/count gate's detection win.

Monotonicity and threshold stability are structural guarantees (proven by
construction in test_gate.py), so they do not need replication. What could be
corpus-specific is the empirical *detection* win (AUROC for catching wrong
answers). This script tests it across the five independently trained e2 models
in ``results/eval_suite/e2/seed_*`` with a leave-one-seed-out protocol:

- fit the monotone (non-negative) relative/count risk on four seeds,
- evaluate detection AUROC on the held-out seed,
- compare against the held-out seed's own e2 outcome-head failure signal
  (1 - success prob) and a raw single-margin control.

The per-seed prediction dumps do not store the full 15-dim trace_inputs, so the
gate here uses the four raw degradation scalars they do store
(agreement, prob_margin, max_attention_mass, mean_entropy). It is still a
non-negative weighted sum of degradation-oriented signals, so it remains monotone
by the same argument as the full gate.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate_gate import _auroc  # noqa: E402
from gate import _fit_nonnegative_logistic  # noqa: E402

EVAL_SUITE = HERE.parent / "results" / "eval_suite" / "e2"
OUT_PATH = HERE / "results" / "relative_count_gate_multiseed.json"


def _degradation_scalars(row: Dict[str, Any]) -> List[float]:
    """Degradation-oriented signals available in every per-seed dump (higher = worse)."""
    return [
        1.0 - float(row.get("agreement", 0.0)),          # disagreement
        1.0 - float(row.get("prob_margin", 0.0)),        # low margin
        1.0 - float(row.get("max_attention_mass", 0.0)), # low attention concentration
        float(row.get("mean_entropy", 0.0)),             # higher entropy
    ]


def _e2_failure_score(row: Dict[str, Any]) -> float:
    """e2's own deployed failure signal for this row: 1 - P(SUCCESS_LIKELY)."""
    probs = row.get("outcome_probs")
    if isinstance(probs, list) and probs:
        return 1.0 - float(probs[0])
    # Fall back to the predicted-failure indicator if probs are absent.
    return 1.0 if int(row.get("outcome_pred", 0)) == 2 else 0.0


def _load_seed(path: str) -> Dict[str, torch.Tensor]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    feats = torch.tensor([_degradation_scalars(r) for r in rows], dtype=torch.float64)
    incorrect = torch.tensor([0 if bool(r["is_correct"]) else 1 for r in rows], dtype=torch.long)
    e2_score = torch.tensor([_e2_failure_score(r) for r in rows], dtype=torch.float64)
    margin_score = torch.tensor([1.0 - float(r.get("prob_margin", 0.0)) for r in rows], dtype=torch.float64)
    return {"feats": feats, "incorrect": incorrect, "e2_score": e2_score, "margin_score": margin_score}


def main() -> None:
    paths = sorted(glob.glob(str(EVAL_SUITE / "seed_*" / "predictions.jsonl")))
    if not paths:
        raise SystemExit(f"no per-seed predictions under {EVAL_SUITE}")
    seeds = {Path(p).parent.name: _load_seed(p) for p in paths}
    seed_names = sorted(seeds)

    folds: List[Dict[str, Any]] = []
    for heldout in seed_names:
        train_feats = torch.cat([seeds[s]["feats"] for s in seed_names if s != heldout], dim=0)
        train_incorrect = torch.cat([seeds[s]["incorrect"] for s in seed_names if s != heldout], dim=0)
        mean = train_feats.mean(dim=0)
        std = train_feats.std(dim=0).clamp(min=1e-6)
        weights, _bias = _fit_nonnegative_logistic((train_feats - mean) / std, train_incorrect.to(dtype=torch.float64))

        test = seeds[heldout]
        gate_risk = ((test["feats"] - mean) / std) @ weights
        fold = {
            "heldout_seed": heldout,
            "n": int(len(test["incorrect"])),
            "gate_auroc": _auroc(gate_risk, test["incorrect"]),
            "e2_outcome_head_auroc": _auroc(test["e2_score"], test["incorrect"]),
            "raw_margin_auroc": _auroc(test["margin_score"], test["incorrect"]),
            "weights": [float(w) for w in weights],
            "weights_nonnegative": bool(torch.all(weights >= 0.0)),
        }
        folds.append(fold)

    def mean(key: str) -> float:
        return float(sum(f[key] for f in folds) / len(folds))

    gate_mean = mean("gate_auroc")
    e2_mean = mean("e2_outcome_head_auroc")
    raw_mean = mean("raw_margin_auroc")
    beats_e2_each = sum(1 for f in folds if f["gate_auroc"] >= f["e2_outcome_head_auroc"])
    beats_raw_each = sum(1 for f in folds if f["gate_auroc"] > f["raw_margin_auroc"])

    # Headline replication claim: across 5 independently trained models, the
    # monotone gate's detection is at least as good as e2's own deployed outcome
    # head. The raw-margin comparison is reported as context, not gated: with only
    # the 4 scalars in the per-seed dumps the gate ties a single margin signal
    # (the multi-evidence lift to 0.821 needs support_ratio/abs_margin, which live
    # only in the full trace cache).
    raw_margin_verdict = (
        "tied" if abs(gate_mean - raw_mean) < 0.01 else ("better" if gate_mean > raw_mean else "worse")
    )
    acceptance = {
        "gate_mean_auroc_at_least_e2": gate_mean >= e2_mean,
        "gate_beats_e2_in_majority_of_seeds": beats_e2_each >= (len(folds) + 1) // 2,
        "all_weights_nonnegative": all(f["weights_nonnegative"] for f in folds),
    }
    acceptance["all_pass"] = all(acceptance.values())

    payload = {
        "scope": "MDGS relative/count gate detection, leave-one-seed across 5 trained e2 models",
        "evidence_level": "multiseed_leave_one_out_detection",
        "note": "monotonicity/threshold-stability are structural (see test_gate.py); this replicates only the empirical detection win",
        "feature_names": ["disagreement", "low_margin", "low_attention_concentration", "mean_entropy"],
        "seeds": seed_names,
        "folds": folds,
        "gate_mean_auroc": gate_mean,
        "e2_outcome_head_mean_auroc": e2_mean,
        "raw_margin_mean_auroc": raw_mean,
        "gate_beats_e2_seed_count": beats_e2_each,
        "gate_beats_raw_seed_count": beats_raw_each,
        "raw_margin_verdict": raw_margin_verdict,
        "acceptance": acceptance,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "gate_mean_auroc": gate_mean,
            "e2_outcome_head_mean_auroc": e2_mean,
            "raw_margin_mean_auroc": raw_mean,
            "raw_margin_verdict": raw_margin_verdict,
            "gate_beats_e2_seed_count": f"{beats_e2_each}/{len(folds)}",
            "acceptance": acceptance,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
