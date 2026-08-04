"""Run DIGIT Extrapolation E25 Deliverable 1.

Trains the E22 dual-parameter architecture across 5 seeds, logs familiarity
exposure scores at epoch 5/10/20/final, runs LTH pruning and retraining,
and reports per-seed results plus the winning ticket count.
"""

from __future__ import annotations

import sys

from experiments.DIGIT.Extrapolation.e25.extrapolation.experiment import run_deliverable1


def _bar(length: int = 70) -> str:
    return "=" * length


def _section(title: str) -> None:
    print()
    print(_bar())
    print(f"  {title}")
    print(_bar())


def main() -> None:
    device = "cpu"
    if len(sys.argv) > 1 and sys.argv[1] in ("cuda", "cpu"):
        device = sys.argv[1]

    print(_bar())
    print("  DIGIT Extrapolation E25 — Deliverable 1")
    print("  Familiarity Density & Lottery Ticket Hypothesis")
    print(_bar())
    print(f"  Device: {device}")
    print(f"  Seeds:  0, 1, 2, 3, 4")
    print()

    all_artifacts = run_deliverable1(seeds=(0, 1, 2, 3, 4), device=device)

    # -----------------------------------------------------------------------
    # Per-seed results table
    # -----------------------------------------------------------------------
    _section("Per-Seed Results")

    header = (
        f"{'Seed':>4}  "
        f"{'Dense Acc':>10}  "
        f"{'Dense Ep':>8}  "
        f"{'Pruned Acc':>10}  "
        f"{'Pruned Ep':>9}  "
        f"{'Ratio':>6}  "
        f"{'Sparsity':>8}  "
        f"{'WinTicket':>9}"
    )
    print(header)
    print("-" * len(header))

    winning_ticket_count = 0
    for art in all_artifacts:
        lth = art.lth
        mark = "YES" if lth.winning_ticket else "NO "
        if lth.winning_ticket:
            winning_ticket_count += 1
        print(
            f"{art.seed:>4}  "
            f"{lth.dense_withheld_accuracy:>10.4f}  "
            f"{lth.dense_stopping_epoch:>8d}  "
            f"{lth.pruned_withheld_accuracy:>10.4f}  "
            f"{lth.pruned_stopping_epoch:>9d}  "
            f"{lth.accuracy_ratio:>6.4f}  "
            f"{lth.global_sparsity:>8.4f}  "
            f"{mark:>9}"
        )

    # -----------------------------------------------------------------------
    # Exposure checkpoint availability
    # -----------------------------------------------------------------------
    _section("Familiarity Exposure Score Checkpoint Availability")
    for art in all_artifacts:
        checkpoints = sorted(art.exposure_layer1.keys())
        print(f"  Seed {art.seed}: checkpoints = {checkpoints}")

    # -----------------------------------------------------------------------
    # Pruning mask statistics
    # -----------------------------------------------------------------------
    _section("Pruning Mask Statistics")
    pm_header = (
        f"{'Seed':>4}  "
        f"{'Threshold':>10}  "
        f"{'L1 Sparsity':>12}  "
        f"{'L2 Sparsity':>12}  "
        f"{'Global':>8}"
    )
    print(pm_header)
    print("-" * len(pm_header))
    for art in all_artifacts:
        pm = art.pruning_mask
        print(
            f"{art.seed:>4}  "
            f"{pm.threshold:>10.6f}  "
            f"{pm.layer1_sparsity:>12.4f}  "
            f"{pm.layer2_sparsity:>12.4f}  "
            f"{pm.global_sparsity:>8.4f}"
        )

    # -----------------------------------------------------------------------
    # Winning ticket summary and INDETERMINATE gate
    # -----------------------------------------------------------------------
    _section("Winning Ticket Summary")
    print(f"  Winning tickets found: {winning_ticket_count} / 5")
    print(f"  Threshold: pruned accuracy >= 99% of dense accuracy")
    print()

    if winning_ticket_count == 0:
        print("  INDETERMINATE")
        print("  Zero seeds found a valid winning ticket at 50% global sparsity.")
        print("  The primary ROC-AUC analysis is not interpretable.")
        print("  Stopping before Deliverable 2 as specified.")
        print()
        print("  Awaiting your decision before proceeding.")
    else:
        print(f"  {winning_ticket_count} seed(s) produced valid winning tickets.")
        print("  Proceeding to Deliverable 2 requires your confirmation.")

    print()
    print(_bar())
    print("  Deliverable 1 complete. Waiting for confirmation.")
    print(_bar())


if __name__ == "__main__":
    main()
