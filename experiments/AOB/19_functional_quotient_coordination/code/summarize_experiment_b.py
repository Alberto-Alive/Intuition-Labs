"""Aggregate Experiment B JSONL runs into seed-mean tables.

Reads every results/experiment_b_runs_*.jsonl, dedupes on
(tag, mode, num_clones, seed, duplicated_evidence) keeping the last row, and
writes results/experiment_b_summary.csv plus a markdown table file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.glob("experiment_b_runs_*.jsonl")):
        tag = path.stem.replace("experiment_b_runs_", "")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            e = r["fqc_extras"]
            rows.append(
                {
                    "tag": tag,
                    "mode": r["mode"],
                    "num_clones": r["num_clones"],
                    "seed": r["seed"],
                    "duplicated_evidence": r.get("duplicated_evidence", False),
                    "test_f1": r["test_f1"],
                    "test_em": r["test_em"],
                    "joint_beats_single": bool(r["joint_beats_single"]),
                    "best_single_clone_f1": r["best_single_clone_f1"],
                    "clone_cosine": r["clone_cosine"],
                    "unique_content_fraction": e["mean_unique_content_fraction"],
                    "content_duplicate_fraction": e["mean_content_duplicate_fraction"],
                    "answer_window_hit_rate": e["answer_window_hit_rate"],
                    "support_coverage": e["mean_support_coverage"],
                }
            )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["tag", "mode", "num_clones", "seed", "duplicated_evidence"], keep="last"
    )
    return df


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    df = load_rows()
    # The screen (seed 0) and medium (seeds 1,2) tags share identical settings.
    df.loc[df.tag.isin(["screen", "medium"]), "tag"] = "standard"
    df.to_csv(RESULTS / "experiment_b_task_level.csv", index=False)

    agg = (
        df.groupby(["tag", "duplicated_evidence", "num_clones", "mode"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_f1=("test_f1", "mean"),
            std_f1=("test_f1", "std"),
            mean_em=("test_em", "mean"),
            joint_beats_single_rate=("joint_beats_single", "mean"),
            mean_unique_content=("unique_content_fraction", "mean"),
            mean_content_dup=("content_duplicate_fraction", "mean"),
            mean_answer_hit=("answer_window_hit_rate", "mean"),
            mean_support_cov=("support_coverage", "mean"),
        )
        .fillna({"std_f1": 0.0})
        .sort_values(["tag", "duplicated_evidence", "num_clones", "mean_f1"], ascending=[True, True, True, False])
    )
    agg.to_csv(RESULTS / "experiment_b_summary.csv", index=False)

    parts = []
    for (tag, dup), group in agg.groupby(["tag", "duplicated_evidence"]):
        parts.append(f"## {tag}{' (duplicated evidence)' if dup else ''}\n")
        parts.append(md_table(group.drop(columns=["tag", "duplicated_evidence"])))
        parts.append("")
    (RESULTS / "experiment_b_tables.md").write_text("\n".join(parts), encoding="utf-8")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
