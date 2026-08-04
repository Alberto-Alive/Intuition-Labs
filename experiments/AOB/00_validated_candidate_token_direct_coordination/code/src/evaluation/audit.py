from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List

from src.agents.types import AttemptBatch


def split_fingerprint(batch: AttemptBatch, include_labels: bool = False) -> str:
    payload = {
        "split": batch.split,
        "example_ids": [str(value) for value in batch.example_ids.tolist()],
        "task_ids": [int(value) for value in batch.task_ids.tolist()],
    }
    if include_labels:
        payload["labels"] = [int(value) for value in batch.labels.tolist()]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def split_overlap_audit(benchmark: str, seed: int, batches: Dict[str, AttemptBatch]) -> Dict[str, object]:
    ids = {split: set(str(value) for value in batch.example_ids.tolist()) for split, batch in batches.items()}
    return {
        "benchmark": benchmark,
        "seed": seed,
        "type": "split_overlap",
        "train_dev_overlap": len(ids["train"] & ids["dev"]),
        "train_test_overlap": len(ids["train"] & ids["test"]),
        "dev_test_overlap": len(ids["dev"] & ids["test"]),
        "train_fingerprint": split_fingerprint(batches["train"]),
        "dev_fingerprint": split_fingerprint(batches["dev"]),
        "test_fingerprint": split_fingerprint(batches["test"]),
    }


def audit_event(
    benchmark: str,
    seed: int,
    event_index: int,
    event: str,
    method: str | None = None,
    condition: str | None = None,
    split_inputs: Iterable[str] = (),
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "event_index": event_index,
        "event": event,
        "method": method,
        "condition": condition,
        "split_inputs": list(split_inputs),
        "test_split_passed": "test" in set(split_inputs),
    }


def summarize_test_access(audit_events: List[Dict[str, object]]) -> Dict[str, object]:
    violations = []
    gates = {}
    for row in audit_events:
        if "event" not in row:
            continue
        key = (str(row["benchmark"]), int(row["seed"]))
        if row["event"] == "test_gate_opened":
            gates[key] = int(row["event_index"])
    for row in audit_events:
        if "event" not in row:
            continue
        if not row.get("test_split_passed"):
            continue
        key = (str(row["benchmark"]), int(row["seed"]))
        gate = gates.get(key)
        if gate is None or int(row["event_index"]) < gate:
            violations.append(row)
    return {
        "type": "test_access_summary",
        "violations": len(violations),
        "violation_events": violations[:10],
    }
