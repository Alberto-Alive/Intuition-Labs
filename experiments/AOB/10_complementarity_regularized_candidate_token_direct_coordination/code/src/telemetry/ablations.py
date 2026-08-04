from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.types import AttemptBatch


def ablate_telemetry_channel(batch: AttemptBatch, channel_name: str) -> AttemptBatch:
    if channel_name not in batch.telemetry_channels:
        raise KeyError(f"telemetry channel not present: {channel_name}")
    hidden = batch.hidden_states - batch.telemetry_channels[channel_name]
    return AttemptBatch(
        split=batch.split,
        example_ids=batch.example_ids.copy(),
        task_ids=batch.task_ids.copy(),
        labels=batch.labels.copy(),
        answers=batch.answers.copy(),
        confidences=batch.confidences.copy(),
        hidden_states=hidden.astype(np.float32),
        visible_texts=[row[:] for row in batch.visible_texts],
        telemetry_channels={
            key: (value * 0.0 if key == channel_name else value).astype(np.float32, copy=True)
            for key, value in batch.telemetry_channels.items()
        },
    )


def ablate_all_available_channels(batches: Dict[str, AttemptBatch]) -> Dict[str, Dict[str, AttemptBatch]]:
    channel_names = sorted(set.intersection(*(set(batch.telemetry_channels) for batch in batches.values())))
    return {
        channel_name: {
            split: ablate_telemetry_channel(batch, channel_name)
            for split, batch in batches.items()
        }
        for channel_name in channel_names
    }

