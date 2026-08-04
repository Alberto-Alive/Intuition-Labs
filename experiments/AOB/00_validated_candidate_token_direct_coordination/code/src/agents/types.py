from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AttemptBatch:
    """Visible attempts plus optional internal telemetry for one split."""

    split: str
    example_ids: np.ndarray
    task_ids: np.ndarray
    labels: np.ndarray
    answers: np.ndarray
    confidences: np.ndarray
    hidden_states: np.ndarray
    visible_texts: List[List[str]]
    telemetry_channels: Dict[str, np.ndarray] = field(default_factory=dict)
    private_agent_views: List[List[str]] = field(default_factory=list)

    @property
    def n_examples(self) -> int:
        return int(self.answers.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.answers.shape[1])

    @property
    def num_layers(self) -> int:
        return int(self.hidden_states.shape[2])

    @property
    def hidden_dim(self) -> int:
        return int(self.hidden_states.shape[3])

    def copy_with_hidden(self, hidden_states: np.ndarray) -> "AttemptBatch":
        return AttemptBatch(
            split=self.split,
            example_ids=self.example_ids.copy(),
            task_ids=self.task_ids.copy(),
            labels=self.labels.copy(),
            answers=self.answers.copy(),
            confidences=self.confidences.copy(),
            hidden_states=hidden_states.astype(np.float32, copy=True),
            visible_texts=[row[:] for row in self.visible_texts],
            telemetry_channels={
                key: value.astype(np.float32, copy=True)
                for key, value in self.telemetry_channels.items()
            },
            private_agent_views=[row[:] for row in self.private_agent_views],
        )

    def copy_with_labels(self, labels: np.ndarray) -> "AttemptBatch":
        return AttemptBatch(
            split=self.split,
            example_ids=self.example_ids.copy(),
            task_ids=self.task_ids.copy(),
            labels=labels.astype(np.int64, copy=True),
            answers=self.answers.copy(),
            confidences=self.confidences.copy(),
            hidden_states=self.hidden_states.copy(),
            visible_texts=[row[:] for row in self.visible_texts],
            telemetry_channels={
                key: value.astype(np.float32, copy=True)
                for key, value in self.telemetry_channels.items()
            },
            private_agent_views=[row[:] for row in self.private_agent_views],
        )
