from __future__ import annotations

from typing import Dict

import numpy as np

from src.agents.types import AttemptBatch


class SingleAgentCoordinator:
    name = "single_agent"
    param_count = 0

    def __init__(self, agent_index: int = 0) -> None:
        self.agent_index = agent_index

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        return None

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        return batch.answers[:, self.agent_index].astype(np.int64)

    def metadata(self) -> Dict[str, int | str]:
        return {"agent_index": self.agent_index}


class IndependentSwarmCoordinator:
    """Best-of-N without labels: choose candidate with highest visible confidence."""

    name = "independent_swarm"
    param_count = 0

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        return None

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        best_agent = np.argmax(batch.confidences, axis=1)
        return batch.answers[np.arange(batch.n_examples), best_agent].astype(np.int64)

    def metadata(self) -> Dict[str, str]:
        return {"selection": "highest_visible_confidence"}

