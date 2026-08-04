from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.agents.types import AttemptBatch
from src.datasets.strict_coordination import StrictCoordinationExample


@dataclass(frozen=True)
class StrictCoordinationAgentConfig:
    n_agents: int = 5
    hidden_dim: int = 32
    num_layers: int = 4
    evidence_signal_strength: float = 1.35
    uncertainty_signal_strength: float = 0.45
    reasoning_mode_signal_strength: float = 0.35
    activation_noise: float = 0.65
    confidence_noise: float = 0.20


class StrictCoordinationSwarm:
    """Agents with private evidence bits and no single-agent solution path."""

    def __init__(
        self,
        config: StrictCoordinationAgentConfig,
        n_evidence_bits: int,
        num_classes: int,
        seed: int,
    ) -> None:
        self.config = config
        self.n_evidence_bits = n_evidence_bits
        self.num_classes = num_classes
        self.seed = seed
        rng = np.random.default_rng(seed + 71_117)
        self.agent_embeddings = rng.normal(0.0, 0.25, size=(config.n_agents, config.hidden_dim))
        self.evidence_directions = np.asarray(
            [_unit(rng.normal(size=config.hidden_dim)) for _ in range(n_evidence_bits)],
            dtype=np.float32,
        )
        self.uncertainty_direction = _unit(rng.normal(size=config.hidden_dim))
        self.reasoning_embeddings = rng.normal(0.0, 0.25, size=(4, config.hidden_dim))
        self.layer_scales = np.linspace(0.35, 1.0, config.num_layers)

    def run(self, splits: Dict[str, List[StrictCoordinationExample]]) -> Dict[str, AttemptBatch]:
        return {split: self._run_split(split, examples) for split, examples in splits.items()}

    def _run_split(self, split: str, examples: List[StrictCoordinationExample]) -> AttemptBatch:
        cfg = self.config
        n = len(examples)
        answers = np.zeros((n, cfg.n_agents), dtype=np.int64)
        confidences = np.zeros((n, cfg.n_agents), dtype=np.float32)
        hidden = np.zeros((n, cfg.n_agents, cfg.num_layers, cfg.hidden_dim), dtype=np.float32)
        channels = {
            name: np.zeros_like(hidden)
            for name in [
                "agent_identity",
                "reasoning_mode",
                "private_evidence",
                "uncertainty",
                "noise",
            ]
        }
        visible_texts: List[List[str]] = []

        for i, example in enumerate(examples):
            row_texts: List[str] = []
            for agent_id in range(cfg.n_agents):
                rng = np.random.default_rng(
                    self.seed
                    + 43_721 * (agent_id + 1)
                    + 11_003 * (example.global_index + 1)
                )
                evidence_id = agent_id % self.n_evidence_bits
                bit = int(example.evidence_bits[evidence_id])
                # Visible answers are deliberately not a coded channel for the
                # private evidence bit. The coordination signal lives in hidden state.
                answer = int(rng.integers(0, self.num_classes))
                confidence = float(np.clip(0.35 + rng.normal(0.0, cfg.confidence_noise), 0.01, 0.99))
                answers[i, agent_id] = answer
                confidences[i, agent_id] = confidence
                hidden_state, hidden_channels = self._hidden_state(
                    rng=rng,
                    agent_id=agent_id,
                    evidence_id=evidence_id,
                    bit=bit,
                    global_index=example.global_index,
                )
                hidden[i, agent_id] = hidden_state
                for channel_name, channel_value in hidden_channels.items():
                    channels[channel_name][i, agent_id] = channel_value
                row_texts.append(
                    " ".join(
                        [
                            f"agent={agent_id}",
                            "task=0",
                            f"answer={answer}",
                            f"confidence={confidence:.3f}",
                            f"rationale_code={(agent_id + example.global_index) % 7}",
                        ]
                    )
                )
            visible_texts.append(row_texts)

        return AttemptBatch(
            split=split,
            example_ids=np.asarray([example.example_id for example in examples], dtype=object),
            task_ids=np.asarray([example.task_id for example in examples], dtype=np.int64),
            labels=np.asarray([example.label for example in examples], dtype=np.int64),
            answers=answers,
            confidences=confidences,
            hidden_states=hidden,
            visible_texts=visible_texts,
            telemetry_channels=channels,
            private_agent_views=[row[:] for row in visible_texts],
        )

    def _hidden_state(
        self,
        rng: np.random.Generator,
        agent_id: int,
        evidence_id: int,
        bit: int,
        global_index: int,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.config
        states = np.zeros((cfg.num_layers, cfg.hidden_dim), dtype=np.float32)
        channels = {
            "agent_identity": np.zeros_like(states),
            "reasoning_mode": np.zeros_like(states),
            "private_evidence": np.zeros_like(states),
            "uncertainty": np.zeros_like(states),
            "noise": np.zeros_like(states),
        }
        bit_value = 1.0 if bit else -1.0
        reasoning_mode = int((agent_id + global_index) % len(self.reasoning_embeddings))
        uncertainty = 0.15 if agent_id >= self.n_evidence_bits else -0.15
        agent_component = self.agent_embeddings[agent_id]
        reasoning_component = cfg.reasoning_mode_signal_strength * self.reasoning_embeddings[reasoning_mode]
        for layer_id, layer_scale in enumerate(self.layer_scales):
            noise = rng.normal(0.0, cfg.activation_noise, size=cfg.hidden_dim)
            channels["agent_identity"][layer_id] = agent_component.astype(np.float32)
            channels["reasoning_mode"][layer_id] = reasoning_component.astype(np.float32)
            channels["private_evidence"][layer_id] = (
                layer_scale * cfg.evidence_signal_strength * bit_value * self.evidence_directions[evidence_id]
            ).astype(np.float32)
            channels["uncertainty"][layer_id] = (
                cfg.uncertainty_signal_strength * uncertainty * self.uncertainty_direction
            ).astype(np.float32)
            channels["noise"][layer_id] = noise.astype(np.float32)
            states[layer_id] = sum(channel[layer_id] for channel in channels.values()).astype(np.float32)
        return states, channels


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / max(float(norm), 1e-8)
