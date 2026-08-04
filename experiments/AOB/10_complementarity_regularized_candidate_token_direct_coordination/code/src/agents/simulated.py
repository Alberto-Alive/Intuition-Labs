from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.agents.types import AttemptBatch
from src.datasets.synthetic import SyntheticExample


@dataclass(frozen=True)
class SimulatedAgentConfig:
    n_agents: int = 5
    hidden_dim: int = 32
    num_layers: int = 4
    activation_signal_strength: float = 1.25
    activation_noise: float = 0.85
    confidence_noise: float = 0.28
    hidden_label_signal_strength: float = 0.0
    task_signal_strength: float = 1.0
    agent_signal_strength: float = 1.0
    agent_quality_signal_strength: float = 0.25
    reasoning_mode_signal_strength: float = 0.35
    stuckness_signal_strength: float = 0.25


class SimulatedAgentSwarm:
    """Cloned synthetic agents with noisy visible confidence and latent telemetry.

    Hidden states encode an internal knows/uncertain signal that determines
    whether the agent samples the correct candidate. This is not an evaluator
    leak: the signal is generated before coordination and no coordinator receives
    labels except through supervised train labels during fit.
    """

    def __init__(
        self,
        config: SimulatedAgentConfig,
        num_tasks: int,
        num_classes: int,
        seed: int,
    ) -> None:
        self.config = config
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.seed = seed
        rng = np.random.default_rng(seed + 10_003)
        self.agent_task_skill = rng.uniform(0.42, 0.78, size=(config.n_agents, num_tasks))
        self.agent_task_skill += rng.normal(0.0, 0.06, size=(config.n_agents, num_tasks))
        self.agent_task_skill = np.clip(self.agent_task_skill, 0.25, 0.92)
        self.task_embeddings = rng.normal(0.0, 0.45, size=(num_tasks, config.hidden_dim))
        self.agent_embeddings = rng.normal(0.0, 0.30, size=(config.n_agents, config.hidden_dim))
        self.label_embeddings = rng.normal(0.0, 0.40, size=(num_classes, config.hidden_dim))
        self.reasoning_mode_embeddings = rng.normal(0.0, 0.35, size=(4, config.hidden_dim))
        self.knowledge_direction = self._unit(rng.normal(size=config.hidden_dim))
        self.uncertainty_direction = self._unit(rng.normal(size=config.hidden_dim))
        self.quality_direction = self._unit(rng.normal(size=config.hidden_dim))
        self.stuckness_direction = self._unit(rng.normal(size=config.hidden_dim))
        self.layer_scales = np.linspace(0.35, 1.0, config.num_layers)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / max(float(norm), 1e-8)

    def run(self, splits: Dict[str, List[SyntheticExample]]) -> Dict[str, AttemptBatch]:
        return {split: self._run_split(split, examples) for split, examples in splits.items()}

    def _run_split(self, split: str, examples: List[SyntheticExample]) -> AttemptBatch:
        n = len(examples)
        cfg = self.config
        answers = np.zeros((n, cfg.n_agents), dtype=np.int64)
        confidences = np.zeros((n, cfg.n_agents), dtype=np.float32)
        hidden = np.zeros((n, cfg.n_agents, cfg.num_layers, cfg.hidden_dim), dtype=np.float32)
        channels = {
            name: np.zeros_like(hidden)
            for name in [
                "task_identity",
                "agent_identity",
                "label_identity",
                "reasoning_mode",
                "correctness_state",
                "uncertainty",
                "agent_quality",
                "stuckness",
                "noise",
            ]
        }
        visible_texts: List[List[str]] = []

        for i, example in enumerate(examples):
            row_texts: List[str] = []
            difficulty = self._difficulty(example.global_index, example.task_id)
            for agent_id in range(cfg.n_agents):
                rng = np.random.default_rng(
                    self.seed
                    + 97_531 * (agent_id + 1)
                    + 31_337 * (example.global_index + 1)
                    + 701 * example.task_id
                )
                base_skill = self.agent_task_skill[agent_id, example.task_id]
                p_correct = np.clip(base_skill - 0.28 * difficulty, 0.10, 0.93)
                knows = bool(rng.random() < p_correct)
                if knows:
                    answer = example.label
                else:
                    offset = int(rng.integers(1, self.num_classes))
                    answer = int((example.label + offset) % self.num_classes)
                confidence = float(
                    np.clip(
                        0.18 + 0.68 * p_correct + rng.normal(0.0, cfg.confidence_noise),
                        0.01,
                        0.99,
                    )
                )
                answers[i, agent_id] = answer
                confidences[i, agent_id] = confidence
                hidden_state, hidden_channels = self._hidden_state(
                    rng=rng,
                    global_index=example.global_index,
                    task_id=example.task_id,
                    agent_id=agent_id,
                    label=example.label,
                    knows=knows,
                    p_correct=float(p_correct),
                    base_skill=float(base_skill),
                    difficulty=float(difficulty),
                )
                hidden[i, agent_id] = hidden_state
                for channel_name, channel_value in hidden_channels.items():
                    channels[channel_name][i, agent_id] = channel_value
                row_texts.append(
                    " ".join(
                        [
                            f"agent={agent_id}",
                            f"task={example.task_id}",
                            f"answer={answer}",
                            f"confidence={confidence:.3f}",
                            f"rationale_code={(answer + agent_id + example.task_id) % 7}",
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

    @staticmethod
    def _difficulty(global_index: int, task_id: int) -> float:
        raw = (19 * global_index + 23 * task_id + 11) % 100
        return float(raw) / 99.0

    def _hidden_state(
        self,
        rng: np.random.Generator,
        global_index: int,
        task_id: int,
        agent_id: int,
        label: int,
        knows: bool,
        p_correct: float,
        base_skill: float,
        difficulty: float,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.config
        states = np.zeros((cfg.num_layers, cfg.hidden_dim), dtype=np.float32)
        channels = {
            "task_identity": np.zeros_like(states),
            "agent_identity": np.zeros_like(states),
            "label_identity": np.zeros_like(states),
            "reasoning_mode": np.zeros_like(states),
            "correctness_state": np.zeros_like(states),
            "uncertainty": np.zeros_like(states),
            "agent_quality": np.zeros_like(states),
            "stuckness": np.zeros_like(states),
            "noise": np.zeros_like(states),
        }
        reasoning_mode = int((global_index + 3 * agent_id + 5 * task_id) % len(self.reasoning_mode_embeddings))
        task_component = cfg.task_signal_strength * self.task_embeddings[task_id]
        agent_component = cfg.agent_signal_strength * self.agent_embeddings[agent_id]
        label_component = cfg.hidden_label_signal_strength * self.label_embeddings[label]
        reasoning_component = cfg.reasoning_mode_signal_strength * self.reasoning_mode_embeddings[reasoning_mode]
        knowledge_value = 1.0 if knows else -1.0
        uncertainty_value = 2.0 * p_correct - 1.0
        quality_value = 2.0 * base_skill - 1.0
        stuckness_value = 2.0 * difficulty - 1.0
        for layer_id, layer_scale in enumerate(self.layer_scales):
            noise = rng.normal(0.0, cfg.activation_noise, size=cfg.hidden_dim)
            channels["task_identity"][layer_id] = task_component.astype(np.float32)
            channels["agent_identity"][layer_id] = agent_component.astype(np.float32)
            channels["label_identity"][layer_id] = label_component.astype(np.float32)
            channels["reasoning_mode"][layer_id] = reasoning_component.astype(np.float32)
            channels["correctness_state"][layer_id] = (
                layer_scale * cfg.activation_signal_strength * knowledge_value * self.knowledge_direction
            ).astype(np.float32)
            channels["uncertainty"][layer_id] = (
                0.45 * layer_scale * uncertainty_value * self.uncertainty_direction
            ).astype(np.float32)
            channels["agent_quality"][layer_id] = (
                cfg.agent_quality_signal_strength * quality_value * self.quality_direction
            ).astype(np.float32)
            channels["stuckness"][layer_id] = (
                cfg.stuckness_signal_strength * stuckness_value * self.stuckness_direction
            ).astype(np.float32)
            channels["noise"][layer_id] = noise.astype(np.float32)
            states[layer_id] = sum(channel[layer_id] for channel in channels.values()).astype(np.float32)
        return states, channels
