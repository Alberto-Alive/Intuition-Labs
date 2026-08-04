from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from src.agents.types import AttemptBatch
from src.datasets.strict_coordination import StrictCoordinationExample
from src.telemetry.hooks import ForwardHookRecorder


@dataclass(frozen=True)
class TransformerStrictConfig:
    model_name_or_path: str = "EleutherAI/pythia-70m-deduped"
    layers: tuple[int, ...] = (1, 3, 5)
    token_positions: tuple[str, ...] = ("evidence", "final")
    batch_size: int = 32
    max_length: int = 96
    device: str = "cuda"
    prompt_template: str = "default"
    evidence_tokens: tuple[str, str] = ("BIT_ZERO", "BIT_ONE")
    mask_token: str = "BIT_MASK"
    answer_format: str = "default"
    visible_evidence: bool = False


class TransformerStrictPartialEvidenceSwarm:
    """Strict partial-evidence agents backed by local transformer hidden states.

    The visible output is intentionally uninformative. The transformer prompt
    contains one private evidence bit for one agent. Hidden states are captured
    via hooks and can be aggregated by activation-aware coordinators.
    """

    def __init__(
        self,
        config: TransformerStrictConfig,
        n_evidence_bits: int,
        num_classes: int,
        seed: int,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers is required for Stage 1 hidden-state capture") from exc

        self.config = config
        self.n_evidence_bits = n_evidence_bits
        self.num_classes = num_classes
        self.seed = seed
        self.device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
        self.evidence_tokens = tuple(str(value) for value in config.evidence_tokens)
        if len(self.evidence_tokens) != 2:
            raise ValueError("evidence_tokens must contain [zero_token, one_token]")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        blocks = _find_transformer_blocks(self.model)
        self.layers = tuple(layer for layer in config.layers if 0 <= layer < len(blocks))
        if not self.layers:
            raise ValueError("no valid transformer layers selected")
        self.blocks = [blocks[layer] for layer in self.layers]
        self.slot_labels = [
            f"{position}:layer_{layer}"
            for position in config.token_positions
            for layer in self.layers
        ]

    def run(self, splits: Dict[str, List[StrictCoordinationExample]]) -> Dict[str, AttemptBatch]:
        return {split: self._run_split(split, examples) for split, examples in splits.items()}

    def run_masked_evidence(self, splits: Dict[str, List[StrictCoordinationExample]]) -> Dict[str, AttemptBatch]:
        return {
            split: self._run_split(split, examples, mask_evidence=True)
            for split, examples in splits.items()
        }

    def run_shuffled_evidence(
        self,
        splits: Dict[str, List[StrictCoordinationExample]],
        seed: int,
    ) -> Dict[str, AttemptBatch]:
        return {
            split: self._run_split(
                split,
                examples,
                evidence_bits_override=_shuffle_evidence_bits(
                    examples,
                    seed=seed + _split_offset(split),
                    n_evidence_bits=self.n_evidence_bits,
                ),
            )
            for split, examples in splits.items()
        }

    def _run_split(
        self,
        split: str,
        examples: List[StrictCoordinationExample],
        evidence_bits_override: Sequence[Sequence[int]] | None = None,
        mask_evidence: bool = False,
    ) -> AttemptBatch:
        prompts: List[str] = []
        evidence_positions: List[int] = []
        prompt_lookup: List[tuple[int, int]] = []
        n_agents = self.n_evidence_bits
        for example_id, example in enumerate(examples):
            evidence_bits = (
                evidence_bits_override[example_id]
                if evidence_bits_override is not None
                else example.evidence_bits
            )
            for agent_id in range(n_agents):
                evidence_id = agent_id % self.n_evidence_bits
                bit = int(evidence_bits[evidence_id])
                prompt = self._prompt(
                    agent_id=agent_id,
                    evidence_id=evidence_id,
                    bit=bit,
                    mask_evidence=mask_evidence,
                )
                prompts.append(prompt)
                evidence_positions.append(
                    self._evidence_token_position(prompt, self.config.mask_token if mask_evidence else self._bit_token(bit))
                )
                prompt_lookup.append((example_id, agent_id))

        captured = self._capture(prompts, evidence_positions)
        hidden_dim = captured.shape[-1]
        hidden = np.zeros((len(examples), n_agents, captured.shape[1], hidden_dim), dtype=np.float32)
        for row_id, (example_id, agent_id) in enumerate(prompt_lookup):
            hidden[example_id, agent_id] = captured[row_id]

        answers = np.zeros((len(examples), n_agents), dtype=np.int64)
        confidences = np.full((len(examples), n_agents), 0.25, dtype=np.float32)
        visible_texts = []
        for example_id, example in enumerate(examples):
            evidence_bits = (
                evidence_bits_override[example_id]
                if evidence_bits_override is not None
                else example.evidence_bits
            )
            visible_texts.append(
                [
                    self._visible_text(
                        agent_id=agent_id,
                        bit=int(evidence_bits[agent_id % self.n_evidence_bits]),
                        reveal_evidence=bool(self.config.visible_evidence),
                    )
                    for agent_id in range(n_agents)
                ]
            )
        private_agent_views = [
            [
                self._prompt(
                    agent_id=agent_id,
                    evidence_id=agent_id % self.n_evidence_bits,
                    bit=int(
                        (
                            evidence_bits_override[example_id]
                            if evidence_bits_override is not None
                            else example.evidence_bits
                        )[agent_id % self.n_evidence_bits]
                    ),
                    mask_evidence=mask_evidence,
                )
                for agent_id in range(n_agents)
            ]
            for example_id, example in enumerate(examples)
        ]
        return AttemptBatch(
            split=split,
            example_ids=np.asarray([example.example_id for example in examples], dtype=object),
            task_ids=np.asarray([example.task_id for example in examples], dtype=np.int64),
            labels=np.asarray([example.label for example in examples], dtype=np.int64),
            answers=answers,
            confidences=confidences,
            hidden_states=hidden,
            visible_texts=visible_texts,
            private_agent_views=private_agent_views,
        )

    def _prompt(self, agent_id: int, evidence_id: int, bit: int, mask_evidence: bool = False) -> str:
        bit_token = str(self.config.mask_token) if mask_evidence else self._bit_token(bit)
        if self.config.prompt_template == "compact":
            return (
                f"Agent {agent_id}; private slot {evidence_id}; value {bit_token}. "
                "Keep the value private. Reply STATUS_UNKNOWN only."
            )
        if self.config.prompt_template == "chat":
            return (
                "<system>Partial evidence coordination. Never reveal private evidence.</system>\n"
                f"<user>Agent {agent_id}, inspect slot {evidence_id}. Evidence token is {bit_token}. "
                "Do not infer the class; output unknown.</user>"
            )
        return (
            "Partial evidence coordination task. "
            f"You are agent {agent_id}. "
            f"You may see only private evidence slot {evidence_id}. "
            f"Private evidence value: {bit_token}. "
            "Do not reveal the private evidence. Do not infer the final class. "
            "Visible reply must be STATUS_UNKNOWN."
        )

    def _visible_text(self, agent_id: int, bit: int, reveal_evidence: bool) -> str:
        fields = {
            "agent": agent_id,
            "task": 0,
            "answer": 0,
            "confidence": 0.250,
            "rationale_code": 0,
            "status": "withheld_private_evidence",
        }
        if reveal_evidence:
            fields["status"] = "revealed_private_evidence"
            fields["private_evidence_bit"] = bit
            fields["private_evidence_token"] = self._bit_token(bit)
        if self.config.answer_format == "json":
            pairs = [
                f'"agent":{fields["agent"]}',
                f'"task":{fields["task"]}',
                f'"answer":{fields["answer"]}',
                f'"confidence":{fields["confidence"]:.3f}',
                f'"rationale_code":{fields["rationale_code"]}',
                f'"status":"{fields["status"]}"',
            ]
            if reveal_evidence:
                pairs.extend(
                    [
                        f'"private_evidence_bit":{fields["private_evidence_bit"]}',
                        f'"private_evidence_token":"{fields["private_evidence_token"]}"',
                    ]
                )
            return "{" + ",".join(pairs) + "}"
        if self.config.answer_format == "terse":
            parts = [
                f"a{fields['agent']}",
                "ans0",
                "conf0.250",
                f"status:{fields['status']}",
            ]
            if reveal_evidence:
                parts.append(f"private_evidence_bit={fields['private_evidence_bit']}")
                parts.append(f"private_evidence_token={fields['private_evidence_token']}")
            return "|".join(parts)
        parts = [
            f"agent={fields['agent']}",
            f"task={fields['task']}",
            f"answer={fields['answer']}",
            f"confidence={fields['confidence']:.3f}",
            f"rationale_code={fields['rationale_code']}",
            f"status={fields['status']}",
        ]
        if reveal_evidence:
            parts.append(f"private_evidence_bit={fields['private_evidence_bit']}")
            parts.append(f"private_evidence_token={fields['private_evidence_token']}")
        return " ".join(parts)

    def _bit_token(self, bit: int) -> str:
        return self.evidence_tokens[int(bit)]

    def _evidence_token_position(self, prompt: str, marker: str) -> int:
        prefix = prompt[: prompt.index(marker)]
        return len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])

    def _capture(self, prompts: Sequence[str], evidence_positions: Sequence[int]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for start in range(0, len(prompts), self.config.batch_size):
            batch_prompts = list(prompts[start : start + self.config.batch_size])
            batch_evidence_positions = list(evidence_positions[start : start + self.config.batch_size])
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad(), ForwardHookRecorder(self.blocks) as recorder:
                _ = self.model(**encoded)
                captured = self._stack_positions(
                    recorder=recorder,
                    attention_mask=encoded["attention_mask"],
                    evidence_positions=batch_evidence_positions,
                )
            rows.append(captured.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)

    def _stack_positions(
        self,
        recorder: ForwardHookRecorder,
        attention_mask: torch.Tensor,
        evidence_positions: Sequence[int],
    ) -> torch.Tensor:
        ordered = [recorder.outputs[index] for index in sorted(recorder.outputs)]
        lengths = attention_mask.sum(dim=1).long()
        final_positions = torch.clamp(lengths - 1, min=0)
        evidence_tensor = torch.as_tensor(evidence_positions, dtype=torch.long, device=attention_mask.device)
        evidence_tensor = torch.minimum(evidence_tensor, final_positions)
        position_map = {
            "evidence": evidence_tensor,
            "final": final_positions,
            "last": final_positions,
        }
        slots: List[torch.Tensor] = []
        batch_index = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        for position_name in self.config.token_positions:
            if position_name in {"prompt_mean", "mean"}:
                mask = attention_mask.to(dtype=ordered[0].dtype).unsqueeze(-1)
                denom = lengths.clamp(min=1).to(dtype=ordered[0].dtype).unsqueeze(-1)
                for layer_output in ordered:
                    slots.append((layer_output * mask).sum(dim=1) / denom)
            else:
                positions = position_map[position_name]
                for layer_output in ordered:
                    slots.append(layer_output[batch_index, positions, :])
        return torch.stack(slots, dim=1)


def _find_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    candidate_paths: Iterable[str] = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
    )
    for path in candidate_paths:
        current: object = model
        found = True
        for part in path.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found and isinstance(current, torch.nn.ModuleList):
            return list(current)
    raise RuntimeError("could not locate transformer blocks for hook capture")


def _bit_token(bit: int) -> str:
    return "BIT_ONE" if int(bit) else "BIT_ZERO"


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 10_000, "test": 20_000}.get(split, 30_000)


def _shuffle_evidence_bits(
    examples: Sequence[StrictCoordinationExample],
    seed: int,
    n_evidence_bits: int,
) -> List[tuple[int, ...]]:
    bits = np.asarray([example.evidence_bits for example in examples], dtype=np.int64)
    if bits.shape[0] <= 1:
        return [tuple(int(value) for value in row[:n_evidence_bits]) for row in bits]
    rng = np.random.default_rng(seed)
    shuffled = bits.copy()
    for bit_id in range(n_evidence_bits):
        perm = rng.permutation(bits.shape[0])
        if np.all(perm == np.arange(bits.shape[0])):
            perm = np.roll(perm, 1)
        shuffled[:, bit_id] = bits[perm, bit_id]
    return [tuple(int(value) for value in row[:n_evidence_bits]) for row in shuffled]
