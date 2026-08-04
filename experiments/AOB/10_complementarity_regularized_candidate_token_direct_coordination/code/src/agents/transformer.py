from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from src.telemetry.hooks import ForwardHookRecorder


@dataclass(frozen=True)
class TransformerCaptureConfig:
    model_name_or_path: str
    device: str = "cuda"
    layers: tuple[int, ...] | None = None
    max_new_tokens: int = 8
    do_sample: bool = False


class TransformerHiddenStateAgent:
    """Optional local open-weight transformer support.

    This class intentionally does not download weights itself. Point
    `model_name_or_path` at an already available Hugging Face model directory or
    a cached model id. Hidden states are captured through forward hooks on the
    model's transformer blocks.
    """

    def __init__(self, config: TransformerCaptureConfig) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "transformers is required for TransformerHiddenStateAgent"
            ) from exc

        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        blocks = _find_transformer_blocks(self.model)
        if config.layers is None:
            self.blocks = blocks
        else:
            self.blocks = [blocks[i] for i in config.layers]

    def generate_with_hidden_states(self, prompts: Sequence[str]) -> tuple[List[str], torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Re-run the full generated sequence once under hooks so all selected
        # block activations align to the final generated token.
        hook_encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad(), ForwardHookRecorder(self.blocks) as recorder:
            _ = self.model(**hook_encoded)
            hidden = recorder.stacked_final_token().detach().cpu()
        return texts, hidden


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
    raise RuntimeError(
        "could not locate transformer blocks; add the architecture path to _find_transformer_blocks"
    )

