"""Shared decoder LLM service.

A single Qwen2.5-Instruct model is loaded once and reused by *every* system, so
the only thing that varies between systems is the context/evidence handed to
the decoder.  Generation is **batched on the GPU** — all (system x question)
prompts for the whole experiment are decoded in large batches, which is what
makes the experiment finish fast on a single card.

An offline deterministic decoder is provided so the full pipeline runs with no
downloads (used for CI/smoke tests via `offline: true`).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---- structured output contract shared by all systems -------------------
FIELDS = ["LABEL", "VALUE", "CITES", "CONFLICT", "ANSWER"]
VALID_LABELS = {"YES", "NO", "CONDITIONAL", "INSUFFICIENT", "VALUE"}


@dataclass
class ParsedAnswer:
    label: str
    value: str
    cites: List[str]
    conflict: bool
    answer: str
    raw: str


def parse_structured(raw: str) -> ParsedAnswer:
    """Lenient parser for the LABEL/VALUE/CITES/CONFLICT/ANSWER contract."""
    def grab(field: str) -> str:
        m = re.search(rf"{field}\s*:\s*(.*)", raw, flags=re.IGNORECASE)
        if not m:
            return ""
        # stop at the next known field on a new line
        val = m.group(1).strip()
        return val.strip()

    label_raw = grab("LABEL").upper()
    label = next((l for l in VALID_LABELS if l in label_raw), "INSUFFICIENT")
    value = grab("VALUE").strip().strip(".")
    if value.upper() in {"NONE", "N/A", ""}:
        value = ""
    cites_raw = grab("CITES")
    cites = [c.strip() for c in re.split(r"[,\s]+", cites_raw) if re.match(r"D\d{3,}", c.strip())]
    conflict = grab("CONFLICT").upper().startswith("Y")
    answer = grab("ANSWER") or raw.strip().splitlines()[-1] if raw.strip() else ""
    return ParsedAnswer(label=label, value=value, cites=cites, conflict=conflict,
                        answer=answer, raw=raw)


class DecoderLLM:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16",
                 offline: bool = False):
        self.model_name = model_name
        self.device = device
        self.offline = offline
        self._model = None
        self._tok = None
        if not offline:
            self._load(dtype)

    def _load(self, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"   # decoder-only batched generation
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype).to(self.device)
        self._model.eval()

    # -- prompt construction ---------------------------------------------
    def chat_text(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return self._tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    # -- batched generation ----------------------------------------------
    def generate(self, system_prompts: List[str], user_prompts: List[str],
                 max_new_tokens: int = 200, batch_size: int = 24) -> List[Dict]:
        """Returns a list of dicts: {raw, input_tokens, output_tokens}.
        Offline mode returns deterministic extractive answers instead."""
        assert len(system_prompts) == len(user_prompts)
        if self.offline:
            return [self._offline_answer(s, u) for s, u in zip(system_prompts, user_prompts)]

        import torch
        texts = [self.chat_text(s, u) for s, u in zip(system_prompts, user_prompts)]
        results: List[Dict] = [None] * len(texts)  # type: ignore
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucket for less padding
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch = [texts[i] for i in idxs]
            enc = self._tok(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096).to(self.device)
            in_lens = enc["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = self._tok.batch_decode(gen, skip_special_tokens=True)
            out_lens = [(row != self._tok.pad_token_id).sum().item() for row in gen]
            for j, i in enumerate(idxs):
                results[i] = {"raw": decoded[j].strip(),
                              "input_tokens": int(in_lens[j]),
                              "output_tokens": int(out_lens[j])}
        return results

    # -- offline deterministic decoder (no model) ------------------------
    def _offline_answer(self, system: str, user: str) -> Dict:
        """Extractive stand-in: emits the first VALUE-looking token from the
        context and an INSUFFICIENT fallback. Only for smoke tests."""
        ctx = user
        # crude: prefer an explicit EVIDENCE_PACKET exact_value if present
        m = re.search(r'"exact_value"\s*:\s*"([^"]+)"', ctx)
        if not m:
            m = re.search(r'(\$?\d[\d,\.]*\s*(?:days|%|TB|months|seconds)?|\d{4}-\d{2}-\d{2}|99\.\d+%)', ctx)
        val = m.group(1) if m else ""
        cites = re.findall(r"D\d{3,}", ctx)[:2]
        label = "VALUE" if val else "INSUFFICIENT"
        raw = (f"LABEL: {label}\nVALUE: {val or 'NONE'}\n"
               f"CITES: {', '.join(cites) or 'NONE'}\nCONFLICT: NO\n"
               f"ANSWER: {val or 'Insufficient evidence.'}")
        return {"raw": raw, "input_tokens": len(ctx.split()), "output_tokens": len(raw.split())}

    def count_tokens(self, text: str) -> int:
        if self.offline or self._tok is None:
            return len(text.split())
        return len(self._tok(text)["input_ids"])
