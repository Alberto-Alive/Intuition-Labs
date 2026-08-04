"""Stability-gated public wrapper for DIGIT models.

The wrapped DIGIT model can retain rich internal primitives, but the public
interface is collapsed to a smaller, more cautious codebook:

- answer: YES / NO / MAYBE
- support: LOW / ADEQUATE
- confidence: LOW / MEDIUM+
- risk: LOW / HIGH

If the public code changes under any one-filter coarsening of the query, the
wrapper returns a guarded abstention-like code instead of the underlying code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

GUARDED_PUBLIC_VECTOR = np.array([2, 0, 0, 2], dtype=np.int64)  # MAYBE / LOW / LOW / HIGH


@dataclass
class StabilityGateConfig:
    max_drop_neighbors: int = 8
    max_public_specificity: int | None = None
    min_public_support_count: int | None = None


def _publicize_primitive_vectors(raw_vectors: np.ndarray) -> np.ndarray:
    public = np.empty_like(raw_vectors)

    answer = raw_vectors[:, 0]
    public[:, 0] = np.where(answer == 0, 0, np.where(answer == 1, 1, 2))

    support = raw_vectors[:, 1]
    public[:, 1] = np.where(support <= 1, 0, 2)

    confidence = raw_vectors[:, 2]
    public[:, 2] = np.where(confidence == 0, 0, 1)

    risk = raw_vectors[:, 3]
    public[:, 3] = np.where(risk == 2, 2, 0)

    return public.astype(np.int64)


def _drop_one_filter_neighbors(
    query_vec: torch.Tensor,
    max_drop_neighbors: int,
) -> List[torch.Tensor]:
    n_fields = int(query_vec.numel() - 1)
    active = [i for i in range(n_fields) if int(query_vec[i].item()) != 0]
    neighbors: List[torch.Tensor] = []
    for idx in active[:max_drop_neighbors]:
        q = query_vec.clone()
        q[idx] = 0
        q[n_fields] = int((q[:n_fields] != 0).sum().item())
        neighbors.append(q)
    return neighbors


class StabilityGatedDigitWrapper:
    """Expose a smaller, stability-gated public interface over DIGIT."""

    def __init__(
        self,
        digit_model,
        private_data,
        gate_cfg: StabilityGateConfig | None = None,
    ):
        self.digit_model = digit_model
        self.private_data = private_data
        self.gate_cfg = gate_cfg or StabilityGateConfig()
        self.cfg = getattr(digit_model, "cfg", None)
        self._device = next(digit_model.parameters()).device

    def _raw_primitive_vectors(self, queries: List[torch.Tensor]) -> np.ndarray:
        q_batch = torch.stack(queries).to(self._device)
        with torch.no_grad():
            out = self.digit_model.generate(q_batch, self.private_data)
        p = out["primitives"]
        return np.stack(
            [
                p.answer_discrete.argmax(-1).cpu().numpy(),
                p.support_discrete.argmax(-1).cpu().numpy(),
                p.confidence_discrete.argmax(-1).cpu().numpy(),
                p.risk_discrete.argmax(-1).cpu().numpy(),
            ],
            axis=1,
        ).astype(np.int64)

    def answer_query(self, query_vec: torch.Tensor) -> Dict:
        if self.gate_cfg.min_public_support_count is not None:
            stats = self.private_data.get_subgroup_stats(query_vec)
            if int(stats["n"]) < int(self.gate_cfg.min_public_support_count):
                final = GUARDED_PUBLIC_VECTOR
                return {
                    "answer": int(final[0]),
                    "support": int(final[1]),
                    "confidence": int(final[2]),
                    "risk": int(final[3]),
                    "primitive_vector": final.copy(),
                    "stability_guarded": True,
                    "stability_neighbor_count": 0,
                    "specificity_guarded": False,
                    "support_count_guarded": True,
                    "support_count": int(stats["n"]),
                }
        if (
            self.gate_cfg.max_public_specificity is not None
            and int((query_vec[:-1] != 0).sum().item()) >= int(self.gate_cfg.max_public_specificity)
        ):
            final = GUARDED_PUBLIC_VECTOR
            return {
                "answer": int(final[0]),
                "support": int(final[1]),
                "confidence": int(final[2]),
                "risk": int(final[3]),
                "primitive_vector": final.copy(),
                "stability_guarded": True,
                "stability_neighbor_count": 0,
                "specificity_guarded": True,
                "support_count_guarded": False,
            }
        neighbors = _drop_one_filter_neighbors(query_vec, self.gate_cfg.max_drop_neighbors)
        raw_vectors = self._raw_primitive_vectors([query_vec] + neighbors)
        public_vectors = _publicize_primitive_vectors(raw_vectors)
        base = public_vectors[0]
        stable = bool(np.all(public_vectors[1:] == base)) if len(public_vectors) > 1 else True
        final = base if stable else GUARDED_PUBLIC_VECTOR
        return {
            "answer": int(final[0]),
            "support": int(final[1]),
            "confidence": int(final[2]),
            "risk": int(final[3]),
            "primitive_vector": final.copy(),
            "stability_guarded": bool(not stable),
            "stability_neighbor_count": len(neighbors),
            "specificity_guarded": False,
            "support_count_guarded": False,
        }

    def get_primitive_vectors(self, queries: List[torch.Tensor]) -> np.ndarray:
        return np.stack([self.answer_query(q)["primitive_vector"] for q in queries], axis=0)
