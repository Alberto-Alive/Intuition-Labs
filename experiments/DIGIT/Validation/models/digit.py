"""ValidationDIGITModel — encoder + executor + bottleneck (no text decoder).

Stripped-down DIGIT for the validation framework. The bottleneck properties
(information gate, discrete primitives) are identical to the full PoC model.
Removing the decoder makes training 3–5× faster and avoids the need for a
vocabulary specific to each dataset.

The four discrete primitives are the only output used for:
  - utility evaluation (primitive prediction accuracy)
  - privacy attack evaluation (MIA uses primitive outputs as attack features)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse encoder and bottleneck from the PoC package
_poc_root = Path(__file__).resolve().parents[2] / "PoC"
if str(_poc_root) not in sys.path:
    sys.path.insert(0, str(_poc_root))

from poc.models.encoder import AdultQueryEncoder as _FieldEncoder  # noqa: E402
from poc.models.bottleneck import PrimitiveBottleneckHead           # noqa: E402
from .executor import GenericSafeExecutor


class ValidationConfig:
    """Minimal config for ValidationDIGITModel.

    Parameters
    ----------
    field_vocab_sizes : list of vocab sizes per query field (incl. specificity)
    num_query_fields  : total number of query fields (len(field_vocab_sizes))
    d_model / nhead / num_encoder_layers / d_ff / dropout : transformer params
    num_answer/support/confidence/risk_classes : bottleneck dims
    min_group_size / variance_buckets / confidence_buckets : executor params
    """

    def __init__(
        self,
        field_vocab_sizes: List[int],
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_answer_classes: int = 6,
        num_support_classes: int = 4,
        num_confidence_classes: int = 3,
        num_risk_classes: int = 3,
        min_group_size: int = 10,
        variance_buckets: int = 4,
        confidence_buckets: int = 3,
    ):
        self.field_vocab_sizes = field_vocab_sizes
        self.num_query_fields = len(field_vocab_sizes)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_answer_classes = num_answer_classes
        self.num_support_classes = num_support_classes
        self.num_confidence_classes = num_confidence_classes
        self.num_risk_classes = num_risk_classes
        self.min_group_size = min_group_size
        self.variance_buckets = variance_buckets
        self.confidence_buckets = confidence_buckets

    @property
    def executor_feature_dim(self) -> int:
        return 1 + self.variance_buckets + self.confidence_buckets + 1 + 3

    @property
    def max_bits_per_query(self) -> float:
        import math
        return math.log2(
            self.num_answer_classes
            * self.num_support_classes
            * self.num_confidence_classes
            * self.num_risk_classes
        )


class ValidationDIGITModel(nn.Module):
    """Encoder + executor + bottleneck only (no text decoder).

    Inputs  : (B, num_query_fields) int64 query tensor + PrivateDataset
    Outputs : primitive logits and discrete one-hot tensors
    """

    def __init__(self, cfg: ValidationConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder (reused from PoC; name is historical, logic is generic)
        self.encoder = _FieldEncoder(
            field_vocab_sizes=cfg.field_vocab_sizes,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_encoder_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )

        # Executor (deterministic)
        self.executor = GenericSafeExecutor(
            min_group_size=cfg.min_group_size,
            variance_buckets=cfg.variance_buckets,
            confidence_buckets=cfg.confidence_buckets,
        )
        for p in self.executor.parameters():
            p.requires_grad = False

        # Bottleneck (reused from PoC)
        self.bottleneck = PrimitiveBottleneckHead(
            d_model=cfg.d_model,
            executor_dim=self.executor.output_dim,
            num_answer=cfg.num_answer_classes,
            num_support=cfg.num_support_classes,
            num_confidence=cfg.num_confidence_classes,
            num_risk=cfg.num_risk_classes,
        )

    def forward(
        self,
        query_fields: torch.Tensor,
        private_dataset,
        mode: str = "gumbel",
        tau: float = 1.0,
    ) -> Dict:
        z_q = self.encoder(query_fields)
        exec_feat = self.executor(query_fields, private_dataset)
        primitives = self.bottleneck(z_q, exec_feat, mode=mode, tau=tau)
        return {
            "primitives": primitives,
            "executor_features": exec_feat,
            "z_q": z_q,
        }

    @torch.no_grad()
    def generate(self, query_fields: torch.Tensor, private_dataset) -> Dict:
        """Inference mode — returns hard discrete primitives."""
        return self.forward(query_fields, private_dataset, mode="hard", tau=1.0)

    @torch.no_grad()
    def get_primitive_indices(self, query_fields: torch.Tensor, private_dataset) -> Dict:
        """Return the argmax primitive indices for a batch of queries."""
        out = self.generate(query_fields, private_dataset)
        p = out["primitives"]
        return {
            "answer":     p.answer_discrete.argmax(dim=-1),
            "support":    p.support_discrete.argmax(dim=-1),
            "confidence": p.confidence_discrete.argmax(dim=-1),
            "risk":       p.risk_discrete.argmax(dim=-1),
        }
