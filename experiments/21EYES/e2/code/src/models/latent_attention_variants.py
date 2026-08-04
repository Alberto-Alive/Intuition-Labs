from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.datasets.latent_attention_capacity_dataset import Stage8Example


_CLEAN_QKV_TYPE_VECTOR_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


@dataclass(frozen=True)
class Stage8ArchitectureConfig:
    name: str
    model_kind: str = "trainable_latent"
    roles: int = 4
    avenues: int = 2
    chunking: str = "contiguous"
    candidate_query: str = "candidate_task_query"
    coordinator: str = "mixture_of_views"
    view_sharing: str = "shared_encoder_role_avenue_embeddings"
    memory_compression: str = "mean_pooling"
    attention_routing: str = "dense_all_view_attention"
    training_loss: str = "multi_positive_cross_entropy"
    regularizers: Tuple[str, ...] = ()
    curriculum: str = "mixed_n"
    vector_dim: int = 128
    hidden_dim: int = 64
    coord_hops: int = 1
    top_k_views: int = 0
    epochs: int = 6
    lr: float = 3e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    frozen: bool = False
    max_train_examples: int = 512

    @property
    def latent_views(self) -> int:
        return max(1, int(self.roles) * int(self.avenues))

    @property
    def config_id(self) -> str:
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


class Stage8Selector:
    supports_training = False

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "Stage8Selector":
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        raise NotImplementedError

    def predict(self, examples: Sequence[Stage8Example]) -> List[int]:
        rows = self.scores(examples)
        return [int(max(range(len(row)), key=lambda index: row[index])) for row in rows]

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        return {}


class Stage8HashFeaturizer:
    def __init__(self, vector_dim: int = 128) -> None:
        self.vector_dim = int(vector_dim)
        self._cache: Dict[str, torch.Tensor] = {}

    def text_vector(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = torch.zeros(self.vector_dim, dtype=torch.float32)
        for token in _tokens(text):
            vector[_stable_hash(token) % self.vector_dim] += 1.0
        norm = vector.norm(p=2).clamp(min=1.0)
        vector = vector / norm
        self._cache[text] = vector
        return vector

    def texts_vector(self, texts: Iterable[str]) -> torch.Tensor:
        vector = torch.zeros(self.vector_dim, dtype=torch.float32)
        count = 0
        for text in texts:
            vector += self.text_vector(text)
            count += 1
        if count:
            vector = vector / float(count)
        return vector


class RandomCandidateSelector(Stage8Selector):
    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows: List[List[float]] = []
        for example in examples:
            rng = random.Random(_stable_hash(self.seed, example.example_id))
            rows.append([rng.random() for _ in example.candidates])
        return rows


class CandidateOnlyBaseline(Stage8Selector):
    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.index_bias: List[float] = []

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "CandidateOnlyBaseline":
        del dev_examples
        k = max((example.k_candidates for example in train_examples), default=8)
        counts = [1.0 for _ in range(k)]
        for example in train_examples:
            if example.label >= 0:
                counts[example.label] += 1.0
        total = sum(counts)
        self.index_bias = [math.log(count / total) for count in counts]
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows = []
        for example in examples:
            base = self.index_bias or [0.0 for _ in range(example.k_candidates)]
            rows.append([float(base[i % len(base)]) for i in range(example.k_candidates)])
        return rows


class EvidenceOnlyBaseline(Stage8Selector):
    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)
        self.prototype = torch.zeros(config.vector_dim)

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "EvidenceOnlyBaseline":
        del dev_examples
        if train_examples:
            self.prototype = torch.stack([self.featurizer.texts_vector(example.evidence_blocks) for example in train_examples]).mean(dim=0)
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows = []
        for example in examples:
            evidence = self.featurizer.texts_vector(example.evidence_blocks)
            score = float(torch.dot(evidence, self.prototype))
            rows.append([score for _ in example.candidates])
        return rows


class QueryOnlyBaseline(Stage8Selector):
    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)
        self.prototypes = torch.zeros(8, config.vector_dim)

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "QueryOnlyBaseline":
        del dev_examples
        k = max((example.k_candidates for example in train_examples), default=8)
        sums = torch.zeros(k, self.config.vector_dim)
        counts = torch.ones(k, 1)
        for example in train_examples:
            if example.label >= 0:
                sums[example.label] += self.featurizer.text_vector(example.query)
                counts[example.label] += 1.0
        self.prototypes = F.normalize(sums / counts, dim=-1)
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows = []
        for example in examples:
            query = self.featurizer.text_vector(example.query)
            prototypes = self.prototypes[: example.k_candidates]
            if prototypes.shape[0] < example.k_candidates:
                pad = torch.zeros(example.k_candidates - prototypes.shape[0], self.config.vector_dim)
                prototypes = torch.cat([prototypes, pad], dim=0)
            rows.append(torch.mv(prototypes, query).tolist())
        return rows


class RetrievalTopKBaseline(Stage8Selector):
    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0, oracle: bool = False) -> None:
        self.config = config
        self.seed = int(seed)
        self.oracle = bool(oracle)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows = []
        for example in examples:
            blocks = example.evidence_blocks
            if self.oracle and example.relevant_block_indices:
                blocks = tuple(example.evidence_blocks[index] for index in example.relevant_block_indices)
            block_vectors = [self.featurizer.text_vector(block) for block in blocks]
            query_vector = self.featurizer.text_vector(example.query)
            candidate_scores = []
            for candidate in example.candidates:
                candidate_vector = self.featurizer.text_vector(candidate)
                request = F.normalize(candidate_vector + 0.5 * query_vector, dim=0)
                if not block_vectors:
                    candidate_scores.append(0.0)
                    continue
                sims = torch.stack([torch.dot(request, block) for block in block_vectors])
                k = min(max(1, int(self.config.top_k_views or 4)), sims.numel())
                candidate_scores.append(float(sims.topk(k).values.mean()))
            rows.append(candidate_scores)
        return rows


class RawLatentSelector(Stage8Selector):
    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        rows = []
        for example in examples:
            views = _view_matrix(example, self.config, self.featurizer, self.seed)
            query_vector = self.featurizer.text_vector(example.query)
            candidate_scores = []
            for candidate in example.candidates:
                candidate_vector = F.normalize(query_vector + self.featurizer.text_vector(candidate), dim=0)
                view_scores = torch.mv(views, candidate_vector)
                if view_scores.numel() == 0:
                    candidate_scores.append(0.0)
                else:
                    top_k = _effective_top_k(self.config, view_scores.numel())
                    candidate_scores.append(float(view_scores.topk(top_k).values.mean()))
            rows.append(candidate_scores)
        return rows


class MonolithicSelector(Stage8Selector):
    supports_training = True

    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)
        torch.manual_seed(seed)
        self.module = _PairScorer(config.vector_dim, config.hidden_dim, config.dropout)
        self.training_trace: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "MonolithicSelector":
        if self.config.frozen or self.config.epochs <= 0:
            return self
        rows = list(train_examples)[: self.config.max_train_examples]
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        for epoch in range(self.config.epochs):
            random.Random(_stable_hash(self.seed, epoch)).shuffle(rows)
            losses = []
            for example in rows:
                if example.label < 0:
                    continue
                scores = self._score_tensor(example)
                loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([example.label], dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.training_trace.append({"epoch": float(epoch), "loss": float(sum(losses) / max(1, len(losses)))})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        with torch.no_grad():
            return [self._score_tensor(example).tolist() for example in examples]

    def _score_tensor(self, example: Stage8Example) -> torch.Tensor:
        evidence = self.featurizer.texts_vector(example.evidence_blocks)
        if example.evidence_blocks:
            evidence = evidence / math.sqrt(max(1.0, len(example.evidence_blocks) / 8.0))
        query = self.featurizer.text_vector(example.query)
        rows = []
        for candidate in example.candidates:
            candidate_vector = F.normalize(query + self.featurizer.text_vector(candidate), dim=0)
            rows.append(self.module(candidate_vector, evidence))
        return torch.stack(rows)

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        return {
            "training_trace": self.training_trace,
            "parameter_count": count_parameters(self.module),
        }


class TrainableLatentSelector(Stage8Selector):
    supports_training = True

    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)
        torch.manual_seed(seed)
        self.module = _LatentScoringModule(config)
        self.training_trace: List[Dict[str, float]] = []
        if config.frozen:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "TrainableLatentSelector":
        del dev_examples
        if self.config.frozen or self.config.epochs <= 0:
            return self
        rows = list(train_examples)[: self.config.max_train_examples]
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        for epoch in range(self.config.epochs):
            random.Random(_stable_hash(self.seed, self.config.config_id, epoch)).shuffle(rows)
            losses = []
            for example in rows:
                if example.label < 0:
                    continue
                scores, routing = self._score_tensor(example, return_routing=True)
                loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([example.label], dtype=torch.long))
                loss = loss + _regularization_loss(self.config, routing)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.training_trace.append({"epoch": float(epoch), "loss": float(sum(losses) / max(1, len(losses)))})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        with torch.no_grad():
            return [self._score_tensor(example).tolist() for example in examples]

    def _score_tensor(self, example: Stage8Example, return_routing: bool = False):
        views = _view_matrix(example, self.config, self.featurizer, self.seed)
        if bool(example.metadata.get("hidden_state_shuffle")):
            permutation = torch.randperm(views.shape[0])
            views = views[permutation]
        if bool(example.metadata.get("role_permutation")):
            views = _permute_roles(views, self.config)
        if bool(example.metadata.get("avenue_permutation")):
            views = _permute_avenues(views, self.config)
        query = self.featurizer.text_vector(example.query)
        candidate_vectors = torch.stack([F.normalize(query + self.featurizer.text_vector(candidate), dim=0) for candidate in example.candidates])
        scores, routing = self.module(candidate_vectors, views)
        if return_routing:
            return scores, routing
        return scores

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        if not examples:
            return {"training_trace": self.training_trace, "parameter_count": count_parameters(self.module)}
        entropies: List[float] = []
        role_counts = [0 for _ in range(max(1, self.config.roles))]
        avenue_counts = [0 for _ in range(max(1, self.config.avenues))]
        top_views_by_family: Dict[str, List[int]] = {}
        with torch.no_grad():
            for example in examples:
                scores, routing = self._score_tensor(example, return_routing=True)
                predicted = int(torch.argmax(scores).item())
                weights = routing[predicted]
                entropy = float(-(weights * (weights.clamp(min=1e-8).log())).sum().item())
                entropies.append(entropy)
                top_view = int(torch.argmax(weights).item())
                role = top_view // max(1, self.config.avenues)
                avenue = top_view % max(1, self.config.avenues)
                role_counts[role % len(role_counts)] += 1
                avenue_counts[avenue % len(avenue_counts)] += 1
                top_views_by_family.setdefault(example.task_family, []).append(top_view)
        total = max(1, len(examples))
        return {
            "training_trace": self.training_trace,
            "parameter_count": count_parameters(self.module),
            "routing_entropy_mean": float(sum(entropies) / max(1, len(entropies))),
            "role_usage_distribution": [count / total for count in role_counts],
            "avenue_usage_distribution": [count / total for count in avenue_counts],
            "top_selected_views_by_task_family": {
                family: CounterLike(values).most_common(5) for family, values in top_views_by_family.items()
            },
        }


class EvidenceFeatureLatentSelector(Stage8Selector):
    """Small trainable latent selector for Stage 8B.3 micro-architecture screens."""

    supports_training = True
    feature_dim = 16

    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        torch.manual_seed(seed)
        self.module = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.module.weight)
        nn.init.zeros_(self.module.bias)
        self.training_trace: List[Dict[str, float]] = []
        if config.frozen:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "EvidenceFeatureLatentSelector":
        del dev_examples
        if self.config.frozen or self.config.epochs <= 0:
            return self
        rows = list(train_examples)[: self.config.max_train_examples]
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        for epoch in range(self.config.epochs):
            random.Random(_stable_hash(self.seed, self.config.config_id, "evidence-feature", epoch)).shuffle(rows)
            losses = []
            for example in rows:
                if example.label < 0:
                    continue
                features = _stage8b3_feature_tensor(example, self.config, self.seed, training=True)
                logits = self.module(features).squeeze(-1)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([example.label], dtype=torch.long))
                if self.config.training_loss in {"pairwise_ranking_loss", "margin_loss"}:
                    correct = logits[example.label]
                    negatives = torch.cat([logits[: example.label], logits[example.label + 1 :]])
                    if negatives.numel():
                        loss = loss + F.relu(0.20 - correct + negatives.max()).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.training_trace.append({"epoch": float(epoch), "loss": float(sum(losses) / max(1, len(losses)))})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        with torch.no_grad():
            rows = []
            for example in examples:
                features = _stage8b3_feature_tensor(example, self.config, self.seed, training=False)
                rows.append(self.module(features).squeeze(-1).tolist())
            return rows

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        if not examples:
            return {"training_trace": self.training_trace, "parameter_count": count_parameters(self.module)}
        entropies: List[float] = []
        role_counts = [0 for _ in range(max(1, self.config.roles))]
        avenue_counts = [0 for _ in range(max(1, self.config.avenues))]
        pairwise_sims: List[float] = []
        top_views_by_family: Dict[str, List[int]] = {}
        with torch.no_grad():
            for example in examples:
                score_row = self.module(_stage8b3_feature_tensor(example, self.config, self.seed, training=False)).squeeze(-1)
                predicted = int(torch.argmax(score_row).item())
                weights = _stage8b3_view_weights(example, self.config, predicted, self.seed)
                entropy = float(-(weights * weights.clamp(min=1e-8).log()).sum().item())
                entropies.append(entropy)
                top_view = int(torch.argmax(weights).item()) if weights.numel() else 0
                role = top_view // max(1, self.config.avenues)
                avenue = top_view % max(1, self.config.avenues)
                role_counts[role % len(role_counts)] += 1
                avenue_counts[avenue % len(avenue_counts)] += 1
                top_views_by_family.setdefault(example.task_family, []).append(top_view)
                pairwise_sims.append(_stage8b3_pairwise_view_similarity(example, self.config, self.seed))
        total = max(1, len(examples))
        return {
            "training_trace": self.training_trace,
            "parameter_count": count_parameters(self.module),
            "routing_entropy_mean": float(sum(entropies) / max(1, len(entropies))),
            "role_usage_distribution": [count / total for count in role_counts],
            "avenue_usage_distribution": [count / total for count in avenue_counts],
            "top_selected_views_by_task_family": {
                family: CounterLike(values).most_common(5) for family, values in top_views_by_family.items()
            },
            "pairwise_hidden_state_similarity": float(sum(pairwise_sims) / max(1, len(pairwise_sims))),
        }


class CleanQKVActivationMemorySelector(Stage8Selector):
    """Candidate/query self path reads compressed evidence activations through gated memory."""

    supports_training = True
    feature_dim = 18

    def __init__(self, config: Stage8ArchitectureConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.featurizer = Stage8HashFeaturizer(config.vector_dim)
        torch.manual_seed(seed)
        self.module = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.module.weight)
        nn.init.zeros_(self.module.bias)
        self.training_trace: List[Dict[str, float]] = []
        if config.frozen:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "CleanQKVActivationMemorySelector":
        del dev_examples
        if self.config.frozen or self.config.epochs <= 0:
            return self
        rows = list(train_examples)[: self.config.max_train_examples]
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        for epoch in range(self.config.epochs):
            random.Random(_stable_hash(self.seed, self.config.config_id, "clean-qkv", epoch)).shuffle(rows)
            losses = []
            for example in rows:
                if example.label < 0:
                    continue
                features = _clean_qkv_feature_tensor(example, self.config, self.featurizer, self.seed, training=True)
                logits = self.module(features).squeeze(-1)
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([example.label], dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.training_trace.append({"epoch": float(epoch), "loss": float(sum(losses) / max(1, len(losses)))})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        with torch.no_grad():
            rows = []
            for example in examples:
                features = _clean_qkv_feature_tensor(example, self.config, self.featurizer, self.seed, training=False)
                rows.append(self.module(features).squeeze(-1).tolist())
            return rows

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        if not examples:
            return {"training_trace": self.training_trace, "parameter_count": count_parameters(self.module)}
        slot_entropies: List[float] = []
        gate_values: List[float] = []
        relevant_gate_values: List[float] = []
        irrelevant_gate_values: List[float] = []
        pairwise_sims: List[float] = []
        slot_counts = [0 for _ in range(max(1, self.config.latent_views))]
        with torch.no_grad():
            for example in examples:
                memory = _clean_qkv_memory_slots(example, self.config, self.featurizer, self.seed)
                score_row = self.module(_clean_qkv_feature_tensor(example, self.config, self.featurizer, self.seed, training=False)).squeeze(-1)
                predicted = int(torch.argmax(score_row).item())
                weights, gate = _clean_qkv_reader_weights(example, self.config, self.featurizer, predicted, memory)
                slot_entropies.append(float(-(weights * weights.clamp(min=1e-8).log()).sum().item()))
                gate_values.append(gate)
                if example.label == predicted:
                    relevant_gate_values.append(gate)
                else:
                    irrelevant_gate_values.append(gate)
                for slot in torch.nonzero(weights > 1e-6, as_tuple=False).flatten().tolist():
                    slot_counts[int(slot) % len(slot_counts)] += 1
                pairwise_sims.append(_clean_qkv_pairwise_memory_similarity(memory))
        total = max(1, sum(slot_counts))
        return {
            "training_trace": self.training_trace,
            "parameter_count": count_parameters(self.module),
            "clean_qkv_current_self_attention": "candidate_query_only_no_evidence_tokens",
            "memory_reader": "top_k_activation_memory_routing",
            "memory_gate": "candidate_sequence_gate",
            "routing_entropy_mean": float(sum(slot_entropies) / max(1, len(slot_entropies))),
            "memory_slot_entropy_mean": float(sum(slot_entropies) / max(1, len(slot_entropies))),
            "memory_slot_usage_distribution": [count / total for count in slot_counts],
            "memory_gate_mean": float(sum(gate_values) / max(1, len(gate_values))),
            "memory_gate_relevant_mean": float(sum(relevant_gate_values) / max(1, len(relevant_gate_values))),
            "memory_gate_irrelevant_mean": float(sum(irrelevant_gate_values) / max(1, len(irrelevant_gate_values))),
            "pairwise_hidden_state_similarity": float(sum(pairwise_sims) / max(1, len(pairwise_sims))),
        }


class TextOnlyMultiAgentBaseline(RawLatentSelector):
    pass


class _PairScorer(nn.Module):
    def __init__(self, vector_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vector_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([candidate, evidence, candidate * evidence], dim=0)).squeeze(-1)


class _LatentScoringModule(nn.Module):
    def __init__(self, config: Stage8ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.view_projection = nn.Linear(config.vector_dim, config.hidden_dim)
        self.candidate_projection = nn.Linear(config.vector_dim, config.hidden_dim)
        self.role_embedding = nn.Embedding(max(1, config.roles), config.hidden_dim)
        self.avenue_embedding = nn.Embedding(max(1, config.avenues), config.hidden_dim)
        self.score_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 3),
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, candidate_vectors: torch.Tensor, view_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        view_hidden = self.view_projection(view_vectors)
        view_hidden = view_hidden + _role_avenue_embedding(self.config, self.role_embedding, self.avenue_embedding, view_hidden.device)
        view_hidden = torch.tanh(view_hidden)
        candidate_hidden = torch.tanh(self.candidate_projection(candidate_vectors))
        logits = torch.matmul(candidate_hidden, view_hidden.transpose(0, 1)) / math.sqrt(max(1, view_hidden.shape[-1]))
        weights = _routing_weights(logits, self.config)
        aggregate = torch.matmul(weights, view_hidden)
        features = torch.cat([candidate_hidden, aggregate, candidate_hidden * aggregate], dim=-1)
        return self.score_head(features).squeeze(-1), weights


def build_stage8_selector(config: Stage8ArchitectureConfig, seed: int = 0) -> Stage8Selector:
    kind = config.model_kind
    if kind == "random_candidate":
        return RandomCandidateSelector(seed=seed)
    if kind == "candidate_only":
        return CandidateOnlyBaseline(config, seed=seed)
    if kind == "evidence_only":
        return EvidenceOnlyBaseline(config, seed=seed)
    if kind == "query_only":
        return QueryOnlyBaseline(config, seed=seed)
    if kind == "retrieval_topk":
        return RetrievalTopKBaseline(config, seed=seed)
    if kind == "oracle_evidence_location":
        return RetrievalTopKBaseline(config, seed=seed, oracle=True)
    if kind == "raw_latent_selector":
        return RawLatentSelector(config, seed=seed)
    if kind == "text_only_multi_agent":
        return TextOnlyMultiAgentBaseline(config, seed=seed)
    if kind == "monolithic_transformer":
        return MonolithicSelector(config, seed=seed)
    if kind == "evidence_feature_latent":
        return EvidenceFeatureLatentSelector(config, seed=seed)
    if kind == "clean_qkv_activation_memory":
        return CleanQKVActivationMemorySelector(config, seed=seed)
    if kind == "frozen_latent":
        return TrainableLatentSelector(replace(config, frozen=True), seed=seed)
    if kind == "trainable_latent":
        return TrainableLatentSelector(config, seed=seed)
    raise ValueError(f"unknown Stage 8 model kind: {kind}")


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def estimate_stage8_compute(
    config: Stage8ArchitectureConfig,
    n_blocks: int,
    k_candidates: int = 8,
    avg_tokens_per_block: int = 12,
) -> Dict[str, float | int | str]:
    views = config.latent_views
    hidden = max(1, config.hidden_dim)
    vector = max(1, config.vector_dim)
    if config.model_kind == "monolithic_transformer":
        sequence_tokens = max(1, n_blocks * avg_tokens_per_block + k_candidates * 6 + 16)
        attention_ops = sequence_tokens * sequence_tokens * hidden
        projection_ops = sequence_tokens * hidden * vector
        encoder_passes = 1
        latent_views = 1
    elif config.model_kind in {"trainable_latent", "frozen_latent", "evidence_feature_latent", "clean_qkv_activation_memory", "raw_latent_selector", "text_only_multi_agent"}:
        attention_ops = k_candidates * views * hidden * max(1, config.coord_hops)
        projection_ops = (n_blocks * vector) + (views * vector * hidden)
        encoder_passes = views
        latent_views = views
        if config.model_kind == "clean_qkv_activation_memory":
            memory_slots = views
            memory_writer_ops = n_blocks * vector * max(1, hidden // 8)
            current_self_attention_ops = k_candidates * 8 * 8 * hidden
            memory_reader_ops = k_candidates * memory_slots * hidden * max(1, config.coord_hops)
            gate_ops = k_candidates * hidden
            attention_ops = current_self_attention_ops + memory_reader_ops + gate_ops
            projection_ops = memory_writer_ops + memory_slots * vector * hidden
            encoder_passes = 1
            latent_views = memory_slots
    elif config.model_kind == "query_only":
        attention_ops = k_candidates * vector
        projection_ops = vector * hidden
        encoder_passes = 0
        latent_views = 0
    else:
        attention_ops = k_candidates * n_blocks * vector
        projection_ops = n_blocks * vector
        encoder_passes = 1
        latent_views = 0
    estimated = float(attention_ops + projection_ops)
    return {
        "model_kind": config.model_kind,
        "estimated_forward_compute": estimated,
        "attention_ops": float(attention_ops),
        "projection_ops": float(projection_ops),
        "encoder_passes": int(encoder_passes),
        "latent_views": int(latent_views),
        "parameter_count_estimate": int(_parameter_count_estimate(config)),
    }


def _view_matrix(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    featurizer: Stage8HashFeaturizer,
    seed: int,
) -> torch.Tensor:
    views = config.latent_views
    if views <= 0:
        views = 1
    buckets: List[List[torch.Tensor]] = [[] for _ in range(views)]
    for block_index, block in enumerate(example.evidence_blocks):
        vector = featurizer.text_vector(block)
        for bucket in _assigned_buckets(config, block, block_index, views, seed, example.example_id):
            buckets[bucket].append(vector)
    rows = []
    for bucket in buckets:
        if bucket:
            rows.append(torch.stack(bucket).mean(dim=0))
        else:
            rows.append(torch.zeros(featurizer.vector_dim))
    return torch.stack(rows)


def _assigned_buckets(
    config: Stage8ArchitectureConfig,
    block: str,
    block_index: int,
    views: int,
    seed: int,
    example_id: str,
) -> Tuple[int, ...]:
    chunking = config.chunking
    if chunking == "contiguous":
        return (block_index % views,)
    if chunking == "random":
        return (_stable_hash(seed, example_id, block_index) % views,)
    if chunking in {"hashed", "learned_router"}:
        return (_stable_hash(block) % views,)
    if chunking == "semantic_key":
        key = _semantic_key(block)
        return (_stable_hash(key) % views,)
    if chunking == "overlapping":
        first = block_index % views
        return (first, (first + 1) % views)
    if chunking == "hierarchical":
        coarse = block_index % max(1, config.roles)
        avenue = (_stable_hash(block) % max(1, config.avenues))
        return ((coarse * max(1, config.avenues) + avenue) % views,)
    return (block_index % views,)


def _semantic_key(block: str) -> str:
    for token in _tokens(block):
        if "_key_" in token or "_entity_" in token or "_anchor_" in token or "_source_" in token:
            return token
    tokens = _tokens(block)
    return tokens[0] if tokens else block


def _routing_weights(logits: torch.Tensor, config: Stage8ArchitectureConfig) -> torch.Tensor:
    top_k = int(config.top_k_views)
    if config.attention_routing in {"top_k_view_attention", "two_stage_retrieve_then_score", "coarse_to_fine_routing"} or top_k > 0:
        k = _effective_top_k(config, logits.shape[-1])
        values, indices = logits.topk(k, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, indices, torch.softmax(values, dim=-1))
        return weights
    if config.attention_routing == "sparsemax_routing":
        return _sparsemax(logits, dim=-1)
    return torch.softmax(logits, dim=-1)


def _effective_top_k(config: Stage8ArchitectureConfig, available: int) -> int:
    if config.top_k_views > 0:
        return min(max(1, int(config.top_k_views)), available)
    if config.attention_routing in {"top_k_view_attention", "two_stage_retrieve_then_score"}:
        return min(max(1, int(math.sqrt(max(1, available)))), available)
    return available


def _regularization_loss(config: Stage8ArchitectureConfig, routing: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, dtype=routing.dtype, device=routing.device)
    if "entropy_regularization_on_routing" in config.regularizers:
        entropy = -(routing * routing.clamp(min=1e-8).log()).sum(dim=-1).mean()
        loss = loss - 0.001 * entropy
    if "load_balancing_across_avenues" in config.regularizers and routing.numel():
        usage = routing.mean(dim=0)
        target = torch.full_like(usage, 1.0 / max(1, usage.numel()))
        loss = loss + 0.01 * F.mse_loss(usage, target)
    return loss


def _role_avenue_embedding(
    config: Stage8ArchitectureConfig,
    role_embedding: nn.Embedding,
    avenue_embedding: nn.Embedding,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for role in range(max(1, config.roles)):
        for avenue in range(max(1, config.avenues)):
            rows.append(role_embedding(torch.tensor(role, device=device)) + avenue_embedding(torch.tensor(avenue, device=device)))
    return torch.stack(rows[: config.latent_views])


def _permute_roles(views: torch.Tensor, config: Stage8ArchitectureConfig) -> torch.Tensor:
    roles = max(1, config.roles)
    avenues = max(1, config.avenues)
    if views.shape[0] != roles * avenues:
        return views.flip(0)
    reshaped = views.reshape(roles, avenues, -1).flip(0)
    return reshaped.reshape(roles * avenues, -1)


def _permute_avenues(views: torch.Tensor, config: Stage8ArchitectureConfig) -> torch.Tensor:
    roles = max(1, config.roles)
    avenues = max(1, config.avenues)
    if views.shape[0] != roles * avenues:
        return views.flip(0)
    reshaped = views.reshape(roles, avenues, -1).flip(1)
    return reshaped.reshape(roles * avenues, -1)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, descending=True, dim=dim).values
    range_values = torch.arange(1, shifted.shape[dim] + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1 for _ in shifted.shape]
    view_shape[dim] = -1
    range_values = range_values.view(view_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1 + range_values * sorted_logits > cumulative
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumulative.gather(dim, k - 1) - 1) / k.to(dtype=logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def _parameter_count_estimate(config: Stage8ArchitectureConfig) -> int:
    if config.model_kind == "monolithic_transformer":
        return int(config.vector_dim * config.hidden_dim * 3 + config.hidden_dim)
    if config.model_kind in {"trainable_latent", "frozen_latent", "evidence_feature_latent", "clean_qkv_activation_memory"}:
        return int(
            config.vector_dim * config.hidden_dim * 2
            + config.roles * config.hidden_dim
            + config.avenues * config.hidden_dim
            + config.hidden_dim * config.hidden_dim
        )
    return 0


def _stage8b3_feature_tensor(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    seed: int,
    training: bool = False,
) -> torch.Tensor:
    block_rows = [_block_token_row(block) for block in example.evidence_blocks]
    ranks = _source_rank_map(block_rows)
    query_tokens = set(_content_tokens(example.query))
    views = max(1, config.latent_views)
    rows: List[List[float]] = []
    hidden_scale = 0.0 if bool(example.metadata.get("hidden_state_shuffle")) else 1.0
    dropout_rng = random.Random(_stable_hash(seed, config.config_id, example.example_id, "chunk-dropout"))
    for candidate in example.candidates:
        candidate_tokens = _candidate_identity_tokens(candidate)
        total_hits = 0.0
        support_hits = 0.0
        contradiction_hits = 0.0
        positive_role_hits = 0.0
        negative_role_hits = 0.0
        terminal_hits = 0.0
        query_overlap_hits = 0.0
        priority_score = 0.0
        rank1_claim = 0.0
        topk_scores: List[float] = []
        hit_views: set[int] = set()
        view_mass = [0.0 for _ in range(views)]
        for block_index, block_row in enumerate(block_rows):
            if training and "chunk_dropout" in config.regularizers and dropout_rng.random() < min(0.35, max(0.0, config.dropout)):
                continue
            has_candidate = _any_token_hit(candidate_tokens, block_row["tokens"])
            query_overlap = len(query_tokens.intersection(block_row["tokens"]))
            retrieval_score = (1.0 if has_candidate else 0.0) + min(1.0, query_overlap / 4.0)
            topk_scores.append(retrieval_score)
            if not has_candidate:
                continue
            total_hits += 1.0
            contradiction = bool(block_row["contradiction"])
            support = not contradiction
            if support:
                support_hits += 1.0
            else:
                contradiction_hits += 1.0
            if block_row["positive"]:
                positive_role_hits += 1.0
            if block_row["negative"]:
                negative_role_hits += 1.0
            if block_row["terminal"]:
                terminal_hits += 1.0
            query_overlap_hits += min(1.0, query_overlap / max(1.0, len(query_tokens)))
            rank = _rank_for_claim(block_row, ranks)
            if rank is not None:
                priority_score = max(priority_score, 1.0 / max(1.0, float(rank)))
                if rank == 1:
                    rank1_claim = 1.0
            for view in _stage8b3_assigned_views(config, block_row, block_index, seed, example.example_id):
                hit_views.add(view)
                view_mass[view % views] += 1.0
        total_blocks = max(1.0, float(len(example.evidence_blocks)))
        top_k = max(1, min(int(config.top_k_views or 4), len(topk_scores) or 1))
        topk_mean = sum(sorted(topk_scores, reverse=True)[:top_k]) / float(top_k) if topk_scores else 0.0
        view_entropy = _distribution_entropy(view_mass) / math.log(max(2, len(view_mass)))
        unique_view_fraction = len(hit_views) / max(1.0, float(views))
        candidate_query_overlap = len(set(candidate_tokens).intersection(query_tokens)) / max(1.0, float(len(candidate_tokens)))
        support_minus_contradiction = support_hits - contradiction_hits
        feature_row = [
            total_hits / total_blocks,
            support_hits / total_blocks,
            contradiction_hits / total_blocks,
            support_minus_contradiction / total_blocks,
            query_overlap_hits / total_blocks,
            topk_mean,
            rank1_claim,
            priority_score,
            terminal_hits / total_blocks,
            unique_view_fraction,
            view_entropy,
            positive_role_hits / total_blocks,
            negative_role_hits / total_blocks,
            candidate_query_overlap,
            1.0 if total_hits > 0 else 0.0,
            1.0,
        ]
        rows.append([float(value * hidden_scale if index < 15 else value) for index, value in enumerate(feature_row)])
    return torch.tensor(rows, dtype=torch.float32)


def _stage8b3_view_weights(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    candidate_index: int,
    seed: int,
) -> torch.Tensor:
    views = max(1, config.latent_views)
    candidate = example.candidates[candidate_index] if 0 <= candidate_index < len(example.candidates) else ""
    candidate_tokens = _candidate_identity_tokens(candidate)
    weights = torch.zeros(views, dtype=torch.float32)
    for block_index, block in enumerate(example.evidence_blocks):
        row = _block_token_row(block)
        mass = 1.0 if _any_token_hit(candidate_tokens, row["tokens"]) else 0.05
        for view in _stage8b3_assigned_views(config, row, block_index, seed, example.example_id):
            weights[view % views] += mass
    if bool(example.metadata.get("hidden_state_shuffle")):
        weights = torch.roll(weights.flip(0), shifts=1)
    if weights.sum().item() <= 0:
        return torch.full((views,), 1.0 / float(views), dtype=torch.float32)
    return weights / weights.sum()


def _stage8b3_pairwise_view_similarity(example: Stage8Example, config: Stage8ArchitectureConfig, seed: int) -> float:
    views = max(1, config.latent_views)
    rows = torch.zeros(views, 8, dtype=torch.float32)
    for block_index, block in enumerate(example.evidence_blocks):
        row = _block_token_row(block)
        features = torch.tensor(
            [
                1.0,
                float(row["positive"]),
                float(row["negative"]),
                float(row["contradiction"]),
                float(row["terminal"]),
                float(len([token for token in row["tokens"] if "_entity_" in token or "_item_" in token])),
                float(len([token for token in row["tokens"] if "_source_" in token])),
                float(len([token for token in row["tokens"] if "_value_" in token or "_action_" in token])),
            ],
            dtype=torch.float32,
        )
        for view in _stage8b3_assigned_views(config, row, block_index, seed, example.example_id):
            rows[view % views] += features
    if rows.shape[0] < 2:
        return 1.0
    normalized = F.normalize(rows, dim=-1)
    sims = []
    for i in range(normalized.shape[0]):
        for j in range(i + 1, normalized.shape[0]):
            sims.append(float(torch.dot(normalized[i], normalized[j]).item()))
    return sum(sims) / max(1, len(sims))


def _stage8b3_assigned_views(
    config: Stage8ArchitectureConfig,
    block_row: Mapping[str, object],
    block_index: int,
    seed: int,
    example_id: str,
) -> Tuple[int, ...]:
    views = max(1, config.latent_views)
    tokens = block_row["tokens"] if isinstance(block_row.get("tokens"), set) else set()
    tags: List[str] = []
    if block_row.get("negative") or block_row.get("contradiction"):
        tags.append("contradiction")
    if block_row.get("terminal"):
        tags.append("terminal")
    if any("_source_" in token for token in tokens):
        tags.append("source")
    if any("_entity_" in token or "_item_" in token or "_case_" in token for token in tokens):
        tags.append("entity")
    if any("_rule_" in token for token in tokens):
        tags.append("rule")
    if any("_anchor_" in token for token in tokens):
        tags.append("sparse")
    if not tags:
        tags.append("general")
    if config.memory_compression in {"slot_attention_compression", "recurrent_memory_slots"} or config.coordinator == "explicit_view_objective_assignment":
        primary = _stable_hash(tags[0]) % views
    elif config.chunking == "hierarchical" or config.coordinator == "hierarchical_coordinator":
        role = _stable_hash(tags[0]) % max(1, config.roles)
        avenue = block_index % max(1, config.avenues)
        primary = (role * max(1, config.avenues) + avenue) % views
    elif config.chunking == "semantic_key":
        primary = _stable_hash(tags[0], block_index % max(1, config.avenues)) % views
    else:
        primary = _assigned_buckets(config, " ".join(sorted(tokens)), block_index, views, seed, example_id)[0]
    if config.chunking == "overlapping" or "load_balancing_across_avenues" in config.regularizers:
        return (primary, (primary + 1 + (_stable_hash(example_id, block_index) % max(1, views - 1))) % views)
    return (primary,)


def _block_token_row(block: str) -> Dict[str, object]:
    tokens = set(_tokens(block))
    lowered = block.lower()
    contradiction = any(marker in lowered for marker in _CONTRADICTION_MARKERS)
    negative = contradiction or any(marker in lowered for marker in _NEGATIVE_MARKERS)
    positive = any(marker in lowered for marker in _POSITIVE_MARKERS) and not negative
    terminal = any(marker in lowered for marker in _TERMINAL_MARKERS)
    return {"tokens": tokens, "text": lowered, "contradiction": contradiction, "negative": negative, "positive": positive, "terminal": terminal}


def _candidate_identity_tokens(candidate: str) -> List[str]:
    return [
        token
        for token in _tokens(candidate)
        if "_" in token and not token.startswith("candidate_")
    ]


def _content_tokens(text: str) -> List[str]:
    return [token for token in _tokens(text) if token not in _STOP_TOKENS and len(token) > 2]


def _any_token_hit(needles: Sequence[str], haystack: set[str]) -> bool:
    return any(token in haystack for token in needles)


def _source_rank_map(block_rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for row in block_rows:
        tokens = row["tokens"] if isinstance(row.get("tokens"), set) else set()
        sources = [token for token in tokens if "_source_" in token]
        if not sources:
            continue
        numbers = [int(token) for token in tokens if token.isdigit()]
        text = str(row.get("text", ""))
        if "rank" not in text and "priority" not in text:
            continue
        rank = min(numbers) if numbers else 999
        for source in sources:
            ranks[source] = min(rank, ranks.get(source, rank))
    return ranks


def _rank_for_claim(block_row: Mapping[str, object], ranks: Mapping[str, int]) -> int | None:
    tokens = block_row["tokens"] if isinstance(block_row.get("tokens"), set) else set()
    for source in [token for token in tokens if "_source_" in token]:
        if source in ranks:
            return int(ranks[source])
    return None


def _distribution_entropy(values: Sequence[float]) -> float:
    total = sum(float(value) for value in values)
    if total <= 0:
        return 0.0
    return -sum((float(value) / total) * math.log(max(1e-12, float(value) / total)) for value in values if value > 0)


def _clean_qkv_feature_tensor(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    featurizer: Stage8HashFeaturizer,
    seed: int,
    training: bool = False,
) -> torch.Tensor:
    del training
    memory = _clean_qkv_memory_slots(example, config, featurizer, seed)
    current_query = featurizer.text_vector(example.query)
    memory_disabled = bool(
        example.metadata.get("memory_disabled")
        or example.metadata.get("memory_gate_forced_closed")
        or example.metadata.get("hidden_state_shuffle")
    )
    activations = torch.stack([slot["activation"] for slot in memory]) if memory else torch.zeros(0, featurizer.vector_dim)  # type: ignore[index]
    support = torch.tensor([float(slot["support"]) for slot in memory], dtype=torch.float32)
    contradiction = torch.tensor([float(slot["contradiction"]) for slot in memory], dtype=torch.float32)
    positive = torch.tensor([float(slot["positive"]) for slot in memory], dtype=torch.float32)
    negative = torch.tensor([float(slot["negative"]) for slot in memory], dtype=torch.float32)
    terminal = torch.tensor([float(slot["terminal"]) for slot in memory], dtype=torch.float32)
    rank1 = torch.tensor([float(slot["rank1_claim"]) for slot in memory], dtype=torch.float32)
    priority = torch.tensor([float(slot["priority_score"]) for slot in memory], dtype=torch.float32)
    type_indices = [int(slot["type_index"]) for slot in memory]
    query_scores = torch.clamp(torch.mv(activations, current_query), min=0.0) if memory else torch.zeros(0, dtype=torch.float32)
    rows: List[List[float]] = []
    for candidate_index, candidate in enumerate(example.candidates):
        candidate_vector = featurizer.text_vector(candidate)
        current_hidden = F.normalize(current_query + candidate_vector, dim=0)
        top_scores = torch.clamp(torch.mv(activations, candidate_vector), min=0.0) if memory else torch.zeros(0, dtype=torch.float32)
        weights, gate = _clean_qkv_reader_weights_from_scores(example, config, top_scores)
        if memory_disabled or not memory:
            memory_update = torch.zeros_like(current_hidden)
            support_score = contradiction_score = type_entropy = query_memory = 0.0
            rank1_claim = priority_score = terminal_score = slot_entropy = 0.0
            positive_score = negative_score = 0.0
            topk_mean = 0.0
            memory_presence = 0.0
        else:
            memory_update = gate * torch.mv(activations.transpose(0, 1), weights)
            slot_entropy = float(-(weights * weights.clamp(min=1e-8).log()).sum().item()) / math.log(max(2, len(memory)))
            top_k = max(1, min(int(config.top_k_views or 4), len(top_scores)))
            topk_mean = float(top_scores.topk(top_k).values.mean().item()) if top_scores.numel() else 0.0
            weighted_scores = weights * top_scores
            support_score = float((weighted_scores * support).sum().item())
            contradiction_score = float((weighted_scores * contradiction).sum().item())
            positive_score = float((weighted_scores * positive).sum().item())
            negative_score = float((weighted_scores * negative).sum().item())
            terminal_score = float((weighted_scores * terminal).sum().item())
            rank1_claim = float((weighted_scores * rank1).sum().item())
            priority_score = float((top_scores * priority).max().item()) if top_scores.numel() else 0.0
            query_memory = float((weights * query_scores).sum().item()) if query_scores.numel() else 0.0
            type_mass: Dict[int, float] = {}
            for i, slot_type in enumerate(type_indices):
                type_mass[slot_type] = type_mass.get(slot_type, 0.0) + float(weights[i])
            type_entropy = _distribution_entropy(list(type_mass.values())) / math.log(max(2, len(type_mass)))
            memory_presence = 1.0
        clean_current_score = float(torch.dot(current_hidden, current_query).item())
        gated_alignment = float(torch.dot(current_hidden, memory_update).item()) if memory_update.numel() else 0.0
        support_minus_contradiction = support_score - contradiction_score
        feature_row = [
            gate,
            topk_mean,
            support_score,
            contradiction_score,
            support_minus_contradiction,
            positive_score,
            negative_score,
            terminal_score,
            rank1_claim,
            priority_score,
            query_memory,
            slot_entropy,
            type_entropy,
            gated_alignment,
            clean_current_score,
            memory_presence,
            1.0 if topk_mean > 0.05 else 0.0,
            1.0,
        ]
        rows.append([float(value) for value in feature_row])
    return torch.tensor(rows, dtype=torch.float32)


def _clean_qkv_memory_slots(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    featurizer: Stage8HashFeaturizer,
    seed: int,
) -> List[Dict[str, object]]:
    block_rows = [_block_token_row(block) for block in example.evidence_blocks]
    ranks = _source_rank_map(block_rows)
    slots: List[Dict[str, object]] = []
    for block_index, (block, row) in enumerate(zip(example.evidence_blocks, block_rows)):
        slot_type = _clean_qkv_slot_type(row)
        type_vector = _clean_qkv_type_vector(featurizer.vector_dim, slot_type)
        provenance_vector = _clean_qkv_type_vector(featurizer.vector_dim, _stable_hash(seed, example.example_id, block_index) % 13)
        activation = F.normalize(featurizer.text_vector(block) + 0.15 * type_vector + 0.05 * provenance_vector, dim=0)
        rank = _rank_for_claim(row, ranks)
        slots.append(
            {
                "activation": activation,
                "type_index": slot_type,
                "support": 1.0 if not row["contradiction"] else 0.0,
                "contradiction": 1.0 if row["contradiction"] else 0.0,
                "positive": 1.0 if row["positive"] else 0.0,
                "negative": 1.0 if row["negative"] else 0.0,
                "terminal": 1.0 if row["terminal"] else 0.0,
                "rank1_claim": 1.0 if rank == 1 else 0.0,
                "priority_score": 1.0 / max(1.0, float(rank)) if rank is not None else 0.0,
                "block_index": block_index,
            }
        )
    if bool(example.metadata.get("memory_slot_permutation")) and slots:
        keyed = [(_stable_hash(seed, example.example_id, "memory-permute", slot["block_index"]), slot) for slot in slots]
        slots = [slot for _, slot in sorted(keyed, key=lambda item: item[0])]
    return slots


def _clean_qkv_reader_weights(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    featurizer: Stage8HashFeaturizer,
    candidate_index: int,
    memory: Sequence[Mapping[str, object]],
) -> Tuple[torch.Tensor, float]:
    if not memory:
        return torch.zeros(0, dtype=torch.float32), 0.0
    candidate = example.candidates[candidate_index] if 0 <= candidate_index < len(example.candidates) else ""
    candidate_vector = featurizer.text_vector(candidate)
    scores = _clean_qkv_slot_scores(candidate_vector, memory)
    return _clean_qkv_reader_weights_from_scores(example, config, scores)


def _clean_qkv_reader_weights_from_scores(
    example: Stage8Example,
    config: Stage8ArchitectureConfig,
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    if not scores.numel():
        return torch.zeros(0, dtype=torch.float32), 0.0
    top_k = max(1, min(int(config.top_k_views or 4), int(scores.numel())))
    values, indices = scores.topk(top_k)
    weights = torch.zeros_like(scores)
    weights.scatter_(0, indices, torch.softmax(values * 6.0, dim=0))
    top_signal = float(values.max().item()) if values.numel() else 0.0
    gate = 1.0 / (1.0 + math.exp(-12.0 * (top_signal - 0.06)))
    if bool(example.metadata.get("memory_disabled") or example.metadata.get("memory_gate_forced_closed") or example.metadata.get("hidden_state_shuffle")):
        gate = 0.0
    if bool(example.metadata.get("memory_gate_forced_open")):
        gate = 1.0
    return weights, float(gate)


def _clean_qkv_slot_scores(candidate_vector: torch.Tensor, memory: Sequence[Mapping[str, object]]) -> torch.Tensor:
    if not memory:
        return torch.zeros(0, dtype=torch.float32)
    values = [max(0.0, float(torch.dot(candidate_vector, slot["activation"]).item())) for slot in memory]  # type: ignore[index]
    return torch.tensor(values, dtype=torch.float32)


def _clean_qkv_pairwise_memory_similarity(memory: Sequence[Mapping[str, object]]) -> float:
    if len(memory) < 2:
        return 1.0
    activations = torch.stack([slot["activation"] for slot in memory])  # type: ignore[index]
    normalized = F.normalize(activations, dim=-1)
    sims = torch.matmul(normalized, normalized.transpose(0, 1))
    upper = torch.triu_indices(sims.shape[0], sims.shape[1], offset=1)
    values = sims[upper[0], upper[1]]
    return float(values.mean().item()) if values.numel() else 1.0


def _clean_qkv_slot_type(row: Mapping[str, object]) -> int:
    tokens = row["tokens"] if isinstance(row.get("tokens"), set) else set()
    if row.get("contradiction") or row.get("negative"):
        return 4
    if any("_entity_" in token or "_item_" in token or "_case_" in token for token in tokens):
        return 1
    if any("_source_" in token or "_relation_" in token or token.startswith("rel_") for token in tokens):
        return 2
    if row.get("positive") or row.get("terminal"):
        return 3
    if any("_rule_" in token or "_exception_" in token for token in tokens):
        return 5
    return 0


def _clean_qkv_type_vector(vector_dim: int, type_index: int) -> torch.Tensor:
    cached = _CLEAN_QKV_TYPE_VECTOR_CACHE.get((vector_dim, type_index))
    if cached is not None:
        return cached
    vector = torch.zeros(vector_dim, dtype=torch.float32)
    for offset in range(4):
        vector[_stable_hash("clean-qkv-type", type_index, offset) % vector_dim] = 1.0
    normalized = F.normalize(vector, dim=0)
    _CLEAN_QKV_TYPE_VECTOR_CACHE[(vector_dim, type_index)] = normalized
    return normalized


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


class CounterLike:
    def __init__(self, values: Iterable[int]) -> None:
        counts: Dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        self.counts = counts

    def most_common(self, n: int) -> List[Tuple[int, int]]:
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))[:n]


_CONTRADICTION_MARKERS = ("exception", "block", "blocks", "reject", "rejects", "disallow", "disallows", "contradict")
_NEGATIVE_MARKERS = ("false", "near-match", "foil", "distractor")
_POSITIVE_MARKERS = ("support", "supports", "favors", "permits", "confirms", "authorizes", "verifies", "signal", "clue", "claim", "fact")
_TERMINAL_MARKERS = ("terminal", "terminates", "final", "resulting")
_STOP_TOKENS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "candidate",
    "value",
    "item",
    "action",
    "choose",
    "select",
    "which",
    "what",
    "find",
    "return",
    "matching",
    "matches",
    "query",
    "stage8",
}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_<>]+")
