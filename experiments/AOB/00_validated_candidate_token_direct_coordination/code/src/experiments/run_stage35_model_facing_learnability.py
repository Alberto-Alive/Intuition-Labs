from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_VALUE_PAIRS,
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    SANITIZED_CANDIDATE_REPRESENTATION,
    SANITIZED_DATASET_SOURCE,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    format_candidate_block,
    model_record_from_example,
)
from src.experiments.architecture_search import _candidate_specs, _fit_baselines
from src.experiments.real_shared_weight_latent_coordination import (
    CandidatewiseScoringModel,
    fit_context_baseline,
    predict_context_baseline,
    _text_tokens,
)
from src.experiments.run_stage3_gpu_hard_validation import ARCHITECTURE, _clear_cuda, _configure_cuda, _stage_from_config
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    _accuracy,
    run_diagnostics,
    stage34_dataset_config,
)
from src.experiments.run_stage34_candidate_sanitization_smoke import _run_seed as _run_stage34_smoke_seed


DEFAULT_CONFIG = "configs/stage34_candidate_sanitization_cuda_smoke.json"
DEFAULT_OUTPUT = "results/stage35_model_facing_learnability.json"
DEFAULT_REPORT = "reports/STAGE35_MODEL_FACING_LEARNABILITY.md"
LEARNABILITY_TARGET = 0.50


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.5 model-facing learnability diagnosis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_stage35(config, config_path=config_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage35(config: Dict[str, object], config_path: Path | None = None) -> Dict[str, object]:
    base_config = _stage35_base_config(config)
    stage = _stage_from_config(base_config)
    dataset_config = stage34_dataset_config(base_config)
    splits = build_multiview_code_patch_splits(dataset_config, seed=0, repo_root=Path("."))
    inspection = _inspect_model_records(splits)
    positive_controls = _positive_control_suite(splits, base_config, suite_name="stage34_sanitized_v3")
    best_control = _best_test_accuracy(positive_controls)
    model_facing_learnable = best_control >= LEARNABILITY_TARGET

    variants = []
    if not model_facing_learnable:
        variant_config = _stage34b_config(config)
        variant_diagnostics = run_diagnostics(variant_config, config_path=config_path)
        variant_splits = build_multiview_code_patch_splits(stage34_dataset_config(variant_config), seed=0, repo_root=Path("."))
        variant_positive = _positive_control_suite(variant_splits, variant_config, suite_name="stage34b_balanced_categories_v3")
        variant_smoke = None
        if _variant_gates_ok(variant_diagnostics):
            variant_smoke = _run_tiny_smoke(variant_config, variant_splits)
        variants.append(
            {
                "name": BALANCED_34B_CANDIDATE_REPRESENTATION,
                "dataset_source": BALANCED_34B_DATASET_SOURCE,
                "dataset_diagnostics": variant_diagnostics,
                "positive_controls": variant_positive,
                "best_positive_control_test_accuracy": _best_test_accuracy(variant_positive),
                "smoke": variant_smoke,
                "fixes_learnability": _best_test_accuracy(variant_positive) >= LEARNABILITY_TARGET,
                "keeps_leakage_baselines_near_chance": _variant_gates_ok(variant_diagnostics),
            }
        )

    stage34b_near_chance = bool(variants and variants[0].get("keeps_leakage_baselines_near_chance") and _smoke_candidate_controls_ok(variants[0].get("smoke")))
    full_validation_justified = bool(
        variants
        and variants[0].get("fixes_learnability")
        and stage34b_near_chance
        and variants[0].get("smoke", {}).get("summary", {}).get("smoke_passed", False)
        and float(variants[0].get("smoke", {}).get("summary", {}).get("mean_test_accuracy", {}).get("trainable", 0.0)) > 0.18
    )
    return {
        "metadata": {
            "stage": "stage3.5_model_facing_learnability",
            "config_path": str(config_path) if config_path else None,
            "base_dataset": SANITIZED_DATASET_SOURCE,
            "architecture": ARCHITECTURE,
            "architecture_changes": "none",
            "scope": "diagnosis plus one-seed smoke only; no full 10-seed validation",
        },
        "inspection": inspection,
        "stage34_positive_controls": positive_controls,
        "stage34_best_positive_control_test_accuracy": best_control,
        "stage34_model_facing_inputs_learnable": model_facing_learnable,
        "stage34_over_sanitized": not model_facing_learnable,
        "stage34b_variants": variants,
        "summary": {
            "model_facing_inputs_learnable": model_facing_learnable,
            "best_positive_control_accuracy": best_control,
            "stage34_over_sanitized": not model_facing_learnable,
            "stage34b_representation_fixes_learnability": bool(variants and variants[0].get("fixes_learnability")),
            "stage34b_keeps_candidate_only_and_view_masked_near_chance": stage34b_near_chance,
            "full_validation_justified": full_validation_justified,
        },
    }


def _stage35_base_config(config: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(config))
    out["stage"] = {
        **dict(out["stage"]),
        "n_train": 256,
        "n_dev": 128,
        "n_test": 256,
        "seeds": [0],
    }
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": SANITIZED_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_sanitized_v3_stage35",
        "candidate_representation": SANITIZED_CANDIDATE_REPRESENTATION,
    }
    return out


def _stage34b_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage35_base_config(config)
    out["dataset"] = BALANCED_34B_DATASET_SOURCE
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage35",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _inspect_model_records(splits: Dict[str, Sequence[MultiViewTaskExample]], limit: int = 50) -> Dict[str, object]:
    samples_by_split = {}
    flags = {
        "candidate_text_nonempty": True,
        "private_views_nonempty": True,
        "candidate_texts_have_meaningful_differences": True,
        "oracle_metadata_separated": True,
        "all_candidates_not_indistinguishable": True,
    }
    for split, examples in splits.items():
        rows = []
        for example in list(examples)[:limit]:
            record = model_record_from_example(example)
            oracle = example_oracle_metadata(example)
            candidate_texts = [candidate.text for candidate in example.candidates]
            candidate_without_slot = [_normalize_candidate_text_for_indistinguishability(text) for text in candidate_texts]
            flags["candidate_text_nonempty"] = flags["candidate_text_nonempty"] and all(bool(text.strip()) for text in candidate_texts)
            flags["private_views_nonempty"] = flags["private_views_nonempty"] and all(bool(view.text.strip()) for view in example.views)
            flags["candidate_texts_have_meaningful_differences"] = flags["candidate_texts_have_meaningful_differences"] and len(set(candidate_without_slot)) > 1
            flags["all_candidates_not_indistinguishable"] = flags["all_candidates_not_indistinguishable"] and len({(candidate.text, candidate.attributes) for candidate in example.candidates}) > 1
            flags["oracle_metadata_separated"] = flags["oracle_metadata_separated"] and example.oracle_metadata is not None
            rows.append(
                {
                    "id": example.id,
                    "gold_index": int(example.label),
                    "model_record": record,
                    "oracle_metadata": oracle,
                    "candidates_before_sanitization": [
                        {
                            "candidate_index": index,
                            "oracle_patch_values": values,
                            "oracle_tuple": oracle.get("candidate_tuples", [])[index],
                        }
                        for index, values in enumerate(oracle.get("candidate_patch_values", []))
                    ],
                    "candidates_after_sanitization": [
                        {
                            "candidate_index": index,
                            "text": candidate.text,
                            "source_path": candidate.source_path,
                            "attributes": list(candidate.attributes),
                        }
                        for index, candidate in enumerate(example.candidates)
                    ],
                }
            )
        samples_by_split[split] = rows
    return {
        "checks": flags,
        "interpretation": _inspection_interpretation(flags),
        "samples_by_split": samples_by_split,
    }


def _normalize_candidate_text_for_indistinguishability(text: str) -> str:
    text = re_sub(r"Patch option \d+", "Patch option <slot>", text)
    text = re_sub(r"slot_marker: SLOT_\d+", "slot_marker: <slot>", text)
    text = re_sub(r"sanitized_uid: [0-9a-f]+", "sanitized_uid: <uid>", text)
    return text


def re_sub(pattern: str, repl: str, text: str) -> str:
    import re

    return re.sub(pattern, repl, text)


def _inspection_interpretation(flags: Dict[str, bool]) -> str:
    if all(flags.values()):
        return "model-facing records contain nontrivial distinguishable inputs"
    return "model-facing records appear over-sanitized or indistinguishable after removing slot/uid artifacts"


def _positive_control_suite(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    config: Dict[str, object],
    suite_name: str,
) -> Dict[str, Dict[str, float]]:
    training = MLPTrainingConfig(epochs=5, batch_size=32, lr=0.002, weight_decay=0.0001, patience=3, hidden_dims=(64,))
    controls = {
        "all_view_bow_candidate_scorer": HashedCandidateScorer(
            text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
            feature_dim=256,
            training=training,
            seed=11,
            hidden_dims=(64,),
        ),
        "all_view_tfidf_logistic_candidate_scorer": TfidfCandidateScorer(
            text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
            max_features=512,
            training=MLPTrainingConfig(epochs=8, batch_size=32, lr=0.01, weight_decay=0.0001, patience=3, hidden_dims=()),
            seed=12,
        ),
        "small_cross_encoder_all_views_candidate": TinyCrossEncoderCandidateScorer(
            text_builder=lambda example, index: _all_views_text(example) + "\n" + example.candidates[index].text,
            seed=13,
            epochs=4,
        ),
        "candidate_pair_compatibility_mlp": CompatibilityFeatureScorer(training=training, seed=14),
        "single_agent_full_context_tiny_transformer": FullContextTinyTransformerControl(config=config, seed=15),
    }
    out = {name: _fit_eval_control(control, splits) for name, control in controls.items()}
    for left in range(4):
        for right in range(left + 1, 4):
            control = HashedCandidateScorer(
                text_builder=lambda example, index, left=left, right=right: _selected_views_text(example, [left, right])
                + "\n"
                + example.candidates[index].text,
                feature_dim=256,
                training=training,
                seed=20 + left * 4 + right,
                hidden_dims=(64,),
            )
            out[f"pairwise_view_text_roles_{left}_{right}"] = _fit_eval_control(control, splits)
    out["_suite"] = {"name": suite_name}  # type: ignore[dict-item]
    return out


def _fit_eval_control(control, splits: Dict[str, Sequence[MultiViewTaskExample]]) -> Dict[str, float]:
    control.fit(splits["train"], splits["dev"])
    return {
        split: _accuracy(control.predict(rows), np.asarray([example.label for example in rows], dtype=np.int64))
        for split, rows in splits.items()
    }


class HashedCandidateScorer:
    def __init__(
        self,
        text_builder: Callable[[MultiViewTaskExample, int], str],
        feature_dim: int,
        training: MLPTrainingConfig,
        seed: int,
        hidden_dims: Iterable[int],
    ) -> None:
        self.text_builder = text_builder
        self.feature_dim = int(feature_dim)
        self.training = training
        self.seed = int(seed)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.model: CandidatewiseScoringModel | None = None

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        torch.manual_seed(self.seed)
        x_train = self._features(train_examples)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = self._features(dev_examples)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = _fit_candidatewise_model(x_train, y_train, x_dev, y_dev, self.training, self.hidden_dims, self.seed)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("control is not fit")
        return _predict_candidatewise_model(self.model, self._features(examples))

    def _features(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        rows = []
        for example in examples:
            rows.append([_hashed_feature_vector(self.text_builder(example, index), self.feature_dim) for index in range(len(example.candidates))])
        return np.asarray(rows, dtype=np.float32)


class TfidfCandidateScorer:
    def __init__(
        self,
        text_builder: Callable[[MultiViewTaskExample, int], str],
        max_features: int,
        training: MLPTrainingConfig,
        seed: int,
    ) -> None:
        self.text_builder = text_builder
        self.max_features = int(max_features)
        self.training = training
        self.seed = int(seed)
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.model: CandidatewiseScoringModel | None = None

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        texts = [self.text_builder(example, index) for example in train_examples for index in range(len(example.candidates))]
        counts = Counter(token for text in texts for token in set(_text_tokens(text)))
        self.vocab = {token: index for index, (token, _count) in enumerate(counts.most_common(self.max_features))}
        df = np.ones(len(self.vocab), dtype=np.float32)
        for text in texts:
            for token in set(_text_tokens(text)):
                if token in self.vocab:
                    df[self.vocab[token]] += 1.0
        self.idf = np.log((1.0 + len(texts)) / df).astype(np.float32)
        x_train = self._features(train_examples)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = self._features(dev_examples)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = _fit_candidatewise_model(x_train, y_train, x_dev, y_dev, self.training, self.training.hidden_dims, self.seed)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("control is not fit")
        return _predict_candidatewise_model(self.model, self._features(examples))

    def _features(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("tf-idf control is not fit")
        rows = []
        for example in examples:
            per_candidate = []
            for index in range(len(example.candidates)):
                vec = np.zeros(len(self.vocab), dtype=np.float32)
                for token in _text_tokens(self.text_builder(example, index)):
                    if token in self.vocab:
                        vec[self.vocab[token]] += 1.0
                vec *= self.idf
                norm = np.linalg.norm(vec)
                per_candidate.append(vec / max(1.0, norm))
            rows.append(per_candidate)
        return np.asarray(rows, dtype=np.float32)


class CompatibilityFeatureScorer:
    def __init__(self, training: MLPTrainingConfig, seed: int) -> None:
        self.training = training
        self.seed = int(seed)
        self.model: CandidatewiseScoringModel | None = None

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        x_train = self._features(train_examples)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = self._features(dev_examples)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = _fit_candidatewise_model(x_train, y_train, x_dev, y_dev, self.training, self.training.hidden_dims, self.seed)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("control is not fit")
        return _predict_candidatewise_model(self.model, self._features(examples))

    def _features(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        rows = []
        for example in examples:
            view_bits = np.asarray(_model_facing_view_bits(example), dtype=np.float32)
            per_candidate = []
            for candidate in example.candidates:
                candidate_bits = np.asarray(_model_facing_candidate_bits(candidate.attributes), dtype=np.float32)
                equality = (candidate_bits == view_bits).astype(np.float32)
                valid = np.asarray([float(np.all(candidate_bits >= 0.0) and np.all(view_bits >= 0.0))], dtype=np.float32)
                if not bool(valid[0]):
                    candidate_bits = np.zeros(4, dtype=np.float32)
                    view_for_features = np.zeros(4, dtype=np.float32)
                    equality = np.zeros(4, dtype=np.float32)
                else:
                    view_for_features = view_bits
                per_candidate.append(np.concatenate([view_for_features, candidate_bits, equality, np.asarray([equality.mean()], dtype=np.float32), valid]))
            rows.append(per_candidate)
        return np.asarray(rows, dtype=np.float32)


class FullContextTinyTransformerControl:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.result = None

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        stage = _stage_from_config(self.config)
        from src.experiments.architecture_search import _agent_config, _training_config

        self.result = fit_context_baseline(
            train_examples=train_examples,
            dev_examples=dev_examples,
            agent_config=_agent_config(stage),
            training_config=_training_config(stage),
            num_classes=8,
            seed=self.seed,
            device="cuda" if torch.cuda.is_available() and str(self.config.get("device")) == "cuda" else "cpu",
            method="stage35_single_agent_full_context",
            prompt_mode="full",
        )

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("control is not fit")
        return predict_context_baseline(self.result, examples)


class TinyCrossEncoderCandidateScorer:
    def __init__(self, text_builder: Callable[[MultiViewTaskExample, int], str], seed: int, epochs: int) -> None:
        self.text_builder = text_builder
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.vocab_size = 2048
        self.max_len = 96
        self.model: _TinyPairEncoder | None = None

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _TinyPairEncoder(self.vocab_size, hidden_dim=32, max_len=self.max_len).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=0.0001)
        y_train = torch.as_tensor([example.label for example in train_examples], dtype=torch.long, device=device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed)
        for _epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(train_examples))
            for start in range(0, len(order), 32):
                idx = order[start : start + 32]
                batch = [train_examples[int(index)] for index in idx]
                token_ids = self._encode_examples(batch, device)
                logits = self.model(token_ids)
                loss = F.cross_entropy(logits, y_train[torch.as_tensor(idx, dtype=torch.long, device=device)])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            dev_pred = self.predict(dev_examples)
            dev_y = np.asarray([example.label for example in dev_examples], dtype=np.int64)
            dev_acc = _accuracy(dev_pred, dev_y)
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 2:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("control is not fit")
        device = next(self.model.parameters()).device
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(examples), 64):
                batch = list(examples[start : start + 64])
                logits = self.model(self._encode_examples(batch, device))
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)

    def _encode_examples(self, examples: Sequence[MultiViewTaskExample], device: torch.device) -> torch.Tensor:
        rows = []
        for example in examples:
            per_candidate = []
            for index in range(len(example.candidates)):
                ids = [_token_id(token, self.vocab_size) for token in _text_tokens(self.text_builder(example, index))[: self.max_len]]
                ids.extend([0] * max(0, self.max_len - len(ids)))
                per_candidate.append(ids[: self.max_len])
            rows.append(per_candidate)
        return torch.as_tensor(rows, dtype=torch.long, device=device)


class _TinyPairEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, candidates, length = token_ids.shape
        flat = token_ids.reshape(batch * candidates, length)
        pos = torch.arange(length, device=token_ids.device)
        x = self.embedding(flat) + self.position(pos).unsqueeze(0)
        encoded = self.encoder(x, src_key_padding_mask=flat.eq(0))
        lengths = flat.ne(0).sum(dim=1).clamp(min=1) - 1
        pooled = encoded[torch.arange(encoded.shape[0], device=token_ids.device), lengths]
        return self.head(pooled).reshape(batch, candidates)


def _fit_candidatewise_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    training: MLPTrainingConfig,
    hidden_dims: Iterable[int],
    seed: int,
) -> CandidatewiseScoringModel:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidatewiseScoringModel(input_dim=x_train.shape[-1], hidden_dims=tuple(hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.lr, weight_decay=training.weight_decay)
    tx = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y_train, dtype=torch.long, device=device)
    dx = torch.as_tensor(x_dev, dtype=torch.float32, device=device)
    dy = torch.as_tensor(y_dev, dtype=torch.long, device=device)
    best_state = None
    best_dev = -1.0
    stale = 0
    rng = np.random.default_rng(seed)
    for _epoch in range(training.epochs):
        model.train()
        for batch_idx in _batches(rng.permutation(len(y_train)), training.batch_size):
            idx = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
            loss = F.cross_entropy(model(tx.index_select(0, idx)), ty.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_acc = float((torch.argmax(model(dx), dim=1) == dy).float().mean().item())
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= training.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model


def _predict_candidatewise_model(model: CandidatewiseScoringModel, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tx.shape[0], 2048):
            preds.append(torch.argmax(model(tx[start : start + 2048]), dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _hashed_feature_vector(text: str, feature_dim: int) -> np.ndarray:
    features = np.zeros(feature_dim, dtype=np.float32)
    for token in _text_tokens(text):
        raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
        index = int(raw[:8], 16) % feature_dim
        sign = 1.0 if int(raw[8:10], 16) % 2 == 0 else -1.0
        features[index] += sign
    return features / max(1.0, float(np.linalg.norm(features)))


def _token_id(token: str, vocab_size: int) -> int:
    raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
    return 1 + (int(raw[:8], 16) % max(1, vocab_size - 1))


def _all_views_text(example: MultiViewTaskExample) -> str:
    return "\n".join(view.text for view in example.views)


def _selected_views_text(example: MultiViewTaskExample, roles: Sequence[int]) -> str:
    return "\n".join(example.views[int(role)].text for role in roles)


def _model_facing_view_bits(example: MultiViewTaskExample) -> List[int]:
    text = "\n".join(view.text.lower() for view in example.views)
    return [_category_bit_from_text(text, pair) for pair in ATTRIBUTE_VALUE_PAIRS]


def _model_facing_candidate_bits(attributes: Sequence[str]) -> List[int]:
    bits = []
    for index, pair in enumerate(ATTRIBUTE_VALUE_PAIRS):
        value = str(attributes[index]) if index < len(attributes) else ""
        bits.append(_category_bit_from_value(value, pair))
    return bits


def _category_bit_from_text(text: str, pair: Sequence[str]) -> int:
    low = str(pair[0]).replace("_", " ").lower()
    high = str(pair[1]).replace("_", " ").lower()
    if high in text:
        return 1
    if low in text:
        return 0
    return -1


def _category_bit_from_value(value: str, pair: Sequence[str]) -> int:
    if value == str(pair[1]):
        return 1
    if value == str(pair[0]):
        return 0
    return -1


def _best_test_accuracy(results: Dict[str, Dict[str, float]]) -> float:
    values = [float(row.get("test", 0.0)) for name, row in results.items() if not name.startswith("_")]
    return max(values or [0.0])


def _variant_gates_ok(diagnostics: Dict[str, object]) -> bool:
    summary = diagnostics.get("summary", {})
    return bool(
        summary.get("leakage_passes", False)
        and float(summary.get("mean_candidate_only_accuracy", 1.0)) <= 0.15
        and float(summary.get("mean_candidate_metadata_only_accuracy", 1.0)) <= 0.15
        and float(summary.get("mean_view_masked_candidates_visible_accuracy", 1.0)) <= 0.18
        and float(summary.get("mean_lexical_overlap_accuracy", 1.0)) <= 0.18
        and float(summary.get("mean_static_frequency_accuracy", 1.0)) <= 0.20
        and float(summary.get("mean_all_role_structured_oracle_accuracy", 0.0)) >= 0.90
    )


def _run_tiny_smoke(config: Dict[str, object], splits: Dict[str, Sequence[MultiViewTaskExample]]) -> Dict[str, object]:
    device, hardware = _configure_cuda(config)
    stage = _stage_from_config(config)
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    _clear_cuda()
    baselines = _fit_baselines(stage, splits, seed=0, device=device)
    row = _run_stage34_smoke_seed(stage, 0, candidate, splits, baselines, device, hardware)
    from src.experiments.run_stage34_candidate_sanitization_smoke import _summary as _smoke_summary

    result = {"seed_rows": [row], "summary": _smoke_summary([row], {"summary": {"pretraining_dataset_validity_passed": True, "leakage_passes": True, "mean_role_pair_only_accuracy": 0.125}})}
    _clear_cuda()
    return result


def _smoke_candidate_controls_ok(smoke: object) -> bool:
    if not isinstance(smoke, dict):
        return False
    acc = smoke.get("summary", {}).get("mean_test_accuracy", {})
    if not isinstance(acc, dict):
        return False
    return float(acc.get("candidate_only", 1.0)) <= 0.18 and float(acc.get("view_masked", 1.0)) <= 0.18


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    controls = result["stage34_positive_controls"]
    variants = result.get("stage34b_variants", [])
    lines = [
        "# Stage 3.5 Model-Facing Learnability Diagnosis",
        "",
        "## Stage 3.4 Inspection",
        "",
        f"- Model-facing inputs learnable: `{summary['model_facing_inputs_learnable']}`",
        f"- Best Stage 3.4 positive-control test accuracy: `{float(summary['best_positive_control_accuracy']):.4f}`",
        f"- Stage 3.4 over-sanitized: `{summary['stage34_over_sanitized']}`",
        f"- Inspection result: `{result['inspection']['interpretation']}`",
        "",
        "## Stage 3.4 Positive Controls",
        "",
        "| control | train | dev | test |",
        "|---|---:|---:|---:|",
    ]
    for name, row in controls.items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | {float(row['train']):.4f} | {float(row['dev']):.4f} | {float(row['test']):.4f} |")
    if variants:
        variant = variants[0]
        diag = variant["dataset_diagnostics"]["summary"]
        smoke = variant.get("smoke") or {}
        smoke_acc = smoke.get("summary", {}).get("mean_test_accuracy", {})
        lines.extend(
            [
                "",
                "## Stage 3.4b Balanced Representation",
                "",
                f"- Representation: `{variant['name']}`",
                f"- Best positive-control test accuracy: `{float(variant['best_positive_control_test_accuracy']):.4f}`",
                f"- Candidate-only baseline: `{float(diag.get('mean_candidate_only_accuracy', 0.0)):.4f}`",
                f"- Candidate-metadata-only baseline: `{float(diag.get('mean_candidate_metadata_only_accuracy', 0.0)):.4f}`",
                f"- View-masked + candidates-visible baseline: `{float(diag.get('mean_view_masked_candidates_visible_accuracy', 0.0)):.4f}`",
                f"- Lexical-overlap baseline: `{float(diag.get('mean_lexical_overlap_accuracy', 0.0)):.4f}`",
                f"- Static frequency baseline: `{float(diag.get('mean_static_frequency_accuracy', 0.0)):.4f}`",
                f"- All-role oracle: `{float(diag.get('mean_all_role_structured_oracle_accuracy', 0.0)):.4f}`",
                "",
                "### Stage 3.4b Positive Controls",
                "",
                "| control | train | dev | test |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, row in variant["positive_controls"].items():
            if name.startswith("_"):
                continue
            lines.append(f"| {name} | {float(row['train']):.4f} | {float(row['dev']):.4f} | {float(row['test']):.4f} |")
        lines.extend(
            [
                "",
                "### Stage 3.4b Tiny Smoke",
                "",
                f"- Trainable: `{float(smoke_acc.get('trainable', 0.0)):.4f}`",
                f"- Frozen: `{float(smoke_acc.get('frozen', 0.0)):.4f}`",
                f"- Text-only: `{float(smoke_acc.get('text_only', 0.0)):.4f}`",
                f"- Raw latent: `{float(smoke_acc.get('raw_latent', 0.0)):.4f}`",
                f"- Candidate-only: `{float(smoke_acc.get('candidate_only', 0.0)):.4f}`",
                f"- View-masked: `{float(smoke_acc.get('view_masked', 0.0)):.4f}`",
                f"- Role-shuffled: `{float(smoke_acc.get('role_labels_shuffled', 0.0)):.4f}`",
                f"- Oracle: `{float(smoke_acc.get('oracle', 0.0)):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stage 3.4b fixes learnability while keeping candidate-only/view-masked near chance: `{summary['stage34b_representation_fixes_learnability'] and summary['stage34b_keeps_candidate_only_and_view_masked_near_chance']}`",
            f"- Full validation justified: `{summary['full_validation_justified']}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
