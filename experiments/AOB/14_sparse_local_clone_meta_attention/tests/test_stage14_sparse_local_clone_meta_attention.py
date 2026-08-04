from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "src"
    / "experiments"
    / "run_stage14_sparse_local_clone_meta_attention.py"
)
SPEC = importlib.util.spec_from_file_location("stage14_sparse_local_clone_meta_attention", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mask_start_boundary_hits_window_boundaries() -> None:
    starts = [
        MODULE._mask_start_for_window(
            sequence_length=64,
            mask_span_length=4,
            strategy="boundary",
            num_clones=4,
            example_index=index,
            seed=0,
        )
        for index in range(3)
    ]
    assert starts == [14, 30, 46]


def test_masked_span_dataset_inserts_mask_tokens() -> None:
    vocab = MODULE.Vocabulary(
        stoi={"<pad>": 0, "<bos>": 1, "<unk>": 2, "<mask>": 3, "a": 4, "b": 5, "c": 6, "d": 7, "e": 8},
        itos=["<pad>", "<bos>", "<unk>", "<mask>", "a", "b", "c", "d", "e"],
    )
    dataset = MODULE.MaskedSpanDataset(
        texts=["a b c d e a b c d e"],
        vocab=vocab,
        sequence_length=8,
        mask_span_length=2,
        mask_strategy="center",
        num_clones=4,
        seed=0,
    )
    row = dataset[0]
    assert row["target_ids"].tolist() == row["input_ids"][3:5].new_tensor([6, 7]).tolist()
    assert row["input_ids"][3].item() == vocab.mask_id
    assert row["input_ids"][4].item() == vocab.mask_id


def test_sparse_attention_keeps_mass_inside_clone_window() -> None:
    attention = MODULE.SparseLocalCloneSelfAttention(
        hidden_size=4,
        num_heads=1,
        num_clones=2,
        max_length=4,
        dropout=0.0,
    )
    hidden = torch.ones(1, 2, 4, 4, dtype=torch.float32)
    clone_token_mask = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ]
        ],
        dtype=torch.bool,
    )
    _out, probs = attention(hidden, clone_token_mask=clone_token_mask)
    coverage = MODULE._clone_window_coverage([probs], MODULE._build_clone_window_mask(num_clones=2, sequence_length=4))
    assert coverage["mean_outside_window_mass"] == 0.0
    assert coverage["max_outside_window_mass"] == 0.0


def test_span_meta_attention_returns_one_weight_vector_per_span_position() -> None:
    meta = MODULE.SpanMetaAttention(hidden_size=4, num_heads=1, span_length=3, dropout=0.0, init_std=0.02)
    clone_states = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
        ],
        dtype=torch.float32,
    )
    meta_out, weights = meta(clone_states)
    assert meta_out.shape == (1, 3, 4)
    assert weights.shape == (1, 3, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3, dtype=weights.dtype))


def test_train_on_text_splits_smoke_runs_and_reports_requested_metrics() -> None:
    text_splits = {
        "train": [
            "import foo from bar if x == y and value returns module patch policy coordinator local window clone extra tokens",
            "class widget uses local clone attention over sparse sequence window and predicts masked span correctly now",
            "patch local import boundary explicit policy function call returns value from module global picture assembly",
            "meta attention assembles local views from sparse clones into one masked span reconstruction head",
        ],
        "dev": [
            "sparse clone window meta attention masked span reconstruction from local views only",
            "private qkv weights and shared mlp blocks encourage separate gradients per clone",
        ],
        "test": [
            "each clone attends only to its fixed local window with hard mask",
            "meta attention integrates clone summaries into one masked span prediction",
        ],
    }
    config = MODULE.Stage14Config(
        corpus=MODULE.CorpusConfig(max_length=8, vocab_size=128, mask_span_length=2, mask_strategy="boundary"),
        training=MODULE.TrainingConfig(
            epochs=2,
            batch_size=2,
            seed=7,
            device="cpu",
            early_stopping_patience=2,
            use_bf16=False,
        ),
    )
    results = MODULE.train_on_text_splits(text_splits, config)
    assert len(results["history"]) >= 1
    assert len(results["final_test"]["clone_perplexity"]) == 4
    assert len(results["final_test"]["attention_entropy_per_clone"]) == 4
    assert len(results["final_test"]["meta_attention_distribution"]["mean_weights"]) == 4
    assert "clone_window_coverage" in results["final_test"]
    assert "target_window_distribution" in results["final_test"]
    assert results["mask_strategy"] == "boundary"
    assert results["mask_span_length"] == 2
    assert math.isfinite(float(results["final_test"]["joint_perplexity"]))
