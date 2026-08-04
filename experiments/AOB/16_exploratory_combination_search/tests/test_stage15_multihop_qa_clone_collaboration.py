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
    / "run_stage15_multihop_qa_clone_collaboration.py"
)
SPEC = importlib.util.spec_from_file_location("stage15_multihop_qa_clone_collaboration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _raw_example(question: str, answer: str, paragraphs: list[tuple[str, list[str]]]) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "context": {
            "title": [title for title, _sentences in paragraphs],
            "sentences": [sentences for _title, sentences in paragraphs],
        },
    }


def test_prepare_hotpot_example_finds_answer_window() -> None:
    example = _raw_example(
        question="Which city hosts the river museum?",
        answer="metropolis",
        paragraphs=[
            ("River Museum", ["The river museum is located in Metropolis near the old bridge."]),
            ("Other", ["Nothing relevant happens here."]),
        ],
    )
    prepared = MODULE._prepare_hotpot_example(
        example,
        question_max_length=16,
        context_length=32,
        window_size=8,
    )
    assert prepared is not None
    assert prepared.answer_tokens == ["metropolis"]
    assert prepared.answer_start >= 0
    assert prepared.answer_window == prepared.answer_start // 8


def test_clone_gather_indices_keep_question_prefix_for_every_clone() -> None:
    indices = MODULE._build_clone_gather_indices(question_segment_length=5, window_size=4, num_clones=3)
    expected_question = [0, 1, 2, 3, 4]
    assert indices.shape == (3, 9)
    for clone_index in range(3):
        assert indices[clone_index, :5].tolist() == expected_question
    assert indices[1, 5:].tolist() == [9, 10, 11, 12]


def test_model_forward_returns_context_length_logits() -> None:
    model = MODULE.MultiHopQACloneModel(
        vocab_size=32,
        question_max_length=4,
        window_size=2,
        config=MODULE.ModelConfig(num_clones=4, hidden_size=8, num_layers=1, num_heads=2, ff_dim=16, dropout=0.0),
    )
    total_length = model.total_length
    input_ids = torch.randint(0, 31, (2, total_length), dtype=torch.long)
    attention_mask = torch.ones(2, total_length, dtype=torch.bool)
    context_mask = torch.ones(2, model.context_length, dtype=torch.bool)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context_mask=context_mask,
        return_clone_states=True,
        return_attention=True,
    )
    assert outputs["start_logits"].shape == (2, model.context_length)
    assert outputs["end_logits"].shape == (2, model.context_length)
    assert outputs["clone_states"].shape == (2, 4, 8)
    assert len(outputs["attention_probs"]) == 1


def test_clone_pooling_uses_context_slice_not_shared_question_prefix() -> None:
    model = MODULE.MultiHopQACloneModel(
        vocab_size=32,
        question_max_length=2,
        window_size=2,
        config=MODULE.ModelConfig(num_clones=2, hidden_size=2, num_layers=1, num_heads=1, ff_dim=4, dropout=0.0),
    )
    clone_hidden = torch.tensor(
        [
            [
                [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [1.0, 0.0], [3.0, 0.0]],
                [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [0.0, 2.0], [0.0, 4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    clone_mask = torch.ones(1, 2, 5, dtype=torch.bool)
    pooled = model.pooled_clone_states(clone_hidden, clone_mask)
    assert torch.allclose(pooled[0, 0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(pooled[0, 1], torch.tensor([0.0, 3.0]))


def test_best_span_respects_max_answer_length() -> None:
    start = torch.tensor([0.0, 0.0, 5.0, 1.0], dtype=torch.float32)
    end = torch.tensor([0.0, 0.0, 1.0, 5.0], dtype=torch.float32)
    mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    best = MODULE._best_span(start, end, mask, max_answer_length=2)
    assert best == (2, 3)


def test_train_on_prepared_splits_smoke_runs_and_reports_requested_metrics() -> None:
    rows = [
        MODULE._prepare_hotpot_example(
            _raw_example(
                question="Which city hosts the river museum?",
                answer="metropolis",
                paragraphs=[
                    ("River Museum", ["The river museum is located in Metropolis near the bridge."]),
                    ("Bridge", ["The old bridge crosses the silver river in the capital district."]),
                ],
            ),
            question_max_length=16,
            context_length=16,
            window_size=4,
        ),
        MODULE._prepare_hotpot_example(
            _raw_example(
                question="Which town contains the observatory?",
                answer="oakridge",
                paragraphs=[
                    ("Observatory", ["The observatory stands in Oakridge above the western forest."]),
                    ("Forest", ["The western forest borders the lake and the watchtower."]),
                ],
            ),
            question_max_length=16,
            context_length=16,
            window_size=4,
        ),
        MODULE._prepare_hotpot_example(
            _raw_example(
                question="Which village owns the lighthouse archive?",
                answer="stonehaven",
                paragraphs=[
                    ("Archive", ["The lighthouse archive is preserved in Stonehaven by the harbor."]),
                    ("Harbor", ["The harbor connects the ferry routes and the old market."]),
                ],
            ),
            question_max_length=16,
            context_length=16,
            window_size=4,
        ),
    ]
    prepared_rows = [row for row in rows if row is not None]
    splits = {
        "train": prepared_rows,
        "dev": prepared_rows[:2],
        "test": prepared_rows[1:],
    }
    config = MODULE.Stage15Config(
        corpus=MODULE.CorpusConfig(
            n_train=3,
            n_dev=2,
            n_test=2,
            vocab_size=128,
            question_max_length=16,
            window_size=4,
            answer_max_length=4,
        ),
        model=MODULE.ModelConfig(num_clones=4, hidden_size=16, num_layers=1, num_heads=2, ff_dim=32, dropout=0.0),
        training=MODULE.TrainingConfig(
            epochs=2,
            batch_size=2,
            seed=7,
            device="cpu",
            early_stopping_patience=2,
            use_bf16=False,
        ),
    )
    results = MODULE.train_on_prepared_splits(splits, config)
    assert len(results["history"]) >= 1
    assert math.isfinite(float(results["final_test"]["f1"]))
    assert len(results["final_test"]["per_clone_f1"]) == 4
    assert len(results["final_test"]["meta_attention_distribution"]["mean_weights"]) == 4
    assert "answer_window_coverage" in results["final_test"]
