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
    / "run_stage15b_question_routed_multihop_qa_clone_collaboration.py"
)
SPEC = importlib.util.spec_from_file_location("stage15b_question_routed_multihop_qa_clone_collaboration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _raw_example(question: str, answer: str, paragraphs: list[tuple[str, list[str]]], supporting: list[tuple[str, int]]) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "context": {
            "title": [title for title, _sentences in paragraphs],
            "sentences": [sentences for _title, sentences in paragraphs],
        },
        "supporting_facts": {
            "title": [title for title, _sid in supporting],
            "sent_id": [sid for _title, sid in supporting],
        },
    }


def test_prepare_hotpot_example_extracts_support_windows() -> None:
    example = _raw_example(
        question="Which city hosts the river museum?",
        answer="metropolis",
        paragraphs=[
            ("River Museum", ["The river museum is located in Metropolis near the old bridge."]),
            ("Bridge", ["The old bridge stands over the silver river."]),
        ],
        supporting=[("River Museum", 0), ("Bridge", 0)],
    )
    prepared = MODULE._prepare_hotpot_example(
        example,
        question_max_length=16,
        context_length=24,
        window_size=4,
    )
    assert prepared is not None
    assert prepared.answer_tokens == ["metropolis"]
    assert len(prepared.support_windows) >= 1


def test_filter_mode_multi_support_requires_two_support_windows() -> None:
    row = MODULE.PreparedHotpotExample(
        question_tokens=["which", "city"],
        context_tokens=["a"] * 16,
        answer_tokens=["metropolis"],
        answer_text="metropolis",
        answer_start=2,
        answer_end=2,
        answer_window=0,
        support_windows=[0, 2],
    )
    assert MODULE._keep_prepared_example(row, "multi_support") is True
    assert MODULE._keep_prepared_example(row, "answer_aligned") is True
    weak = MODULE.PreparedHotpotExample(
        question_tokens=row.question_tokens,
        context_tokens=row.context_tokens,
        answer_tokens=row.answer_tokens,
        answer_text=row.answer_text,
        answer_start=row.answer_start,
        answer_end=row.answer_end,
        answer_window=row.answer_window,
        support_windows=[1],
    )
    assert MODULE._keep_prepared_example(weak, "multi_support") is False


def test_oracle_support_routing_repeats_support_windows() -> None:
    model = MODULE.QuestionRoutedMultiHopQACloneModel(
        vocab_size=32,
        question_max_length=4,
        window_size=2,
        config=MODULE.ModelConfig(num_clones=4, hidden_size=8, num_layers=1, num_heads=2, ff_dim=16, dropout=0.0, routing_mode="oracle_support"),
    )
    weights, selected = model._oracle_route_support(
        support_windows=[[1, 3]],
        answer_window=torch.tensor([2], dtype=torch.long),
        device=torch.device("cpu"),
    )
    assert weights.shape == (1, 4, 4)
    assert selected.tolist()[0] == [1, 3, 1, 3]


def test_model_forward_returns_routed_window_outputs() -> None:
    model = MODULE.QuestionRoutedMultiHopQACloneModel(
        vocab_size=64,
        question_max_length=4,
        window_size=2,
        config=MODULE.ModelConfig(num_clones=4, hidden_size=8, num_layers=1, num_heads=2, ff_dim=16, dropout=0.0),
    )
    total_length = model.total_length
    input_ids = torch.randint(0, 63, (2, total_length), dtype=torch.long)
    attention_mask = torch.ones(2, total_length, dtype=torch.bool)
    context_mask = torch.ones(2, model.context_length, dtype=torch.bool)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context_mask=context_mask,
        support_windows=[[0, 1], [2, 3]],
        answer_window=torch.tensor([0, 1], dtype=torch.long),
        return_clone_states=True,
        return_attention=True,
    )
    assert outputs["start_logits"].shape == (2, model.context_length)
    assert outputs["end_logits"].shape == (2, model.context_length)
    assert outputs["clone_states"].shape == (2, 4, 8)
    assert outputs["selected_windows"].shape == (2, 4)
    assert len(outputs["attention_probs"]) == 1


def test_learned_distinct_routing_selects_unique_windows_at_eval() -> None:
    model = MODULE.QuestionRoutedMultiHopQACloneModel(
        vocab_size=64,
        question_max_length=4,
        window_size=2,
        config=MODULE.ModelConfig(
            num_clones=4,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            routing_mode="learned_distinct",
        ),
    )
    model.eval()
    input_ids = torch.randint(0, 63, (1, model.total_length), dtype=torch.long)
    attention_mask = torch.ones(1, model.total_length, dtype=torch.bool)
    context_mask = torch.ones(1, model.context_length, dtype=torch.bool)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context_mask=context_mask,
        support_windows=[[0, 1]],
        answer_window=torch.tensor([0], dtype=torch.long),
    )
    selected = outputs["selected_windows"][0].tolist()
    assert len(selected) == len(set(selected))


def test_train_on_prepared_splits_smoke_runs_and_reports_routing_metrics() -> None:
    rows = [
        MODULE._prepare_hotpot_example(
            _raw_example(
                question="Which city hosts the river museum?",
                answer="metropolis",
                paragraphs=[
                    ("River Museum", ["The river museum is located in Metropolis near the bridge."]),
                    ("Bridge", ["The old bridge crosses the silver river in the capital district."]),
                ],
                supporting=[("River Museum", 0), ("Bridge", 0)],
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
                supporting=[("Observatory", 0), ("Forest", 0)],
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
                supporting=[("Archive", 0), ("Harbor", 0)],
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
    config = MODULE.Stage15bConfig(
        corpus=MODULE.CorpusConfig(
            n_train=3,
            n_dev=2,
            n_test=2,
            vocab_size=128,
            question_max_length=16,
            window_size=4,
            answer_max_length=4,
        ),
        model=MODULE.ModelConfig(
            num_clones=4,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            ff_dim=32,
            dropout=0.0,
            routing_mode="oracle_support",
        ),
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
    assert "routing" in results["final_test"]
