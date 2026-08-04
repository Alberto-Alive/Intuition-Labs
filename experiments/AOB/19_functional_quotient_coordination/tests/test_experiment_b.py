import sys
from pathlib import Path

import pytest
import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import run_experiment_b as B  # noqa: E402

S16 = B.S16


def make_model(routing_mode: str, num_clones: int = 4, window_size: int = 4, vocab_size: int = 64):
    config = S16.ModelConfig(
        num_clones=num_clones,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        ff_dim=32,
        dropout=0.0,
        routing_mode=routing_mode,
        communication_mode="continuous",
    )
    torch.manual_seed(0)
    model = B.FQCRoutedModel(
        vocab_size=vocab_size,
        question_max_length=4,
        window_size=window_size,
        config=config,
    )
    model.eval()
    return model


def make_batch(model, distinct_windows: bool = True, duplicate_pair: tuple | None = None):
    total = model.total_length
    q_len = model.question_segment_length
    w = model.num_windows
    ws = model.window_size
    input_ids = torch.zeros(1, total, dtype=torch.long)
    input_ids[0, :q_len] = torch.arange(1, q_len + 1)
    for wdx in range(w):
        base = 10 + wdx * ws if distinct_windows else 10
        input_ids[0, q_len + wdx * ws : q_len + (wdx + 1) * ws] = torch.arange(base, base + ws)
    if duplicate_pair is not None:
        a, b = duplicate_pair
        src = input_ids[0, q_len + a * ws : q_len + (a + 1) * ws].clone()
        input_ids[0, q_len + b * ws : q_len + (b + 1) * ws] = src
    attention_mask = torch.ones(1, total, dtype=torch.bool)
    return input_ids, attention_mask


def route(model, input_ids, attention_mask, support_windows=None, answer_window=None):
    support_windows = support_windows if support_windows is not None else [[]]
    answer_window = answer_window if answer_window is not None else torch.tensor([-1])
    with torch.no_grad():
        _, _, selected, _ = model._build_clone_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            support_windows=support_windows,
            answer_window=answer_window,
        )
    return selected[0].tolist()


def test_fqc_ignores_ground_truth_routing_inputs():
    model = make_model("fqc")
    input_ids, attention_mask = make_batch(model)
    picks_a = route(model, input_ids, attention_mask, [[0, 1]], torch.tensor([2]))
    picks_b = route(model, input_ids, attention_mask, [[3]], torch.tensor([0]))
    assert picks_a == picks_b


def test_identical_content_windows_have_similarity_one():
    model = make_model("fqc")
    input_ids, attention_mask = make_batch(model, duplicate_pair=(0, 2))
    route(model, input_ids, attention_mask)
    sim = model._fqc_window_similarity()
    assert float(sim[0, 0, 2]) == pytest.approx(1.0, abs=1e-5)
    assert float(sim[0, 0, 1]) < 0.99


def test_quotient_spreads_identical_clones():
    # Zeroed clone identities make every clone pick the same window under
    # independent routing; the quotient must spread them without any
    # ground-truth signal.
    model = make_model("fqc")
    model.router_clone_identity.weight.data.zero_()
    input_ids, attention_mask = make_batch(model)
    picks = route(model, input_ids, attention_mask)
    assert len(set(picks)) >= 3

    control = make_model("fqc_no_quotient")
    control.router_clone_identity.weight.data.zero_()
    control_picks = route(control, input_ids, attention_mask)
    assert len(set(control_picks)) == 1


def test_no_reassign_detects_but_does_not_move():
    model = make_model("fqc_no_reassign")
    model.router_clone_identity.weight.data.zero_()
    input_ids, attention_mask = make_batch(model)
    picks = route(model, input_ids, attention_mask)
    assert len(set(picks)) == 1


def test_lexical_similarity_matches_token_overlap():
    model = make_model("fqc_lexical")
    input_ids, attention_mask = make_batch(model, duplicate_pair=(1, 3))
    route(model, input_ids, attention_mask)
    sim = model._fqc_window_similarity()
    assert float(sim[0, 1, 3]) == pytest.approx(1.0)
    assert float(sim[0, 0, 1]) == pytest.approx(0.0)


def test_stock_modes_unchanged():
    model = make_model("learned_distinct")
    input_ids, attention_mask = make_batch(model)
    picks = route(model, input_ids, attention_mask)
    assert len(set(picks)) == model.config.num_clones  # forced no-replacement


def test_duplicated_evidence_transform_preserves_answer_and_supports():
    ws, w = 4, 4
    context = [f"t{i}" for i in range(ws * w)]
    answer_tokens = context[2 * ws : 2 * ws + 2]
    row = S16.PreparedHotpotExample(
        question_tokens=["q"],
        context_tokens=context,
        answer_tokens=answer_tokens,
        answer_text=" ".join(answer_tokens),
        answer_start=2 * ws,
        answer_end=2 * ws + 1,
        answer_window=2,
        support_windows=[1, 2],
    )
    out = B.duplicated_evidence_transform([row], window_size=ws, num_windows=w, seed=0)
    assert len(out) == 1
    new = out[0]
    assert new.context_tokens[new.answer_start : new.answer_end + 1] == answer_tokens
    assert new.answer_window == new.answer_start // ws
    # Only two unique windows remain, each tiled twice.
    unique_windows = {tuple(new.context_tokens[i * ws : (i + 1) * ws]) for i in range(w)}
    assert len(unique_windows) == 2
    # All copies of the original support windows are listed.
    assert len(new.support_windows) == 4
