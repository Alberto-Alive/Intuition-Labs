from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Grid = List[List[int]]


@dataclass(frozen=True)
class ArcPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcVerificationExample:
    id: str
    task_id: str
    split: str
    train_pairs: Tuple[ArcPair, ...]
    test_input: Grid
    gold_output: Grid
    candidates: Tuple[Grid, ...]
    label: int
    negative_types: Tuple[str, ...]
    public_source_tags: Tuple[str, ...]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class ArcVerificationDatasetConfig:
    dataset_root: str = r"W:\HocusPocus\ARC-AGI-2"
    data_split: str = "training"
    num_candidates: int = 8
    train_tasks: int = 96
    dev_tasks: int = 32
    test_tasks: int = 32
    include_evaluation: bool = False
    max_train_pairs: int = 6
    max_grid_size: int = 30
    negative_difficulty: str = "mixed_hard"


def build_arc_verification_splits(
    config: ArcVerificationDatasetConfig,
    seed: int,
) -> Dict[str, List[ArcVerificationExample]]:
    tasks = load_arc_tasks(config)
    usable = [task for task in tasks if task["test_pairs"]]
    rng = np.random.default_rng(seed)
    order = np.array([task["task_id"] for task in usable], dtype=object)
    rng.shuffle(order)
    by_id = {task["task_id"]: task for task in usable}

    n_train = min(int(config.train_tasks), len(order))
    n_dev = min(int(config.dev_tasks), max(0, len(order) - n_train))
    n_test = min(int(config.test_tasks), max(0, len(order) - n_train - n_dev))
    split_ids = {
        "train": list(order[:n_train]),
        "dev": list(order[n_train : n_train + n_dev]),
        "test": list(order[n_train + n_dev : n_train + n_dev + n_test]),
    }
    splits: Dict[str, List[ArcVerificationExample]] = {"train": [], "dev": [], "test": []}
    for split, ids in split_ids.items():
        for task_id in ids:
            task = by_id[str(task_id)]
            for test_index, test_pair in enumerate(task["test_pairs"]):
                example_seed = seed + _stable_int(f"{task_id}:{test_index}") % 1_000_000_000
                example_rng = np.random.default_rng(example_seed)
                train_pairs = tuple(task["train_pairs"][: int(config.max_train_pairs)])
                candidates, label, negative_types = generate_candidate_set(
                    train_pairs=train_pairs,
                    test_input=test_pair.input,
                    gold_output=test_pair.output,
                    num_candidates=int(config.num_candidates),
                    rng=example_rng,
                    difficulty=str(config.negative_difficulty),
                )
                metadata = {
                    "candidate_count": int(config.num_candidates),
                    "gold_shape": list(_shape(test_pair.output)),
                    "test_input_shape": list(_shape(test_pair.input)),
                    "num_train_pairs": len(train_pairs),
                    "palette_size": len(_palette(test_pair.output)),
                    "task_family": "unknown",
                    "transformation_type": infer_transformation_type(train_pairs, test_pair.input, test_pair.output),
                    "candidate_generator_version": "arc1_balanced_hard_negatives_v1",
                    "model_visible_candidate_sources": False,
                }
                public_tags = tuple("candidate_grid" for _ in candidates)
                splits[split].append(
                    ArcVerificationExample(
                        id=f"{task_id}__test{test_index}",
                        task_id=str(task_id),
                        split=split,
                        train_pairs=train_pairs,
                        test_input=test_pair.input,
                        gold_output=test_pair.output,
                        candidates=tuple(candidates),
                        label=int(label),
                        negative_types=tuple(negative_types),
                        public_source_tags=public_tags,
                        metadata=metadata,
                    )
                )
    return splits


def load_arc_tasks(config: ArcVerificationDatasetConfig) -> List[Dict[str, object]]:
    root = Path(config.dataset_root)
    task_files = list(_candidate_task_files(root, config.data_split))
    if config.include_evaluation:
        task_files.extend(_candidate_task_files(root, "evaluation"))
    if not task_files:
        raise FileNotFoundError(f"no ARC task JSON files found under {root}")
    tasks: List[Dict[str, object]] = []
    for path in sorted(set(task_files)):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or "train" not in raw or "test" not in raw:
            continue
        train_pairs = tuple(
            ArcPair(input=_clean_grid(row["input"]), output=_clean_grid(row["output"]))
            for row in raw.get("train", [])
            if isinstance(row, dict) and "input" in row and "output" in row
        )
        test_pairs = tuple(
            ArcPair(input=_clean_grid(row["input"]), output=_clean_grid(row["output"]))
            for row in raw.get("test", [])
            if isinstance(row, dict) and "input" in row and "output" in row
        )
        if train_pairs and test_pairs:
            tasks.append({"task_id": path.stem, "path": str(path), "train_pairs": train_pairs, "test_pairs": test_pairs})
    return tasks


def _candidate_task_files(root: Path, split: str) -> Iterable[Path]:
    candidates = [
        root / "data" / split,
        root / split,
        root / "data",
        root,
    ]
    for folder in candidates:
        if folder.is_dir():
            yield from folder.glob("*.json")


def generate_candidate_set(
    train_pairs: Sequence[ArcPair],
    test_input: Grid,
    gold_output: Grid,
    num_candidates: int,
    rng: np.random.Generator,
    difficulty: str = "mixed_hard",
) -> Tuple[List[Grid], int, List[str]]:
    gold = _array(gold_output)
    test = _array(test_input)
    candidates: List[np.ndarray] = [gold.copy()]
    negative_types = ["gold"]
    seen = {_grid_key(gold)}

    generators = _negative_generators_for_difficulty(difficulty)
    attempts = 0
    while len(candidates) < int(num_candidates) and attempts < int(num_candidates) * 80:
        fn = generators[attempts % len(generators)]
        attempts += 1
        candidate, family = fn(train_pairs, test, gold, rng)
        candidate = _resize_to(candidate, gold.shape, fill=_dominant_color(gold))
        key = _grid_key(candidate)
        if key in seen or np.array_equal(candidate, gold):
            continue
        seen.add(key)
        candidates.append(candidate)
        negative_types.append(family)

    while len(candidates) < int(num_candidates):
        candidate, family = _negative_random_balanced(train_pairs, test, gold, rng)
        candidate = _resize_to(candidate, gold.shape, fill=_dominant_color(gold))
        key = _grid_key(candidate)
        if key in seen or np.array_equal(candidate, gold):
            candidate = _force_single_cell_change(gold, rng)
            key = _grid_key(candidate)
        if key in seen:
            candidate = _salted_balanced_grid(gold, rng, salt=len(candidates))
            key = _grid_key(candidate)
        seen.add(key)
        candidates.append(candidate)
        negative_types.append(family)

    order = rng.permutation(len(candidates))
    shuffled = [candidates[int(index)].astype(np.int64).tolist() for index in order]
    shuffled_types = [negative_types[int(index)] for index in order]
    label = int(np.where(order == 0)[0][0])
    return shuffled, label, shuffled_types


def _negative_generators_for_difficulty(difficulty: str):
    difficulty = str(difficulty)
    if difficulty == "easy_random":
        return (_negative_random_balanced,)
    if difficulty == "palette_matched":
        return (_negative_random_balanced, _negative_color_swap, _negative_color_drop_region)
    if difficulty == "object_count_matched":
        return (_negative_object_remove, _negative_object_extra, _negative_random_balanced)
    if difficulty == "changed_cell_count_matched":
        return (_negative_difference_preserve_input, _negative_difference_over_transform, _negative_cross_train_diff, _negative_random_balanced)
    if difficulty == "single_perturbation":
        return (_negative_color_swap, _negative_geometry_shift, _negative_object_remove, _negative_difference_preserve_input)
    if difficulty == "adversarial_hard":
        return (
            _negative_color_swap,
            _negative_color_drop_region,
            _negative_geometry_shift,
            _negative_geometry_flip,
            _negative_object_remove,
            _negative_object_extra,
            _negative_difference_preserve_input,
            _negative_difference_over_transform,
            _negative_cross_train_output,
            _negative_cross_train_diff,
        )
    return (
        _negative_color_swap,
        _negative_color_drop_region,
        _negative_geometry_shift,
        _negative_geometry_flip,
        _negative_object_remove,
        _negative_object_extra,
        _negative_difference_preserve_input,
        _negative_difference_over_transform,
        _negative_cross_train_output,
        _negative_cross_train_diff,
        _negative_random_balanced,
    )


def leakage_audit_rows(examples: Sequence[ArcVerificationExample]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example in examples:
        gold = _array(example.gold_output)
        candidate_shapes = [_shape(candidate) for candidate in example.candidates]
        gold_count = sum(int(np.array_equal(_array(candidate), gold)) for candidate in example.candidates)
        source_visible = bool(example.metadata.get("model_visible_candidate_sources", True))
        rows.append(
            {
                "type": "arc1_leakage_audit",
                "id": example.id,
                "task_id": example.task_id,
                "split": example.split,
                "gold_candidate_count": int(gold_count),
                "all_candidates_same_gold_size": all(tuple(shape) == tuple(_shape(example.gold_output)) for shape in candidate_shapes),
                "candidate_order_label": int(example.label),
                "candidate_source_ids_visible_to_model": source_visible,
                "task_metadata_visible_to_model": False,
                "solved_output_metadata_visible_outside_candidates": False,
                "pass": bool(gold_count == 1 and not source_visible and all(tuple(shape) == tuple(_shape(example.gold_output)) for shape in candidate_shapes)),
            }
        )
    return rows


def candidate_metadata_only_accuracy(examples: Sequence[ArcVerificationExample]) -> float:
    if not examples:
        return 0.0
    # Public source tags intentionally contain no source-family or gold marker.
    predictions = [0 for _ in examples]
    return float(np.mean([int(pred == example.label) for pred, example in zip(predictions, examples)]))


def apply_arc_control(
    examples: Sequence[ArcVerificationExample],
    control: str,
    seed: int,
) -> List[ArcVerificationExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_only":
        return [_candidate_only_example(example) for example in examples]
    if control == "examples_only_no_candidates":
        return [_examples_only_example(example) for example in examples]
    if control == "train_pair_shuffle":
        return _train_pair_shuffle(examples, rng)
    if control == "cross_task_train_pair_shuffle":
        return _cross_task_train_pair_shuffle(examples, rng)
    if control == "candidate_evidence_mismatch":
        return _candidate_evidence_mismatch(examples, rng)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle(example, rng) for example in examples]
    if control == "color_permutation":
        return [_color_permutation(example, rng) for example in examples]
    if control == "coordinate_shift_padding":
        return [_coordinate_shift_padding(example) for example in examples]
    if control == "randomized_labels":
        return [_randomized_label(example, rng) for example in examples]
    raise ValueError(f"unknown ARC control: {control}")


def dataset_summary(splits: Dict[str, Sequence[ArcVerificationExample]]) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    return {
        "num_examples": {split: len(rows) for split, rows in splits.items()},
        "num_tasks": {split: len({example.task_id for example in rows}) for split, rows in splits.items()},
        "candidate_count": sorted({len(example.candidates) for example in all_examples}),
        "grid_sizes": _count_strings([f"{_shape(example.gold_output)[0]}x{_shape(example.gold_output)[1]}" for example in all_examples]),
        "train_pair_counts": _count_strings([str(len(example.train_pairs)) for example in all_examples]),
        "palette_sizes": _count_strings([str(len(_palette(example.gold_output))) for example in all_examples]),
        "candidate_generator_recall": 1.0 if all(sum(np.array_equal(_array(c), _array(e.gold_output)) for c in e.candidates) == 1 for e in all_examples) else 0.0,
    }


def infer_transformation_type(train_pairs: Sequence[ArcPair], test_input: Grid, gold_output: Grid) -> str:
    if _shape(test_input) != _shape(gold_output):
        return "resize_or_construct"
    diffs = []
    for pair in train_pairs:
        if _shape(pair.input) == _shape(pair.output):
            a = _array(pair.input)
            b = _array(pair.output)
            diffs.append(float(np.mean(a != b)))
    if not diffs:
        return "same_size_unknown"
    mean_diff = float(np.mean(diffs))
    if mean_diff == 0.0:
        return "identity_like"
    if mean_diff < 0.15:
        return "sparse_edit"
    if mean_diff < 0.55:
        return "partial_rewrite"
    return "dense_rewrite"


def _candidate_only_example(example: ArcVerificationExample) -> ArcVerificationExample:
    blank_test = _blank_like(example.test_input)
    return replace(example, train_pairs=tuple(), test_input=blank_test, metadata={**example.metadata, "control": "candidate_only"})


def _examples_only_example(example: ArcVerificationExample) -> ArcVerificationExample:
    blank = _blank_like(example.gold_output)
    candidates = tuple(blank for _ in example.candidates)
    return replace(example, candidates=candidates, metadata={**example.metadata, "control": "examples_only_no_candidates"})


def _train_pair_shuffle(examples: Sequence[ArcVerificationExample], rng: np.random.Generator) -> List[ArcVerificationExample]:
    outputs = [pair.output for example in examples for pair in example.train_pairs]
    if not outputs:
        return list(examples)
    shuffled: List[ArcVerificationExample] = []
    for example in examples:
        pairs = []
        for pair in example.train_pairs:
            output = outputs[int(rng.integers(0, len(outputs)))]
            pairs.append(ArcPair(input=pair.input, output=output))
        shuffled.append(replace(example, train_pairs=tuple(pairs), metadata={**example.metadata, "control": "train_pair_shuffle"}))
    return shuffled


def _cross_task_train_pair_shuffle(examples: Sequence[ArcVerificationExample], rng: np.random.Generator) -> List[ArcVerificationExample]:
    if len(examples) < 2:
        return list(examples)
    shuffled = []
    for index, example in enumerate(examples):
        other_index = int(rng.integers(0, len(examples) - 1))
        if other_index >= index:
            other_index += 1
        other = examples[other_index]
        shuffled.append(replace(example, train_pairs=other.train_pairs, metadata={**example.metadata, "control": "cross_task_train_pair_shuffle"}))
    return shuffled


def _candidate_evidence_mismatch(examples: Sequence[ArcVerificationExample], rng: np.random.Generator) -> List[ArcVerificationExample]:
    if len(examples) < 2:
        return list(examples)
    mismatched = []
    for index, example in enumerate(examples):
        other_index = int(rng.integers(0, len(examples) - 1))
        if other_index >= index:
            other_index += 1
        other = examples[other_index]
        mismatched.append(
            replace(
                example,
                train_pairs=other.train_pairs,
                test_input=other.test_input,
                metadata={**example.metadata, "control": "candidate_evidence_mismatch", "evidence_from_task_id": other.task_id},
            )
        )
    return mismatched


def _candidate_order_shuffle(example: ArcVerificationExample, rng: np.random.Generator) -> ArcVerificationExample:
    order = rng.permutation(len(example.candidates))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    candidates = tuple(example.candidates[int(index)] for index in order)
    negative_types = tuple(example.negative_types[int(index)] for index in order)
    public_tags = tuple(example.public_source_tags[int(index)] for index in order)
    label = int(np.where(order == int(example.label))[0][0])
    return replace(
        example,
        candidates=candidates,
        label=label,
        negative_types=negative_types,
        public_source_tags=public_tags,
        metadata={**example.metadata, "control": "candidate_order_shuffle_with_gold_remap"},
    )


def _color_permutation(example: ArcVerificationExample, rng: np.random.Generator) -> ArcVerificationExample:
    colors = list(range(10))
    perm = colors[:]
    rng.shuffle(perm)
    mapping = {color: perm[color] for color in colors}
    return replace(
        example,
        train_pairs=tuple(ArcPair(input=_map_grid(pair.input, mapping), output=_map_grid(pair.output, mapping)) for pair in example.train_pairs),
        test_input=_map_grid(example.test_input, mapping),
        gold_output=_map_grid(example.gold_output, mapping),
        candidates=tuple(_map_grid(candidate, mapping) for candidate in example.candidates),
        metadata={**example.metadata, "control": "color_permutation"},
    )


def _coordinate_shift_padding(example: ArcVerificationExample) -> ArcVerificationExample:
    return replace(
        example,
        train_pairs=tuple(ArcPair(input=_pad_grid(pair.input), output=_pad_grid(pair.output)) for pair in example.train_pairs),
        test_input=_pad_grid(example.test_input),
        gold_output=_pad_grid(example.gold_output),
        candidates=tuple(_pad_grid(candidate) for candidate in example.candidates),
        metadata={**example.metadata, "control": "coordinate_shift_padding"},
    )


def _randomized_label(example: ArcVerificationExample, rng: np.random.Generator) -> ArcVerificationExample:
    return replace(
        example,
        label=int(rng.integers(0, len(example.candidates))),
        metadata={**example.metadata, "control": "randomized_labels"},
    )


def _negative_color_swap(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    out = gold.copy()
    colors = [color for color in _palette_array(out) if color != _dominant_color(out)]
    if len(colors) < 2:
        colors = _palette_array(out)
    if len(colors) >= 2:
        a, b = rng.choice(colors, size=2, replace=False)
        mask_a = out == int(a)
        mask_b = out == int(b)
        out[mask_a] = int(b)
        out[mask_b] = int(a)
    else:
        out = _force_single_cell_change(out, rng)
    return out, "color_swapped_palette"


def _negative_color_drop_region(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    out = gold.copy()
    colors = [color for color in _palette_array(out) if color != _dominant_color(out)]
    target = int(rng.choice(colors)) if colors else int(_dominant_color(out))
    positions = np.argwhere(out == target)
    if len(positions) == 0:
        return _force_single_cell_change(out, rng), "color_reduced_region"
    keep = max(1, len(positions) // 2)
    chosen = positions[rng.choice(len(positions), size=keep, replace=False)]
    replacement = int(_dominant_color(out))
    for row, col in chosen:
        out[int(row), int(col)] = replacement
    return out, "color_reduced_region"


def _negative_geometry_shift(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    dy = int(rng.choice([-2, -1, 1, 2]))
    dx = int(rng.choice([-2, -1, 1, 2]))
    fill = int(_dominant_color(gold))
    out = np.full_like(gold, fill)
    src_r0 = max(0, -dy)
    src_r1 = min(gold.shape[0], gold.shape[0] - dy)
    src_c0 = max(0, -dx)
    src_c1 = min(gold.shape[1], gold.shape[1] - dx)
    dst_r0 = max(0, dy)
    dst_r1 = dst_r0 + max(0, src_r1 - src_r0)
    dst_c0 = max(0, dx)
    dst_c1 = dst_c0 + max(0, src_c1 - src_c0)
    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = gold[src_r0:src_r1, src_c0:src_c1]
    return out, "geometry_shifted_objects"


def _negative_geometry_flip(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    if bool(rng.integers(0, 2)):
        return np.fliplr(gold).copy(), "geometry_flipped_objects"
    return np.flipud(gold).copy(), "geometry_flipped_objects"


def _negative_object_remove(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    out = gold.copy()
    components = _components(out)
    if not components:
        return _force_single_cell_change(out, rng), "object_missing"
    component = components[int(rng.integers(0, len(components)))]
    bg = int(_dominant_color(out))
    for row, col in component:
        out[row, col] = bg
    return out, "object_missing"


def _negative_object_extra(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    out = gold.copy()
    components = _components(out)
    if not components:
        return _force_single_cell_change(out, rng), "object_extra"
    component = components[int(rng.integers(0, len(components)))]
    dy = int(rng.choice([-3, -2, 2, 3]))
    dx = int(rng.choice([-3, -2, 2, 3]))
    for row, col in component:
        rr = row + dy
        cc = col + dx
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = gold[row, col]
    return out, "object_extra"


def _negative_difference_preserve_input(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    out = gold.copy()
    if test.shape != gold.shape:
        return _negative_random_balanced(train_pairs, test, gold, rng)[0], "difference_preserve_input_too_much"
    changed = np.argwhere(test != gold)
    if len(changed) == 0:
        return _force_single_cell_change(out, rng), "difference_preserve_input_too_much"
    chosen = changed[rng.choice(len(changed), size=max(1, len(changed) // 2), replace=False)]
    for row, col in chosen:
        out[int(row), int(col)] = test[int(row), int(col)]
    return out, "difference_preserve_input_too_much"


def _negative_difference_over_transform(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs
    out = gold.copy()
    if test.shape != gold.shape:
        return _force_single_cell_change(out, rng), "difference_over_transform_unchanged"
    unchanged = np.argwhere(test == gold)
    changed_colors = [int(gold[row, col]) for row, col in np.argwhere(test != gold)]
    fill = int(rng.choice(changed_colors)) if changed_colors else int(_dominant_color(gold))
    if len(unchanged) == 0:
        return _force_single_cell_change(out, rng), "difference_over_transform_unchanged"
    chosen = unchanged[rng.choice(len(unchanged), size=max(1, min(len(unchanged), max(1, gold.size // 12))), replace=False)]
    for row, col in chosen:
        out[int(row), int(col)] = fill
    return out, "difference_over_transform_unchanged"


def _negative_cross_train_output(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    if not train_pairs:
        return _negative_random_balanced(train_pairs, test, gold, rng)
    pair = train_pairs[int(rng.integers(0, len(train_pairs)))]
    return _resize_to(_array(pair.output), gold.shape, fill=_dominant_color(gold)), "cross_example_train_output_adapted"


def _negative_cross_train_diff(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    if not train_pairs or test.shape != gold.shape:
        return _negative_random_balanced(train_pairs, test, gold, rng)
    out = test.copy()
    compatible = [pair for pair in train_pairs if _shape(pair.input) == _shape(pair.output)]
    if not compatible:
        return _negative_random_balanced(train_pairs, test, gold, rng)
    pair = compatible[int(rng.integers(0, len(compatible)))]
    inp = _array(pair.input)
    outp = _array(pair.output)
    diff = np.argwhere(inp != outp)
    if len(diff) == 0:
        return _force_single_cell_change(gold, rng), "cross_example_wrong_region_transform"
    offset_r = int(rng.integers(0, max(1, gold.shape[0])))
    offset_c = int(rng.integers(0, max(1, gold.shape[1])))
    for row, col in diff:
        rr = (int(row) + offset_r) % gold.shape[0]
        cc = (int(col) + offset_c) % gold.shape[1]
        out[rr, cc] = outp[int(row), int(col)]
    return out, "cross_example_wrong_region_transform"


def _negative_random_balanced(
    train_pairs: Sequence[ArcPair],
    test: np.ndarray,
    gold: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    del train_pairs, test
    flat = gold.reshape(-1).copy()
    rng.shuffle(flat)
    out = flat.reshape(gold.shape).copy()
    if np.array_equal(out, gold):
        out = _force_single_cell_change(out, rng)
    return out, "random_balanced_palette_counts"


def _salted_balanced_grid(gold: np.ndarray, rng: np.random.Generator, salt: int) -> np.ndarray:
    out = gold.copy()
    if out.size == 0:
        return out
    row = int((salt * 7 + rng.integers(0, max(1, out.shape[0]))) % out.shape[0])
    col = int((salt * 11 + rng.integers(0, max(1, out.shape[1]))) % out.shape[1])
    colors = list(range(10))
    out[row, col] = int((int(out[row, col]) + 1 + salt) % len(colors))
    return out


def _components(grid: np.ndarray) -> List[List[Tuple[int, int]]]:
    if grid.size == 0:
        return []
    bg = int(_dominant_color(grid))
    seen = np.zeros(grid.shape, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if seen[row, col] or int(grid[row, col]) == bg:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            component: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(component)
    return components


def _resize_to(grid: np.ndarray, shape: Tuple[int, int], fill: int) -> np.ndarray:
    out = np.full(shape, int(fill), dtype=np.int64)
    h = min(shape[0], grid.shape[0])
    w = min(shape[1], grid.shape[1])
    out[:h, :w] = grid[:h, :w]
    return out


def _force_single_cell_change(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = grid.copy()
    if out.size == 0:
        return out
    row = int(rng.integers(0, out.shape[0]))
    col = int(rng.integers(0, out.shape[1]))
    out[row, col] = int((int(out[row, col]) + 1) % 10)
    return out


def _array(grid: Grid) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def _clean_grid(grid: object) -> Grid:
    rows = []
    for row in grid if isinstance(grid, list) else []:
        rows.append([int(value) for value in row])
    return rows


def _shape(grid: Grid) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _palette(grid: Grid) -> List[int]:
    return sorted({int(value) for row in grid for value in row})


def _palette_array(grid: np.ndarray) -> List[int]:
    return sorted(int(value) for value in np.unique(grid))


def _dominant_color(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _grid_key(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.astype(np.int8, copy=False).tobytes() + str(tuple(grid.shape)).encode("ascii")).hexdigest()


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _blank_like(grid: Grid) -> Grid:
    h, w = _shape(grid)
    return [[0 for _ in range(w)] for _ in range(h)]


def _map_grid(grid: Grid, mapping: Dict[int, int]) -> Grid:
    return [[int(mapping.get(int(value), int(value))) for value in row] for row in grid]


def _pad_grid(grid: Grid) -> Grid:
    arr = _array(grid)
    if arr.size == 0:
        return grid
    bg = int(_dominant_color(arr))
    out = np.full((arr.shape[0] + 1, arr.shape[1] + 1), bg, dtype=np.int64)
    out[1:, 1:] = arr
    return out.tolist()


def _count_strings(values: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
