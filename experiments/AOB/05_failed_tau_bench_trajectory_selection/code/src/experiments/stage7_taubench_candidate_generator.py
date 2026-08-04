from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.datasets.taubench_trajectory_selection_dataset import (
    STAGE7_NUM_CANDIDATES,
    SelectorVisibleTrajectory,
    TauBenchTrajectoryCandidate,
    write_taubench_candidates_jsonl,
)


BENCHMARK = "stage7_taubench_trajectory_selector"
DEFAULT_OUTPUT = Path("results/stage7a_taubench_smoke_candidates.jsonl")
DEFAULT_RAW_ROOT = Path("results/stage7_taubench_raw_trajectories")
DEFAULT_EVAL_ROOT = Path("results/stage7_taubench_eval_metadata")


@dataclass(frozen=True)
class TauBenchGeneratorConfig:
    name: str
    agent: str = "llm_agent"
    agent_llm: str = "gpt-4.1-mini"
    agent_llm_args: Dict[str, object] | None = None
    extra_args: Sequence[str] = ()

    def to_cli_args(self) -> List[str]:
        args = ["--agent", self.agent, "--agent-llm", self.agent_llm]
        if self.agent_llm_args:
            args.extend(["--agent-llm-args", json.dumps(self.agent_llm_args, sort_keys=True)])
        args.extend(str(value) for value in self.extra_args)
        return args


DEFAULT_GENERATOR_CONFIGS: Sequence[TauBenchGeneratorConfig] = (
    TauBenchGeneratorConfig("base_tool_agent_temp0", agent_llm_args={"temperature": 0.0}),
    TauBenchGeneratorConfig("base_tool_agent_temp02", agent_llm_args={"temperature": 0.2}),
    TauBenchGeneratorConfig("base_tool_agent_temp05", agent_llm_args={"temperature": 0.5}),
    TauBenchGeneratorConfig("planning_agent", agent_llm_args={"temperature": 0.2, "prompt_variant": "planning"}),
    TauBenchGeneratorConfig("policy_aware_agent", agent_llm_args={"temperature": 0.2, "prompt_variant": "policy_aware"}),
    TauBenchGeneratorConfig("conservative_agent", agent_llm_args={"temperature": 0.0, "prompt_variant": "conservative"}),
    TauBenchGeneratorConfig("retry_agent", agent_llm_args={"temperature": 0.2, "retry_on_tool_error": True}),
    TauBenchGeneratorConfig("alternate_prompt_agent", agent_llm_args={"temperature": 0.4, "prompt_variant": "alternate"}),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 7 tau-bench K=8 trajectory candidate pools.")
    parser.add_argument("--domain", default="retail", choices=("retail", "airline"))
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-set-name", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--tau2-command", default="tau2")
    parser.add_argument("--tau2-workdir", default=None)
    parser.add_argument("--user-llm", default="gpt-4.1-mini")
    parser.add_argument("--user-llm-args", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=707_000)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Build a local synthetic tau-like pool. Not benchmark evidence.")
    args = parser.parse_args()

    task_ids = args.task_ids if args.task_ids else [str(index) for index in range(int(args.n_tasks))]
    if args.synthetic_smoke:
        candidates = build_synthetic_stage7_candidate_records(
            domain=str(args.domain),
            task_ids=task_ids,
            raw_root=Path(args.raw_root),
            eval_root=Path(args.eval_root),
            seed=int(args.seed),
        )
    else:
        candidates = generate_taubench_candidate_records(
            domain=str(args.domain),
            task_ids=task_ids,
            task_split_name=str(args.task_split_name),
            task_set_name=args.task_set_name,
            raw_root=Path(args.raw_root),
            eval_root=Path(args.eval_root),
            tau2_command=str(args.tau2_command),
            tau2_workdir=None if args.tau2_workdir is None else Path(args.tau2_workdir),
            user_llm=str(args.user_llm),
            user_llm_args=json.loads(args.user_llm_args) if args.user_llm_args else {},
            max_steps=int(args.max_steps),
            max_concurrency=int(args.max_concurrency),
            seed=int(args.seed),
            generator_configs=DEFAULT_GENERATOR_CONFIGS,
        )
    write_taubench_candidates_jsonl(args.output, candidates)
    print(f"stage7 candidate generator: wrote {len(candidates)} candidates to {args.output}")


def generate_taubench_candidate_records(
    domain: str,
    task_ids: Sequence[str],
    task_split_name: str,
    task_set_name: str | None,
    raw_root: Path,
    eval_root: Path,
    tau2_command: str,
    tau2_workdir: Path | None,
    user_llm: str,
    user_llm_args: Dict[str, object],
    max_steps: int,
    max_concurrency: int,
    seed: int,
    generator_configs: Sequence[TauBenchGeneratorConfig] = DEFAULT_GENERATOR_CONFIGS,
) -> List[TauBenchTrajectoryCandidate]:
    if len(generator_configs) != STAGE7_NUM_CANDIDATES:
        raise ValueError(f"Stage 7 requires exactly K={STAGE7_NUM_CANDIDATES} generator configs")
    if not _command_available(tau2_command, tau2_workdir):
        raise RuntimeError(
            "tau2 CLI is not available. Install the official tau2/tau3 benchmark first, or use --synthetic-smoke only for local tests."
        )
    raw_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 707_101)
    rows: List[TauBenchTrajectoryCandidate] = []
    for task_position, task_id in enumerate(task_ids):
        generated_for_task: List[TauBenchTrajectoryCandidate] = []
        for slot, generator in enumerate(generator_configs):
            run_seed = int(seed + task_position * 1000 + slot)
            save_name = f"stage7_{domain}_task_{_safe_name(task_id)}_{generator.name}_{run_seed}"
            started = time.perf_counter()
            command = [
                tau2_command,
                "run",
                "--domain",
                domain,
                "--user-llm",
                user_llm,
                "--num-trials",
                "1",
                "--task-ids",
                str(task_id),
                "--task-split-name",
                task_split_name,
                "--max-steps",
                str(max_steps),
                "--max-concurrency",
                str(max_concurrency),
                "--seed",
                str(run_seed),
                "--save-to",
                save_name,
                "--log-level",
                "ERROR",
                "--auto-resume",
            ]
            if task_set_name:
                command.extend(["--task-set-name", task_set_name])
            if user_llm_args:
                command.extend(["--user-llm-args", json.dumps(user_llm_args, sort_keys=True)])
            command.extend(generator.to_cli_args())
            completed = subprocess.run(
                command,
                cwd=None if tau2_workdir is None else tau2_workdir,
                text=True,
                capture_output=True,
                timeout=max(60, int(max_steps) * 30),
            )
            run_dir = _candidate_run_dir(tau2_workdir, save_name)
            raw_record = _load_first_tau2_record(run_dir)
            if raw_record is None:
                raw_record = {
                    "task_id": str(task_id),
                    "domain": domain,
                    "command": command,
                    "returncode": int(completed.returncode),
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                    "missing_tau2_result_record": True,
                }
            raw_path = raw_root / domain / str(task_id) / f"slot_{slot}_{generator.name}_{run_seed}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True), encoding="utf-8")
            eval_meta_path = eval_root / domain / str(task_id) / f"slot_{slot}_{generator.name}_{run_seed}.json"
            eval_meta_path.parent.mkdir(parents=True, exist_ok=True)
            official_success = _extract_official_success(raw_record)
            eval_meta_path.write_text(
                json.dumps(
                    {
                        "candidate_slot": slot,
                        "generator_name": generator.name,
                        "tau2_returncode": int(completed.returncode),
                        "official_success_from_generation_record": official_success,
                        "latency_seconds": float(time.perf_counter() - started),
                        "raw_record_path": str(raw_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            generated_for_task.append(
                TauBenchTrajectoryCandidate(
                    task_id=str(task_id),
                    domain=domain,
                    candidate_id=f"{domain}-{task_id}-cand-{slot}",
                    candidate_order_index=slot,
                    generator_name=generator.name,
                    generator_config=json.dumps(asdict(generator), sort_keys=True),
                    seed=run_seed,
                    raw_trajectory_path=str(raw_path),
                    selector_visible_trajectory=sanitize_tau2_trajectory(raw_record),
                    official_success=official_success,
                    official_eval_metadata_path=str(eval_meta_path),
                    metadata={
                        "candidate_order_randomized": False,
                        "candidate_order_seed": int(seed),
                        "tau_bench_task_split_name": task_split_name,
                        "tau_bench_task_set_name": task_set_name or "default",
                        "generation_latency_seconds": float(time.perf_counter() - started),
                        "selector_visible_excludes_generator_identity": True,
                    },
                )
            )
        order = rng.permutation(len(generated_for_task))
        for order_index, old_index in enumerate(order.tolist()):
            rows.append(replace(generated_for_task[int(old_index)], candidate_order_index=int(order_index)))
    return rows


def build_synthetic_stage7_candidate_records(
    domain: str,
    task_ids: Sequence[str],
    raw_root: Path,
    eval_root: Path,
    seed: int,
) -> List[TauBenchTrajectoryCandidate]:
    """Build deterministic tau-like trajectories for tests. These are not benchmark evidence."""

    rng = np.random.default_rng(seed + 707_202)
    generator_names = [config.name for config in DEFAULT_GENERATOR_CONFIGS]
    rows: List[TauBenchTrajectoryCandidate] = []
    for task_index, task_id in enumerate(task_ids):
        task_type = _synthetic_task_type(domain, task_index)
        goal_entity = f"{domain[:3].upper()}-{task_index:04d}"
        policy_token = f"verify_{(task_index + seed) % 5}"
        pass_slots = _synthetic_pass_slots(task_index, rng)
        task_candidates: List[TauBenchTrajectoryCandidate] = []
        policy_text = (
            f"{domain} policy for {task_type}: verify customer identity with {policy_token}; "
            "use the matching entity id; do not claim completion after a tool error."
        )
        public_request = f"Customer asks to {task_type.replace('_', ' ')} for entity {goal_entity} after verification {policy_token}."
        for slot, generator_name in enumerate(generator_names):
            success = slot in pass_slots
            visible = _synthetic_visible_trajectory(
                domain=domain,
                task_type=task_type,
                goal_entity=goal_entity,
                policy_token=policy_token,
                candidate_slot=slot,
                success=success,
            )
            raw = {
                "benchmark": BENCHMARK,
                "synthetic_smoke_only": True,
                "domain": domain,
                "task_id": str(task_id),
                "candidate_slot": slot,
                "generator_name": generator_name,
                "success": bool(success),
                "selector_visible_trajectory": visible.to_record(),
            }
            raw_path = raw_root / domain / str(task_id) / f"candidate_{slot}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
            eval_path = eval_root / domain / str(task_id) / f"candidate_{slot}.json"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            eval_path.write_text(
                json.dumps(
                    {
                        "synthetic_smoke_only": True,
                        "official_success": bool(success),
                        "label_source": "deterministic_local_synthetic_stage7_smoke",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            split = "train" if task_index % 5 in {0, 1, 2} else "dev" if task_index % 5 == 3 else "test"
            task_candidates.append(
                TauBenchTrajectoryCandidate(
                    task_id=str(task_id),
                    domain=domain,
                    candidate_id=f"{domain}-{task_id}-cand-{slot}",
                    candidate_order_index=slot,
                    generator_name=generator_name,
                    generator_config=json.dumps({"synthetic": True, "slot": slot, "name": generator_name}, sort_keys=True),
                    seed=int(seed + task_index * 100 + slot),
                    raw_trajectory_path=str(raw_path),
                    selector_visible_trajectory=visible,
                    official_success=bool(success),
                    official_eval_metadata_path=str(eval_path),
                    metadata={
                        "split": split,
                        "task_type": task_type,
                        "policy_text": policy_text,
                        "public_task_request": public_request,
                        "candidate_order_randomized": False,
                        "candidate_order_seed": int(seed),
                        "tau_bench_task_split_name": "synthetic_smoke",
                        "tau_bench_task_set_name": "synthetic_smoke",
                        "publishable_claim_allowed": False,
                        "synthetic_smoke_only": True,
                        "selector_visible_excludes_generator_identity": True,
                    },
                )
            )
        order = rng.permutation(STAGE7_NUM_CANDIDATES)
        for order_index, old_index in enumerate(order.tolist()):
            old = task_candidates[int(old_index)]
            rows.append(
                replace(
                    old,
                    candidate_order_index=int(order_index),
                    metadata={**old.metadata, "candidate_order_randomized": True, "candidate_order": [int(value) for value in order.tolist()]},
                )
            )
    return rows


def sanitize_tau2_trajectory(raw_record: Dict[str, object]) -> SelectorVisibleTrajectory:
    messages = _extract_messages(raw_record)
    user_messages: List[str] = []
    assistant_messages: List[str] = []
    tool_observations: List[str] = []
    tool_calls: List[Dict[str, object]] = []
    for message in messages:
        role = str(message.get("role", message.get("type", ""))).lower()
        content = _message_content(message)
        if role == "user":
            user_messages.append(content)
        elif role in {"assistant", "agent"}:
            assistant_messages.append(content)
            for call in _message_tool_calls(message):
                tool_calls.append(_sanitize_tool_call(call))
        elif role in {"tool", "observation", "environment"}:
            tool_observations.append(_sanitize_observation(content))
        else:
            if "tool" in message:
                tool_calls.append(_sanitize_tool_call(_safe_dict(message.get("tool"))))
            if "observation" in message:
                tool_observations.append(_sanitize_observation(str(message.get("observation"))))
    if not tool_calls:
        tool_calls.extend(_extract_tool_calls(raw_record))
    if not tool_observations:
        tool_observations.extend(_extract_observations(raw_record))
    final_response = next((text for text in reversed(assistant_messages) if text.strip()), "")
    return SelectorVisibleTrajectory(
        user_messages=tuple(_drop_forbidden_selector_fields(user_messages)),
        assistant_messages=tuple(_drop_forbidden_selector_fields(assistant_messages)),
        tool_calls=tuple(tool_calls),
        tool_observations_sanitized=tuple(_drop_forbidden_selector_fields(tool_observations)),
        final_response=_drop_forbidden_selector_text(final_response),
    )


def _synthetic_visible_trajectory(
    domain: str,
    task_type: str,
    goal_entity: str,
    policy_token: str,
    candidate_slot: int,
    success: bool,
) -> SelectorVisibleTrajectory:
    wrong_entity = f"{goal_entity}-WRONG"
    entity = goal_entity if success else wrong_entity
    tool_name = _synthetic_success_tool(domain, task_type) if success else _synthetic_failure_tool(domain, task_type, candidate_slot)
    error_observation = f"tool_error: entity {entity} not eligible for requested operation"
    ok_observation = f"tool_result: {tool_name} completed for {goal_entity}; verification={policy_token}; state=updated"
    observations = (ok_observation,) if success else (f"tool_result: lookup returned {entity}", error_observation)
    final = (
        f"I verified {policy_token} and completed the {task_type.replace('_', ' ')} for {goal_entity}."
        if success
        else f"I completed the {task_type.replace('_', ' ')} for {entity}."
    )
    return SelectorVisibleTrajectory(
        user_messages=(
            f"Please {task_type.replace('_', ' ')} for {goal_entity}.",
            f"My verification code is {policy_token}.",
        ),
        assistant_messages=(
            "I will verify the customer and inspect the relevant record.",
            "I will now apply the allowed action based on the policy.",
        ),
        tool_calls=(
            {"tool_name": f"{domain}_lookup", "arguments": {"entity_id": goal_entity, "attempt_slot": candidate_slot}},
            {
                "tool_name": tool_name,
                "arguments": {"entity_id": entity, "verification": policy_token if success else "missing", "attempt_slot": candidate_slot},
            },
        ),
        tool_observations_sanitized=observations,
        final_response=final,
    )


def _synthetic_pass_slots(index: int, rng: np.random.Generator) -> Sequence[int]:
    if index % 9 == 0:
        return ()
    first = int((index * 5 + 2) % STAGE7_NUM_CANDIDATES)
    if index % 7 == 0:
        second = int((first + 3 + int(rng.integers(0, 2))) % STAGE7_NUM_CANDIDATES)
        return (first, second) if second != first else (first,)
    return (first,)


def _synthetic_task_type(domain: str, index: int) -> str:
    retail = ("cancel_order", "modify_address", "return_item", "exchange_item", "status_lookup")
    airline = ("cancel_reservation", "change_flight", "book_flight", "refund_ticket", "baggage_lookup")
    values = airline if domain == "airline" else retail
    return values[index % len(values)]


def _synthetic_success_tool(domain: str, task_type: str) -> str:
    return f"{domain}_{task_type}"


def _synthetic_failure_tool(domain: str, task_type: str, slot: int) -> str:
    if slot % 3 == 0:
        return f"{domain}_status_lookup"
    if slot % 3 == 1:
        return f"{domain}_{task_type}_without_verification"
    return f"{domain}_unsupported_action"


def _command_available(command: str, cwd: Path | None) -> bool:
    try:
        completed = subprocess.run([command, "--help"], cwd=cwd, text=True, capture_output=True, timeout=20)
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _candidate_run_dir(workdir: Path | None, save_name: str) -> Path:
    root = Path.cwd() if workdir is None else workdir
    return root / "data" / "simulations" / save_name


def _load_first_tau2_record(run_dir: Path) -> Dict[str, object] | None:
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.rglob("*.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record = _find_simulation_record(data)
        if record is not None:
            return record
    return None


def _find_simulation_record(data: object) -> Dict[str, object] | None:
    if isinstance(data, dict):
        if any(key in data for key in ("task_id", "reward", "messages", "trajectory", "events", "simulation")):
            return data
        for value in data.values():
            found = _find_simulation_record(value)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_simulation_record(value)
            if found is not None:
                return found
    return None


def _extract_official_success(record: Dict[str, object]) -> bool | None:
    for key in ("success", "reward", "task_reward", "environment_reward", "passed"):
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, (int, float)):
                return float(value) > 0.0
            text = str(value).lower()
            if text in {"true", "success", "passed", "1"}:
                return True
            if text in {"false", "failed", "0"}:
                return False
    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        return _extract_official_success(evaluation)
    return None


def _extract_messages(record: Dict[str, object]) -> List[Dict[str, object]]:
    for key in ("messages", "trajectory", "conversation", "turns", "events", "steps"):
        value = record.get(key)
        rows = _coerce_message_rows(value)
        if rows:
            return rows
    rows: List[Dict[str, object]] = []
    _walk_for_messages(record, rows)
    return rows


def _coerce_message_rows(value: object) -> List[Dict[str, object]]:
    if isinstance(value, list):
        return [_safe_dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("messages", "turns", "events", "steps"):
            rows = _coerce_message_rows(value.get(key))
            if rows:
                return rows
    return []


def _walk_for_messages(value: object, out: List[Dict[str, object]]) -> None:
    if isinstance(value, dict):
        if ("role" in value or "type" in value) and ("content" in value or "message" in value or "text" in value):
            out.append(dict(value))
        for child in value.values():
            _walk_for_messages(child, out)
    elif isinstance(value, list):
        for child in value:
            _walk_for_messages(child, out)


def _message_content(message: Dict[str, object]) -> str:
    for key in ("content", "message", "text", "response"):
        if key in message and message[key] is not None:
            value = message[key]
            if isinstance(value, str):
                return _drop_forbidden_selector_text(value)
            return _drop_forbidden_selector_text(json.dumps(value, sort_keys=True))
    return ""


def _message_tool_calls(message: Dict[str, object]) -> List[Dict[str, object]]:
    for key in ("tool_calls", "tools", "actions"):
        value = message.get(key)
        if isinstance(value, list):
            return [_safe_dict(row) for row in value]
        if isinstance(value, dict):
            return [dict(value)]
    return []


def _extract_tool_calls(record: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    _walk_for_key(record, {"tool_call", "tool_calls", "action", "actions"}, rows)
    return [_sanitize_tool_call(row) for row in rows[:64]]


def _extract_observations(record: Dict[str, object]) -> List[str]:
    rows: List[Dict[str, object]] = []
    _walk_for_key(record, {"observation", "observations", "tool_result", "tool_results"}, rows)
    values = []
    for row in rows[:64]:
        values.append(_sanitize_observation(json.dumps(row, sort_keys=True)))
    return values


def _walk_for_key(value: object, names: set[str], out: List[Dict[str, object]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in names:
                if isinstance(child, list):
                    out.extend(_safe_dict(row) for row in child if isinstance(row, dict))
                elif isinstance(child, dict):
                    out.append(dict(child))
                else:
                    out.append({str(key): str(child)})
            _walk_for_key(child, names, out)
    elif isinstance(value, list):
        for child in value:
            _walk_for_key(child, names, out)


def _sanitize_tool_call(call: Dict[str, object]) -> Dict[str, object]:
    allowed = {}
    for key, value in call.items():
        if _forbidden_key(key):
            continue
        if isinstance(value, dict):
            allowed[str(key)] = {str(k): _drop_forbidden_selector_text(str(v)) for k, v in value.items() if not _forbidden_key(k)}
        elif isinstance(value, list):
            allowed[str(key)] = [_drop_forbidden_selector_text(str(child)) for child in value[:20]]
        else:
            allowed[str(key)] = _drop_forbidden_selector_text(str(value))
    return allowed


def _sanitize_observation(text: str) -> str:
    return _drop_forbidden_selector_text(str(text))[:4000]


def _drop_forbidden_selector_fields(values: Iterable[str]) -> List[str]:
    return [_drop_forbidden_selector_text(value) for value in values]


def _drop_forbidden_selector_text(text: str) -> str:
    value = str(text)
    forbidden = (
        "official_success",
        "official_eval",
        "evaluator_verdict",
        "hidden_goal_state",
        "database_diff",
        "candidate_rank",
        "generator_name",
        "generator_config",
    )
    for token in forbidden:
        value = value.replace(token, "[redacted]")
    return value


def _forbidden_key(key: object) -> bool:
    text = str(key).lower()
    return any(token in text for token in ("official", "evaluator", "hidden_goal", "database_diff", "generator"))


def _safe_dict(value: object) -> Dict[str, object]:
    return dict(value) if isinstance(value, dict) else {"value": str(value)}


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:120]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
