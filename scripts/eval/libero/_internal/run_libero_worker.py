#!/usr/bin/env python3
"""Evaluate one shard of the original 40-task LIBERO benchmark with WAVE-0."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_one
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def language_from_task_name(task_name: str) -> str:
    if task_name and task_name[0].isupper():
        marker = task_name.find("SCENE")
        if marker >= 0:
            offset = 8 if task_name.startswith("SCENE10", marker) else 7
            return " ".join(task_name[marker + offset :].split("_"))
    return " ".join(task_name.split("_"))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError(f"Task catalog must be a non-empty JSON list: {path}")
    for task in tasks:
        task["task_id"] = int(task["task_id"])
        task.setdefault("classification_id", task["task_id"] + 1)
        task.setdefault("category", "Original")
        task.setdefault("difficulty_level", None)
    return tasks


def result_key(suite: str, task_id: int) -> str:
    return f"{suite}:{task_id}"


def shard_tasks(tasks: list[dict[str, Any]], worker_id: int, num_workers: int) -> list[dict[str, Any]]:
    return [task for index, task in enumerate(tasks) if index % num_workers == worker_id]


def load_done_keys(path: Path, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    accepted = {"ok"} if retry_failed else {"ok", "failed"}
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") in accepted:
                done.add(result_key(row["suite"], int(row["task_id"])))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


class FakeSuite:
    """Expose one original LIBERO task through the suite API used by LiberoEnv."""

    def __init__(self, suite_name: str, task_name: str):
        self.tasks = [
            Task(
                name=task_name,
                language=language_from_task_name(task_name),
                problem="Libero",
                problem_folder=suite_name,
                bddl_file=f"{task_name}.bddl",
                init_states_file=f"{task_name}.pruned_init",
            )
        ]
        self.n_tasks = 1
        self.name = suite_name

    def get_task(self, index: int) -> Task:
        return self.tasks[index]


def make_original_task_env(suite: str, task_name: str) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    suite_obj = FakeSuite(suite, task_name)

    def make_one() -> LiberoEnv:
        return LiberoEnv(
            task_suite=suite_obj,
            task_id=0,
            task_suite_name=suite,
            obs_type="pixels_agent_pos",
            init_states=True,
            observation_height=256,
            observation_width=256,
        )

    return {suite: {0: gym.vector.SyncVectorEnv([make_one])}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--world-model-stage", default="inference")
    parser.add_argument("--compile-model", default="false", choices=["true", "false"])
    parser.add_argument("--n-action-steps", type=int, default=10)
    args = parser.parse_args()

    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    if args.n_episodes < 1 or args.n_action_steps < 1:
        parser.error("n-episodes and n-action-steps must be positive")

    all_tasks = load_tasks(Path(args.tasks_json))
    assigned = shard_tasks(all_tasks, args.worker_id, args.num_workers)
    if args.max_tasks > 0:
        assigned = assigned[: args.max_tasks]

    output_jsonl = Path(args.output_jsonl)
    done = load_done_keys(output_jsonl, args.retry_failed)
    pending = [task for task in assigned if result_key(task["suite"], task["task_id"]) not in done]
    print(
        f"[worker {args.worker_id}/{args.num_workers}] assigned={len(assigned)} "
        f"done={len(assigned) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    set_seed(args.seed)
    device = get_safe_torch_device(args.device, log=True)
    env_cfg = LiberoEnvConfig(
        task="libero_goal",
        task_ids=[0],
        obs_type="pixels_agent_pos",
        init_states=True,
        observation_height=256,
        observation_width=256,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    if getattr(policy_cfg, "type", None) != "wave0":
        raise ValueError(f"Expected a WAVE-0 checkpoint (type=wave0), got {getattr(policy_cfg, 'type', None)!r}")
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = device.type if hasattr(device, "type") else str(device)
    if hasattr(policy_cfg, "dtype"):
        policy_cfg.dtype = "bfloat16"
    if hasattr(policy_cfg, "world_model_stage"):
        policy_cfg.world_model_stage = args.world_model_stage
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = args.compile_model == "true"
    if hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = args.n_action_steps

    # LIBERO maps its two cameras to image/image2; no RoboTwin-style rename is needed.
    rename_map: dict[str, str] = {}
    print(
        f"[worker {args.worker_id}] loading WAVE-0 from {args.policy_path} "
        f"stage={getattr(policy_cfg, 'world_model_stage', '?')} "
        f"compile={getattr(policy_cfg, 'compile_model', '?')} "
        f"n_action_steps={getattr(policy_cfg, 'n_action_steps', '?')}",
        flush=True,
    )
    started_loading = time.time()
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    print(f"[worker {args.worker_id}] policy ready in {time.time() - started_loading:.1f}s", flush=True)

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if getattr(policy.config, "use_amp", False) and device.type == "cuda"
        else nullcontext()
    )
    for index, task in enumerate(pending, start=1):
        suite = task["suite"]
        task_id = int(task["task_id"])
        started = time.time()
        row: dict[str, Any] = {
            "suite": suite,
            "task_id": task_id,
            "classification_id": task.get("classification_id", task_id + 1),
            "name": task["name"],
            "category": task.get("category", "Original"),
            "difficulty_level": task.get("difficulty_level"),
            "worker_id": args.worker_id,
            "seed": args.seed,
            "n_episodes": args.n_episodes,
        }
        envs = None
        try:
            envs = make_original_task_env(suite, task["name"])
            with torch.no_grad(), autocast:
                metrics = eval_one(
                    envs[suite][0],
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=0,
                    videos_dir=None,
                    return_episode_data=False,
                    start_seed=args.seed,
                )
            successes = [bool(value) for value in metrics["successes"]]
            seeds = metrics.get("seeds", [])
            row.update(
                {
                    "status": "ok",
                    "successes": successes,
                    "seeds": [int(seed) for seed in seeds if seed is not None],
                    "pc_success": 100.0 * sum(successes) / len(successes),
                    "sum_rewards": [float(value) for value in metrics["sum_rewards"]],
                    "max_rewards": [float(value) for value in metrics["max_rewards"]],
                    "elapsed_s": time.time() - started,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep other tasks running
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                    "successes": [],
                    "seeds": [],
                    "pc_success": None,
                    "elapsed_s": time.time() - started,
                }
            )
        finally:
            if envs is not None:
                try:
                    close_envs(envs)
                except Exception:  # noqa: BLE001
                    pass
        append_jsonl(output_jsonl, row)
        print(
            f"[worker {args.worker_id}] {index}/{len(pending)} {suite}:{task_id} "
            f"{row['status']} success={row.get('pc_success')} elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
