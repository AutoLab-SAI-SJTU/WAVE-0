#!/usr/bin/env python3
"""Validate and aggregate WAVE-0 original LIBERO worker JSONL shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(run_root: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((run_root / "shards").glob("worker_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    by_key[(row["suite"], int(row["task_id"]))] = row
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    return list(by_key.values())


def validate_row(row: dict[str, Any], expected_episodes: int) -> tuple[bool, str]:
    if row.get("status") != "ok":
        return False, str(row.get("error", "worker failed"))
    successes = row.get("successes")
    seeds = row.get("seeds")
    if not isinstance(successes, list) or len(successes) != expected_episodes:
        return False, f"expected {expected_episodes} successes, got {len(successes) if isinstance(successes, list) else type(successes).__name__}"
    if any(type(value) is not bool for value in successes):
        return False, "successes must contain only booleans"
    if not isinstance(seeds, list) or len(seeds) != expected_episodes:
        return False, f"expected {expected_episodes} seeds, got {len(seeds) if isinstance(seeds, list) else type(seeds).__name__}"
    if any(type(value) is not int for value in seeds) or len(set(seeds)) != expected_episodes:
        return False, "seeds must be unique integers"
    expected_rate = 100.0 * sum(successes) / expected_episodes
    if not math.isclose(float(row.get("pc_success", -1.0)), expected_rate, abs_tol=1e-6):
        return False, "pc_success does not match successes"
    return True, ""


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_config = json.loads((run_root / "run_config.json").read_text())
    episodes = int(run_config["n_episodes"])
    rows = load_rows(run_root)
    validated: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        ok, reason = validate_row(row, episodes)
        row = dict(row)
        row["validation_error"] = reason
        if ok:
            validated.append(row)
        else:
            invalid.append(row)

    expected = args.expected_tasks
    complete = len(rows) == expected and len(validated) == expected and not invalid
    all_successes = [value for row in validated for value in row["successes"]]
    task_rates = [float(row["pc_success"]) for row in validated]
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)

    per_suite: dict[str, Any] = {}
    for suite, suite_rows in sorted(by_suite.items()):
        good = [row for row in suite_rows if row in validated]
        per_suite[suite] = {
            "completed_tasks": len(good),
            "total_tasks": len(suite_rows),
            "failed_tasks": len(suite_rows) - len(good),
            "episodes": len(good) * episodes,
            "successes": sum(sum(row["successes"]) for row in good),
            "success_rate": mean([float(row["pc_success"]) for row in good]),
        }

    summary = {
        "schema_version": 1,
        "complete": complete,
        "total_tasks": expected,
        "completed_tasks": len(validated),
        "failed_tasks": expected - len(validated),
        "episodes_per_task": episodes,
        "total_episodes": expected * episodes,
        "completed_episodes": len(all_successes),
        "successful_episodes": sum(all_successes),
        "overall_success_rate": 100.0 * sum(all_successes) / len(all_successes) if all_successes else None,
        "macro_task_success_rate": mean(task_rates),
        "invalid_rows": [
            {"suite": row.get("suite"), "task_id": row.get("task_id"), "error": row["validation_error"]}
            for row in invalid
        ],
        "per_suite": per_suite,
        "protocol": {
            "seed": run_config["seed"],
            "chunk_size": run_config["chunk_size"],
            "n_action_steps": run_config["n_action_steps"],
            "world_model_stage": run_config["world_model_stage"],
            "compile_model": run_config["compile_model"],
            "wam_loss_warmup_steps": 1000,
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (run_root / "summary.csv").open("w", newline="") as stream:
        fields = ["suite", "task_id", "status", "pc_success", "n_episodes", "worker_id", "elapsed_s", "validation_error", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["suite"], int(item["task_id"]))):
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"Completed: {len(validated)}/{expected}")
    print(f"Failed or invalid: {expected - len(validated)}")
    if summary["macro_task_success_rate"] is not None:
        print(f"Macro task success: {summary['macro_task_success_rate']:.2f}%")
    if summary["overall_success_rate"] is not None:
        print(f"Overall episode success: {summary['overall_success_rate']:.2f}%")
    print(f"Wrote {run_root / 'summary.json'}")
    print(f"Wrote {run_root / 'summary.csv'}")
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
