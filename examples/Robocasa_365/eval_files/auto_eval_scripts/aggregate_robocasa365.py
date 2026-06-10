#!/usr/bin/env python3
"""Aggregate RoboCasa-365 per-task eval JSON files into leaderboard summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from examples.Robocasa_365.eval_files.auto_eval_scripts.gen_target_tasks import (
        TARGET_SPLITS,
        TaskRow,
        build_task_rows,
    )
except ImportError:  # pragma: no cover - useful when run from this directory directly
    from gen_target_tasks import TARGET_SPLITS, TaskRow, build_task_rows


def eval_dir_for_ckpt(ckpt: str) -> Path:
    return Path(ckpt).with_suffix(".eval")


def result_path(eval_dir: Path, env_name: str) -> Path:
    return eval_dir / f"{env_name.replace('/', '_')}.json"


def read_tasks_tsv(path: str) -> list[TaskRow]:
    rows: list[TaskRow] = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, parts in enumerate(reader, start=1):
            if not parts or all(not part.strip() for part in parts):
                continue
            if parts[0] == "split":
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 TSV columns, got {len(parts)}")
            split, task_name, horizon = parts
            rows.append(TaskRow(split=split, task_name=task_name, horizon=int(horizon)))
    return rows


def load_task_rows(args: argparse.Namespace) -> list[TaskRow]:
    if args.tasks_tsv:
        return read_tasks_tsv(args.tasks_tsv)
    return build_task_rows(args.task_filter, check_gym=not args.no_check_gym)


def load_successes(path: Path) -> tuple[list[bool] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return None, f"json_error: {exc}"

    successes = data.get("successes")
    if not isinstance(successes, list):
        return None, "successes_not_list"
    return [bool(item) for item in successes], None


def mean_or_none(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def aggregate(rows: Sequence[TaskRow], ckpt: str, n_episodes: int) -> dict[str, Any]:
    eval_dir = eval_dir_for_ckpt(ckpt)
    allowlist = {row.env_name for row in rows}

    completed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for row in rows:
        path = result_path(eval_dir, row.env_name)
        successes, error = load_successes(path)
        if error is not None:
            incomplete.append(
                {
                    "split": row.split,
                    "task_name": row.task_name,
                    "env_name": row.env_name,
                    "horizon": row.horizon,
                    "path": str(path),
                    "reason": error,
                }
            )
            continue
        assert successes is not None
        if len(successes) != n_episodes:
            incomplete.append(
                {
                    "split": row.split,
                    "task_name": row.task_name,
                    "env_name": row.env_name,
                    "horizon": row.horizon,
                    "path": str(path),
                    "reason": f"expected {n_episodes} successes, got {len(successes)}",
                }
            )
            continue

        successes_int = sum(1 for item in successes if item)
        completed.append(
            {
                "split": row.split,
                "task_name": row.task_name,
                "env_name": row.env_name,
                "horizon": row.horizon,
                "path": str(path),
                "n_episodes": n_episodes,
                "successes": successes_int,
                "success_rate": successes_int / n_episodes if n_episodes else 0.0,
            }
        )

    stale_json = []
    if eval_dir.exists():
        for path in sorted(eval_dir.glob("*.json")):
            if path.name == "_summary.json":
                continue
            env_name = path.stem
            if env_name.startswith("robocasa_"):
                env_name = "robocasa/" + env_name[len("robocasa_") :]
            if env_name not in allowlist:
                stale_json.append(str(path))

    by_split: dict[str, Any] = {}
    for split in TARGET_SPLITS:
        split_rows = [row for row in rows if row.split == split]
        split_completed = [row for row in completed if row["split"] == split]
        rates = [row["success_rate"] for row in split_completed]
        by_split[split] = {
            "expected_tasks": len(split_rows),
            "completed_tasks": len(split_completed),
            "incomplete_tasks": len(split_rows) - len(split_completed),
            "success_rate": mean_or_none(rates),
        }

    overall_rates = [row["success_rate"] for row in completed]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ckpt": ckpt,
        "eval_dir": str(eval_dir),
        "n_episodes": n_episodes,
        "leaderboard_ready": len(completed) == len(rows) and len(rows) == 50,
        "expected_tasks": len(rows),
        "completed_tasks": len(completed),
        "incomplete_tasks": len(incomplete),
        "overall_success_rate": mean_or_none(overall_rates),
        "splits": by_split,
        "tasks": completed,
        "incomplete": incomplete,
        "ignored_stale_json": stale_json,
        "task_table": [asdict(row) for row in rows],
    }
    return summary


def write_summary(summary: dict[str, Any]) -> Path:
    eval_dir = Path(summary["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / "_summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return out


def fmt_rate(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:6.2f}%"


def print_report(summary: dict[str, Any]) -> None:
    print("\n=== RoboCasa-365 Summary ===")
    print(f"ckpt: {summary['ckpt']}")
    print(f"eval_dir: {summary['eval_dir']}")
    print(
        "tasks: "
        f"{summary['completed_tasks']}/{summary['expected_tasks']} complete, "
        f"{summary['incomplete_tasks']} incomplete"
    )
    print(f"overall: {fmt_rate(summary['overall_success_rate'])}")
    print(f"leaderboard_ready: {summary['leaderboard_ready']}")

    print("\nSplit                 complete   success_rate")
    print("--------------------  --------  ------------")
    for split in TARGET_SPLITS:
        item = summary["splits"].get(split, {})
        complete = f"{item.get('completed_tasks', 0)}/{item.get('expected_tasks', 0)}"
        print(f"{split:<20}  {complete:>8}  {fmt_rate(item.get('success_rate')):>12}")

    print("\nTask                                    split                 sr")
    print("--------------------------------------  --------------------  --------")
    for task in sorted(summary["tasks"], key=lambda item: (item["split"], item["task_name"])):
        print(f"{task['task_name']:<38}  {task['split']:<20}  {fmt_rate(task['success_rate']):>8}")

    if summary["incomplete"]:
        print("\nIncomplete tasks")
        for item in summary["incomplete"]:
            print(f"  {item['env_name']}: {item['reason']}")
    if summary["ignored_stale_json"]:
        print("\nIgnored stale JSON")
        for path in summary["ignored_stale_json"]:
            print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path used by simulation_env.py")
    parser.add_argument("--tasks-tsv", default=None, help="Task allowlist TSV from gen_target_tasks.py")
    parser.add_argument("--task-filter", default="all", help="Used only when --tasks-tsv is omitted")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--no-check-gym", action="store_true", help="Skip gym registration check when generating tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_task_rows(args)
    summary = aggregate(rows, args.ckpt, args.n_episodes)
    out = write_summary(summary)
    print_report(summary)
    print()
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
