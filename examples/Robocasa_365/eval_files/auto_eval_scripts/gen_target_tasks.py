#!/usr/bin/env python3
"""Generate the RoboCasa-365 target task table for batch evaluation.

The default output is the leaderboard target50 set: 18 atomic_seen,
16 composite_seen, and 16 composite_unseen tasks. Each row is:

    split<TAB>task_name<TAB>horizon

The evaluation itself is pure simulation, so downloaded datasets are not part
of the default filtering. ``--require-downloaded`` is only a dev-mode helper.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TARGET_SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
EXPECTED_COUNTS = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}
TASK_FILTER_TO_SPLITS = {
    "all": TARGET_SPLITS,
    "target50": TARGET_SPLITS,
    "atomic": ("atomic_seen",),
    "atomic_seen": ("atomic_seen",),
    "composite": ("composite_seen", "composite_unseen"),
    "composite_seen": ("composite_seen",),
    "composite_unseen": ("composite_unseen",),
}


@dataclass(frozen=True)
class TaskRow:
    split: str
    task_name: str
    horizon: int

    @property
    def env_name(self) -> str:
        return self.task_name if self.task_name.startswith("robocasa/") else f"robocasa/{self.task_name}"


def _import_registry(register_gym: bool = True):
    try:
        # RoboCasa/robosuite can print import-time warnings to stdout. Keep stdout
        # reserved for TSV data so callers can safely pipe it to other tools.
        with contextlib.redirect_stdout(sys.stderr):
            import robocasa  # noqa: F401

            if register_gym:
                import robocasa.wrappers.gym_wrapper  # noqa: F401

            from robocasa.utils import dataset_registry
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Failed to import robocasa dataset registry. Run this with the "
            "RoboCasa-365 Python environment and PYTHONPATH set for the clone.\\n"
            f"Original error: {exc}"
        ) from exc
    return dataset_registry


def split_names(task_filter: str) -> tuple[str, ...]:
    try:
        return tuple(TASK_FILTER_TO_SPLITS[task_filter])
    except KeyError as exc:
        allowed = ", ".join(sorted(TASK_FILTER_TO_SPLITS))
        raise ValueError(f"Unknown task filter {task_filter!r}. Expected one of: {allowed}") from exc


def _task_config(registry, task_name: str) -> dict:
    if task_name in registry.ATOMIC_TASK_DATASETS:
        return registry.ATOMIC_TASK_DATASETS[task_name]
    if task_name in registry.COMPOSITE_TASK_DATASETS:
        return registry.COMPOSITE_TASK_DATASETS[task_name]
    raise KeyError(f"Task {task_name!r} is in TARGET_TASKS but not in the dataset registries")


def build_task_rows(
    task_filter: str = "all",
    *,
    check_gym: bool = True,
    limit: int | None = None,
) -> list[TaskRow]:
    registry = _import_registry(register_gym=check_gym)
    rows: list[TaskRow] = []
    for split in split_names(task_filter):
        for task_name in registry.TARGET_TASKS[split]:
            config = _task_config(registry, task_name)
            try:
                horizon = int(config["horizon"])
            except KeyError as exc:
                raise KeyError(f"Task {task_name!r} has no horizon in the registry") from exc
            rows.append(TaskRow(split=split, task_name=task_name, horizon=horizon))

    if limit is not None and limit > 0:
        rows = rows[:limit]

    if check_gym:
        check_gym_registration(rows)
    return rows


def check_leaderboard_shape(rows: Sequence[TaskRow]) -> None:
    counts = {split: 0 for split in TARGET_SPLITS}
    for row in rows:
        if row.split in counts:
            counts[row.split] += 1

    problems = []
    for split, expected in EXPECTED_COUNTS.items():
        got = counts.get(split, 0)
        if got != expected:
            problems.append(f"{split}: expected {expected}, got {got}")
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        problems.append(f"total: expected {sum(EXPECTED_COUNTS.values())}, got {len(rows)}")

    if problems:
        raise SystemExit("Leaderboard target50 task table is incomplete: " + "; ".join(problems))


def check_gym_registration(rows: Iterable[TaskRow]) -> None:
    try:
        import gymnasium as gym
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to import gymnasium for env registration checks: {exc}") from exc

    missing = []
    for row in rows:
        try:
            gym.spec(row.env_name)
        except Exception:  # noqa: BLE001
            missing.append(row.env_name)

    if missing:
        raise SystemExit(
            "The following RoboCasa gym env ids are not registered:\n"
            + "\n".join(f"  {name}" for name in missing)
        )


def _dataset_roots(cli_root: str | None) -> list[Path]:
    roots: list[Path] = []
    if cli_root:
        roots.append(Path(cli_root).expanduser())
    for key in ("ROBOCASA_DATASET_ROOT", "ROBOCASA_DATASETS", "DATASET_ROOT"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(
        [
            Path.home() / ".robocasa" / "datasets",
            Path.home() / ".cache" / "robocasa" / "datasets",
        ]
    )

    unique: list[Path] = []
    seen = set()
    for root in roots:
        resolved = str(root)
        if resolved not in seen:
            unique.append(root)
            seen.add(resolved)
    return unique


def _downloaded_target_path(registry, task_name: str, roots: Sequence[Path]) -> Path | None:
    config = _task_config(registry, task_name)
    rel = config.get("target", {}).get("human_path")
    if not rel:
        return None
    for root in roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def filter_downloaded(rows: Sequence[TaskRow], dataset_root: str | None) -> list[TaskRow]:
    registry = _import_registry(register_gym=False)
    roots = _dataset_roots(dataset_root)
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise SystemExit(
            "--require-downloaded was set, but no dataset root exists. Pass "
            "--dataset-root or set ROBOCASA_DATASET_ROOT. Checked: "
            + ", ".join(str(root) for root in roots)
        )

    kept = []
    skipped = []
    for row in rows:
        if _downloaded_target_path(registry, row.task_name, existing_roots) is not None:
            kept.append(row)
        else:
            skipped.append(row.task_name)

    if skipped:
        print(
            "[gen_target_tasks] dev filter skipped tasks without downloaded target data: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
    return kept


def write_tsv(rows: Sequence[TaskRow], output: str, *, header: bool) -> None:
    out = sys.stdout if output == "-" else open(output, "w", newline="")
    try:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        if header:
            writer.writerow(["split", "task_name", "horizon"])
        for row in rows:
            writer.writerow([row.split, row.task_name, row.horizon])
    finally:
        if out is not sys.stdout:
            out.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="-", help="Output TSV path, or '-' for stdout")
    parser.add_argument(
        "--task-filter",
        default="all",
        choices=sorted(TASK_FILTER_TO_SPLITS),
        help="Subset to emit. 'all' is the target50 leaderboard set.",
    )
    parser.add_argument(
        "--mode",
        default="leaderboard",
        choices=("leaderboard", "dev"),
        help="leaderboard enforces the full target50 shape when task-filter=all.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Dev-only row limit for smoke tests")
    parser.add_argument("--require-downloaded", action="store_true", help="Dev-only filter to downloaded target data")
    parser.add_argument("--dataset-root", default=None, help="Root used by --require-downloaded")
    parser.add_argument("--no-check-gym", action="store_true", help="Skip gym env registration checks")
    parser.add_argument("--no-header", action="store_true", help="Do not write a TSV header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "leaderboard" and (args.task_filter != "all" or args.limit):
        raise SystemExit("leaderboard mode requires --task-filter all and no --limit")
    if args.require_downloaded and args.mode != "dev":
        raise SystemExit("--require-downloaded is dev-only; leaderboard mode must keep all target50 tasks")

    rows = build_task_rows(
        args.task_filter,
        check_gym=not args.no_check_gym,
        limit=args.limit if args.limit > 0 else None,
    )
    if args.require_downloaded:
        rows = filter_downloaded(rows, args.dataset_root)
    if args.mode == "leaderboard" and args.task_filter == "all":
        check_leaderboard_shape(rows)

    write_tsv(rows, args.output, header=not args.no_header)
    print(f"[gen_target_tasks] wrote {len(rows)} tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
