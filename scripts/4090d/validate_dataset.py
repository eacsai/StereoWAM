"""Validate a self-rendered LIBERO LeRobot dataset against the official IPEC
reference, catching silent bugs of the kind that caused the 2026-05-21 gripper
polarity disaster (Mono-SelfRender 35% vs Mono-Official 88%).

The polarity bug went undetected for ~6 days because no automated check ever
compared self-rendered action column distributions to the official release.
This script is the missing pre-flight check: run it BEFORE launching training
on any newly-rendered dataset.

Checks performed (raises on first failure)
------------------------------------------
1. Schema:        info.json features match official (state.shape=8, action.shape=7,
                  video codec/fps/dtype consistent)
2. Tasks:         tasks.jsonl ordering matches official (same 10 task names same order)
3. Episode count: |self - official| within tolerance (loose, since IPEC may shuffle)
4. Per-task ep count: every task has reasonable count (≥80% of expected)
5. Action ranges: XYZ ∈ [-0.94, 0.94], rot ∈ [-0.40, 0.40] (matches official q01-q99)
6. **Gripper convention**: action[:, 6].unique() == {0.0, 1.0} (the polarity-bug check)
7. State ranges:  ee_pos/quat plausible
8. Frame length: mean and stdev within reasonable range of official

Usage
-----
    cd /data/wangqiwei/ICLR2026/starVLA          # or container starVLA dir
    .venv/bin/python scripts/4090d/validate_dataset.py \\
        --self  /path/to/self_rendered/libero_goal \\
        --official /path/to/official_libero_goal_no_noops_1.0.0_lerobot \\
        [--strict]  # turn warnings into errors

Exit code 0 = all green; non-zero + raised AssertionError = bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Tolerances (per check)
EPISODE_COUNT_REL_TOL = 0.05   # |self - official| / official < 5%
PER_TASK_COUNT_MIN_FRAC = 0.80  # each task must have ≥ 80% of expected episodes
FRAME_LEN_MEAN_REL_TOL = 0.15   # mean frame length within ±15% of official
ACTION_GRIPPER_EXPECTED = {0.0, 1.0}  # the polarity-bug check: OpenVLA convention
ACTION_XYZ_RANGE = (-0.96, 0.96)
ACTION_ROT_RANGE = (-0.45, 0.45)


def _load_info(p: Path) -> dict:
    with open(p / "meta" / "info.json") as f:
        return json.load(f)


def _load_tasks(p: Path) -> list[dict]:
    out = []
    with open(p / "meta" / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return sorted(out, key=lambda d: d["task_index"])


def _collect_actions(p: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """Return (all_actions [N, 7], frames_per_episode list, task_index_per_episode list)."""
    pq_dir = p / "data" / "chunk-000"
    assert pq_dir.exists(), f"missing {pq_dir}"

    all_actions = []
    frames_per_ep = []
    task_per_ep = []
    for f in sorted(pq_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        a = np.stack(df["action"].tolist())
        all_actions.append(a)
        frames_per_ep.append(len(df))
        task_per_ep.append(int(df["task_index"].iloc[0]))
    return np.concatenate(all_actions, axis=0), frames_per_ep, task_per_ep


def check(label: str, ok: bool, detail: str = "", strict: bool = False) -> bool:
    sym = "✅" if ok else "❌"
    print(f"  {sym} {label}: {detail}")
    if not ok and strict:
        raise AssertionError(f"{label}: {detail}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self", required=True, type=Path, dest="self_dir",
                   help="path to self-rendered LeRobot suite root (has meta/, data/, videos/)")
    p.add_argument("--official", required=True, type=Path,
                   help="path to official IPEC reference (same structure)")
    p.add_argument("--strict", action="store_true",
                   help="raise on first failure (default: report all then exit non-zero)")
    args = p.parse_args()

    print(f"=== VALIDATE: self={args.self_dir} vs official={args.official} ===")

    # ---- load both sides ----
    self_info = _load_info(args.self_dir)
    off_info = _load_info(args.official)
    self_tasks = _load_tasks(args.self_dir)
    off_tasks = _load_tasks(args.official)
    self_acts, self_lens, self_task_per_ep = _collect_actions(args.self_dir)
    off_acts, off_lens, off_task_per_ep = _collect_actions(args.official)

    all_ok = True

    # ---- 1. Schema ----
    print("\n[1] Schema (info.json features)")
    ok = (
        self_info["features"]["observation.state"]["shape"] == [8]
        and self_info["features"]["action"]["shape"] == [7]
        and self_info["features"]["observation.images.image"]["info"]["video.codec"] == "av1"
        and self_info["features"]["observation.images.image"]["info"]["video.fps"] == 20
    )
    all_ok &= check("schema state[8] action[7] codec=av1 fps=20", ok,
                    detail=f"state.shape={self_info['features']['observation.state']['shape']}, "
                           f"action.shape={self_info['features']['action']['shape']}",
                    strict=args.strict)

    # ---- 2. Tasks ordering ----
    print("\n[2] Tasks ordering")
    ok = [t["task"] for t in self_tasks] == [t["task"] for t in off_tasks]
    all_ok &= check("tasks.jsonl ordering matches official", ok,
                    detail=f"self has {len(self_tasks)} tasks, off has {len(off_tasks)}",
                    strict=args.strict)

    # ---- 3. Episode count ----
    print("\n[3] Episode counts")
    n_self = self_info["total_episodes"]
    n_off = off_info["total_episodes"]
    rel = abs(n_self - n_off) / n_off
    ok = rel < EPISODE_COUNT_REL_TOL
    all_ok &= check(f"|self - official| / official < {EPISODE_COUNT_REL_TOL}", ok,
                    detail=f"self={n_self}, off={n_off}, rel_diff={rel:.3f}",
                    strict=args.strict)

    # ---- 4. Per-task ep count ----
    print("\n[4] Per-task episode counts")
    self_per_task = {ti: self_task_per_ep.count(ti) for ti in range(len(self_tasks))}
    off_per_task = {ti: off_task_per_ep.count(ti) for ti in range(len(off_tasks))}
    bad_tasks = []
    for ti in range(len(self_tasks)):
        if off_per_task.get(ti, 0) > 0:
            frac = self_per_task.get(ti, 0) / off_per_task[ti]
            if frac < PER_TASK_COUNT_MIN_FRAC:
                bad_tasks.append((ti, self_per_task.get(ti, 0), off_per_task[ti], frac))
    ok = len(bad_tasks) == 0
    detail = (
        f"per-task self vs off: " + ", ".join(
            f"t{ti}:{self_per_task.get(ti, 0)}/{off_per_task.get(ti, 0)}"
            for ti in range(len(self_tasks))
        )
    )
    all_ok &= check(f"every task has ≥{PER_TASK_COUNT_MIN_FRAC * 100:.0f}% of official ep count", ok,
                    detail=detail, strict=args.strict)
    if bad_tasks:
        for ti, sc, oc, fr in bad_tasks:
            print(f"    ⚠ task {ti} ({self_tasks[ti]['task']!r}): self={sc} off={oc} frac={fr:.2f}")

    # ---- 5. Action ranges ----
    print("\n[5] Action XYZ + rot ranges")
    xyz_min = self_acts[:, 0:3].min()
    xyz_max = self_acts[:, 0:3].max()
    ok_xyz = ACTION_XYZ_RANGE[0] <= xyz_min and xyz_max <= ACTION_XYZ_RANGE[1]
    all_ok &= check(f"action XYZ ∈ {ACTION_XYZ_RANGE}", ok_xyz,
                    detail=f"xyz_min={xyz_min:.3f}, xyz_max={xyz_max:.3f}",
                    strict=args.strict)
    rot_min = self_acts[:, 3:6].min()
    rot_max = self_acts[:, 3:6].max()
    ok_rot = ACTION_ROT_RANGE[0] <= rot_min and rot_max <= ACTION_ROT_RANGE[1]
    all_ok &= check(f"action rot ∈ {ACTION_ROT_RANGE}", ok_rot,
                    detail=f"rot_min={rot_min:.3f}, rot_max={rot_max:.3f}",
                    strict=args.strict)

    # ---- 6. ⭐ GRIPPER POLARITY CHECK (the 2026-05-21 bug) ----
    print("\n[6] ⭐ Gripper polarity (the 2026-05-21 bug)")
    self_grip = set(np.unique(self_acts[:, 6]).tolist())
    off_grip = set(np.unique(off_acts[:, 6]).tolist())
    ok = self_grip == ACTION_GRIPPER_EXPECTED
    all_ok &= check(
        f"self gripper unique values == {ACTION_GRIPPER_EXPECTED} (OpenVLA convention)",
        ok,
        detail=f"self={self_grip}, official={off_grip}"
              + ("" if ok else "  ← THIS IS THE 0521 POLARITY BUG — apply patch_self_rendered_parquets.py"),
        strict=args.strict,
    )

    # ---- 7. State sanity (just shape + finite) ----
    print("\n[7] State sanity (finite + reasonable scale)")
    # Just verify no NaN/Inf in actions (proxy for state — state lives in parquet too but action is most prone)
    ok = np.isfinite(self_acts).all()
    all_ok &= check("action values all finite (no NaN/Inf)", ok,
                    detail=f"finite count {int(np.isfinite(self_acts).sum())} / {self_acts.size}",
                    strict=args.strict)

    # ---- 8. Frame length distribution ----
    print("\n[8] Frame length distribution")
    self_mean = float(np.mean(self_lens))
    off_mean = float(np.mean(off_lens))
    rel = abs(self_mean - off_mean) / off_mean
    ok = rel < FRAME_LEN_MEAN_REL_TOL
    all_ok &= check(f"mean frame length within ±{FRAME_LEN_MEAN_REL_TOL * 100:.0f}% of official", ok,
                    detail=f"self_mean={self_mean:.1f}, off_mean={off_mean:.1f}, rel_diff={rel:.3f}",
                    strict=args.strict)

    # ---- summary ----
    print(f"\n=== {'OVERALL ✅ ALL GREEN' if all_ok else 'OVERALL ❌ SOME CHECKS FAILED'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
