"""Patch self-rendered LIBERO LeRobot parquets to fix gripper polarity bug.

Background
----------
scripts/4090d/regenerate_libero_stereo.py records the raw LIBERO demo action
straight to parquet. Raw LIBERO demo gripper col ∈ {-1, +1} (LIBERO sim
convention: -1=open, +1=close).

The official IPEC LeRobot release applies an OpenVLA-style polarity transform
in the RLDS-to-LeRobot conversion step:

    g_lerobot = (1 - g_libero) / 2     # -1 → 1 (open), +1 → 0 (close)

so its gripper col ∈ {0, 1} with 1=open (OpenVLA convention).

Our regenerate_libero_stereo.py MISSED that transform → self-rendered parquets
have gripper ∈ {-1, +1}, causing eval_libero.py's default _binarize_gripper_open
(which expects v ∈ [0, 1] threshold-0.5 + invert) to misinterpret model output
→ gripper polarity flipped every step → catastrophic eval success rate collapse
(verified: Z 35% with default convention vs 80% with libero_raw hack).

This script applies the missing transform in-place to all parquet files under
the given root, optionally writing to a separate output dir.

Usage (container)
-----------------
    cd /mnt/data/wangqiwei/wangqiwei/starVLA
    python scripts/4090d/patch_self_rendered_parquets.py \\
        --root /mnt/data/wangqiwei/wangqiwei/libero_goal_stereo_openvla/libero_goal \\
        --dry-run                     # check distribution before/after, no write

    # Then drop --dry-run for in-place patch:
    python scripts/4090d/patch_self_rendered_parquets.py \\
        --root /mnt/data/wangqiwei/wangqiwei/libero_goal_stereo_openvla/libero_goal \\
        --inplace

Post-patch steps
----------------
    1. Re-compute meta/stats_gr00t.json (gripper col range changed):
       (handled by trainer auto-cache invalidation, OR rebuild via starVLA's
       stats-rebuild utility — see starVLA/dataloader/gr00t_lerobot for hooks)
    2. Retrain Mono-SelfRender (and Stereo-SelfRender) on patched parquets
    3. Eval new ckpt with DEFAULT openvla gripper convention (no special flag
       needed) — matches Mono-Official eval pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def patch_parquet(in_path: Path, out_path: Path, dry_run: bool = False) -> dict:
    """Apply (1-g)/2 to action[:, 6] (gripper col) and write to out_path.

    Returns stats dict for verification (pre/post unique values, frame count).
    """
    df = pd.read_parquet(in_path)
    actions = np.stack(df["action"].tolist())  # (T, 7)
    assert actions.shape[1] == 7, f"expected 7-dim action, got {actions.shape[1]} in {in_path}"

    pre_unique = sorted(set(actions[:, 6].tolist()))
    pre_min, pre_max = float(actions[:, 6].min()), float(actions[:, 6].max())

    # The transform: g_new = (1 - g_old) / 2
    # -1 (LIBERO open) → 1 (OpenVLA "is_open" flag set)
    # +1 (LIBERO close) → 0 (OpenVLA "is_open" flag clear)
    actions[:, 6] = (1.0 - actions[:, 6]) / 2.0

    post_unique = sorted(set(actions[:, 6].tolist()))
    post_min, post_max = float(actions[:, 6].min()), float(actions[:, 6].max())

    # Validate: post-transform must be subset of {0.0, 1.0} for well-formed
    # LIBERO demos (where raw gripper ∈ {-1, +1}). Anything else means the
    # source parquet was already patched OR had non-discrete gripper values.
    bad_vals = [v for v in post_unique if v not in (0.0, 1.0)]
    assert not bad_vals, (
        f"post-transform unexpected gripper values {bad_vals} in {in_path}; "
        f"raw values were {pre_unique} — may have been already patched"
    )

    stats = {
        "in": str(in_path),
        "out": str(out_path),
        "frames": len(df),
        "gripper_pre_unique": pre_unique,
        "gripper_pre_range": (pre_min, pre_max),
        "gripper_post_unique": post_unique,
        "gripper_post_range": (post_min, post_max),
    }

    if dry_run:
        return stats

    # Write back. action is a list-of-arrays column in pandas; reconstruct.
    df["action"] = [actions[i] for i in range(len(df))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="self-rendered dataset root, e.g. .../libero_goal_stereo_openvla/libero_goal")
    p.add_argument("--out-root", type=Path, default=None,
                   help="output root; if not set + --inplace given, write in-place (RECOMMENDED PATH)")
    p.add_argument("--inplace", action="store_true",
                   help="DEPRECATED: overwrite parquets in --root. Requires --allow-inplace to actually run. "
                        "Use --out-root for safer transactional patching that preserves originals.")
    p.add_argument("--allow-inplace", action="store_true",
                   help="Acknowledgement flag required to actually mutate --root. Set this only after backing up "
                        "the raw parquets to a side dir; re-running an in-place patch on already-patched data "
                        "produces silent data corruption (0 → 0.5, 1 → 0). With --allow-inplace, the post-write "
                        "value-range assert in patch_parquet() will refuse to re-patch already-patched files.")
    p.add_argument("--dry-run", action="store_true",
                   help="report stats only, no write (always safe to run)")
    args = p.parse_args()

    if args.dry_run:
        out_root = args.root  # ignored
    elif args.inplace:
        if args.out_root is not None:
            sys.exit("ERROR: --inplace and --out-root are mutually exclusive")
        if not args.allow_inplace:
            sys.exit(
                "ERROR: --inplace requires --allow-inplace to confirm. In-place is dangerous because\n"
                "  (a) accidentally re-running corrupts already-patched data ({0,1} → {0.5, 0})\n"
                "  (b) loses the raw source without a separate backup step.\n"
                "Prefer:  --out-root <NEW_DIR>  (transactional, safe to retry).\n"
                "If you really need in-place AND have backed up --root first:  add --allow-inplace."
            )
        out_root = args.root
    else:
        if args.out_root is None:
            sys.exit("ERROR: must pass either --out-root <DIR> (recommended), --inplace --allow-inplace, or --dry-run")
        out_root = args.out_root

    parquet_dir = args.root / "data" / "chunk-000"
    if not parquet_dir.exists():
        sys.exit(f"ERROR: {parquet_dir} not found — is --root the right dataset path?")

    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    print(f"[patch] found {len(parquet_files)} parquets under {parquet_dir}")
    print(f"[patch] mode: {'DRY-RUN (no writes)' if args.dry_run else f'WRITE to {out_root}'}")

    all_stats = []
    for i, in_path in enumerate(parquet_files):
        rel = in_path.relative_to(args.root)
        out_path = out_root / rel
        st = patch_parquet(in_path, out_path, dry_run=args.dry_run)
        all_stats.append(st)
        if i < 3 or i == len(parquet_files) - 1:
            print(f"  [{i:3d}] {in_path.name}: gripper {st['gripper_pre_unique']} → {st['gripper_post_unique']} ({st['frames']} frames)")
        elif i == 3:
            print(f"  ... ({len(parquet_files) - 4} more parquets)")

    # Overall sanity: all post unique values should be subset of {0.0, 1.0}
    all_post = sorted({v for st in all_stats for v in st["gripper_post_unique"]})
    print(f"[patch] overall post-transform gripper unique across ALL parquets: {all_post}")
    assert set(all_post).issubset({0.0, 1.0}), f"unexpected post values: {all_post}"

    total_frames = sum(st["frames"] for st in all_stats)
    print(f"[patch] total frames patched: {total_frames}")
    print("[patch] DONE")


if __name__ == "__main__":
    main()
