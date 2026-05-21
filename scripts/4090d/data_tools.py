"""data_tools.py — unified self-rendered LIBERO LeRobot dataset operations.

Replaces (merged 2026-05-21 per codex PR review task bk8nlokfw, finding F3):
  - scripts/4090d/patch_self_rendered_parquets.py
  - scripts/4090d/validate_dataset.py

Why merged
----------
Previously the polarity-fix pipeline was 3 separate scripts + a hand-edited
manifest JSON. Provenance was "documented by comments" rather than enforced by
code; any contributor bypassing one step (e.g. retraining on un-patched data,
or evaluating with the wrong --gripper-convention flag) silently re-introduced
the 2026-05-21 35%-vs-80% catastrophe. A single CLI guarantees the
manifest-check pre-flight runs before any long-running train/eval starts.

Subcommands
-----------
  patch           apply OpenVLA polarity transform (1-g)/2 to action[:, 6]
  validate        8-check sanity vs official IPEC reference
  manifest-write  write dataset_manifest.json declaring this dataset's contract
  manifest-check  fail-closed pre-flight check (call from train/eval launchers)

Quick reference
---------------
# Polarity-patch (transactional — keep originals)
python scripts/4090d/data_tools.py patch \
    --root /path/to/self_rendered/libero_goal \
    --out-root /path/to/self_rendered_patched/libero_goal

# Validate vs official
python scripts/4090d/data_tools.py validate \
    --self /path/to/self_rendered_patched/libero_goal \
    --official /path/to/official_libero_goal_no_noops_1.0.0_lerobot \
    --strict

# Write manifest after patch+validate green
python scripts/4090d/data_tools.py manifest-write \
    --root /path/to/self_rendered_patched/libero_goal \
    --convention openvla \
    --rendered-by scripts/4090d/regenerate_libero_stereo.py

# Pre-launch gate (use in train.sh / eval launcher)
python scripts/4090d/data_tools.py manifest-check \
    --root /path/to/dataset \
    --expect-convention openvla
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

ALLOWED_CONVENTIONS = ("openvla", "libero_raw")
ALLOWED_CONVENTIONS_SET = set(ALLOWED_CONVENTIONS)
GRIPPER_VALUES_BY_CONVENTION = {
    "openvla":    {0.0, 1.0},   # IPEC release after (1-g)/2 transform
    "libero_raw": {-1.0, 1.0},  # raw LIBERO sim convention pre-transform
}

# 8-check tolerances (used by validate subcommand)
EPISODE_COUNT_REL_TOL = 0.05
PER_TASK_COUNT_MIN_FRAC = 0.80
FRAME_LEN_MEAN_REL_TOL = 0.15
ACTION_XYZ_RANGE = (-0.96, 0.96)
ACTION_ROT_RANGE = (-0.45, 0.45)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

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


def _collect_actions(p: Path):
    pq_dir = p / "data" / "chunk-000"
    assert pq_dir.exists(), f"missing {pq_dir}"
    all_actions, frames_per_ep, task_per_ep = [], [], []
    for f in sorted(pq_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        a = np.stack(df["action"].tolist())
        all_actions.append(a)
        frames_per_ep.append(len(df))
        task_per_ep.append(int(df["task_index"].iloc[0]))
    return np.concatenate(all_actions, axis=0), frames_per_ep, task_per_ep


def _check_print(label: str, ok: bool, detail: str = "", strict: bool = False) -> bool:
    sym = "✅" if ok else "❌"
    print(f"  {sym} {label}: {detail}")
    if not ok and strict:
        raise AssertionError(f"{label}: {detail}")
    return ok


# --------------------------------------------------------------------------- #
# Subcommand: patch
# --------------------------------------------------------------------------- #

def _patch_parquet(in_path: Path, out_path: Path, dry_run: bool = False) -> dict:
    df = pd.read_parquet(in_path)
    actions = np.stack(df["action"].tolist())
    assert actions.shape[1] == 7, f"expected 7-dim action, got {actions.shape[1]} in {in_path}"

    pre_unique = sorted(set(actions[:, 6].tolist()))
    actions[:, 6] = (1.0 - actions[:, 6]) / 2.0
    post_unique = sorted(set(actions[:, 6].tolist()))

    bad = [v for v in post_unique if v not in (0.0, 1.0)]
    assert not bad, (
        f"post-transform unexpected gripper values {bad} in {in_path}; "
        f"raw values were {pre_unique} — may already be patched"
    )

    stats = {
        "in": str(in_path),
        "out": str(out_path),
        "frames": len(df),
        "gripper_pre_unique": pre_unique,
        "gripper_post_unique": post_unique,
    }
    if dry_run:
        return stats

    df["action"] = [actions[i] for i in range(len(df))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return stats


def cmd_patch(args) -> int:
    if args.dry_run:
        out_root = args.root
    elif args.inplace:
        if args.out_root is not None:
            sys.exit("ERROR: --inplace and --out-root are mutually exclusive")
        if not args.allow_inplace:
            sys.exit(
                "ERROR: --inplace requires --allow-inplace to confirm. In-place is dangerous:\n"
                "  (a) accidentally re-running corrupts already-patched data ({0,1} → {0.5, 0})\n"
                "  (b) loses the raw source without a separate backup step.\n"
                "Prefer: --out-root <NEW_DIR>  (transactional, safe to retry)."
            )
        out_root = args.root
    else:
        if args.out_root is None:
            sys.exit("ERROR: must pass --out-root <DIR> (recommended) or --inplace --allow-inplace or --dry-run")
        out_root = args.out_root

    parquet_dir = args.root / "data" / "chunk-000"
    if not parquet_dir.exists():
        sys.exit(f"ERROR: {parquet_dir} not found — wrong --root?")

    files = sorted(parquet_dir.glob("episode_*.parquet"))
    print(f"[patch] {len(files)} parquets under {parquet_dir}")
    print(f"[patch] mode: {'DRY-RUN' if args.dry_run else f'WRITE to {out_root}'}")

    all_stats = []
    for i, in_path in enumerate(files):
        rel = in_path.relative_to(args.root)
        st = _patch_parquet(in_path, out_root / rel, dry_run=args.dry_run)
        all_stats.append(st)
        if i < 3 or i == len(files) - 1:
            print(f"  [{i:3d}] {in_path.name}: {st['gripper_pre_unique']} → {st['gripper_post_unique']} ({st['frames']} frames)")
        elif i == 3:
            print(f"  ... ({len(files) - 4} more parquets)")

    all_post = sorted({v for st in all_stats for v in st["gripper_post_unique"]})
    assert set(all_post).issubset({0.0, 1.0}), f"unexpected post values: {all_post}"
    print(f"[patch] overall post unique: {all_post}")
    print(f"[patch] total frames patched: {sum(st['frames'] for st in all_stats)}")
    # Finding A (codex 2026-05-21): transactional --out-root mode previously only
    # wrote parquets and left meta/ + videos/ behind, producing a dataset dir that
    # could not be opened by any consumer. Symlink the siblings so out_root is a
    # complete, usable dataset (videos can be GBs — symlink avoids duplicate copy).
    if not args.dry_run and out_root != args.root:
        for sibling in ("meta", "videos"):
            src_path = (args.root / sibling).resolve()
            dst = out_root / sibling
            if not src_path.exists():
                continue
            if dst.exists() or dst.is_symlink():
                # Finding F5b (codex 2026-05-21): patch rerun should not silently accept
                # a stale sibling from a previous run pointing somewhere else.
                if dst.is_symlink():
                    existing = dst.resolve()
                    if existing != src_path:
                        sys.exit(
                            f"ERROR: {dst} already exists as symlink → {existing}, but this run\n"
                            f"  wants {sibling}/ to point at {src_path}. Either delete {dst} first \n"
                            f"  (if stale), or use a fresh --out-root."
                        )
                    # same target: skip silently (idempotent rerun)
                    continue
                else:
                    sys.exit(
                        f"ERROR: {dst} already exists as a real {sibling} directory, not a \n"
                        f"  symlink. Refusing to mix real + linked siblings in out_root. \n"
                        f"  Either delete {dst} or use a fresh --out-root."
                    )
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src_path, target_is_directory=True)
            print(f"[patch] linked {sibling}/ from {args.root} into {out_root}")
    print("[patch] DONE")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: validate
# --------------------------------------------------------------------------- #

def cmd_validate(args) -> int:
    print(f"=== VALIDATE: self={args.self_dir} vs official={args.official} ===")

    self_info = _load_info(args.self_dir)
    off_info = _load_info(args.official)
    self_tasks = _load_tasks(args.self_dir)
    off_tasks = _load_tasks(args.official)
    self_acts, self_lens, self_task_per_ep = _collect_actions(args.self_dir)
    off_acts, off_lens, off_task_per_ep = _collect_actions(args.official)

    all_ok = True

    print("\n[1] Schema")
    ok = (
        self_info["features"]["observation.state"]["shape"] == [8]
        and self_info["features"]["action"]["shape"] == [7]
        and self_info["features"]["observation.images.image"]["info"]["video.codec"] == "av1"
        and self_info["features"]["observation.images.image"]["info"]["video.fps"] == 20
    )
    all_ok &= _check_print("state[8] action[7] codec=av1 fps=20", ok, strict=args.strict)

    print("\n[2] Tasks ordering")
    ok = [t["task"] for t in self_tasks] == [t["task"] for t in off_tasks]
    all_ok &= _check_print("tasks.jsonl ordering matches official", ok,
                            detail=f"self={len(self_tasks)} off={len(off_tasks)}", strict=args.strict)

    print("\n[3] Episode counts")
    n_self, n_off = self_info["total_episodes"], off_info["total_episodes"]
    rel = abs(n_self - n_off) / n_off
    all_ok &= _check_print(f"|self-off|/off < {EPISODE_COUNT_REL_TOL}", rel < EPISODE_COUNT_REL_TOL,
                            detail=f"self={n_self} off={n_off} rel={rel:.3f}", strict=args.strict)

    print("\n[4] Per-task counts")
    self_pt = {ti: self_task_per_ep.count(ti) for ti in range(len(self_tasks))}
    off_pt = {ti: off_task_per_ep.count(ti) for ti in range(len(off_tasks))}
    bad = []
    for ti in range(len(self_tasks)):
        if off_pt.get(ti, 0) > 0:
            frac = self_pt.get(ti, 0) / off_pt[ti]
            if frac < PER_TASK_COUNT_MIN_FRAC:
                bad.append((ti, self_pt.get(ti, 0), off_pt[ti], frac))
    all_ok &= _check_print(f"per-task ≥{PER_TASK_COUNT_MIN_FRAC*100:.0f}% of official", len(bad) == 0,
                            detail=f"bad_tasks={bad}" if bad else "all ok", strict=args.strict)

    print("\n[5] Action ranges")
    xyz_min, xyz_max = self_acts[:, :3].min(), self_acts[:, :3].max()
    rot_min, rot_max = self_acts[:, 3:6].min(), self_acts[:, 3:6].max()
    all_ok &= _check_print(f"XYZ ∈ {ACTION_XYZ_RANGE}",
                            ACTION_XYZ_RANGE[0] <= xyz_min and xyz_max <= ACTION_XYZ_RANGE[1],
                            detail=f"min={xyz_min:.3f} max={xyz_max:.3f}", strict=args.strict)
    all_ok &= _check_print(f"rot ∈ {ACTION_ROT_RANGE}",
                            ACTION_ROT_RANGE[0] <= rot_min and rot_max <= ACTION_ROT_RANGE[1],
                            detail=f"min={rot_min:.3f} max={rot_max:.3f}", strict=args.strict)

    print("\n[6] ⭐ Gripper polarity (the 2026-05-21 bug)")
    self_grip = set(np.unique(self_acts[:, 6]).tolist())
    off_grip = set(np.unique(off_acts[:, 6]).tolist())
    expected = GRIPPER_VALUES_BY_CONVENTION["openvla"]
    ok = self_grip == expected
    all_ok &= _check_print(f"self gripper unique == {expected} (openvla)", ok,
                            detail=(f"self={self_grip} off={off_grip}"
                                    + ("" if ok else " ← 0521 POLARITY BUG — run ")),
                            strict=args.strict)

    print("\n[7] Action finite")
    all_ok &= _check_print("all action values finite", np.isfinite(self_acts).all(), strict=args.strict)

    print("\n[8] Frame length")
    self_mean, off_mean = float(np.mean(self_lens)), float(np.mean(off_lens))
    rel = abs(self_mean - off_mean) / off_mean
    all_ok &= _check_print(f"mean within ±{FRAME_LEN_MEAN_REL_TOL*100:.0f}%",
                            rel < FRAME_LEN_MEAN_REL_TOL,
                            detail=f"self={self_mean:.1f} off={off_mean:.1f} rel={rel:.3f}", strict=args.strict)

    print(f"\n=== {'ALL GREEN ✅' if all_ok else 'FAILED ❌'} ===")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# Subcommand: manifest-write
# --------------------------------------------------------------------------- #

def cmd_manifest_write(args) -> int:
    info = _load_info(args.root)
    self_acts, self_lens, _ = _collect_actions(args.root)
    observed_grip = sorted(set(np.unique(self_acts[:, 6]).tolist()))
    expected_grip = sorted(GRIPPER_VALUES_BY_CONVENTION[args.convention])

    if set(observed_grip) != set(expected_grip):
        sys.exit(
            f"ERROR: declared convention='{args.convention}' expects gripper values "
            f"{expected_grip}, but parquet has {observed_grip}. "
            f"Either run  first or declare the correct convention."
        )

    manifest = {
        "dataset_id": args.root.name,
        "suite": args.suite or args.root.name,
        "rendered_by": args.rendered_by,
        "stereo_baseline_m": args.stereo_baseline_m,
        "schema": {
            "state": {"shape": info["features"]["observation.state"]["shape"]},
            "action": {"shape": info["features"]["action"]["shape"],
                       "layout": "delta_xyz[3] + delta_rot[3] + gripper[1]"},
            "total_episodes": info.get("total_episodes"),
            "total_frames": int(sum(self_lens)),
        },
        "gripper_convention": args.convention,
        "gripper_col_values": observed_grip,
        "recommended_eval": {
            "gripper_convention_flag": args.convention,
        },
        "history": [f"{date.today().isoformat()}: manifest written by data_tools.py manifest-write"]
                   + ([args.note] if args.note else []),
    }
    out = args.root / "dataset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest-write] wrote {out}")
    print(json.dumps(manifest, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: manifest-check  (the F2 fail-closed gate)
# --------------------------------------------------------------------------- #

def cmd_manifest_check(args) -> int:
    manifest_path = args.root / "dataset_manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"ERROR: manifest missing at {manifest_path}. "
            f"Self-rendered datasets MUST declare gripper_convention before training/eval. "
            f"Run: data_tools.py manifest-write --root {args.root} --convention <openvla|libero_raw> ..."
        )
    with open(manifest_path) as f:
        m = json.load(f)

    got = m.get("gripper_convention")
    if got not in ALLOWED_CONVENTIONS_SET:
        sys.exit(f"ERROR: manifest gripper_convention='{got}' not in {ALLOWED_CONVENTIONS}")

    if got != args.expect_convention:
        sys.exit(
            f"ERROR: gripper_convention mismatch.\n"
            f"  manifest at {manifest_path} declares '{got}'\n"
            f"  but launcher expects '{args.expect_convention}'.\n"
            f"This is the 2026-05-21 polarity-bug guard — refusing to start.\n"
            f"Either (a) re-patch the data with data_tools.py patch, or \n"
            f"       (b) pass --expect-convention {got} if you know what you are doing."
        )

    schema = m.get("schema", {})
    act_shape = schema.get("action", {}).get("shape")
    if act_shape != [args.expect_action_dim]:
        sys.exit(f"ERROR: action shape {act_shape} != expected [{args.expect_action_dim}]")
    state_shape = schema.get("state", {}).get("shape")
    if state_shape != [args.expect_state_dim]:
        sys.exit(f"ERROR: state shape {state_shape} != expected [{args.expect_state_dim}]")

    if args.live:
        # Finding B (codex 2026-05-21): static JSON validation misses corruption that
        # only shows up when the dataset is actually loaded. Spot-check 1 parquet (action
        # shape + gripper convention) and 1 video file (exists + nonzero size) so the
        # pre-launch gate refuses a 30k-step run when something is silently broken.
        pq_dir = args.root / "data" / "chunk-000"
        parquets = sorted(pq_dir.glob("episode_*.parquet"))
        if not parquets:
            sys.exit(f"ERROR [live]: no parquets found under {pq_dir}")
        df = pd.read_parquet(parquets[0])
        for col in ("action", "observation.state", "task_index"):
            if col not in df.columns:
                sys.exit(f"ERROR [live]: parquet {parquets[0].name} missing column {col!r}")
        a = np.stack(df["action"].tolist())
        if a.shape[1] != args.expect_action_dim:
            sys.exit(f"ERROR [live]: parquet action dim {a.shape[1]} != expected {args.expect_action_dim}")
        observed_grip = set(np.unique(a[:, 6]).tolist())
        expected_grip = GRIPPER_VALUES_BY_CONVENTION[got]
        if not observed_grip.issubset(expected_grip):
            sys.exit(
                f"ERROR [live]: parquet gripper values {sorted(observed_grip)} not subset of "
                f"{sorted(expected_grip)} for declared convention='{got}'. Manifest and parquet disagree."
            )
        # Finding F5a (codex 2026-05-21): live-check must cover ALL declared video
        # modalities. A stereo dataset with right_view missing/empty would silently
        # pass if we only check primary.
        schema = m.get("schema", {})
        video_keys = schema.get("video_keys_available") or ["observation.images.image"]
        video_summary = []
        for vkey in video_keys:
            vid_dir = args.root / "videos" / "chunk-000" / vkey
            if not vid_dir.exists():
                sys.exit(f"ERROR [live]: video dir missing for declared modality {vkey!r}: {vid_dir}")
            videos = sorted(vid_dir.glob("episode_*.mp4"))
            if not videos:
                sys.exit(f"ERROR [live]: no .mp4 files under {vid_dir} (modality {vkey!r})")
            sz = videos[0].stat().st_size
            if sz < 1024:
                sys.exit(f"ERROR [live]: first video {videos[0].name} suspiciously small ({sz} bytes) for modality {vkey!r}")
            video_summary.append(f"{vkey.split('.')[-1]}={videos[0].name}({sz//1024}KB)")
        print(f"[manifest-check live] parquet={parquets[0].name} action={a.shape} grip={sorted(observed_grip)} "
              f"| videos: {', '.join(video_summary)}")

    print(f"[manifest-check] ✅ {args.root}: convention={got}, action={act_shape}, state={state_shape}")
    return 0


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(prog="data_tools.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("patch", help="apply (1-g)/2 polarity fix to gripper col")
    sp.add_argument("--root", required=True, type=Path)
    sp.add_argument("--out-root", type=Path, default=None)
    sp.add_argument("--inplace", action="store_true")
    sp.add_argument("--allow-inplace", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_patch)

    sp = sub.add_parser("validate", help="8-check sanity vs official IPEC reference")
    sp.add_argument("--self", dest="self_dir", required=True, type=Path)
    sp.add_argument("--official", required=True, type=Path)
    sp.add_argument("--strict", action="store_true")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("manifest-write", help="write dataset_manifest.json")
    sp.add_argument("--root", required=True, type=Path)
    sp.add_argument("--convention", required=True, choices=ALLOWED_CONVENTIONS)
    sp.add_argument("--rendered-by", required=True)
    sp.add_argument("--suite", default=None)
    sp.add_argument("--stereo-baseline-m", type=float, default=0.06)
    sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_manifest_write)

    sp = sub.add_parser("manifest-check", help="fail-closed pre-flight gate for train/eval")
    sp.add_argument("--root", required=True, type=Path)
    sp.add_argument("--expect-convention", required=True, choices=ALLOWED_CONVENTIONS)
    sp.add_argument("--expect-action-dim", type=int, default=7)
    sp.add_argument("--expect-state-dim", type=int, default=8)
    sp.add_argument("--live", action="store_true",
                    help="also load 1 parquet + check 1 video file to catch corruption "
                         "that static JSON validation misses (Codex 2026-05-21 finding B)")
    sp.set_defaults(func=cmd_manifest_check)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
