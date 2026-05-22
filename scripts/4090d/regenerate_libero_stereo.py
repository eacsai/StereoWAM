"""
regenerate_libero_stereo.py
============================

OpenVLA-faithful LIBERO regenerator with stereo right_view added.

Pipeline (verbatim from openvla/experiments/robot/libero/regenerate_libero_dataset.py):
  - env = OffScreenRenderEnv(bddl_file, camera_heights=256, camera_widths=256)
  - env.seed(0)                                  # OpenVLA: "important - affects object positions"
  - env.reset()
  - obs = env.set_init_state(orig_states[0])
  - for _ in range(10): env.step([0,0,0,0,0,0,-1])  # settle sim
  - for action in orig_actions:
      if is_noop(action, prev): continue           # OpenVLA no-op filter
      record obs["agentview_image"], obs["robot0_eye_in_hand_image"]
      record env.sim.render(camera_name="rightview")  # OUR addition
      obs, reward, done, info = env.step(action.tolist())

Augmentation:
  - rightview camera injected via _stereo_render_utils.make_stereo_env (baseline 0.06)
  - all 3 views get 180-deg rotation ([::-1,::-1]) BEFORE encoding, matching
    moojink/rlds_dataset_builder/LIBERO_Goal_dataset_builder.py line 46-47

Output: LeRobot format directly (parquet + 3 mp4 + meta), matching IPEC layout but
with extra `observation.images.right_view` videos.

Reuses helpers from SSF/render_scripts/render_stereo_libero_hdf5.py for
parquet/meta writing (those are format-only, not flow-specific).

Usage:
  python regenerate_libero_stereo.py \
    --suite libero_goal \
    --libero_raw_data_dir /data/wangqiwei/ICLR2026/data \
    --libero_target_dir /data/wangqiwei/ICLR2026/data/libero_stereo_openvla \
    --num_tasks 1 --num_demos 1 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Locate stereo render utils dir (parallel to this file: starVLA/SSF/render_scripts/).
THIS_DIR = Path(__file__).resolve().parent
RENDER_SCRIPTS_DIR = THIS_DIR.parent.parent / "SSF" / "render_scripts"
sys.path.insert(0, str(RENDER_SCRIPTS_DIR))

# Reuse helpers from existing state-replay renderer (format/IO only; main flow differs).
from render_stereo_libero_hdf5 import (  # type: ignore  # noqa: E402
    FPS,
    discover_hdf5_files,
    task_language_from_hdf5,
    list_demo_keys,
    write_parquet_for_episode,
    write_meta,
)
from _stereo_render_utils import make_stereo_env, write_video  # type: ignore  # noqa: E402


# OpenVLA constants (openvla/experiments/robot/libero/regenerate_libero_dataset.py:43)
IMAGE_RESOLUTION = 256


def is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    """Verbatim from openvla/experiments/robot/libero/regenerate_libero_dataset.py:46.

    A frame is a no-op iff:
      (1) |action[:-1]| < threshold  (positional dims near zero)
      (2) gripper dim equals previous timestep's gripper dim
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def get_libero_dummy_action() -> list:
    """Verbatim from openvla/experiments/robot/libero/libero_utils.py:28."""
    return [0, 0, 0, 0, 0, 0, -1]


def _build_state8_from_obs(ee_pos_list, ee_axisangle_list, gripper_qpos_list) -> np.ndarray:
    """Build (T, 8) state vector matching IPEC LeRobot layout:
       state8[:, 0:3] = ee_pos        (3D end-effector position in world frame)
       state8[:, 3:6] = ee_axisangle  (3D axis-angle from quaternion)
       state8[:, 6:8] = gripper_qpos  (2 finger joints)
    IPEC labels state[6] as "pad" but the actual values are the second
    gripper finger (signed pair around zero) -- we keep that label for schema
    compatibility but the numeric content is gripper_qpos[0:2].
    """
    ee_pos = np.stack(ee_pos_list, axis=0)              # (T, 3)
    ee_axisangle = np.stack(ee_axisangle_list, axis=0)  # (T, 3)
    gripper_qpos = np.stack(gripper_qpos_list, axis=0)  # (T, 2)
    return np.concatenate([ee_pos, ee_axisangle, gripper_qpos], axis=1).astype(np.float32)


def render_one_demo_openvla(env, demo, resolution: int):
    """OpenVLA-faithful action-replay + stereo right_view + 180-deg rotation.

    Returns: dict of arrays (all 180-deg rotated, T = orig_T minus no-op count):
      agentview:    (T, H, W, 3) uint8
      eye_in_hand:  (T, H, W, 3) uint8
      right_view:   (T, H, W, 3) uint8
      actions:      (T, 7) float32
      state8:       (T, 8) float32
      done:         bool (did the action-replay end with done=True?)
    """
    import robosuite.utils.transform_utils as T_utils

    orig_states = demo["states"][:]                       # (T_orig, 79)
    orig_actions = demo["actions"][:].astype(np.float32)  # (T_orig, 7)

    # OpenVLA: seed -> reset -> set_init_state -> 10 dummy steps to settle sim.
    env.seed(0)
    env.reset()
    obs = env.set_init_state(orig_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())

    # Buffers
    agentview, eye_in_hand, right_view = [], [], []
    ee_pos_list, ee_axisangle_list, gripper_qpos_list = [], [], []
    actions_kept = []

    prev_action = None
    done = False

    for action in orig_actions:
        if is_noop(action, prev_action):
            continue

        # Record current obs (BEFORE stepping with this action).
        # Empirically (verified against IPEC official mp4 PSNR=37dB on image; user
        # visual check showed initial extra [::-1] on right_view caused upside-down):
        # both obs["..._image"] and env.sim.render() return frames in the SAME raw
        # MuJoCo bottom-up orientation, and OpenVLA's rlds_dataset_builder applies
        # a uniform [::-1, ::-1] 180-deg rotation across all views. So all 3 views
        # use the SAME single-step [::-1, ::-1] -- no extra per-view flip.
        rv_raw = env.sim.render(camera_name="rightview", height=resolution, width=resolution)
        agentview.append(obs["agentview_image"][::-1, ::-1])
        eye_in_hand.append(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        right_view.append(rv_raw[::-1, ::-1])

        # state components for state8 (IPEC layout: ee_pos[3] + ee_axisangle[3] + gripper_qpos[2])
        ee_pos_list.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        ee_axisangle_list.append(T_utils.quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32))
        gripper_qpos_list.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))

        # Apply OpenVLA → LeRobot gripper polarity transform AT SOURCE.
        # Raw LIBERO sim demo gripper col ∈ {-1, +1} (-1=open, +1=close).
        # Official IPEC LeRobot release applies (1-g)/2 → {0, 1} (1=open, 0=close,
        # OpenVLA "is_open" probability convention). Our regen previously appended
        # raw LIBERO action verbatim, which caused the 2026-05-21 polarity bug
        # (self-rendered model emit ±1 but eval_libero.py default assumed [0,1]
        # → every gripper command got inverted → catastrophic 35% vs 88%).
        # Apply the same transform here so newly-rendered data is OpenVLA-correct
        # by construction; eval can use default openvla convention without a flag.
        # NOTE: pass the env.step the ORIGINAL libero-convention action (sim needs
        # ±1), but RECORD the transformed action to parquet.
        action_to_record = action.copy().astype(np.float32)
        action_to_record[-1] = (1.0 - action_to_record[-1]) / 2.0  # -1→1 (open), +1→0 (close)
        actions_kept.append(action_to_record)
        prev_action = action  # is_noop comparison uses ORIGINAL gripper convention

        obs, reward, done, info = env.step(action.tolist())

    state8 = _build_state8_from_obs(ee_pos_list, ee_axisangle_list, gripper_qpos_list)

    # Trajectory drift indicator (codex r3 finding): compare final ee_pos against
    # the hdf5 demo's recorded final ee_pos. Open-loop action replay can silently
    # drift if stereo XML injection / sim version / float-nondeterm shifts the
    # trajectory while still ending in done=True. This is a logged warning only
    # (not a skip), since our 37dB PSNR vs official IPEC empirically shows drift
    # is small in practice; the indicator surfaces any future degradation.
    final_drift_ee = None
    try:
        hdf5_ee_pos = demo["obs/ee_pos"][:]  # (T_orig, 3) -- LIBERO native
        # hdf5 last ee_pos (before any no-op trimming); our last ee_pos is from kept frames
        final_drift_ee = float(np.linalg.norm(ee_pos_list[-1] - hdf5_ee_pos[-1]))
    except Exception as e:
        # Some hdf5 schemas may not expose obs/ee_pos at this path -- skip silently.
        pass

    return {
        "agentview": np.stack(agentview, axis=0),
        "eye_in_hand": np.stack(eye_in_hand, axis=0),
        "right_view": np.stack(right_view, axis=0),
        "actions": np.stack(actions_kept, axis=0).astype(np.float32),
        "state8": state8,
        "done": bool(done),
        "final_drift_ee": final_drift_ee,
    }


def _ensure_clean(dst_root: Path, overwrite: bool):
    if dst_root.exists():
        if overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite to clobber")
    dst_root.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=str, default="libero_goal")
    p.add_argument("--libero_raw_data_dir", type=str, required=True,
                   help="Dir containing <task_name>_demo.hdf5 (flat) "
                        "or <hdf5_root>/<suite>/<task_name>_demo.hdf5 (nested)")
    p.add_argument("--libero_target_dir", type=str, required=True,
                   help="Output dir; will write <target_dir>/<suite>/...")
    p.add_argument("--baseline", type=float, default=0.06,
                   help="Stereo baseline in meters (default 6 cm, human IPD)")
    p.add_argument("--num_tasks", type=int, default=-1, help="-1 for all")
    p.add_argument("--num_demos", type=int, default=-1, help="-1 for all per task")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_failed", action="store_true",
                   help="Keep episodes where done=False (default: skip them, matching OpenVLA)")
    p.add_argument("--ipec_ref_dir", type=str,
                   default="/data/wangqiwei/ICLR2026/data/libero_official/libero",
                   help="Optional: dir containing IPEC <suite>_no_noops_1.0.0_lerobot/meta/tasks.jsonl. "
                        "If present, task_index ordering is taken from IPEC tasks.jsonl (so we match "
                        "the official IPEC LeRobot dataset exactly). Falls back to LIBERO benchmark "
                        "internal order if the file is missing.")
    return p.parse_args()


def main():
    args = parse_args()

    # Build language -> absolute_bddl_path lookup from the LIBERO benchmark registry.
    # Matches openvla/experiments/robot/libero/libero_utils.py:get_libero_env: the bddl
    # file argument to OffScreenRenderEnv must be the absolute path under
    # get_libero_path("bddl_files")/<task.problem_folder>/<task.bddl_file>.
    import os as _os
    from libero.libero import benchmark, get_libero_path
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl_root = get_libero_path("bddl_files")
    lang_to_bddl = {}
    for ti in range(task_suite.n_tasks):
        task = task_suite.get_task(ti)
        lang_to_bddl[task.language] = _os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    hdf5_root = Path(args.libero_raw_data_dir)
    dst_root = Path(args.libero_target_dir) / args.suite
    _ensure_clean(dst_root, args.overwrite)

    # Discover hdf5 files (support both flat root or nested <root>/<suite>/).
    all_hdf5 = discover_hdf5_files(hdf5_root, args.suite)
    if not all_hdf5:
        all_hdf5 = sorted(hdf5_root.glob("*_demo.hdf5"))

    # Build language -> hdf5 path lookup
    lang_to_h5 = {}
    for h in all_hdf5:
        with h5py.File(h, "r") as h5:
            lang_to_h5[task_language_from_hdf5(h5)] = h

    # Determine task ordering: prefer IPEC tasks.jsonl (matches official IPEC LeRobot
    # dataset exactly), fall back to LIBERO benchmark internal order if absent.
    ipec_tasks_jsonl = Path(args.ipec_ref_dir) / f"{args.suite}_no_noops_1.0.0_lerobot" / "meta" / "tasks.jsonl"
    if ipec_tasks_jsonl.exists():
        ipec_task_langs = []
        with open(ipec_tasks_jsonl) as f:
            for line in f:
                r = json.loads(line)
                ipec_task_langs.append(r["task"])
        ordering_source = f"IPEC tasks.jsonl ({ipec_tasks_jsonl})"
        ordered_langs = ipec_task_langs
    else:
        ordered_langs = [task_suite.get_task(ti).language for ti in range(task_suite.n_tasks)]
        ordering_source = "LIBERO benchmark internal (IPEC tasks.jsonl not found)"

    hdf5_files = []
    for lang in ordered_langs:
        h = lang_to_h5.get(lang)
        if h is None:
            print(f"  [warn] no hdf5 for task language={lang!r}; skipping")
            continue
        hdf5_files.append(h)
    if args.num_tasks > 0:
        hdf5_files = hdf5_files[:args.num_tasks]

    print(f"[openvla-stereo] suite={args.suite}  tasks={len(hdf5_files)}  "
          f"baseline={args.baseline}  resolution={IMAGE_RESOLUTION}  rotate=180deg")
    print(f"[openvla-stereo] hdf5_root={hdf5_root}")
    print(f"[openvla-stereo] dst_root={dst_root}")
    print(f"[openvla-stereo] task ordering = {ordering_source}")

    task_to_index: dict = {}
    episode_rows: list = []
    total_frames = 0
    global_index_offset = 0
    skipped_no_done = 0
    episode_global_counter = 0  # sequential episode_index, matching IPEC episodes.jsonl pattern
    t_run_start = time.time()

    for task_idx, h5_path in enumerate(hdf5_files):
        with h5py.File(h5_path, "r") as h5:
            task_lang = task_language_from_hdf5(h5)
            task_to_index[task_lang] = task_idx
            demo_keys = list_demo_keys(h5)
            if args.num_demos > 0:
                demo_keys = demo_keys[:args.num_demos]

            bddl_path = lang_to_bddl.get(task_lang)
            if bddl_path is None:
                raw = h5["data"].attrs["bddl_file_name"]
                bddl_path = raw.replace("libero/libero/", "")
                print(f"  [warn] no lang match for {task_lang!r}; using attrs bddl: {bddl_path}")

            print(f"[openvla-stereo] task {task_idx}: {task_lang!r}  "
                  f"demos={len(demo_keys)}  bddl={Path(bddl_path).name}")

            env = make_stereo_env(bddl_file_name=bddl_path,
                                  baseline=args.baseline,
                                  resolution=IMAGE_RESOLUTION)
            try:
                for demo_idx_str in demo_keys:
                    demo = h5[f"data/{demo_idx_str}"]
                    demo_idx = int(demo_idx_str.split("_")[1])
                    # episode_index uses global sequential counter (assigned AFTER done check
                    # below), matching IPEC episodes.jsonl pattern where ep_index is the
                    # cumulative # of accepted (post-no-op-filter, done=True) demos.
                    t0 = time.time()

                    out = render_one_demo_openvla(env, demo, resolution=IMAGE_RESOLUTION)

                    # OpenVLA only keeps episodes where done=True at end
                    if not out["done"] and not args.keep_failed:
                        skipped_no_done += 1
                        print(f"  [demo {demo_idx:02d}] SKIPPED (done=False, T={out['actions'].shape[0]})")
                        continue

                    # Assign sequential episode_index AFTER all filters pass.
                    episode_index = episode_global_counter
                    episode_global_counter += 1
                    T = out["actions"].shape[0]

                    # video out paths
                    base = dst_root / "videos" / "chunk-000"
                    img_path = base / "observation.images.image" / f"episode_{episode_index:06d}.mp4"
                    wrist_path = base / "observation.images.wrist_image" / f"episode_{episode_index:06d}.mp4"
                    right_path = base / "observation.images.right_view" / f"episode_{episode_index:06d}.mp4"
                    parquet_path = dst_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"

                    for p in (img_path.parent, wrist_path.parent, right_path.parent, parquet_path.parent):
                        p.mkdir(parents=True, exist_ok=True)

                    write_video(out["agentview"], img_path, fps=FPS)
                    write_video(out["eye_in_hand"], wrist_path, fps=FPS)
                    write_video(out["right_view"], right_path, fps=FPS)

                    write_parquet_for_episode(
                        out_path=parquet_path,
                        state8=out["state8"],
                        actions=out["actions"],
                        episode_index=episode_index,
                        task_index=task_idx,
                        global_index_offset=global_index_offset,
                    )

                    episode_rows.append({
                        "episode_index": episode_index,
                        "tasks": [task_lang],
                        "length": T,
                    })
                    total_frames += T
                    global_index_offset += T

                    dt = time.time() - t0
                    drift_str = ""
                    if out["final_drift_ee"] is not None:
                        d = out["final_drift_ee"]
                        flag = " ⚠️ DRIFT" if d > 0.05 else ""
                        drift_str = f"  drift_ee={d:.4f}m{flag}"
                    print(f"  [demo {demo_idx:02d}] T={T} ep_idx={episode_index} dt={dt:.1f}s{drift_str}")
            finally:
                # OffScreenRenderEnv close (best-effort)
                try:
                    env.close()
                except Exception:
                    pass

    # Write meta (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, modality.json).
    write_meta(dst_root, task_to_index, episode_rows, total_frames)
    total_dt = time.time() - t_run_start
    print(f"[openvla-stereo] DONE: {len(episode_rows)} episodes  "
          f"{total_frames} frames  total={total_dt:.1f}s  "
          f"skipped_no_done={skipped_no_done}")


if __name__ == "__main__":
    main()
