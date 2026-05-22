"""
render_stereo_libero_hdf5.py
============================

Re-render LIBERO HDF5 demos into a stereo LeRobot dataset by replaying the
per-frame MuJoCo state recorded in the HDF5 files (bit-exact) with TWO
mounted cameras separated by `--baseline` meters.

Bypasses the IPEC-mono -> init_state mapping problem: instead of trying to
recover (episode_idx -> init_state_idx) from the pruned IPEC parquet, we go
straight to the source HDF5 demos which carry the full (init_state, states,
actions) tuple per demonstration.

Per-frame reproduction recipe
-----------------------------
  env.set_init_state(states[0])            # seed sim with the saved init state
  for t in range(T):
      env.regenerate_obs_from_state(states[t])
      left  = env.sim.render(camera_name="agentview",          ...)[::-1]
      right = env.sim.render(camera_name="rightview",          ...)[::-1]
      wrist = env.sim.render(camera_name="robot0_eye_in_hand", ...)[::-1]

`regenerate_obs_from_state` calls `set_state_from_flattened + sim.forward +
_update_observables(force=True)`, so the sim is fully consistent with the
recorded state at frame t. No action accumulation -> no drift.

LeRobot 8-dim state mapping (verified by IPEC cross-check on libero_spatial)
---------------------------------------------------------------------------
state8 = concat(obs/ee_pos[3], obs/ee_ori[3], obs/gripper_states[2])
       = (x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper, gripper)
NB: IPEC's modality.json labels state[6] as "pad" but the actual values are
the second gripper finger (signed pair around zero). We keep that label for
schema compatibility but the numeric content is gripper_states[0:2].
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

# Reuse the stereo-camera-injection plumbing from the original script.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _stereo_render_utils import (  # type: ignore  # noqa: E402
    compute_right_camera_pose,
    _inject_camera_into_xml_string,
    make_stereo_env,
    render_step,
    write_video,
)


# -----------------------------------------------------------------------------
# Constants / schema
# -----------------------------------------------------------------------------

FPS = 20

# IPEC LeRobot info.json describes wrist/image with codec av1 + yuv420p but we
# write libx264 + yuv420p (HF reader accepts both; av1 is slow to encode).
INFO_FEATURES_TEMPLATE = {
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": 256, "video.width": 256, "video.codec": "av1",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "video.fps": FPS, "video.channels": 3, "has_audio": False,
        },
    },
    "observation.images.image": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": 256, "video.width": 256, "video.codec": "av1",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "video.fps": FPS, "video.channels": 3, "has_audio": False,
        },
    },
    "observation.images.right_view": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": 256, "video.width": 256, "video.codec": "av1",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "video.fps": FPS, "video.channels": 3, "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32", "shape": [8],
        "names": {"motors": ["x","y","z","axis_angle1","axis_angle2","axis_angle3","gripper","gripper"]},
    },
    "action": {
        "dtype": "float32", "shape": [7],
        "names": {"motors": ["x","y","z","axis_angle1","axis_angle2","axis_angle3","gripper"]},
    },
    "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
    "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
    "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
    "index":         {"dtype": "int64",   "shape": [1], "names": None},
    "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
}

MODALITY_TEMPLATE = {
    "state": {
        "x":           {"start": 0, "end": 1},
        "y":           {"start": 1, "end": 2},
        "z":           {"start": 2, "end": 3},
        "roll":        {"start": 3, "end": 4},
        "pitch":       {"start": 4, "end": 5},
        "yaw":         {"start": 5, "end": 6},
        "pad":         {"start": 6, "end": 7},
        "gripper":     {"start": 7, "end": 8},
    },
    "action": {
        "x":       {"start": 0, "end": 1},
        "y":       {"start": 1, "end": 2},
        "z":       {"start": 2, "end": 3},
        "roll":    {"start": 3, "end": 4},
        "pitch":   {"start": 4, "end": 5},
        "yaw":     {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "primary_image": {"original_key": "observation.images.image"},
        "right_view":    {"original_key": "observation.images.right_view"},
        "wrist_image":   {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------

def discover_hdf5_files(hdf5_root: Path, suite: str) -> list[Path]:
    return sorted((hdf5_root / suite).glob("*_demo.hdf5"))


def task_language_from_hdf5(h5: h5py.File) -> str:
    pinfo = json.loads(h5["data"].attrs["problem_info"])
    return pinfo["language_instruction"]


def list_demo_keys(h5: h5py.File) -> list[str]:
    # Sort by integer suffix so demo_0, demo_1, ..., demo_49 are in order
    keys = [k for k in h5["data"].keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def build_state8(demo: h5py.Group, t_slice: slice = slice(None)) -> np.ndarray:
    ee_pos = demo["obs/ee_pos"][t_slice]                # (T, 3)
    ee_ori = demo["obs/ee_ori"][t_slice]                # (T, 3) axis-angle
    grip   = demo["obs/gripper_states"][t_slice]        # (T, 2)
    state8 = np.concatenate([ee_pos, ee_ori, grip], axis=1).astype(np.float32)
    return state8


# -----------------------------------------------------------------------------
# Per-demo rendering
# -----------------------------------------------------------------------------

def render_one_demo(
    h5_demo: h5py.Group,
    env,
    resolution: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Replay the per-frame MuJoCo state and harvest left/right/wrist videos.

    Returns (left_frames, right_frames, wrist_frames), each length T.
    """
    states = h5_demo["states"][:]      # (T, 92)
    T = states.shape[0]

    env.seed(0)
    env.reset()
    # Bit-exact init: env.set_init_state internally calls
    # set_state_from_flattened + sim.forward + update_observables.
    env.set_init_state(states[0])

    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []

    for t in range(T):
        # regenerate_obs_from_state does: set_state -> sim.forward ->
        # _update_observables(force=True). This is the cleanest way to
        # render at the exact recorded state with no action accumulation.
        env.regenerate_obs_from_state(states[t])
        views = render_step(env, resolution)
        left_frames.append(views["left"])
        right_frames.append(views["right"])
        wrist_frames.append(views["wrist"])

    return left_frames, right_frames, wrist_frames


# -----------------------------------------------------------------------------
# LeRobot writers
# -----------------------------------------------------------------------------

def write_parquet_for_episode(
    out_path: Path,
    state8: np.ndarray,        # (T, 8) float32
    actions: np.ndarray,       # (T, 7) float32
    episode_index: int,
    task_index: int,
    global_index_offset: int,
) -> None:
    T = state8.shape[0]
    assert actions.shape[0] == T, f"state/action length mismatch: {T} vs {actions.shape[0]}"

    frame_index = np.arange(T, dtype=np.int64)
    timestamp = (frame_index.astype(np.float32) / float(FPS))
    episode_index_arr = np.full(T, episode_index, dtype=np.int64)
    index_arr = np.arange(global_index_offset, global_index_offset + T, dtype=np.int64)
    task_index_arr = np.full(T, task_index, dtype=np.int64)

    # Per-row object dtype (arrays) for state and action columns -- matches IPEC
    # parquet layout (read back with df['observation.state'].iloc[i] -> ndarray).
    df = pd.DataFrame({
        "observation.state": list(state8),
        "action":            list(actions),
        "timestamp":         timestamp,
        "frame_index":       frame_index,
        "episode_index":     episode_index_arr,
        "index":             index_arr,
        "task_index":        task_index_arr,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def write_meta(
    dst_root: Path,
    task_to_index: dict[str, int],
    episode_rows: list[dict],
    total_frames: int,
) -> None:
    meta = dst_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episode_rows),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": len(episode_rows) * 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": INFO_FEATURES_TEMPLATE,
    }
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    with open(meta / "modality.json", "w") as f:
        json.dump(MODALITY_TEMPLATE, f, indent=2)

    # tasks.jsonl
    with open(meta / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # episodes.jsonl
    with open(meta / "episodes.jsonl", "w") as f:
        for row in episode_rows:
            f.write(json.dumps(row) + "\n")

    # episodes_stats.jsonl: starVLA's dataloader doesn't strictly require it
    # but it's part of LeRobot v2.1 spec. Write minimal {count} per episode.
    with open(meta / "episodes_stats.jsonl", "w") as f:
        for row in episode_rows:
            f.write(json.dumps({
                "episode_index": row["episode_index"],
                "stats": {"count": [row["length"]]},
            }) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--num_demos", type=int, default=-1,
                   help="Demos PER TASK (-1 = all 50)")
    p.add_argument("--num_tasks", type=int, default=-1,
                   help="Tasks to process (-1 = all)")
    p.add_argument("--baseline", type=float, default=0.06)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--hdf5_root", type=str,
        default="/data/wangqiwei/ICLR2026/data/libero_hdf5",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="/data/wangqiwei/ICLR2026/starVLA/playground/Datasets/LEROBOT_LIBERO_STEREO_DATA",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Wipe data/videos/meta of an existing output_dir/<suite> before re-rendering. Without this flag, a populated directory aborts the run to prevent stale-episode contamination of dataset statistics.",
    )
    return p.parse_args()



def _ensure_clean_output(dst_root, overwrite: bool) -> None:
    """Reject reuse of a non-empty stereo dataset directory unless --overwrite.

    Rerunning the renderer with  and then only
    rewriting metadata leaves stale parquet/video files from prior runs
    (e.g., different --num_demos, --baseline, --resolution, or after a partial
    crash). Downstream dataset statistics glob  so the stale
    parquets silently contaminate normalization while 
    describes a different dataset.

    Behavior:
      * If dst_root does not exist → create and proceed.
      * If dst_root exists but is empty → proceed.
      * If dst_root has any of data/ / videos/ / meta/ populated:
          - overwrite=False (default) → raise SystemExit with clear message.
          - overwrite=True → wipe data/ / videos/ / meta/ atomically per-dir.
    """
    import shutil
    dst_root = Path(dst_root)
    SUBDIRS = ("data", "videos", "meta")
    populated = []
    if dst_root.exists():
        for sub in SUBDIRS:
            d = dst_root / sub
            if d.exists() and any(d.iterdir()):
                populated.append(sub)
    if populated and not overwrite:
        raise SystemExit(
            f"[stereo-render] refuse to write to non-empty output: {dst_root} "
            f"(populated subdirs: {populated}). Pass --overwrite to wipe and re-render."
        )
    if populated and overwrite:
        for sub in SUBDIRS:
            d = dst_root / sub
            if d.exists():
                print(f"[stereo-render] [--overwrite] removing {d}", flush=True)
                shutil.rmtree(d)
    dst_root.mkdir(parents=True, exist_ok=True)

def main() -> None:
    args = parse_args()
    hdf5_root = Path(args.hdf5_root)
    dst_root = Path(args.output_dir) / args.suite
    _ensure_clean_output(dst_root, args.overwrite)

    # LIBERO benchmark gives us the BDDL path. Build a (language -> bddl_path)
    # map from the benchmark suite so we can match HDF5 (which carries the
    # language) to a bddl file deterministically.
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    lang_to_bddl: dict[str, str] = {}
    for i in range(task_suite.n_tasks):
        t = task_suite.get_task(i)
        bddl = str(task_suite.get_task_bddl_file_path(i))
        lang_to_bddl[t.language] = bddl

    hdf5_files = discover_hdf5_files(hdf5_root, args.suite)
    if args.num_tasks > 0:
        hdf5_files = hdf5_files[: args.num_tasks]

    print(f"[hdf5-render] suite={args.suite}  tasks={len(hdf5_files)}  "
          f"baseline={args.baseline}  resolution={args.resolution}")
    print(f"[hdf5-render] hdf5_root={hdf5_root}")
    print(f"[hdf5-render] dst_root={dst_root}")

    task_to_index: dict[str, int] = {}
    episode_rows: list[dict] = []
    total_frames = 0
    global_index_offset = 0
    t_run_start = time.time()

    for task_idx, h5_path in enumerate(hdf5_files):
        with h5py.File(h5_path, "r") as h5:
            task_lang = task_language_from_hdf5(h5)
            task_to_index[task_lang] = task_idx
            demo_keys = list_demo_keys(h5)
            if args.num_demos > 0:
                demo_keys = demo_keys[: args.num_demos]

            bddl_path = lang_to_bddl.get(task_lang)
            if bddl_path is None:
                # Fallback: derive bddl filename from HDF5 attrs
                raw = h5["data"].attrs["bddl_file_name"]
                bddl_path = raw.replace("libero/libero/", "")  # let LIBERO resolve
                # In practice we should always hit lang_to_bddl; warn instead.
                print(f"  [warn] no lang match for {task_lang!r}; "
                      f"using attrs bddl: {bddl_path}")

            print(f"[hdf5-render] task {task_idx}: {task_lang!r}  "
                  f"demos={len(demo_keys)}  bddl={Path(bddl_path).name}")

            # Build one env per task -- bddl is constant within a task. Initial
            # reset_from_xml_string inside make_stereo_env preserves rightview.
            env = make_stereo_env(
                bddl_file_name=bddl_path,
                baseline=args.baseline,
                resolution=args.resolution,
            )
            try:
                for demo_idx_str in demo_keys:
                    demo = h5[f"data/{demo_idx_str}"]
                    demo_idx = int(demo_idx_str.split("_")[1])
                    episode_index = task_idx * 50 + demo_idx
                    t0 = time.time()

                    actions = demo["actions"][:].astype(np.float32)  # (T, 7)
                    state8 = build_state8(demo)                       # (T, 8)
                    T = actions.shape[0]
                    assert state8.shape[0] == T

                    left, right, wrist = render_one_demo(
                        demo, env, args.resolution
                    )

                    # Write videos
                    base = f"episode_{episode_index:06d}.mp4"
                    write_video(
                        left,
                        dst_root / "videos/chunk-000/observation.images.image" / base,
                        fps=FPS,
                    )
                    write_video(
                        right,
                        dst_root / "videos/chunk-000/observation.images.right_view" / base,
                        fps=FPS,
                    )
                    write_video(
                        wrist,
                        dst_root / "videos/chunk-000/observation.images.wrist_image" / base,
                        fps=FPS,
                    )

                    # Write parquet
                    write_parquet_for_episode(
                        out_path=dst_root / "data" / "chunk-000"
                                 / f"episode_{episode_index:06d}.parquet",
                        state8=state8,
                        actions=actions,
                        episode_index=episode_index,
                        task_index=task_idx,
                        global_index_offset=global_index_offset,
                    )

                    episode_rows.append({
                        "episode_index": episode_index,
                        "tasks": [task_lang],
                        "length": int(T),
                    })
                    total_frames += T
                    global_index_offset += T
                    dt = time.time() - t0
                    print(f"  [demo {demo_idx:02d}] T={T} ep_idx={episode_index} "
                          f"dt={dt:.1f}s")
            finally:
                env.close()

    # Final meta pass after all episodes are known
    write_meta(dst_root, task_to_index, episode_rows, total_frames)

    total_dt = time.time() - t_run_start
    n_ep = len(episode_rows)
    print(f"[hdf5-render] DONE: {n_ep} episodes  {total_frames} frames  "
          f"total={total_dt:.1f}s  mean_ep={total_dt/max(1,n_ep):.1f}s/ep")


if __name__ == "__main__":
    main()
