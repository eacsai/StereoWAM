#!/usr/bin/env python3
"""Build episode->scene-flow sidecar index and audit RGB alignment.

The mapping deliberately uses HDF5 native demo-key order, matching the LeRobot
conversion path. Do not replace it with numeric demo sorting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FLIPS = ("identity", "flip_ud", "flip_lr", "rot180")
CAMERA_TO_VIDEO_KEY = {
    "agentview": "video.primary_image",
    "primary": "video.primary_image",
    "primary_image": "video.primary_image",
    "right_view": "video.right_view",
    "eye_in_hand": "video.wrist_image",
    "wrist": "video.wrist_image",
    "wrist_image": "video.wrist_image",
}


def task_slug(task: str) -> str:
    return str(task).replace(" ", "_")


def apply_flip(image: np.ndarray, flip: str) -> np.ndarray:
    if flip == "identity":
        return image
    if flip == "flip_ud":
        return image[::-1, ...]
    if flip == "flip_lr":
        return image[:, ::-1, ...]
    if flip == "rot180":
        return image[::-1, ::-1, ...]
    raise ValueError(f"unknown flip {flip!r}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks(lerobot_root: Path) -> dict[int, str]:
    tasks = {}
    for row in load_jsonl(lerobot_root / "meta" / "tasks.jsonl"):
        tasks[int(row["task_index"])] = row.get("task") or row.get("text") or row.get("name")
    return tasks


def episode_task(episode: dict, tasks: dict[int, str]) -> str:
    if "task" in episode and isinstance(episode["task"], str):
        return episode["task"]
    if "tasks" in episode and episode["tasks"]:
        t0 = episode["tasks"][0]
        # Some LeRobot exports store the task STRING directly in episode["tasks"];
        # others store an integer index into the tasks table. Handle both.
        if isinstance(t0, str):
            return t0
        return tasks[int(t0)]
    if "task_index" in episode:
        return tasks[int(episode["task_index"])]
    raise KeyError(f"could not infer task for episode {episode}")


def hdf5_native_demo_keys(hdf5_path: Path) -> list[str]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        return list(f["data"].keys())


def sidecar_metadata(sidecar_path: Path) -> dict:
    with np.load(sidecar_path, allow_pickle=False) as npz:
        metadata = {"pair_count": int(npz["flow_3d"].shape[0])}
        if "camera_name" in npz:
            value = npz["camera_name"]
            if isinstance(value, np.ndarray):
                value = value.item() if value.shape == () else value.tolist()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            metadata["camera_name"] = str(value)
        return metadata


def build_index(lerobot_root: Path, raw_hdf5_root: Path, sidecar_root: Path) -> dict:
    tasks = load_tasks(lerobot_root)
    episodes = load_jsonl(lerobot_root / "meta" / "episodes.jsonl")

    grouped: dict[str, list[dict]] = {}
    for episode in episodes:
        grouped.setdefault(episode_task(episode, tasks), []).append(episode)

    records = {}
    missing_hdf5 = []
    missing_sidecar = []
    for task, task_episodes in grouped.items():
        slug = task_slug(task)
        stem = slug
        hdf5_path = raw_hdf5_root / f"{slug}_demo.hdf5"
        if not hdf5_path.exists():
            # libero_10 etc.: HDF5/npz carry a scene prefix (e.g. KITCHEN_SCENE6_) that the
            # lerobot task text strips. Suffix-match the real file and reuse its actual stem.
            cands = sorted(raw_hdf5_root.glob(f"*{slug}_demo.hdf5"))
            if len(cands) == 1:
                hdf5_path = cands[0]
                stem = hdf5_path.name[: -len("_demo.hdf5")]
            else:
                missing_hdf5.append(str(hdf5_path) + (f" (ambiguous:{len(cands)})" if cands else ""))
                continue
        demo_keys = hdf5_native_demo_keys(hdf5_path)
        for within_task_idx, episode in enumerate(task_episodes):
            if within_task_idx >= len(demo_keys):
                continue
            demo_key = demo_keys[within_task_idx]
            sidecar = sidecar_root / f"{stem}_demo_{demo_key}_sceneflow.npz"
            episode_index = int(episode["episode_index"])
            if not sidecar.exists():
                missing_sidecar.append(str(sidecar))
                continue
            metadata = sidecar_metadata(sidecar)
            records[str(episode_index)] = {
                "task": task,
                "task_slug": slug,
                "hdf5_stem": stem,
                "demo_key": demo_key,
                "sidecar": str(sidecar.resolve()),
                "pair_count": metadata["pair_count"],
            }
            if "camera_name" in metadata:
                records[str(episode_index)]["camera_name"] = metadata["camera_name"]

    return {
        "schema_version": 1,
        "lerobot_root": str(lerobot_root.resolve()),
        "raw_hdf5_root": str(raw_hdf5_root.resolve()),
        "sidecar_root": str(sidecar_root.resolve()),
        "episodes": records,
        "missing_hdf5_count": len(missing_hdf5),
        "missing_sidecar_count": len(missing_sidecar),
        "missing_hdf5_first": missing_hdf5[:20],
        "missing_sidecar_first": missing_sidecar[:20],
    }


def build_training_dataset(args):
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    cfg = OmegaConf.load(args.config_yaml).datasets.vla_data
    cfg.data_root_dir = str(args.lerobot_root.parent)
    cfg.scene_flow = cfg.get("scene_flow", {}) or {}
    cfg.scene_flow.enabled = False
    return make_LeRobotSingleDataset(
        data_root_dir=args.lerobot_root.parent,
        data_name=args.lerobot_root.name,
        robot_type=args.robot_type,
        delete_pause_frame=bool(cfg.get("delete_pause_frame", False)),
        data_cfg=cfg,
    )


def load_hdf5_camera_rgb(raw_hdf5_root: Path, record: dict, base_index: int, camera_name: str) -> np.ndarray:
    import h5py

    hdf5_path = raw_hdf5_root / f"{record.get('hdf5_stem', record['task_slug'])}_demo.hdf5"
    hdf5_key = f"obs/{camera_name}_rgb"
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["data"][record["demo_key"]][hdf5_key][base_index])


def sidecar_camera_name(record: dict, args) -> str:
    return str(record.get("camera_name") or args.sidecar_camera_name or "agentview")


def find_training_sample(dataset, episode_id: int, base_index: int) -> tuple[dict, int]:
    for sample_index, (trajectory_id, step_index) in enumerate(dataset.all_steps):
        if int(trajectory_id) == int(episode_id) and int(step_index) == int(base_index):
            return dataset[sample_index], sample_index
    raise KeyError(f"episode {episode_id} base_index {base_index} is not present in the training dataset")


def infer_training_image_index(dataset, camera_name: str, args) -> tuple[int, str, int]:
    video_keys = list(dataset.modality_keys["video"])
    video_key = args.primary_video_key or CAMERA_TO_VIDEO_KEY.get(str(camera_name).lower())
    if video_key is None:
        raise RuntimeError(
            f"cannot infer training video key for sidecar camera_name={camera_name!r}; "
            "pass --primary-video-key explicitly"
        )
    if video_key not in video_keys:
        raise RuntimeError(
            f"sidecar camera_name={camera_name!r} maps to {video_key!r}, but training video keys are {video_keys}"
        )

    delta_indices = np.asarray(dataset.delta_indices[video_key])
    zero_positions = np.flatnonzero(delta_indices == 0)
    frame_offset = int(zero_positions[-1]) if len(zero_positions) else len(delta_indices) - 1
    image_index = video_keys.index(video_key) * len(delta_indices) + frame_offset
    if args.primary_image_index is not None and int(args.primary_image_index) != image_index:
        raise RuntimeError(
            f"--primary-image-index={args.primary_image_index} disagrees with training layout: "
            f"{video_key} at camera-major image index {image_index} "
            f"(video_keys={video_keys}, delta_indices={delta_indices.tolist()})"
        )
    return image_index, video_key, frame_offset


def draw_flow_overlay(training_rgb: np.ndarray, record: dict, base_index: int, flip: str, arrow_scale_px: float) -> Image.Image:
    overlay = Image.fromarray(training_rgb).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    with np.load(record["sidecar"], allow_pickle=False) as npz:
        if base_index >= npz["flow_3d"].shape[0]:
            return overlay
        flow = apply_flip(np.asarray(npz["flow_3d"][base_index], dtype=np.float32), flip)
        dynamic = apply_flip(np.asarray(npz["dynamic_mask"][base_index], dtype=bool), flip)

    h_src, w_src = dynamic.shape
    h_dst, w_dst = training_rgb.shape[:2]
    stride = max(1, h_src // 16)
    for y in range(stride // 2, h_src, stride):
        for x in range(stride // 2, w_src, stride):
            if not dynamic[y, x]:
                continue
            x0 = int(round(x * (w_dst - 1) / max(w_src - 1, 1)))
            y0 = int(round(y * (h_dst - 1) / max(h_src - 1, 1)))
            dx = float(flow[y, x, 0]) * arrow_scale_px
            dy = float(flow[y, x, 1]) * arrow_scale_px
            x1 = int(round(x0 + dx))
            y1 = int(round(y0 + dy))
            draw.line((x0, y0, x1, y1), fill=(255, 32, 32), width=2)
            draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=(255, 240, 0))
    return overlay


def audit_alignment(args, index_data: dict) -> dict:
    if args.audit_samples <= 0:
        raise RuntimeError("scene-flow index build requires --audit-samples > 0; refusing an unaudited identity default")

    dataset = build_training_dataset(args)
    records = index_data["episodes"]
    out_dir = Path(args.audit_png_dir) if args.audit_png_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    audited = []
    for episode_key in sorted(records, key=lambda x: int(x))[: args.audit_samples]:
        record = records[episode_key]
        episode_id = int(episode_key)
        base_index = int(args.audit_base_index)
        camera_name = sidecar_camera_name(record, args)
        sample, sample_index = find_training_sample(dataset, episode_id, base_index)
        image_index, video_key, frame_offset = infer_training_image_index(dataset, camera_name, args)
        training_rgb = np.asarray(sample["image"][image_index].convert("RGB"))
        source_rgb = load_hdf5_camera_rgb(args.raw_hdf5_root, record, base_index, camera_name)
        source_rgb = np.asarray(Image.fromarray(source_rgb).resize(training_rgb.shape[1::-1]))

        diffs = {}
        for flip in FLIPS:
            candidate = apply_flip(source_rgb, flip)
            diffs[flip] = float(np.mean(np.abs(candidate.astype(np.float32) - training_rgb.astype(np.float32))))
        ranked = sorted(diffs.items(), key=lambda item: item[1])
        best_flip = ranked[0][0]
        unique = len(ranked) == 1 or (ranked[1][1] - ranked[0][1]) > args.unique_margin
        record["sidecar_to_training_flip"] = best_flip
        record["alignment_audit"] = {
            "audited": True,
            "base_index": base_index,
            "dataset_sample_index": sample_index,
            "camera_name": camera_name,
            "training_video_key": video_key,
            "training_image_index": image_index,
            "training_frame_offset": frame_offset,
            "rgb_mean_abs_diff": diffs,
            "unique_minimum": unique,
        }
        audited.append({"episode_index": episode_id, "best_flip": best_flip, "unique_minimum": unique})

        if out_dir:
            tiles = [Image.fromarray(training_rgb)]
            labels = ["training_primary"]
            for flip in FLIPS:
                tiles.append(Image.fromarray(apply_flip(source_rgb, flip)))
                labels.append(f"{flip}:{diffs[flip]:.2f}")
            tiles.append(draw_flow_overlay(training_rgb, record, base_index, best_flip, args.flow_arrow_scale_px))
            labels.append(f"flow_on_primary:{best_flip}")
            tile_w, tile_h = tiles[0].size
            canvas = Image.new("RGB", (tile_w * len(tiles), tile_h + 18), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (tile, label) in enumerate(zip(tiles, labels)):
                canvas.paste(tile, (i * tile_w, 18))
                draw.text((i * tile_w + 2, 2), label, fill=(0, 0, 0))
            canvas.save(out_dir / f"episode_{episode_id:06d}_base_{base_index:04d}_alignment.png")

    if audited:
        if not all(item["unique_minimum"] for item in audited):
            raise RuntimeError(f"alignment audit did not find a unique best flip for all samples: {audited}")
        flips = {item["best_flip"] for item in audited}
        if len(flips) != 1:
            raise RuntimeError(f"alignment audit found inconsistent flips across samples: {audited}")
        global_flip = next(iter(flips))
        expected_flip = str(args.expected_flip).lower()
        if expected_flip not in {"", "any", "none"} and global_flip != expected_flip:
            raise RuntimeError(
                f"alignment audit found global_flip={global_flip!r}, expected {expected_flip!r}; "
                "identity or another unexpected flip means the sidecar/training image convention must be inspected"
            )
        for record in records.values():
            if "alignment_audit" not in record:
                record["sidecar_to_training_flip"] = global_flip
                record["alignment_audit"] = {
                    "audited": False,
                    "propagated_global_flip": global_flip,
                    "source": "alignment_audit_consensus",
                }
        return {
            "enabled": True,
            "global_flip": global_flip,
            "expected_flip": expected_flip,
            "audited": audited,
        }

    raise RuntimeError("alignment audit did not audit any samples; refusing to write an enabled index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--raw-hdf5-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-samples", type=int, default=8)
    parser.add_argument("--config-yaml", type=Path, default=Path("examples/LIBERO/train_files/starvla_cotrain_libero.yaml"))
    parser.add_argument("--robot-type", type=str, default="libero")
    parser.add_argument("--primary-image-index", type=int, default=None)
    parser.add_argument("--primary-video-key", type=str, default=None)
    parser.add_argument("--sidecar-camera-name", type=str, default=None)
    parser.add_argument("--expected-flip", type=str, default="rot180", choices=(*FLIPS, "any", "none"))
    parser.add_argument("--audit-base-index", type=int, default=0)
    parser.add_argument("--unique-margin", type=float, default=1e-3)
    parser.add_argument("--audit-png-dir", type=Path, default=None)
    parser.add_argument("--flow-arrow-scale-px", type=float, default=1500.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_data = build_index(args.lerobot_root, args.raw_hdf5_root, args.sidecar_root)
    index_data["expected_sidecar_to_training_flip"] = str(args.expected_flip).lower()
    index_data["alignment_audit"] = audit_alignment(args, index_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(index_data, f, indent=2)
    print(
        f"[scene-flow] wrote {args.output} with {len(index_data['episodes'])} indexed episodes; "
        f"missing_sidecar_count={index_data['missing_sidecar_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
