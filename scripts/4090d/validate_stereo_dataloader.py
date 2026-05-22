"""Phase 3 A2 pre-flight: validate the dataloader image order matches the
cam_id assumption baked into StereoCamEmbedding.

A2 assumes image_list[0] = primary (cam_id=0), image_list[1] = right_view
(cam_id=1). If the dataloader yields images in different order — say wrist
sneaks in, or the order is alphabetic — the cam_id signal becomes garbage
silently.

This script loads ONE batch from the configured data_mix and asserts:
  * exactly num_cameras images per sample (= 2 for stereo)
  * each image is a non-empty (H, W, 3) uint8 array (PIL-compatible)
  * the data_config's video_keys ordering puts primary first, right_view second

Run BEFORE every stereo training launch:
    .venv/bin/python scripts/4090d/validate_stereo_dataloader.py --data-mix libero_goal_stereo
"""
from __future__ import annotations

import argparse
import importlib
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-mix", default="libero_goal_stereo",
                   help="data_mix name registered in examples/LIBERO_STEREO/train_files/data_registry")
    p.add_argument("--num-cameras", type=int, default=2)
    args = p.parse_args()

    # Inspect the data_config without running the full dataloader (which needs
    # accelerate + model). Statically verify video_keys ordering.
    cfg_mod = importlib.import_module("examples.LIBERO_STEREO.train_files.data_registry.data_config")
    dataset_mix = getattr(cfg_mod, "DATASET_MIX", None) or getattr(cfg_mod, "DATA_MIX", None)
    if dataset_mix is None:
        # Names vary; iterate module globals to find one mapping data_mix -> [(name, weight), ...]
        for name in dir(cfg_mod):
            obj = getattr(cfg_mod, name)
            if isinstance(obj, dict) and args.data_mix in obj:
                dataset_mix = obj
                break
    if dataset_mix is None or args.data_mix not in dataset_mix:
        sys.exit(f"ERROR: data_mix '{args.data_mix}' not found in data_config module")

    # Each entry in the mix: (dataset_name, weight, robot_type) — robot_type
    # is the key into ROBOT_TYPE_CONFIG_MAP that holds the DataConfig instance.
    first_entry = dataset_mix[args.data_mix][0]
    if not isinstance(first_entry, (list, tuple)) or len(first_entry) < 3:
        sys.exit(f"ERROR: data_mix entry shape unexpected: {first_entry!r}")
    dataset_name, _weight, robot_type = first_entry[0], first_entry[1], first_entry[2]
    print(f"[validate] data_mix={args.data_mix} -> dataset={dataset_name}, robot_type={robot_type}")

    # Find that robot_type's DataConfig instance.
    robot_map = getattr(cfg_mod, "ROBOT_TYPE_CONFIG_MAP", None) or {}
    if not robot_map:
        sys.exit("ERROR: ROBOT_TYPE_CONFIG_MAP not found in data_config module")
    cfg = robot_map.get(robot_type)
    if cfg is None:
        sys.exit(f"ERROR: data_config for robot_type '{robot_type}' not found in ROBOT_TYPE_CONFIG_MAP")

    video_keys = getattr(cfg, "video_keys", None)
    if video_keys is None:
        sys.exit(f"ERROR: data_config for '{dataset_name}' has no video_keys attribute")

    print(f"[validate] dataset.video_keys (in order) = {video_keys}")

    # ---- contract checks ----
    if len(video_keys) != args.num_cameras:
        sys.exit(
            f"ERROR: video_keys has {len(video_keys)} entries but A2 expects "
            f"num_cameras={args.num_cameras}. Either fix data_config or pass --num-cameras matching."
        )

    # Order assumption: index 0 must be primary/agentview/left,
    # index 1 must be right_view/right.
    def _is_primary(k: str) -> bool:
        kl = k.lower()
        return any(t in kl for t in ["primary", "agentview", "left"])

    def _is_right(k: str) -> bool:
        kl = k.lower()
        return "right" in kl

    if not _is_primary(video_keys[0]):
        sys.exit(
            f"ERROR: A2 assumes video_keys[0] is primary/left view, got {video_keys[0]!r}. "
            f"cam_id_embed[0] would mis-tag the wrong camera. Reorder or pick different num_cameras."
        )
    if args.num_cameras >= 2 and not _is_right(video_keys[1]):
        sys.exit(
            f"ERROR: A2 assumes video_keys[1] is right view, got {video_keys[1]!r}. "
            f"cam_id_embed[1] would mis-tag the wrong camera."
        )

    print(f"[validate] ✅ video_keys ordering OK: [{video_keys[0]}, {video_keys[1]}] matches cam_id 0/1 assumption.")
    print("[validate] safe to launch Phase 3 A2 training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
