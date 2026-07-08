#!/usr/bin/env python3
"""Smoke-test vectorized render_world_view against the old reference renderer."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data/wangqiwei/ICLR2026/starVLA")
PROBE_DIR = ROOT / "scripts" / "tools"
PRECOMP_DIR = ROOT / "scripts" / "4090d"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    parser.add_argument("--n-rows", type=int, default=5)
    parser.add_argument("--suite", default="libero_object_no_noops_1.0.0_lerobot")
    parser.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW")
    parser.add_argument("--data-mix", default="libero_all_sfstereo_leftprimary")
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--ffs-repo-dir", default="/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
    parser.add_argument("--ffs-model-path", default="/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth")
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--level-table", choices=["off", "auto", "always"], default="auto")
    return parser.parse_args()


def _import_runtime(args: argparse.Namespace):
    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    for p in (str(ROOT), str(PROBE_DIR), str(PRECOMP_DIR), str(Path(args.ffs_repo_dir).resolve())):
        if p not in sys.path:
            sys.path.insert(0, p)
    import probe_orthogonal_multiview_render as probe  # noqa: WPS433
    import precompute_ortho_view_cache as precomp  # noqa: WPS433
    return probe, precomp


def _load_suite(args: argparse.Namespace, precomp):
    cfg = precomp._load_cfg(args)
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    mixture_spec = DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]
    available = [name for name, _w, _rt in mixture_spec]
    if args.suite not in available:
        raise ValueError(f"suite {args.suite!r} not in data_mix {cfg.datasets.vla_data.data_mix!r}; available={available}")
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
    by_name = {single.dataset_name: single for single in dataset.datasets}
    if args.suite not in by_name:
        raise ValueError(f"suite {args.suite!r} loaded but not in dataset.datasets; got {list(by_name)}")
    return by_name[args.suite]


def _geometry_for_sample(sample, renderer, args, probe, precomp):
    images = sample["image"]
    primary = precomp._to_np_256(images[0])
    left_view = precomp._to_np_256(images[1])
    right_view_pos_w, right_view_R_mjc_to_w = probe.load_right_view_pose(
        precomp._probe_suite_name(sample["dataset_name"])
    )

    disp = renderer.run_ffs_disparity(left_view, primary)
    torch.cuda.synchronize()
    depth, valid = probe.disparity_to_depth(disp, probe.FX, probe.BASELINE_M)
    pts, cols = probe.unproject(depth, left_view, valid, probe.VALID_LO, probe.VALID_HI)
    pts_w = probe.camera_to_world(pts, right_view_pos_w, right_view_R_mjc_to_w)
    pts_w, table_deg, table_inliers, leveled, level_reason = probe.maybe_level_table(
        pts_w,
        cols,
        args.level_table,
    )
    origin_w = pts_w.mean(0)
    pts_rel, cols_rel = probe.workspace_crop(pts_w, cols, origin_w)
    return pts_rel, cols_rel, {
        "table_deg": None if np.isnan(table_deg) else float(table_deg),
        "table_inliers": int(table_inliers),
        "leveled": bool(leveled),
        "level_reason": str(level_reason),
    }


def _foreground_iou(old: np.ndarray, new: np.ndarray) -> float:
    old_fg = np.any(old != 0, axis=2)
    new_fg = np.any(new != 0, axis=2)
    union = int(np.logical_or(old_fg, new_fg).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(old_fg, new_fg).sum()) / float(union)


def _compare_view(row: int, view: str, pts_rel, cols_rel, probe):
    t0 = time.perf_counter()
    old = probe.render_world_view_ref(pts_rel, cols_rel, view)
    ref_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = probe.render_world_view(pts_rel, cols_rel, view)
    vec_s = time.perf_counter() - t0

    equal = bool(np.array_equal(old, new))
    diff_count = 0
    iou = 1.0
    if not equal:
        diff_mask = np.any(old != new, axis=2)
        diff_count = int(diff_mask.sum())
        iou = _foreground_iou(old, new)
        coords = np.argwhere(diff_mask)[:5]
        print(f"  DIFF row={row} view={view} pixels={diff_count} foreground_iou={iou:.6f}")
        for y, x in coords:
            print(f"    y={int(y)} x={int(x)} old={old[y, x].tolist()} new={new[y, x].tolist()}")

    print(
        f"  row={row} view={view:5s} equal={equal} "
        f"ref={ref_s * 1e3:.1f}ms vec={vec_s * 1e3:.1f}ms "
        f"speedup={ref_s / max(vec_s, 1e-9):.1f}x diff_pixels={diff_count} fg_iou={iou:.6f}",
        flush=True,
    )
    return equal, ref_s, vec_s


def main() -> int:
    args = _parse_args()
    probe, precomp = _import_runtime(args)
    if not torch.cuda.is_available():
        raise RuntimeError("smoke_render_vec requires CUDA for FFS geometry")

    print("SMOKE_RENDER_VEC: loading cfg + dataset", flush=True)
    t0 = time.perf_counter()
    target = _load_suite(args, precomp)
    print(f"SMOKE_RENDER_VEC: dataset loaded in {time.perf_counter() - t0:.1f}s; suite={target.dataset_name} rows={len(target.all_steps)}", flush=True)

    renderer = precomp.OrthoRenderer(args)
    print("SMOKE_RENDER_VEC: warming up FFS", flush=True)
    renderer._ensure_ffs()
    z = np.zeros((probe.H, probe.W, 3), dtype=np.uint8)
    _ = renderer.run_ffs_disparity(z, z)
    torch.cuda.synchronize()

    n = min(args.n_rows, len(target.all_steps))
    all_equal = True
    ref_times = []
    vec_times = []
    for row in range(n):
        sample = target[row]
        pts_rel, cols_rel, info = _geometry_for_sample(sample, renderer, args, probe, precomp)
        print(
            f"SMOKE_RENDER_VEC row={row} pts={len(pts_rel)} "
            f"level={info['leveled']}:{info['level_reason']} inliers={info['table_inliers']}",
            flush=True,
        )
        for view in ("top", "front", "side"):
            equal, ref_s, vec_s = _compare_view(row, view, pts_rel, cols_rel, probe)
            all_equal = all_equal and equal
            ref_times.append(ref_s)
            vec_times.append(vec_s)

    ref_ms = float(np.mean(ref_times) * 1e3)
    vec_ms = float(np.mean(vec_times) * 1e3)
    print("=" * 72)
    print(
        f"SMOKE_RENDER_VEC_SUMMARY n_rows={n} views={len(ref_times)} "
        f"array_equal_all={all_equal} ref_mean={ref_ms:.1f}ms/view "
        f"vec_mean={vec_ms:.1f}ms/view speedup={ref_ms / max(vec_ms, 1e-9):.1f}x"
    )
    if all_equal:
        print("SMOKE_RENDER_VEC_OK")
        return 0
    print("SMOKE_RENDER_VEC_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
