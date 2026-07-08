#!/usr/bin/env python3
"""Profile precompute_ortho_view_cache per-stage timing breakdown.

Reuses OrthoRenderer + probe_orthogonal_multiview_render stages, wraps each
in time.perf_counter, runs N rows of libero_object, prints per-stage mean ms.

Per-row stages timed:
  1. dataset_getitem   - dataset[row] (av1 decode + _pack_sample), NOT counted in total
  2. ffs_disparity     - renderer.run_ffs_disparity
  3. geometry          - disparity_to_depth + unproject + camera_to_world
                         + maybe_level_table + workspace_crop
  4. render_world_view - probe.render_world_view x3 (top/front/side)
  5. legend_compose    - make_legend + compose_grid
  6. sqlite_insert     - _png_bytes(grid/axo) + INSERT OR REPLACE
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data/wangqiwei/ICLR2026/starVLA")
PROBE_DIR = ROOT / "scripts" / "tools"
PRECOMP_DIR = ROOT / "scripts" / "4090d"
for p in (str(ROOT), str(PROBE_DIR), str(PRECOMP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import probe_orthogonal_multiview_render as probe  # noqa: E402
import precompute_ortho_view_cache as precomp  # noqa: E402


STAGES = [
    "dataset_getitem",
    "ffs_disparity",
    "geometry",
    "render_world_view",
    "legend_compose",
    "sqlite_insert",
]


def _run_geometry(renderer, sample, args):
    """render() stages 2-5 inlined so we can time each sub-piece."""
    images = sample["image"]
    primary = precomp._to_np_256(images[0])
    left_view = precomp._to_np_256(images[1])
    suite_name = sample["dataset_name"]
    right_view_pos_w, right_view_R_mjc_to_w = probe.load_right_view_pose(
        precomp._probe_suite_name(suite_name)
    )

    t0 = time.perf_counter()
    disp = renderer.run_ffs_disparity(left_view, primary)
    torch.cuda.synchronize()
    t_ffs = time.perf_counter() - t0

    t0 = time.perf_counter()
    depth, valid = probe.disparity_to_depth(disp, probe.FX, probe.BASELINE_M)
    pts, cols = probe.unproject(depth, left_view, valid, probe.VALID_LO, probe.VALID_HI)
    pts_w = probe.camera_to_world(pts, right_view_pos_w, right_view_R_mjc_to_w)
    pts_w, table_deg, table_inliers, leveled, level_reason = probe.maybe_level_table(
        pts_w, cols, args.level_table
    )
    origin_w = pts_w.mean(0)
    pts_rel, cols_rel = probe.workspace_crop(pts_w, cols, origin_w)
    t_geo = time.perf_counter() - t0

    t0 = time.perf_counter()
    top = probe.render_world_view(pts_rel, cols_rel, "top")
    front = probe.render_world_view(pts_rel, cols_rel, "front")
    side = probe.render_world_view(pts_rel, cols_rel, "side")
    t_render = time.perf_counter() - t0

    t0 = time.perf_counter()
    legend = probe.make_legend()
    grid = probe.compose_grid(top, front, side, legend)
    t_legend = time.perf_counter() - t0

    axo = np.zeros((int(args.axo_size), int(args.axo_size), 3), dtype=np.uint8)
    info = {
        "num_points": int(len(pts_rel)),
        "table_deg": None if np.isnan(table_deg) else float(table_deg),
        "table_inliers": int(table_inliers),
        "leveled": bool(leveled),
        "level_reason": str(level_reason),
    }
    return {
        "ffs_disparity": t_ffs,
        "geometry": t_geo,
        "render_world_view": t_render,
        "legend_compose": t_legend,
        "_grid": grid,
        "_axo": axo,
        "_info": info,
    }


def main() -> int:
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
    parser.add_argument("--axo-size", type=int, default=448)
    parser.add_argument("--level-table", choices=["off", "auto", "always"], default="auto")
    args = parser.parse_args()
    args.cache_dir = "/tmp/profile_ortho_dummy"

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if not torch.cuda.is_available():
        raise RuntimeError("profile requires CUDA")

    print("PROFILE: loading cfg + dataset (slow lerobot load, NOT timed)", flush=True)
    t0 = time.perf_counter()
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
    target = by_name[args.suite]
    print(f"PROFILE: dataset loaded in {time.perf_counter()-t0:.1f}s; suite={target.dataset_name} rows={len(target.all_steps)}", flush=True)

    renderer = precomp.OrthoRenderer(args)
    print("PROFILE: warming up FFS model (load + 1 forward so row-0 isn't polluted)...", flush=True)
    t0 = time.perf_counter()
    renderer._ensure_ffs()
    _z = np.zeros((probe.H, probe.W, 3), dtype=np.uint8)
    _ = renderer.run_ffs_disparity(_z, _z)
    torch.cuda.synchronize()
    print(f"PROFILE: FFS warmup done in {time.perf_counter()-t0:.1f}s", flush=True)

    tmp_db = Path("/tmp/profile_ortho_tmp.sqlite")
    tmp_db.unlink(missing_ok=True)
    conn = precomp._ensure_db(tmp_db)

    n = min(args.n_rows, len(target.all_steps))
    timings = {s: [] for s in STAGES}

    print(f"PROFILE: timing {n} rows...", flush=True)
    for row in range(n):
        t0 = time.perf_counter()
        sample = target[row]
        t_get = time.perf_counter() - t0
        timings["dataset_getitem"].append(t_get)

        out = _run_geometry(renderer, sample, args)
        grid, axo, info = out["_grid"], out["_axo"], out["_info"]

        t0 = time.perf_counter()
        conn.execute(
            "INSERT OR REPLACE INTO ortho_views(row,traj_id,base_index,grid_png,axo_png,render_info) "
            "VALUES (?,?,?,?,?,?)",
            (
                int(row),
                json.dumps(precomp._cache_key_part(sample["traj_id"]), sort_keys=True),
                json.dumps(precomp._cache_key_part(sample["base_index"]), sort_keys=True),
                sqlite3.Binary(precomp._png_bytes(grid)),
                sqlite3.Binary(precomp._png_bytes(axo)),
                json.dumps(info, sort_keys=True),
            ),
        )
        conn.commit()
        t_sql = time.perf_counter() - t0
        timings["sqlite_insert"].append(t_sql)

        for s in ("ffs_disparity", "geometry", "render_world_view", "legend_compose"):
            timings[s].append(out[s])

        row_total = sum(timings[s][-1] for s in STAGES if s != "dataset_getitem")
        print(
            f"  row {row}: get={t_get*1e3:.0f}ms ffs={out['ffs_disparity']*1e3:.0f}ms "
            f"geo={out['geometry']*1e3:.0f}ms render={out['render_world_view']*1e3:.0f}ms "
            f"legend={out['legend_compose']*1e3:.0f}ms sql={t_sql*1e3:.0f}ms "
            f"-> total(no get)={row_total*1e3:.0f}ms pts={info['num_points']}",
            flush=True,
        )

    conn.close()
    tmp_db.unlink(missing_ok=True)

    print("\n" + "=" * 72)
    print(f"PROFILE SUMMARY: n_rows={n} suite={args.suite}")
    print("=" * 72)
    means = {}
    for s in STAGES:
        arr = np.array(timings[s]) * 1e3
        means[s] = float(arr.mean())
        print(f"  {s:20s}: mean={arr.mean():7.1f}ms  min={arr.min():7.1f}ms  max={arr.max():7.1f}ms")
    total_no_get = sum(means[s] for s in STAGES if s != "dataset_getitem")
    print("-" * 72)
    print(f"  {'TOTAL (no get)':20s}: mean={total_no_get:7.1f}ms/row")
    print(f"  throughput        : {1000.0/max(total_no_get,1e-3):.3f} rows/sec (excl dataset load)")
    print("-" * 72)
    print("  bottleneck (excl dataset_getitem):")
    ranked = sorted(((s, means[s]) for s in STAGES if s != "dataset_getitem"), key=lambda x: -x[1])
    for s, ms in ranked:
        pct = 100.0 * ms / max(total_no_get, 1e-9)
        print(f"    {s:20s}: {ms:7.1f}ms  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
