#!/usr/bin/env python3
"""R2(a) verification: does re-rendering a cached frame reproduce the cache grid?

Reuses the EXACT precompute code path (precompute._load_cfg + OrthoRenderer) that
the eval server will reuse, and compares its output against the ground-truth
sqlite cache that trained the model. Proves renderer + FFS determinism and that
the reusable renderer is faithful to what the checkpoint actually consumed.

Run on 4090d in the starVLA .venv (needs FFS + get_vla_dataset). READ-ONLY.
"""
import io
import os
import sys
import types
import pickle
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image

STARVLA = "/data/wangqiwei/ICLR2026/starVLA"
os.chdir(STARVLA)
sys.path.insert(0, STARVLA)
sys.path.insert(0, os.path.join(STARVLA, "scripts/4090d"))
sys.path.insert(0, os.path.join(STARVLA, "scripts/tools"))

import precompute_ortho_view_cache as pc  # reuse _load_cfg + OrthoRenderer
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

CACHE_DIR = "playground/Caches/ortho_views_leftprimary_probe_v1"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
N_PER_SUITE = 3

args = types.SimpleNamespace(
    config_yaml="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
    data_root=pc.DEFAULT_DATA_ROOT,
    data_mix=pc.DEFAULT_DATA_MIX,
    video_backend="torchvision_av",
    ffs_repo_dir=os.environ.get("FFS_REPO_DIR", pc.DEFAULT_FFS_REPO_DIR),
    ffs_model_path=os.environ.get("FFS_MODEL_PATH", pc.DEFAULT_FFS_MODEL),
    valid_iters=8,
    max_disp=192,
    axo_size=448,
    level_table="auto",
)
os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir

cfg = pc._load_cfg(args)
dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
by_name = {single.dataset_name: single for single in dataset.datasets}
renderer = pc.OrthoRenderer(args)


def cache_grid(suite_dir, row):
    db = sqlite3.connect(f"file:{suite_dir/'ortho_views.sqlite'}?mode=ro", uri=True)
    cur = db.execute("SELECT grid_png FROM ortho_views WHERE row=?", (int(row),))
    blob = cur.fetchone()[0]
    db.close()
    return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))


print("=" * 100)
print("R2(a) RENDERER<->CACHE REPRODUCIBILITY (re-render stored frame vs sqlite cache grid)")
print("=" * 100)
worst_frac = 0.0
discrepancy = {}
for suite in SUITES:
    # cache subdir is named by the FULL dataset_name (e.g. libero_object_no_noops_1.0.0_lerobot)
    key = next((k for k in by_name if k.startswith(suite)), None)
    if key is None:
        print(f"[{suite}] NO dataset match in {list(by_name)}"); continue
    suite_dir = Path(CACHE_DIR) / key
    single = by_name[key]
    done = np.load(suite_dir / "done.npy", mmap_mode="r")
    done_sum = int(np.asarray(done).sum())
    done_set = set(np.nonzero(np.asarray(done))[0].tolist())
    # rows actually PRESENT in sqlite
    db = sqlite3.connect(f"file:{suite_dir/'ortho_views.sqlite'}?mode=ro", uri=True)
    db_rows = [r[0] for r in db.execute("SELECT row FROM ortho_views ORDER BY row")]
    db.close()
    db_set = set(db_rows)
    missing = sorted(done_set - db_set)  # done=True but NOT in sqlite (would crash training if sampled)
    discrepancy[suite] = (done_sum, len(db_rows), len(missing), missing[:10])
    if len(db_rows) == 0:
        print(f"[{suite}] no sqlite rows"); continue
    # pick rows GUARANTEED present in sqlite for the reproducibility compare
    picks = [db_rows[i] for i in np.linspace(0, len(db_rows) - 1, N_PER_SUITE).astype(int)]
    for row in picks:
        sample = single[int(row)]
        grid, _, info = renderer.render(sample)
        cg = cache_grid(suite_dir, int(row))
        my = np.asarray(grid)
        if my.shape != cg.shape:
            print(f"[{key} row{row}] SHAPE MISMATCH mine={my.shape} cache={cg.shape}"); continue
        diff = np.abs(my.astype(np.int16) - cg.astype(np.int16))
        frac_diff = float((diff.max(axis=-1) > 0).mean())
        maxdiff = int(diff.max())
        exact = bool(np.array_equal(my, cg))
        worst_frac = max(worst_frac, frac_diff)
        tag = "EXACT" if exact else ("NEAR" if (frac_diff < 0.01 and maxdiff <= 8) else "MISMATCH")
        print(f"[{key} row{row}] exact={exact} frac_diff_px={frac_diff*100:.4f}%  maxdiff={maxdiff}  pts={info.get('num_points')}  {tag}")

print("\n" + "-" * 100)
print("DONE.NPY vs SQLITE discrepancy per suite (done_true, sqlite_rows, missing_from_sqlite, first_missing):")
for s, (ds, sr, mc, mrows) in discrepancy.items():
    print(f"  {s}: done={ds} sqlite={sr} MISSING={mc} first_missing={mrows}")

print("\n" + "=" * 100)
print(f"WORST frac differing px = {worst_frac*100:.4f}%")
print("VERDICT:", "RENDERER_REPRODUCES_CACHE (faithful+deterministic)" if worst_frac < 0.01
      else "RENDERER_DRIFT (investigate FFS determinism / input path)")
print("=" * 100)
