#!/usr/bin/env python3
"""Smoke-test the pooled FFS net[0] cache as a live replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TNet0FFSMixin  # noqa: E402
from starVLA.model.modules.stereo.depth_token_inject import DepthTokenProjector  # noqa: E402
from starVLA.model.modules.stereo.ffs_net0_cache import (  # noqa: E402
    FFS_NET0_CACHE_FILENAME,
    FFS_NET0_CACHE_IMAGE_SHA_FILENAME,
    _all_steps_sha256,
    _cache_key_part,
    _image_sha,
    build_ffs_net0_cache_meta,
    read_ffs_net0_pooled_cache_batch,
    run_ffs_net0_cache_startup_check,
)

DEFAULT_CACHE_DIR = ROOT / "_codex_outputs" / "ffs_net0_cache_smoke"
DEFAULT_FFS_REPO_DIR = "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo"
DEFAULT_FFS_MODEL = (
    "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo/"
    "weights/20-30-48/model_best_bp2_serialize.pth"
)
DEFAULT_FFS_SHA256 = "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"


class _FFSNet0CacheSmokeModel(QwenGR00TNet0FFSMixin, nn.Module):
    def __init__(self, cfg: dict, pool_hw: int) -> None:
        super().__init__()
        self._init_frozen_ffs_net0(cfg, "[FFSNet0CacheSmoke]")
        self.pool = DepthTokenProjector.build_pool(pool_hw)

    def live_pooled(self, batch_images):
        raw = self._compute_ffs_feature(batch_images)
        return self.pool(raw).to(dtype=torch.float32)


class _SingleRowDataset:
    dataset_name = "smoke_suite"
    video_backend = "synthetic"
    delete_pause_frame = False

    def __init__(self, images):
        self.images = images
        self.all_steps = [(0, 0)]

    def __len__(self):
        return 1

    def __getitem__(self, row):
        if int(row) != 0:
            raise IndexError(row)
        return {
            "image": self.images,
            "traj_id": 0,
            "base_index": 0,
        }


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_ffs_cfg(args: argparse.Namespace, cache_dir: Path) -> dict:
    return {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": int(args.gru_hidden_dim),
        "ffs_image_size": int(args.image_size),
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "pool_hw": int(args.pool_hw),
        "num_depth_tokens": int(args.num_depth_tokens),
        "ffs_cache_dir": str(cache_dir),
    }


def _make_synthetic_images(size: int):
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    right = np.stack([(x % 256), (y % 256), ((x + y) % 256)], axis=-1).astype(np.uint8)
    left = np.roll(right, shift=3, axis=1).copy()
    return [right, left]


def _write_tiny_cache(
    *,
    cache_dir: Path,
    pooled: torch.Tensor,
    images,
    ffs_cfg: dict,
    ffs_model_sha256: str,
    tamper: bool = False,
) -> _SingleRowDataset:
    suite_dir = cache_dir / "smoke_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    sample_dataset = _SingleRowDataset(images)

    row = 0
    traj_id = 0
    base_index = 0
    net0 = np.lib.format.open_memmap(
        suite_dir / FFS_NET0_CACHE_FILENAME,
        mode="w+",
        dtype=np.float32,
        shape=(1, *tuple(int(x) for x in pooled.shape[1:])),
    )
    row_value = pooled[0].detach().cpu().to(torch.float32).numpy()
    if tamper:
        row_value = row_value.copy()
        flat = row_value.reshape(-1)
        flat[0] += max(1.0, abs(float(flat[0])) + 1.0)
    net0[row] = row_value
    net0.flush()

    done = np.lib.format.open_memmap(suite_dir / "done.npy", mode="w+", dtype=np.bool_, shape=(1,))
    done[:] = True
    done.flush()

    image_sha = np.lib.format.open_memmap(
        suite_dir / FFS_NET0_CACHE_IMAGE_SHA_FILENAME,
        mode="w+",
        dtype="S64",
        shape=(1, 2),
    )
    shas = [_image_sha(img) for img in images]
    image_sha[row] = np.asarray([sha.encode("ascii") for sha in shas], dtype="S64")
    image_sha.flush()

    index = {(_cache_key_part(traj_id), _cache_key_part(base_index)): row}
    with open(suite_dir / "index.pkl", "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    all_steps_sha = _all_steps_sha256([(traj_id, base_index)])
    meta = build_ffs_net0_cache_meta(
        suite="smoke_suite",
        count=1,
        all_steps_sha256=all_steps_sha,
        image_sha256_samples=[
            {
                "row": row,
                "traj_id": _cache_key_part(traj_id),
                "base_index": _cache_key_part(base_index),
                "image_sha256": shas,
            }
        ],
        raw_observed_shape=(int(ffs_cfg["gru_hidden_dim"]), 64, 64),
        pooled_shape=tuple(int(x) for x in pooled.shape[1:]),
        pool_hw=int(ffs_cfg["pool_hw"]),
        num_depth_tokens=int(ffs_cfg["num_depth_tokens"]),
        ffs_cfg=ffs_cfg,
        ffs_model_path=str(ffs_cfg["ffs_model_path"]),
        ffs_model_sha256=ffs_model_sha256,
        data_root_dir="synthetic",
        data_mix="synthetic",
        video_backend="synthetic",
        delete_pause_frame=False,
    )
    with open(suite_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return sample_dataset


def _assert_default_off_split_parity(raw_net0: torch.Tensor, pool_hw: int, num_tokens: int) -> None:
    projector = DepthTokenProjector(
        in_ch=int(raw_net0.shape[1]),
        llm_dim=32,
        num_tokens=int(num_tokens),
        pool_hw=int(pool_hw),
    ).to(raw_net0.device)
    old_path = projector(raw_net0)
    split_path = projector.project_pooled(projector.pool(raw_net0).to(dtype=torch.float32))
    if not torch.equal(old_path, split_path):
        raise RuntimeError("default-off projector split parity failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-model-path", default=os.environ.get("FFS_MODEL_PATH", DEFAULT_FFS_MODEL))
    parser.add_argument("--ffs-expected-sha256", default=os.environ.get("FFS_SHA256", DEFAULT_FFS_SHA256))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gru-hidden-dim", type=int, default=16)
    parser.add_argument("--pool-hw", type=int, default=8)
    parser.add_argument("--num-depth-tokens", type=int, default=64)
    args = parser.parse_args()

    if int(args.pool_hw) * int(args.pool_hw) != int(args.num_depth_tokens):
        raise ValueError(
            f"pool_hw*pool_hw must equal num_depth_tokens; got {args.pool_hw}x{args.pool_hw} vs {args.num_depth_tokens}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("smoke_ffs_net0_cache.py requires CUDA")

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if args.ffs_repo_dir and args.ffs_repo_dir not in sys.path:
        sys.path.insert(0, args.ffs_repo_dir)

    cache_dir = Path(args.cache_dir)
    if cache_dir.exists() and args.overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ffs_cfg = _build_ffs_cfg(args, cache_dir)
    model = _FFSNet0CacheSmokeModel(ffs_cfg, pool_hw=int(args.pool_hw)).cuda().eval()
    images = _make_synthetic_images(int(args.image_size))
    ffs_sha = _sha256_file(args.ffs_model_path)

    with torch.inference_mode():
        raw = model._compute_ffs_feature([images])
        pooled = model.pool(raw).to(dtype=torch.float32)
    _assert_default_off_split_parity(raw, int(args.pool_hw), int(args.num_depth_tokens))
    dataset = _write_tiny_cache(
        cache_dir=cache_dir,
        pooled=pooled,
        images=images,
        ffs_cfg=ffs_cfg,
        ffs_model_sha256=ffs_sha,
    )

    def no_live(_images):
        raise RuntimeError("cache-HIT smoke unexpectedly used live fallback")

    cached, stats = read_ffs_net0_pooled_cache_batch(
        cache_dir=str(cache_dir),
        batch_images=[images],
        sample_ids=[("smoke_suite", 0, 0)],
        handles={},
        ffs_sha256=ffs_sha,
        ffs_cfg=ffs_cfg,
        pool_hw=int(args.pool_hw),
        num_depth_tokens=int(args.num_depth_tokens),
        device=pooled.device,
        live_fallback_fn=no_live,
        dataset_or_none={"smoke_suite": dataset},
        expected_data_root="synthetic",
        expected_data_mix="synthetic",
        expected_video_backend="synthetic",
        expected_delete_pause_frame=False,
    )
    if stats.hits != 1 or stats.row_misses or stats.whole_batch_live:
        raise RuntimeError(f"cache-HIT stats mismatch: {stats}")
    if not torch.equal(cached, pooled):
        raise RuntimeError("cache-HIT tensor is not byte-exact to live pooled net0")

    run_ffs_net0_cache_startup_check(
        cache_dir=str(cache_dir),
        dataset_or_none={"smoke_suite": dataset},
        handles={},
        ffs_sha256=ffs_sha,
        ffs_cfg=ffs_cfg,
        pool_hw=int(args.pool_hw),
        num_depth_tokens=int(args.num_depth_tokens),
        device=pooled.device,
        compute_pooled_fn=model.live_pooled,
        expected_data_root="synthetic",
        expected_data_mix="synthetic",
        expected_video_backend="synthetic",
        expected_delete_pause_frame=False,
    )

    tampered_dir = cache_dir.parent / f"{cache_dir.name}_tampered"
    if tampered_dir.exists():
        shutil.rmtree(tampered_dir)
    tampered_dataset = _write_tiny_cache(
        cache_dir=tampered_dir,
        pooled=pooled,
        images=images,
        ffs_cfg={**ffs_cfg, "ffs_cache_dir": str(tampered_dir)},
        ffs_model_sha256=ffs_sha,
        tamper=True,
    )
    try:
        run_ffs_net0_cache_startup_check(
            cache_dir=str(tampered_dir),
            dataset_or_none={"smoke_suite": tampered_dataset},
            handles={},
            ffs_sha256=ffs_sha,
            ffs_cfg={**ffs_cfg, "ffs_cache_dir": str(tampered_dir)},
            pool_hw=int(args.pool_hw),
            num_depth_tokens=int(args.num_depth_tokens),
            device=pooled.device,
            compute_pooled_fn=model.live_pooled,
            expected_data_root="synthetic",
            expected_data_mix="synthetic",
            expected_video_backend="synthetic",
            expected_delete_pause_frame=False,
        )
    except RuntimeError as exc:
        if "FFS cache invalid" not in str(exc):
            raise
    else:
        raise RuntimeError("tampered startup equality smoke did not raise")

    print("SMOKE_FFS_NET0_CACHE_OK cache_hit=1 default_off_split_parity=1 startup_check=pass tamper_raises=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
