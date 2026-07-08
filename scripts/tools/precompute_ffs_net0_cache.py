#!/usr/bin/env python3
"""Precompute pooled frozen FFS net[0] cache for depth-token training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TNet0FFSMixin  # noqa: E402
from starVLA.model.modules.stereo.depth_token_inject import DepthTokenProjector  # noqa: E402
from starVLA.model.modules.stereo.ffs_net0_cache import (  # noqa: E402
    FFS_NET0_CACHE_FILENAME,
    FFS_NET0_CACHE_IMAGE_SHA_FILENAME,
    _all_steps_from_index,
    _all_steps_sha256,
    _cache_key_part,
    _image_sha,
    build_ffs_net0_cache_meta,
    validate_ffs_net0_cache,
)

DEFAULT_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "Fast-FoundationStereo")
DEFAULT_FFS_MODEL = os.path.join(DEFAULT_FFS_REPO_DIR, "weights/20-30-48/model_best_bp2_serialize.pth")
DEFAULT_FFS_SHA256 = "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
DEFAULT_DATA_ROOT = "playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW"
DEFAULT_DATA_MIX = "libero_all_sfstereo_leftprimary"
DEFAULT_GRU_HIDDEN_DIM = 16
DEFAULT_POOL_HW = 8
DEFAULT_NUM_DEPTH_TOKENS = 64


class _FFSNet0PooledBench(QwenGR00TNet0FFSMixin, nn.Module):
    def __init__(self, cfg: dict, pool_hw: int) -> None:
        super().__init__()
        self._init_frozen_ffs_net0(cfg, "[FFSNet0CachePrecompute]")
        self.pool = DepthTokenProjector.build_pool(pool_hw)

    def compute_raw_and_pooled(self, batch_images):
        raw = self._compute_ffs_feature(batch_images)
        pooled = self.pool(raw).to(dtype=torch.float32)
        return raw, pooled


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_suites(value: str, available: list[str]) -> list[str]:
    if value.strip().lower() == "all":
        return list(available)
    wanted = [x.strip() for x in value.split(",") if x.strip()]
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"requested suites not in data_mix: {missing}; available={available}")
    return wanted


def _assert_libero_leftprimary_only(data_mix: str, mixture_spec: list[tuple[str, float, str]], suites: list[str]) -> None:
    if str(data_mix) != DEFAULT_DATA_MIX:
        raise ValueError(f"FFS net0 cache is scoped to {DEFAULT_DATA_MIX}; got data_mix={data_mix!r}")
    selected = {suite for suite in suites}
    bad = [
        (name, robot_type)
        for name, _weight, robot_type in mixture_spec
        if name in selected and (
            not str(name).startswith("libero_") or robot_type != "libero_franka_sfstereo_leftprimary"
        )
    ]
    if bad:
        raise ValueError(f"FFS net0 cache only supports LIBERO leftprimary stereo suites; bad={bad}")


def _human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def _df_guard(cache_dir: Path, required_bytes: int, multiplier: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    need = int(required_bytes * float(multiplier))
    print(
        "DF_GUARD "
        f"path={cache_dir} free={_human_bytes(usage.free)} required={_human_bytes(required_bytes)} "
        f"required_with_margin={_human_bytes(need)}"
    )
    if usage.free < need:
        raise RuntimeError(f"not enough free space under {cache_dir}: free={usage.free} need={need}")


def _load_cfg(args: argparse.Namespace):
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.load(args.config_yaml)
    cli = {
        "datasets": {
            "vla_data": {
                "data_root_dir": args.data_root,
                "data_mix": args.data_mix,
            }
        }
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(cli))
    cfg = apply_config_compat(cfg)
    scene_flow_cfg = cfg.datasets.vla_data.get("scene_flow", None)
    if scene_flow_cfg is not None:
        was_enabled = bool(scene_flow_cfg.get("enabled", False))
        was_gt_only = bool(scene_flow_cfg.get("gt_only_sampler", False))
        if was_enabled or was_gt_only:
            print(
                "CACHE_PRECOMPUTE_FULL_COVERAGE "
                "forcing datasets.vla_data.scene_flow.enabled=false and gt_only_sampler=false"
            )
        OmegaConf.update(cfg, "datasets.vla_data.scene_flow.enabled", False, merge=True)
        OmegaConf.update(cfg, "datasets.vla_data.scene_flow.gt_only_sampler", False, merge=True)
    return cfg


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


def _open_net0(path: Path, shape: tuple[int, ...], resume: bool):
    if path.is_file() and resume:
        net0 = np.load(path, mmap_mode="r+")
        if tuple(net0.shape) != shape or net0.dtype != np.float32:
            raise RuntimeError(f"existing net0 has shape={net0.shape} dtype={net0.dtype}, expected {shape} fp32")
        return net0
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _open_done(path: Path, count: int, resume: bool):
    if path.is_file() and resume:
        done = np.load(path, mmap_mode="r+")
        if tuple(done.shape) != (count,):
            raise RuntimeError(f"existing done.npy has shape={done.shape}, expected {(count,)}")
        if done.dtype != np.dtype(np.bool_):
            raise RuntimeError(f"existing done.npy dtype={done.dtype}, expected bool")
        return done
    done = np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(count,))
    done[:] = False
    done.flush()
    return done


def _open_image_sha(path: Path, count: int, num_images: int, resume: bool):
    shape = (int(count), int(num_images))
    if path.is_file() and resume:
        image_sha = np.load(path, mmap_mode="r+")
        if tuple(image_sha.shape) != shape or image_sha.dtype != np.dtype("S64"):
            raise RuntimeError(
                f"existing image SHA sidecar has shape={image_sha.shape} dtype={image_sha.dtype}, "
                f"expected {shape} S64"
            )
        return image_sha
    image_sha = np.lib.format.open_memmap(path, mode="w+", dtype="S64", shape=shape)
    image_sha[:] = b""
    image_sha.flush()
    return image_sha


def _store_image_shas(image_sha, row: int, shas: list[str]) -> None:
    image_sha[int(row)] = np.asarray([str(sha).encode("ascii") for sha in shas], dtype="S64")


def _sample_image_shas(dataset, rows: list[int]) -> list[dict]:
    out = []
    for row in rows:
        sample = dataset[row]
        shas = [_image_sha(img) for img in sample["image"]]
        out.append(
            {
                "row": int(row),
                "traj_id": _cache_key_part(sample["traj_id"]),
                "base_index": _cache_key_part(sample["base_index"]),
                "image_sha256": shas,
            }
        )
    return out


def _normalized_all_steps(dataset) -> list[tuple[object, object]]:
    return [
        (_cache_key_part(traj_id), _cache_key_part(base_index))
        for traj_id, base_index in dataset.all_steps
    ]


def _assert_resume_all_steps_match_index(index_path: Path, expected_all_steps: list[tuple[object, object]], suite: str) -> None:
    with open(index_path, "rb") as fh:
        existing_index = pickle.load(fh)
    try:
        existing_all_steps = _all_steps_from_index(existing_index, len(expected_all_steps))
    except RuntimeError as exc:
        raise RuntimeError(
            f"refusing to resume FFS net0 cache for suite={suite}: existing index.pkl row order "
            f"does not match current dataset all_steps ({exc})"
        ) from exc
    if existing_all_steps == expected_all_steps:
        return

    first_mismatch = None
    for row, (existing, expected) in enumerate(zip(existing_all_steps, expected_all_steps)):
        if existing != expected:
            first_mismatch = (row, existing, expected)
            break
    if first_mismatch is None:
        first_mismatch = (min(len(existing_all_steps), len(expected_all_steps)), None, None)
    row, existing, expected = first_mismatch
    raise RuntimeError(
        f"refusing to resume FFS net0 cache for suite={suite}: dataset all_steps row order changed "
        f"before rewriting index.pkl (row={row}, existing={existing!r}, current={expected!r}, "
        f"existing_len={len(existing_all_steps)}, current_len={len(expected_all_steps)})"
    )


def _assert_full_coverage_precompute_dataset(dataset) -> None:
    if bool(getattr(dataset, "scene_flow_enabled", False)) or bool(getattr(dataset, "scene_flow_gt_only_sampler", False)):
        raise RuntimeError(
            "FFS net0 cache precompute must run without scene-flow filtering. "
            "Scene-flow training may sample a gt-only subset, so this cache must be "
            "built over full trajectory coverage as a superset."
        )
    steps_path = Path(getattr(dataset, "dataset_path")) / "meta" / "steps_data_index.pkl"
    expected_key_fn = getattr(dataset, "_get_steps_config_key", None)
    if steps_path.is_file() and callable(expected_key_fn):
        with open(steps_path, "rb") as fh:
            cached_steps = pickle.load(fh)
        cached_key = cached_steps.get("config_key") if isinstance(cached_steps, dict) else None
        expected_key = expected_key_fn()
        if cached_key is not None and str(cached_key) != str(expected_key):
            raise RuntimeError(
                f"FFS net0 cache precompute found a stale steps cache for {dataset.dataset_name}: "
                f"{steps_path} config_key={cached_key} expected={expected_key}. "
                "Delete/rebuild it without scene-flow filtering before precomputing."
            )


def _write_meta(
    path: Path,
    *,
    dataset,
    cfg,
    args: argparse.Namespace,
    ffs_cfg: dict,
    ffs_sha256: str,
    count: int,
    all_steps_sha256: str,
    image_shas: list[dict],
    raw_observed_shape: tuple[int, int, int],
    pooled_shape: tuple[int, int, int],
) -> None:
    meta = build_ffs_net0_cache_meta(
        suite=dataset.dataset_name,
        count=count,
        all_steps_sha256=all_steps_sha256,
        image_sha256_samples=image_shas,
        raw_observed_shape=raw_observed_shape,
        pooled_shape=pooled_shape,
        pool_hw=int(args.pool_hw),
        num_depth_tokens=int(args.num_depth_tokens),
        ffs_cfg=ffs_cfg,
        ffs_model_path=args.ffs_model_path,
        ffs_model_sha256=ffs_sha256,
        data_root_dir=str(cfg.datasets.vla_data.data_root_dir),
        data_mix=str(cfg.datasets.vla_data.data_mix),
        video_backend=str(dataset.video_backend),
        delete_pause_frame=bool(dataset.delete_pause_frame),
    )
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def _assert_resume_cache_consistent(
    *,
    cache_dir: Path,
    suite: str,
    dataset,
    ffs_sha256: str,
    ffs_cfg: dict,
    cfg,
) -> None:
    validate_ffs_net0_cache(
        cache_dir,
        [suite],
        dataset_or_none={suite: dataset},
        ffs_sha256=ffs_sha256,
        ffs_cfg=ffs_cfg,
        pool_hw=int(ffs_cfg["pool_hw"]),
        num_depth_tokens=int(ffs_cfg["num_depth_tokens"]),
        require_all_done=False,
        expected_data_root=str(cfg.datasets.vla_data.data_root_dir),
        expected_data_mix=str(cfg.datasets.vla_data.data_mix),
        expected_video_backend=getattr(dataset, "video_backend", None),
        expected_delete_pause_frame=getattr(dataset, "delete_pause_frame", None),
    )


def precompute_suite(model, ffs_cfg: dict, dataset, suite_dir: Path, args: argparse.Namespace, cfg) -> None:
    _assert_full_coverage_precompute_dataset(dataset)
    count = len(dataset.all_steps)
    pooled_shape = (int(args.gru_hidden_dim), int(args.pool_hw), int(args.pool_hw))
    net0_shape = (count, *pooled_shape)
    suite_dir.mkdir(parents=True, exist_ok=True)
    _df_guard(suite_dir, count * np.dtype(np.float32).itemsize * int(np.prod(pooled_shape)), args.free_space_multiplier)

    net0_path = suite_dir / FFS_NET0_CACHE_FILENAME
    done_path = suite_dir / "done.npy"
    index_path = suite_dir / "index.pkl"
    meta_path = suite_dir / "meta.json"
    image_sha_path = suite_dir / FFS_NET0_CACHE_IMAGE_SHA_FILENAME

    all_steps = _normalized_all_steps(dataset)
    all_steps_sha = _all_steps_sha256(all_steps)
    ffs_sha256 = _sha256_file(args.ffs_model_path)
    structural_paths = (net0_path, done_path, index_path, meta_path, image_sha_path)
    if args.resume and any(path.exists() for path in structural_paths):
        if not all(path.exists() for path in structural_paths):
            missing = [str(path) for path in structural_paths if not path.exists()]
            raise RuntimeError(f"partial FFS net0 cache structure for suite={dataset.dataset_name}; missing={missing}")
        _assert_resume_all_steps_match_index(index_path, all_steps, dataset.dataset_name)
        _assert_resume_cache_consistent(
            cache_dir=suite_dir.parent,
            suite=dataset.dataset_name,
            dataset=dataset,
            ffs_sha256=ffs_sha256,
            ffs_cfg=ffs_cfg,
            cfg=cfg,
        )

    if not args.resume:
        for stale_path in (meta_path, image_sha_path):
            if stale_path.exists():
                stale_path.unlink()

    net0 = _open_net0(net0_path, net0_shape, resume=args.resume)
    done = _open_done(done_path, count, resume=args.resume)
    image_sha = _open_image_sha(
        image_sha_path,
        count=count,
        num_images=int(ffs_cfg["num_cameras"]),
        resume=args.resume,
    )
    index = {(traj_id, base_index): row for row, (traj_id, base_index) in enumerate(all_steps)}
    with open(index_path, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    sha_rows = sorted(set([0, max(0, count // 2), max(0, count - 1)])) if count else []
    image_shas = _sample_image_shas(dataset, sha_rows)
    raw_observed_shape = None
    if args.resume and meta_path.is_file():
        with open(meta_path, "r") as fh:
            raw_observed_shape = tuple(int(x) for x in json.load(fh)["raw_observed_shape"])

    start = time.time()
    limit = int(args.limit_rows) if int(args.limit_rows) > 0 else count
    limit = min(limit, count)
    written = int(done[:limit].sum())
    print(f"CACHE_SUITE_START suite={dataset.dataset_name} count={count} already_done={written} limit={limit}")
    for row in range(limit):
        if bool(done[row]) and args.resume:
            continue
        sample = dataset[row]
        images = sample["image"]
        if len(images) != 2:
            raise RuntimeError(f"{dataset.dataset_name} row={row} expected 2 primary,left_view images, got {len(images)}")
        row_image_shas = [_image_sha(img) for img in images]
        with torch.inference_mode():
            raw, pooled = model.compute_raw_and_pooled([images])
        observed_raw_shape = tuple(int(x) for x in raw.shape[1:])
        if observed_raw_shape[0] != int(args.gru_hidden_dim):
            raise RuntimeError(
                f"{dataset.dataset_name} row={row} raw net0 channels {observed_raw_shape[0]} "
                f"!= configured {args.gru_hidden_dim}"
            )
        if raw_observed_shape is None:
            raw_observed_shape = observed_raw_shape
            _write_meta(
                meta_path,
                dataset=dataset,
                cfg=cfg,
                args=args,
                ffs_cfg=ffs_cfg,
                ffs_sha256=ffs_sha256,
                count=count,
                all_steps_sha256=all_steps_sha,
                image_shas=image_shas,
                raw_observed_shape=raw_observed_shape,
                pooled_shape=pooled_shape,
            )
            print(
                "CACHE_RAW_SHAPE "
                f"suite={dataset.dataset_name} raw_observed_shape={raw_observed_shape} pooled_shape={pooled_shape}"
            )
        elif observed_raw_shape != raw_observed_shape:
            raise RuntimeError(
                f"{dataset.dataset_name} row={row} raw net0 shape {observed_raw_shape} "
                f"!= cached raw_observed_shape {raw_observed_shape}"
            )
        if tuple(int(x) for x in pooled.shape) != (1, *pooled_shape):
            raise RuntimeError(f"unexpected pooled shape for row={row}: {tuple(pooled.shape)} expected={(1, *pooled_shape)}")
        net0[row] = pooled[0].detach().cpu().to(torch.float32).numpy()
        _store_image_shas(image_sha, row, row_image_shas)
        done[row] = True
        written += 1
        if written % int(args.log_every) == 0 or row == limit - 1:
            elapsed = max(time.time() - start, 1e-6)
            rate = written / elapsed
            remaining = max(limit - written, 0)
            eta = remaining / rate if rate > 0 else 0.0
            net0.flush()
            image_sha.flush()
            done.flush()
            print(
                "CACHE_PROGRESS "
                f"suite={dataset.dataset_name} done={written}/{limit} "
                f"rate={rate:.3f}_rows_per_sec eta_sec={eta:.1f}"
            )
    net0.flush()
    image_sha.flush()
    done.flush()
    if raw_observed_shape is None and count:
        raise RuntimeError(
            f"{dataset.dataset_name} has no computed rows and no meta raw_observed_shape; "
            "rerun with --no-resume or a higher --limit-rows"
        )
    print(f"CACHE_SUITE_DONE suite={dataset.dataset_name} done={int(done[:limit].sum())}/{limit}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--suites", default="all")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-model-path", default=os.environ.get("FFS_MODEL_PATH", DEFAULT_FFS_MODEL))
    parser.add_argument("--ffs-expected-sha256", default=os.environ.get("FFS_SHA256", DEFAULT_FFS_SHA256))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gru-hidden-dim", type=int, default=DEFAULT_GRU_HIDDEN_DIM)
    parser.add_argument("--pool-hw", type=int, default=DEFAULT_POOL_HW)
    parser.add_argument("--num-depth-tokens", type=int, default=DEFAULT_NUM_DEPTH_TOKENS)
    parser.add_argument("--free-space-multiplier", type=float, default=1.2)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if int(args.pool_hw) * int(args.pool_hw) != int(args.num_depth_tokens):
        raise ValueError(
            f"pool_hw*pool_hw must equal num_depth_tokens; got {args.pool_hw}x{args.pool_hw} vs {args.num_depth_tokens}"
        )

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if args.ffs_repo_dir and args.ffs_repo_dir not in sys.path:
        sys.path.insert(0, args.ffs_repo_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("precompute_ffs_net0_cache.py requires CUDA")

    cfg = _load_cfg(args)
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    mixture_spec = DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]
    available = [name for name, _weight, _robot_type in mixture_spec]
    suites = _parse_suites(args.suites, available)
    _assert_libero_leftprimary_only(str(cfg.datasets.vla_data.data_mix), mixture_spec, suites)
    # Cache coverage invariant: precompute intentionally uses the plain dataset
    # without scene-flow filtering. C/scene-flow runs may enable gt_only_sampler=True
    # and train on a subset, so this cache must remain a full-trajectory superset.
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
    by_name = {single.dataset_name: single for single in dataset.datasets}
    ffs_cfg = _build_ffs_cfg(args, Path(args.cache_dir))
    model = _FFSNet0PooledBench(ffs_cfg, pool_hw=int(args.pool_hw)).cuda().eval()

    for suite in suites:
        precompute_suite(
            model,
            ffs_cfg,
            by_name[suite],
            Path(args.cache_dir) / suite,
            args,
            cfg,
        )
    print("CACHE_PRECOMPUTE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
