#!/usr/bin/env python3
"""Precompute Method #10 Version-A Utonia per-patch grids for LIBERO training."""
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
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.utonia_pointcloud import (  # noqa: E402
    _all_steps_sha256,
    _cache_key_part,
    _image_sha,
    _utonia_geometry_meta,
    validate_utonia_cache,
)

DEFAULT_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "Fast-FoundationStereo")
DEFAULT_FFS_MODEL = os.path.join(DEFAULT_FFS_REPO_DIR, "weights/20-30-48/model_best_bp2_serialize.pth")
DEFAULT_UTONIA_CKPT = "./playground/Pretrained_models/Utonia/utonia.pth"
DEFAULT_DATA_ROOT = "playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW"
DEFAULT_DATA_MIX = "libero_all_sfstereo_leftprimary"
GRID_SHAPE = (1387, 8, 8)


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
        raise ValueError(
            f"Utonia Version-A cache is scoped to {DEFAULT_DATA_MIX}; got data_mix={data_mix!r}"
        )
    selected = {suite for suite in suites}
    bad = [
        (name, robot_type)
        for name, _weight, robot_type in mixture_spec
        if name in selected and (
            not str(name).startswith("libero_") or robot_type != "libero_franka_sfstereo_leftprimary"
        )
    ]
    if bad:
        raise ValueError(f"Utonia Version-A cache only supports LIBERO leftprimary stereo suites; bad={bad}")


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
        raise RuntimeError(
            f"not enough free space under {cache_dir}: free={usage.free} need={need}"
        )


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
    return cfg


def _build_pc_cfg(args: argparse.Namespace, cache_dir: Path) -> dict:
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import (
        DEFAULT_UTONIA_POINTCLOUD_CFG,
    )

    pc_cfg = dict(DEFAULT_UTONIA_POINTCLOUD_CFG)
    pc_cfg.update(
        {
            "ffs_model_path": args.ffs_model_path,
            "ffs_expected_sha256": args.ffs_expected_sha256,
            "utonia_ckpt_path": args.utonia_ckpt_path,
            "utonia_expected_sha256": args.utonia_expected_sha256,
            "utonia_cache_dir": str(cache_dir),
            "utonia_batched": False,
            "utonia_scale": float(args.utonia_scale),
            "utonia_enable_flash": bool(args.utonia_enable_flash),
            "ffs_image_size": int(args.image_size),
            "image_width": int(args.image_size),
            "image_height": int(args.image_size),
            "backproject_stride": int(args.backproject_stride),
            "fovy_degrees": float(args.fovy_degrees),
            "baseline_m": float(args.baseline_m),
            "depth_min": float(args.depth_min),
            "depth_max": float(args.depth_max),
            "disp_eps": float(args.disp_eps),
        }
    )
    return pc_cfg


def _open_grid(path: Path, shape: tuple[int, ...], resume: bool):
    if path.is_file() and resume:
        grid = np.load(path, mmap_mode="r+")
        if tuple(grid.shape) != shape or grid.dtype != np.float16:
            raise RuntimeError(f"existing grid has shape={grid.shape} dtype={grid.dtype}, expected {shape} fp16")
        return grid
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


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


def _write_meta(
    path: Path,
    *,
    dataset,
    cfg,
    args: argparse.Namespace,
    ffs_sha256: str,
    utonia_sha256: str,
    count: int,
    all_steps_sha256: str,
    image_shas: list[dict],
    geometry: dict,
) -> None:
    meta = {
        "suite": dataset.dataset_name,
        "dims": list(GRID_SHAPE),
        "dtype": "float16",
        "count": int(count),
        "all_steps_sha256": all_steps_sha256,
        "video_backend": dataset.video_backend,
        "delete_pause_frame": bool(dataset.delete_pause_frame),
        "data_mix": str(cfg.datasets.vla_data.data_mix),
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "ffs_model_path": args.ffs_model_path,
        "ffs_model_sha256": ffs_sha256,
        "utonia_ckpt_path": args.utonia_ckpt_path,
        "utonia_ckpt_sha256": utonia_sha256,
        "view_order": geometry["view_order"],
        "reference_view": geometry["reference_view"],
        "net0_frame": geometry["net0_frame"],
        "unrotate": geometry["unrotate"],
        "geometry": geometry,
        "image_sha256_samples": image_shas,
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def _normalized_all_steps(dataset) -> list[tuple[object, object]]:
    return [
        (_cache_key_part(traj_id), _cache_key_part(base_index))
        for traj_id, base_index in dataset.all_steps
    ]


def _assert_resume_cache_consistent(
    *,
    cache_dir: Path,
    suite: str,
    dataset,
    ffs_sha256: str,
    utonia_sha256: str,
    pc_cfg: dict,
    cfg,
) -> None:
    validate_utonia_cache(
        cache_dir,
        [suite],
        dataset_or_none={suite: dataset},
        ffs_sha256=ffs_sha256,
        utonia_sha256=utonia_sha256,
        pc_cfg=pc_cfg,
        grid_hw=(8, 8),
        require_all_done=False,
        expected_data_root=str(cfg.datasets.vla_data.data_root_dir),
        expected_data_mix=str(cfg.datasets.vla_data.data_mix),
        expected_video_backend=getattr(dataset, "video_backend", None),
    )


def precompute_suite(model, pc_cfg: dict, dataset, suite_dir: Path, args: argparse.Namespace, cfg) -> None:

    count = len(dataset.all_steps)
    suite_dir.mkdir(parents=True, exist_ok=True)
    _df_guard(suite_dir, count * np.dtype(np.float16).itemsize * int(np.prod(GRID_SHAPE)), args.free_space_multiplier)

    grid_path = suite_dir / "grid.f16.npy"
    done_path = suite_dir / "done.npy"
    index_path = suite_dir / "index.pkl"
    meta_path = suite_dir / "meta.json"

    all_steps = _normalized_all_steps(dataset)
    all_steps_sha = _all_steps_sha256(all_steps)
    geometry = _utonia_geometry_meta(pc_cfg, (8, 8))
    ffs_sha256 = _sha256_file(args.ffs_model_path)
    utonia_sha256 = _sha256_file(args.utonia_ckpt_path)
    if args.resume and (grid_path.is_file() or done_path.is_file()):
        _assert_resume_cache_consistent(
            cache_dir=suite_dir.parent,
            suite=dataset.dataset_name,
            dataset=dataset,
            ffs_sha256=ffs_sha256,
            utonia_sha256=utonia_sha256,
            pc_cfg=pc_cfg,
            cfg=cfg,
        )

    grid = _open_grid(grid_path, (count, *GRID_SHAPE), resume=args.resume)
    done = _open_done(done_path, count, resume=args.resume)
    index = {
        (_cache_key_part(traj_id), _cache_key_part(base_index)): row
        for row, (traj_id, base_index) in enumerate(dataset.all_steps)
    }
    with open(index_path, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    sha_rows = sorted(set([0, max(0, count // 2), max(0, count - 1)])) if count else []
    _write_meta(
        meta_path,
        dataset=dataset,
        cfg=cfg,
        args=args,
        ffs_sha256=ffs_sha256,
        utonia_sha256=utonia_sha256,
        count=count,
        all_steps_sha256=all_steps_sha,
        image_shas=_sample_image_shas(dataset, sha_rows),
        geometry=geometry,
    )

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
        with torch.inference_mode():
            out = model.compute_utonia_grid([images], pc_cfg, grid_hw=(8, 8), deterministic=True)
        if tuple(out.shape) != (1, *GRID_SHAPE):
            raise RuntimeError(f"unexpected grid shape for row={row}: {tuple(out.shape)}")
        grid[row] = out[0].detach().cpu().to(torch.float16).numpy()
        done[row] = True
        written += 1
        if written % int(args.log_every) == 0 or row == limit - 1:
            elapsed = max(time.time() - start, 1e-6)
            rate = written / elapsed
            remaining = max(limit - written, 0)
            eta = remaining / rate if rate > 0 else 0.0
            grid.flush()
            done.flush()
            print(
                "CACHE_PROGRESS "
                f"suite={dataset.dataset_name} done={written}/{limit} "
                f"rate={rate:.3f}_rows_per_sec eta_sec={eta:.1f}"
            )
    grid.flush()
    done.flush()
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
    parser.add_argument("--utonia-ckpt-path", default=os.environ.get("UTONIA_CKPT_PATH", DEFAULT_UTONIA_CKPT))
    parser.add_argument("--ffs-expected-sha256", default=os.environ.get("FFS_SHA256"))
    parser.add_argument("--utonia-expected-sha256", default=os.environ.get("UTONIA_SHA256"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--backproject-stride", type=int, default=4)
    parser.add_argument("--fovy-degrees", type=float, default=45.0)
    parser.add_argument("--baseline-m", type=float, default=0.06)
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--disp-eps", type=float, default=1e-3)
    parser.add_argument("--utonia-scale", type=float, default=4.0)
    parser.add_argument("--utonia-enable-flash", action="store_true")
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--free-space-multiplier", type=float, default=1.2)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if args.ffs_repo_dir and args.ffs_repo_dir not in sys.path:
        sys.path.insert(0, args.ffs_repo_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("precompute_utonia_cache.py requires CUDA")

    cfg = _load_cfg(args)
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.modules.stereo.utonia_pointcloud import _BenchModel

    mixture_spec = DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]
    available = [name for name, _weight, _robot_type in mixture_spec]
    suites = _parse_suites(args.suites, available)
    _assert_libero_leftprimary_only(str(cfg.datasets.vla_data.data_mix), mixture_spec, suites)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
    by_name = {single.dataset_name: single for single in dataset.datasets}
    pc_cfg = _build_pc_cfg(args, Path(args.cache_dir))
    model = _BenchModel(pc_cfg).cuda().eval()

    for suite in suites:
        precompute_suite(
            model,
            pc_cfg,
            by_name[suite],
            Path(args.cache_dir) / suite,
            args,
            cfg,
        )
    print("CACHE_PRECOMPUTE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
