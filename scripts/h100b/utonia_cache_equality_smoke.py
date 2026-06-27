#!/usr/bin/env python3
"""GPU smoke for Method #10A Utonia offline cache determinism and read fallback."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.utonia_pointcloud import (  # noqa: E402
    _all_steps_sha256,
    _image_sha,
    _utonia_geometry_meta,
)

DEFAULT_FFS_REPO_DIR = "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo"
DEFAULT_FFS_MODEL = (
    "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo/"
    "weights/20-30-48/model_best_bp2_serialize.pth"
)
DEFAULT_UTONIA_CKPT = "./playground/Pretrained_models/Utonia/utonia.pth"
GRID_SHAPE = (1387, 8, 8)


def _build_cfg(args: argparse.Namespace, cache_dir: Path | None) -> dict:
    return {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.image_size,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "utonia_ckpt_path": args.utonia_ckpt_path,
        "utonia_expected_sha256": args.utonia_expected_sha256,
        "utonia_scale": float(args.utonia_scale),
        "utonia_enable_flash": bool(args.utonia_enable_flash),
        "utonia_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "utonia_batched": False,
        "fovy_degrees": 45.0,
        "baseline_m": 0.06,
        "image_width": args.image_size,
        "image_height": args.image_size,
        "depth_min": 0.05,
        "depth_max": 3.0,
        "disp_eps": 1e-3,
        "backproject_stride": args.backproject_stride,
    }


def _write_cache(
    cache_dir: Path,
    suite: str,
    grids: torch.Tensor,
    pc_cfg: dict,
    examples: list,
    model,
) -> list[tuple[str, object, object]]:
    suite_dir = cache_dir / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    n = int(grids.shape[0])
    grid = np.lib.format.open_memmap(
        suite_dir / "grid.f16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n, *GRID_SHAPE),
    )
    grid[:] = grids.detach().cpu().to(torch.float16).numpy()
    grid.flush()
    done = np.lib.format.open_memmap(suite_dir / "done.npy", mode="w+", dtype=np.bool_, shape=(n,))
    done[:] = True
    done.flush()
    index = {(i, 0): i for i in range(n)}
    with open(suite_dir / "index.pkl", "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    all_steps = [(i, 0) for i in range(n)]
    sha_rows = sorted(set([0, max(0, n // 2), max(0, n - 1)])) if n else []
    image_shas = [
        {
            "row": int(row),
            "traj_id": int(row),
            "base_index": 0,
            "image_sha256": [_image_sha(img) for img in examples[row]],
        }
        for row in sha_rows
    ]

    geom = _utonia_geometry_meta(pc_cfg, (8, 8))
    with open(suite_dir / "meta.json", "w") as fh:
        json.dump(
            {
                "suite": suite,
                "dims": list(GRID_SHAPE),
                "dtype": "float16",
                "count": n,
                "all_steps_sha256": _all_steps_sha256(all_steps),
                "ffs_model_sha256": getattr(model, "_ffs_actual_sha256", None),
                "utonia_ckpt_sha256": getattr(model, "_utonia_actual_sha256", None),
                # leftprimary: validate_utonia_cache reads these 4 keys at meta TOP-LEVEL
                # (mirror precompute_utonia_cache.py), not only inside "geometry".
                "view_order": geom["view_order"],
                "reference_view": geom["reference_view"],
                "net0_frame": geom["net0_frame"],
                "unrotate": geom["unrotate"],
                "geometry": geom,
                "image_sha256_samples": image_shas,
            },
            fh,
            sort_keys=True,
        )
    return [(suite, np.int64(i), np.int64(0)) for i in range(n)]


def _diff_report(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> tuple[float, float, float, float]:
    if not a.numel():
        return 0.0, 0.0, 0.0, float(atol)
    diff = (a.float() - b.float()).abs()
    ref_max = float(b.float().abs().max().detach().cpu())
    max_abs = float(diff.max().detach().cpu())
    max_rel = max_abs / max(ref_max, 1e-12)
    allowed = float(atol) + float(rtol) * ref_max
    return max_abs, max_rel, ref_max, allowed


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float, code: int) -> None:
    max_abs, max_rel, ref_max, allowed = _diff_report(a, b, atol, rtol)
    print(
        f"{name} max_abs_diff={max_abs:.6e} max_rel_to_ref={max_rel:.6e} "
        f"ref_max_abs={ref_max:.6e} allowed={allowed:.6e}"
    )
    if max_abs >= allowed:
        print(f"UTONIA_CACHE_SMOKE_FAIL reason={name.lower()}_close")
        raise SystemExit(code)


def _framework_sample_ids(examples: list[dict]):
    from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TFFSBase

    captured = []

    class Harness:
        _sample_ids_from_examples = QwenGR00TFFSBase._sample_ids_from_examples
        _encode_last_hidden_with_ffs = QwenGR00TFFSBase._encode_last_hidden_with_ffs
        _encode_last_hidden_with_optional_sample_ids = QwenGR00TFFSBase._encode_last_hidden_with_optional_sample_ids

        def __init__(self):
            self.qwen_vl_interface = SimpleNamespace(
                build_qwenvl_inputs=lambda images, instructions: {"images": images, "instructions": instructions}
            )

        def _prepare_ffs_for_vlm(self, batch_images, sample_ids=None):
            captured.append(sample_ids)

        def _run_qwenvl_forward(self, qwen_inputs):
            return torch.zeros(1)

        def _cleanup_ffs_after_vlm(self):
            return None

    harness = Harness()
    sample_ids = harness._sample_ids_from_examples(examples)
    harness._encode_last_hidden_with_optional_sample_ids(
        [example["image"] for example in examples],
        [example["lang"] for example in examples],
        sample_ids=sample_ids,
    )
    if captured != [sample_ids]:
        raise RuntimeError(f"sample_ids were not threaded through optional encode: captured={captured} ids={sample_ids}")
    return sample_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="tmp/utonia_cache_smoke")
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-model-path", default=os.environ.get("FFS_MODEL_PATH", DEFAULT_FFS_MODEL))
    parser.add_argument("--utonia-ckpt-path", default=os.environ.get("UTONIA_CKPT_PATH", DEFAULT_UTONIA_CKPT))
    parser.add_argument("--ffs-expected-sha256", default=os.environ.get("FFS_SHA256"))
    parser.add_argument("--utonia-expected-sha256", default=os.environ.get("UTONIA_SHA256"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--backproject-stride", type=int, default=4)
    parser.add_argument("--utonia-scale", type=float, default=4.0)
    parser.add_argument("--utonia-enable-flash", action="store_true")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--allow-random-positive-flake", action="store_true")
    args = parser.parse_args()

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if args.ffs_repo_dir and args.ffs_repo_dir not in sys.path:
        sys.path.insert(0, args.ffs_repo_dir)
    if not torch.cuda.is_available():
        print("UTONIA_CACHE_SMOKE_FAIL reason=no_cuda")
        return 2

    from starVLA.model.modules.stereo.utonia_pointcloud import _BenchModel, _synthetic_pair

    work_dir = (ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    cache_dir = work_dir / "cache"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(args, cache_dir)
    model = _BenchModel(cfg).cuda().eval()
    k = max(int(args.k), 3)
    examples = [_synthetic_pair(7000 + i, args.image_size) for i in range(k)]

    with torch.inference_mode():
        a = model.compute_utonia_grid([examples[0]], cfg, grid_hw=(8, 8), deterministic=True)
        b = model.compute_utonia_grid([examples[0]], cfg, grid_hw=(8, 8), deterministic=True)
    _assert_close("UTONIA_CACHE_DETERMINISM", a, b, args.atol, args.rtol, 3)

    no_cache_cfg = _build_cfg(args, None)
    no_cache_model = _BenchModel(no_cache_cfg).cuda().eval()
    if no_cache_model._utonia_cache_dir is not None or no_cache_model._utonia_transform_deterministic is not None:
        print("UTONIA_CACHE_SMOKE_FAIL reason=cache_off_deterministic_engaged")
        return 4
    if bool(getattr(no_cache_model, "_utonia_shuffle_orders_forced", False)):
        print("UTONIA_CACHE_SMOKE_FAIL reason=cache_off_shuffle_forced")
        return 5
    with torch.inference_mode():
        random_a = no_cache_model.compute_utonia_grid([examples[0]], no_cache_cfg, grid_hw=(8, 8), deterministic=False)
        random_b = no_cache_model.compute_utonia_grid([examples[0]], no_cache_cfg, grid_hw=(8, 8), deterministic=False)
    rand_abs, rand_rel, rand_ref, _rand_allowed = _diff_report(random_a, random_b, args.atol, args.rtol)
    print(
        "UTONIA_CACHE_RANDOM_POSITIVE_CONTROL "
        f"max_abs_diff={rand_abs:.6e} max_rel_to_ref={rand_rel:.6e} ref_max_abs={rand_ref:.6e}"
    )
    if rand_abs <= args.atol:
        msg = "UTONIA_CACHE_RANDOM_POSITIVE_WARN" if args.allow_random_positive_flake else "UTONIA_CACHE_SMOKE_FAIL"
        print(f"{msg} reason=random_path_not_observably_random")
        if not args.allow_random_positive_flake:
            return 6
    if (
        getattr(no_cache_model, "_utonia_cache_hits", 0)
        or getattr(no_cache_model, "_utonia_cache_row_misses", 0)
        or getattr(no_cache_model, "_utonia_cache_whole_batch_live", 0)
    ):
        print("UTONIA_CACHE_SMOKE_FAIL reason=cache_off_touched_cache_counters")
        return 7

    with torch.inference_mode():
        fresh = model.compute_utonia_grid(examples, cfg, grid_hw=(8, 8), deterministic=True)
    sample_ids = _write_cache(cache_dir, "smoke_suite", fresh, cfg, examples, model)
    with torch.inference_mode():
        hits_before = int(getattr(model, "_utonia_cache_hits", 0))
        cached = model.compute_utonia_grid(
            examples,
            cfg,
            grid_hw=(8, 8),
            sample_ids=sample_ids,
            deterministic=True,
        )
        hits_after = int(getattr(model, "_utonia_cache_hits", 0))
    print(
        "UTONIA_CACHE_EQUALITY "
        f"shape={tuple(cached.shape)} dtype={cached.dtype} device={cached.device} "
    )
    if tuple(cached.shape) != tuple(fresh.shape) or cached.dtype != fresh.dtype or cached.device != fresh.device:
        print("UTONIA_CACHE_SMOKE_FAIL reason=shape_dtype_device")
        return 8
    if hits_after - hits_before != len(sample_ids):
        print(
            "UTONIA_CACHE_SMOKE_FAIL reason=cache_hit_counter "
            f"delta={hits_after - hits_before} expected={len(sample_ids)}"
        )
        return 9
    _assert_close("UTONIA_CACHE_EQUALITY", cached, fresh, args.atol, args.rtol, 10)

    example_dicts = [
        {
            "image": examples[i],
            "lang": "pick up the object",
            "action": np.zeros((1, 7), dtype=np.float16),
            "state": np.zeros((1, 7), dtype=np.float16),
            "dataset_name": "smoke_suite",
            "traj_id": np.int64(i),
            "base_index": np.int64(0),
        }
        for i in range(k)
    ]
    framework_ids = _framework_sample_ids(example_dicts)
    hits_before = int(getattr(model, "_utonia_cache_hits", 0))
    with torch.inference_mode():
        model.compute_utonia_grid(
            [example["image"] for example in example_dicts],
            cfg,
            grid_hw=(8, 8),
            sample_ids=framework_ids,
            deterministic=True,
        )
    if int(getattr(model, "_utonia_cache_hits", 0)) - hits_before != k:
        print("UTONIA_CACHE_SMOKE_FAIL reason=id_plumbing_no_cache_hit")
        return 11
    missing_id = dict(example_dicts[0])
    missing_id.pop("base_index")
    if _framework_sample_ids([missing_id]) is not None:
        print("UTONIA_CACHE_SMOKE_FAIL reason=missing_id_not_none")
        return 12
    live_before = int(getattr(model, "_utonia_cache_whole_batch_live", 0))
    with torch.inference_mode():
        model.compute_utonia_grid([missing_id["image"]], cfg, grid_hw=(8, 8), sample_ids=None, deterministic=True)
    if int(getattr(model, "_utonia_cache_whole_batch_live", 0)) <= live_before:
        print("UTONIA_CACHE_SMOKE_FAIL reason=missing_id_live_fallback_not_counted")
        return 13

    mixed_ids = [sample_ids[0], ("smoke_suite", 999999, 0), sample_ids[2]]
    with torch.inference_mode():
        mixed = model.compute_utonia_grid(
            [examples[0], examples[1], examples[2]],
            cfg,
            grid_hw=(8, 8),
            sample_ids=mixed_ids,
            deterministic=True,
        )
        miss_fresh = model._compute_utonia_grid_live([examples[1]], cfg, grid_hw=(8, 8), deterministic=True)
    _assert_close("UTONIA_CACHE_MIXED_HIT0", mixed[0:1], fresh[0:1], args.atol, args.rtol, 14)
    _assert_close("UTONIA_CACHE_MIXED_MISS1", mixed[1:2], miss_fresh, args.atol, args.rtol, 15)
    _assert_close("UTONIA_CACHE_MIXED_HIT2", mixed[2:3], fresh[2:3], args.atol, args.rtol, 16)

    with torch.inference_mode():
        miss = model.compute_utonia_grid(
            [examples[0]],
            cfg,
            grid_hw=(8, 8),
            sample_ids=[("smoke_suite", 999999, 0)],
            deterministic=True,
        )
    print(f"UTONIA_CACHE_MISS_FALLBACK shape={tuple(miss.shape)} dtype={miss.dtype} device={miss.device}")
    if tuple(miss.shape) != (1, *GRID_SHAPE):
        print("UTONIA_CACHE_SMOKE_FAIL reason=miss_shape")
        return 17
    print("UTONIA_CACHE_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
