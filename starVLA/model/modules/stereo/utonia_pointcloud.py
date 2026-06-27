"""Utonia point-cloud construction and frozen feature extraction for Method #10."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TNet0FFSMixin

logger = logging.getLogger(__name__)


UTONIA_FEATURE_DIM = 1386


def _maybe_cache_dir(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _cache_key_part(value):
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _normalize_sample_id(sample_id):
    if sample_id is None:
        return None
    if len(sample_id) != 3:
        raise ValueError(f"sample_id must be (dataset_name,traj_id,base_index), got {sample_id!r}")
    dataset_name, traj_id, base_index = sample_id
    if dataset_name is None or traj_id is None or base_index is None:
        return None
    return str(dataset_name), _cache_key_part(traj_id), _cache_key_part(base_index)


def _image_sha(image) -> str:
    if torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _all_steps_sha256(all_steps: Sequence[Tuple[object, object]]) -> str:
    h = hashlib.sha256()
    for traj_id, base_index in all_steps:
        h.update(repr((_cache_key_part(traj_id), _cache_key_part(base_index))).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _all_steps_from_index(index: dict, count: int) -> List[Tuple[object, object]]:
    rows: List[Optional[Tuple[object, object]]] = [None] * int(count)
    for key, row in index.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise RuntimeError(f"cache index key must be (traj_id,base_index), got {key!r}")
        row = int(row)
        if row < 0 or row >= int(count):
            raise RuntimeError(f"cache index row out of range: key={key!r} row={row} count={count}")
        if rows[row] is not None:
            raise RuntimeError(f"cache index duplicate row {row}: {rows[row]!r} and {key!r}")
        rows[row] = (_cache_key_part(key[0]), _cache_key_part(key[1]))
    if any(row is None for row in rows):
        missing = [i for i, row in enumerate(rows) if row is None][:8]
        raise RuntimeError(f"cache index has holes at rows={missing}")
    return [row for row in rows if row is not None]


def _all_steps_from_dataset(dataset) -> List[Tuple[object, object]]:
    return [
        (_cache_key_part(traj_id), _cache_key_part(base_index))
        for traj_id, base_index in dataset.all_steps
    ]


def _utonia_geometry_meta(pc_cfg, grid_hw: Tuple[int, int]) -> dict:
    keys = (
        "utonia_scale",
        "utonia_enable_flash",
        "utonia_feature_dim",
        "ffs_image_size",
        "fovy_degrees",
        "baseline_m",
        "image_width",
        "image_height",
        "backproject_stride",
        "depth_min",
        "depth_max",
        "disp_eps",
    )
    legacy = any(pc_cfg.get(key, None) is not None for key in ("primary_idx", "right_view_idx", "primary_cam_id"))
    meta = {
        "grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
        "view_order": ["right_view", "primary"] if legacy else ["primary", "left_view"],
        "reference_view": "legacy_primary_after_unrotate" if legacy else "left_view",
        "net0_frame": "legacy_primary_rotated" if legacy else "left_view",
        "unrotate": bool(legacy),
    }
    for key in keys:
        if key == "utonia_feature_dim":
            value = pc_cfg.get(key, UTONIA_FEATURE_DIM)
        elif key in pc_cfg:
            value = pc_cfg[key]
        else:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        meta[key] = value
    return meta


def _dataset_for_cache_suite(dataset_or_none, suite: str):
    if dataset_or_none is None:
        return None
    if isinstance(dataset_or_none, Mapping):
        return dataset_or_none.get(suite)
    if getattr(dataset_or_none, "dataset_name", None) == suite:
        return dataset_or_none
    return None


def _require_cache_meta_value(meta: dict, suite: str, key: str, expected) -> None:
    if expected is None:
        raise RuntimeError(
            f"Utonia cache cannot validate meta {key} for suite={suite}; "
            "the loaded checkpoint SHA was not recorded"
        )
    cached = meta.get(key)
    if cached is None:
        raise RuntimeError(f"Utonia cache meta missing {key} for suite={suite}")
    if str(cached) != str(expected):
        raise RuntimeError(
            f"Utonia cache meta {key} mismatch for suite={suite}: cached={cached} loaded={expected}"
        )


def _require_optional_meta_value(meta: dict, suite: str, key: str, expected) -> None:
    if expected is None:
        return
    cached = meta.get(key)
    if cached is None:
        raise RuntimeError(f"Utonia cache meta missing {key} for suite={suite}")
    if str(cached) != str(expected):
        raise RuntimeError(
            f"Utonia cache meta {key} mismatch for suite={suite}: cached={cached} expected={expected}"
        )


def _validate_image_sha_samples(
    *,
    suite: str,
    meta: dict,
    index: dict,
    count: int,
    dataset=None,
) -> Dict[Tuple[object, object], List[str]]:
    samples = meta.get("image_sha256_samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(
            f"Utonia cache meta image_sha256_samples must be a non-empty list for suite={suite}"
        )

    image_sha_by_key: Dict[Tuple[object, object], List[str]] = {}
    for pos, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError(f"Utonia cache image SHA sample #{pos} must be a dict for suite={suite}")
        if "traj_id" not in sample or "base_index" not in sample:
            raise RuntimeError(f"Utonia cache image SHA sample #{pos} missing traj_id/base_index for suite={suite}")
        key = (_cache_key_part(sample["traj_id"]), _cache_key_part(sample["base_index"]))
        if key not in index:
            raise RuntimeError(f"Utonia cache image SHA sample key={key!r} missing from index for suite={suite}")

        row = sample.get("row", index[key])
        try:
            row = int(row)
        except (TypeError, ValueError):
            raise RuntimeError(f"Utonia cache image SHA sample key={key!r} has invalid row={row!r}")
        if row < 0 or row >= int(count):
            raise RuntimeError(f"Utonia cache image SHA sample key={key!r} row={row} out of range count={count}")
        if int(index[key]) != row:
            raise RuntimeError(
                f"Utonia cache image SHA sample key={key!r} row mismatch: meta={row} index={index[key]}"
            )

        shas = sample.get("image_sha256")
        if not isinstance(shas, list) or not shas or not all(_is_sha256(x) for x in shas):
            raise RuntimeError(
                f"Utonia cache image SHA sample key={key!r} must contain non-empty 64-hex image_sha256 list"
            )
        shas = [str(x) for x in shas]
        image_sha_by_key[key] = shas

        if dataset is not None:
            live_sample = dataset[row]
            live_key = (
                _cache_key_part(live_sample.get("traj_id")),
                _cache_key_part(live_sample.get("base_index")),
            )
            if live_key != key:
                raise RuntimeError(
                    f"Utonia cache image SHA sample row={row} key mismatch for suite={suite}: "
                    f"meta={key!r} dataset={live_key!r}"
                )
            actual = [_image_sha(img) for img in live_sample["image"]]
            if actual != shas:
                raise RuntimeError(
                    f"Utonia cache image SHA mismatch for suite={suite} row={row} key={key!r}: "
                    f"cached={shas} live={actual}"
                )
    return image_sha_by_key


def validate_utonia_cache(
    cache_dir: str | Path,
    suites: Sequence[str],
    dataset_or_none=None,
    ffs_sha256: Optional[str] = None,
    utonia_sha256: Optional[str] = None,
    pc_cfg=None,
    grid_hw: Tuple[int, int] = (8, 8),
    require_all_done: bool = True,
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
) -> Dict[str, dict]:
    """Fail-closed validator for Method #10A Utonia cache suites.

    When dataset_or_none is provided, sampled image SHAs and all-steps hashes are
    checked against the live dataset. Without a dataset, the validator still
    rejects malformed or missing SHA metadata but cannot re-hash image bytes.
    """
    if pc_cfg is None:
        raise RuntimeError("validate_utonia_cache requires pc_cfg so cache geometry can be checked")

    cache_root = Path(cache_dir).expanduser()
    handles: Dict[str, dict] = {}
    expected_tail = (
        int(pc_cfg.get("utonia_feature_dim", UTONIA_FEATURE_DIM)) + 1,
        int(grid_hw[0]),
        int(grid_hw[1]),
    )
    expected_geometry = _utonia_geometry_meta(pc_cfg, grid_hw)

    for suite in suites:
        suite = str(suite)
        suite_dir = cache_root / suite
        grid_path = suite_dir / "grid.f16.npy"
        index_path = suite_dir / "index.pkl"
        meta_path = suite_dir / "meta.json"
        done_path = suite_dir / "done.npy"
        missing_files = []
        if not suite_dir.is_dir():
            missing_files.append(str(suite_dir))
        for path in (grid_path, index_path, meta_path, done_path):
            if not path.is_file():
                missing_files.append(str(path))
        if missing_files:
            raise FileNotFoundError(f"Utonia cache structurally absent for suite={suite}: {missing_files}")

        with open(index_path, "rb") as fh:
            index = pickle.load(fh)
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        grid = np.load(grid_path, mmap_mode="r")
        done = np.load(done_path, mmap_mode="r")

        if tuple(grid.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"Utonia cache grid shape tail {tuple(grid.shape[1:])} != expected {expected_tail} "
                f"for {grid_path}"
            )
        count = int(grid.shape[0])
        if tuple(done.shape) != (count,):
            raise RuntimeError(
                f"Utonia cache done.npy shape {tuple(done.shape)} != rows {(count,)} for {done_path}"
            )
        if done.dtype != np.dtype(np.bool_):
            raise RuntimeError(f"Utonia cache done.npy dtype must be bool for suite={suite}; got {done.dtype}")
        if require_all_done and not bool(np.all(done)):
            incomplete = int(count - int(np.asarray(done, dtype=np.bool_).sum()))
            raise RuntimeError(f"Utonia cache suite={suite} is incomplete: {incomplete}/{count} rows done=False")

        meta_count = meta.get("count")
        try:
            meta_count_int = int(meta_count)
        except (TypeError, ValueError):
            meta_count_int = -1
        if meta_count_int != count or len(index) != count:
            raise RuntimeError(
                f"Utonia cache count mismatch for suite={suite}: "
                f"meta_count={meta_count} index={len(index)} grid={count}"
            )
        if str(meta.get("dtype")) != "float16" or grid.dtype != np.float16:
            raise RuntimeError(
                f"Utonia cache dtype mismatch for suite={suite}: meta={meta.get('dtype')} grid={grid.dtype}"
            )
        if meta.get("dims") != list(expected_tail):
            raise RuntimeError(
                f"Utonia cache dims mismatch for suite={suite}: meta={meta.get('dims')} expected={list(expected_tail)}"
            )
        if str(meta.get("suite")) != suite:
            raise RuntimeError(f"Utonia cache suite meta mismatch: dir={suite} meta={meta.get('suite')}")

        index_all_steps = _all_steps_from_index(index, count)
        index_sha = _all_steps_sha256(index_all_steps)
        if meta.get("all_steps_sha256") != index_sha:
            raise RuntimeError(
                f"Utonia cache all_steps_sha256 mismatch for suite={suite}: "
                f"meta={meta.get('all_steps_sha256')} index={index_sha}"
            )

        dataset = _dataset_for_cache_suite(dataset_or_none, suite)
        if dataset is not None:
            dataset_steps = _all_steps_from_dataset(dataset)
            dataset_sha = _all_steps_sha256(dataset_steps)
            if dataset_sha != index_sha:
                raise RuntimeError(
                    f"Utonia cache all_steps_sha256 mismatch for suite={suite}: "
                    f"dataset={dataset_sha} index={index_sha}"
                )
            _require_optional_meta_value(meta, suite, "video_backend", getattr(dataset, "video_backend", None))

        _require_cache_meta_value(meta, suite, "ffs_model_sha256", ffs_sha256)
        _require_cache_meta_value(meta, suite, "utonia_ckpt_sha256", utonia_sha256)
        _require_optional_meta_value(meta, suite, "data_root_dir", expected_data_root)
        _require_optional_meta_value(meta, suite, "data_mix", expected_data_mix)
        _require_optional_meta_value(meta, suite, "video_backend", expected_video_backend)

        cached_geometry = meta.get("geometry", None)
        if cached_geometry is None:
            raise RuntimeError(f"Utonia cache meta missing geometry for suite={suite}: {meta_path}")
        if cached_geometry != expected_geometry:
            keys = sorted(set(cached_geometry.keys()) | set(expected_geometry.keys()))
            mismatches = {
                key: (cached_geometry.get(key), expected_geometry.get(key))
                for key in keys
                if cached_geometry.get(key) != expected_geometry.get(key)
            }
            raise RuntimeError(f"Utonia cache geometry mismatch for suite={suite}: {mismatches}")
        for key in ("view_order", "reference_view", "net0_frame", "unrotate"):
            if meta.get(key) != expected_geometry.get(key):
                raise RuntimeError(
                    f"Utonia cache {key} mismatch for suite={suite}: "
                    f"meta={meta.get(key)!r} expected={expected_geometry.get(key)!r}"
                )

        image_sha_by_key = _validate_image_sha_samples(
            suite=suite,
            meta=meta,
            index=index,
            count=count,
            dataset=dataset,
        )
        handles[suite] = {
            "grid": grid,
            "index": index,
            "meta": meta,
            "done": done,
            "image_sha_by_key": image_sha_by_key,
        }
    return handles


@dataclass
class PointCloudSample:
    coord: torch.Tensor
    color: torch.Tensor
    normal: torch.Tensor
    pixel_xy: torch.Tensor
    invalid_ratio: float
    clamp_ratio: float
    sampled_pixels: int


@dataclass
class UtoniaPointCloudBatch:
    samples: List[PointCloudSample]
    point_features: List[torch.Tensor]
    disparity: torch.Tensor


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_rgb_tensor_255(img, image_size: int) -> torch.Tensor:
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    if not torch.is_tensor(img):
        img = to_tensor(img)
        scale_to_255 = True
    elif img.ndim == 3 and img.shape[0] not in (1, 3) and img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
        scale_to_255 = torch.is_floating_point(img)
    else:
        scale_to_255 = torch.is_floating_point(img)
    img = img.float()
    img = resize(img.unsqueeze(0)).squeeze(0)
    if scale_to_255:
        img = img * 255.0
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    if img.shape[0] != 3:
        raise ValueError(f"expected RGB image tensor with 3 channels, got {tuple(img.shape)}")
    return img.clamp(0.0, 255.0)


def _make_deterministic_grid_sample(transform_mod, source_grid_sample):
    class DeterministicFirstGridSample(transform_mod.GridSample):
        def __call__(self, data_dict):
            assert "coord" in data_dict.keys()
            scaled_coord = data_dict["coord"] / np.array(self.grid_size)
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            min_coord = min_coord * np.array(self.grid_size)
            key = self.hash(grid_coord)
            idx_sort = np.argsort(key, kind="stable")
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            idx_select = np.cumsum(np.insert(count, 0, 0)[:-1])
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                idx_unique = np.unique(np.append(idx_unique, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = transform_mod.index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = scaled_coord - grid_coord - 0.5
                if self.project_displacement:
                    displacement = np.sum(displacement * data_dict["normal"], axis=-1, keepdims=True)
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

    det = DeterministicFirstGridSample(
        grid_size=source_grid_sample.grid_size,
        hash_type="fnv",
        mode="train",
        return_inverse=source_grid_sample.return_inverse,
        return_grid_coord=source_grid_sample.return_grid_coord,
        return_min_coord=source_grid_sample.return_min_coord,
        return_displacement=source_grid_sample.return_displacement,
        project_displacement=source_grid_sample.project_displacement,
    )
    det.hash = source_grid_sample.hash
    return det


def _build_utonia_default_transform(utonia, *, scale: float, deterministic: bool):
    transform = utonia.transform.default(
        scale=float(scale),
        apply_z_positive=True,
        normalize_coord=False,
    )
    if not deterministic:
        return transform
    replaced = False
    for i, t in enumerate(transform.transforms):
        if isinstance(t, utonia.transform.GridSample):
            transform.transforms[i] = _make_deterministic_grid_sample(utonia.transform, t)
            replaced = True
            break
    if not replaced:
        raise RuntimeError("Utonia default transform did not contain GridSample")
    return transform


def backproject_disparity_to_pointcloud(
    *,
    disparity: torch.Tensor,
    left_rgb_255: torch.Tensor,
    fovy_degrees: float = 45.0,
    baseline_m: float = 0.06,
    image_width: int = 256,
    image_height: int = 256,
    backproject_stride: int = 4,
    depth_min: float = 0.05,
    depth_max: float = 3.0,
    disp_eps: float = 1e-3,
) -> List[PointCloudSample]:
    if disparity.ndim != 4 or int(disparity.shape[1]) != 1:
        raise ValueError(f"disparity must be (B,1,H,W), got {tuple(disparity.shape)}")
    if left_rgb_255.shape[:2] != (disparity.shape[0], 3):
        raise ValueError(
            f"left_rgb_255 must be (B,3,H,W), got {tuple(left_rgb_255.shape)} for B={disparity.shape[0]}"
        )

    B, _, H, W = disparity.shape
    if H != int(image_height) or W != int(image_width):
        raise ValueError(
            f"disparity size {(H, W)} does not match configured {(image_height, image_width)}"
        )
    stride = max(int(backproject_stride), 1)
    device = disparity.device
    dtype = torch.float32
    f = (float(image_height) / 2.0) / math.tan(math.radians(float(fovy_degrees) / 2.0))
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0

    ys = torch.arange(0, H, stride, device=device, dtype=dtype)
    xs = torch.arange(0, W, stride, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    flat_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    sample_count = int(flat_xy.shape[0])

    samples: List[PointCloudSample] = []
    for b in range(B):
        disp = disparity[b, 0, ::stride, ::stride].float().reshape(-1)
        positive_disp = disp > float(disp_eps)
        depth_raw = torch.empty_like(disp)
        depth_raw[positive_disp] = float(f) * float(baseline_m) / disp[positive_disp].clamp_min(float(disp_eps))
        depth_raw[~positive_disp] = float("inf")
        in_depth_range = (depth_raw >= float(depth_min)) & (depth_raw <= float(depth_max))
        valid = positive_disp & in_depth_range
        invalid_ratio = 1.0 - (float(valid.float().mean().detach().cpu()) if valid.numel() else 0.0)
        if not bool(valid.any()):
            empty = disparity.new_zeros((0, 3), dtype=torch.float32)
            samples.append(
                PointCloudSample(
                    coord=empty,
                    color=empty,
                    normal=empty,
                    pixel_xy=empty[:, :2],
                    invalid_ratio=invalid_ratio,
                    clamp_ratio=0.0,
                    sampled_pixels=sample_count,
                )
            )
            continue
        z = depth_raw[valid]
        clamp_ratio = 0.0
        xy = flat_xy[valid]
        x = (xy[:, 0] - cx) * z / float(f)
        y = (xy[:, 1] - cy) * z / float(f)
        coord = torch.stack([x, y, z], dim=-1).to(dtype=torch.float32)
        color = left_rgb_255[b, :, ::stride, ::stride].permute(1, 2, 0).reshape(-1, 3)[valid]
        color = color.to(device=device, dtype=torch.float32).clamp(0.0, 255.0)
        normal = torch.zeros_like(coord)
        samples.append(
            PointCloudSample(
                coord=coord,
                color=color,
                normal=normal,
                pixel_xy=xy.to(dtype=torch.float32),
                invalid_ratio=invalid_ratio,
                clamp_ratio=clamp_ratio,
                sampled_pixels=sample_count,
            )
        )
    return samples


def pool_point_features_to_grid(
    *,
    samples: Sequence[PointCloudSample],
    point_features: Sequence[torch.Tensor],
    grid_hw: Tuple[int, int] = (8, 8),
    image_width: int = 256,
    image_height: int = 256,
    feature_dim: int = UTONIA_FEATURE_DIM,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_hw must be positive, got {grid_hw}")
    if len(samples) != len(point_features):
        raise ValueError(f"samples/features length mismatch: {len(samples)} vs {len(point_features)}")
    B = len(samples)
    device = point_features[0].device if point_features else torch.device("cpu")
    dtype = point_features[0].dtype if point_features else torch.float32
    out = torch.zeros(B, feature_dim + 1, grid_h, grid_w, device=device, dtype=dtype)
    counts_out = torch.zeros(B, grid_h, grid_w, device=device, dtype=torch.long)
    cell_w = float(image_width) / float(grid_w)
    cell_h = float(image_height) / float(grid_h)

    for b, (sample, feats) in enumerate(zip(samples, point_features)):
        if feats.ndim != 2 or int(feats.shape[-1]) != int(feature_dim):
            raise ValueError(
                f"point feature sample {b} must be (N,{feature_dim}), got {tuple(feats.shape)}"
            )
        if feats.numel() == 0:
            continue
        xy = sample.pixel_xy.to(device=feats.device, dtype=torch.float32)
        if int(xy.shape[0]) != int(feats.shape[0]):
            raise ValueError(
                f"point feature sample {b} N={feats.shape[0]} does not match pixels {xy.shape[0]}"
            )
        cx = torch.floor(xy[:, 0] / cell_w).long().clamp_(0, grid_w - 1)
        cy = torch.floor(xy[:, 1] / cell_h).long().clamp_(0, grid_h - 1)
        flat = cy * grid_w + cx
        sums = feats.new_zeros((grid_h * grid_w, feature_dim))
        counts = torch.zeros(grid_h * grid_w, device=feats.device, dtype=torch.long)
        sums.index_add_(0, flat, feats)
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        occupied = counts > 0
        if bool(occupied.any()):
            avg = sums[occupied] / counts[occupied].to(dtype=feats.dtype).unsqueeze(-1)
            flat_grid = feats.new_zeros((grid_h * grid_w, feature_dim))
            flat_grid[occupied] = avg
            out[b, :feature_dim] = flat_grid.t().reshape(feature_dim, grid_h, grid_w)
            out[b, feature_dim] = occupied.to(dtype=feats.dtype).reshape(grid_h, grid_w)
            counts_out[b] = counts.reshape(grid_h, grid_w).to(device=counts_out.device)
    return out, counts_out


def pad_point_features(
    point_features: Sequence[torch.Tensor],
    feature_dim: int = UTONIA_FEATURE_DIM,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not point_features:
        return torch.zeros(0, 1, feature_dim), torch.zeros(0, 1, dtype=torch.bool)
    device = point_features[0].device
    dtype = point_features[0].dtype
    max_n = max(max(int(feats.shape[0]), 1) for feats in point_features)
    batch = torch.zeros(len(point_features), max_n, feature_dim, device=device, dtype=dtype)
    mask = torch.zeros(len(point_features), max_n, device=device, dtype=torch.bool)
    for b, feats in enumerate(point_features):
        if feats.ndim != 2 or int(feats.shape[-1]) != int(feature_dim):
            raise ValueError(f"point features must be (N,{feature_dim}), got {tuple(feats.shape)}")
        n = int(feats.shape[0])
        if n:
            batch[b, :n] = feats
            mask[b, :n] = True
        else:
            mask[b, 0] = True
    return batch, mask


class UtoniaPointCloudMixin:
    """Mixin for frameworks that use frozen FFS disparity plus frozen Utonia."""

    def _init_frozen_ffs_for_disparity(self, ffs_cfg, label: str) -> None:
        self._init_frozen_ffs_net0(ffs_cfg, label, capture_net0=False)

    def compute_ffs_disparity(self, batch_images: List) -> torch.Tensor:
        return self._compute_ffs_disparity(batch_images)

    def _init_frozen_utonia(self, pc_cfg, label: str) -> None:
        ckpt_path = str(pc_cfg.get("utonia_ckpt_path"))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"{label} Utonia checkpoint not found: {ckpt_path}")
        self._utonia_ckpt_path = ckpt_path
        self._utonia_cache_dir = _maybe_cache_dir(pc_cfg.get("utonia_cache_dir", None))
        expected_sha256 = pc_cfg.get("utonia_expected_sha256", None)
        self._utonia_expected_sha256 = str(expected_sha256) if expected_sha256 else None
        actual = _sha256_file(ckpt_path) if expected_sha256 or self._utonia_cache_dir is not None else None
        self._utonia_actual_sha256 = actual
        if expected_sha256:
            if actual != str(expected_sha256):
                raise RuntimeError(
                    f"{label} Utonia SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info("%s Utonia SHA256 verified (%s)", label, str(expected_sha256)[:12])

        import utonia

        enable_flash = bool(pc_cfg.get("utonia_enable_flash", False))
        logger.info("%s loading frozen Utonia from %s (enable_flash=%s)", label, ckpt_path, enable_flash)
        self.utonia = utonia.load(
            ckpt_path,
            custom_config=dict(enable_flash=enable_flash, enc_patch_size=[1024] * 5),
        )
        self.utonia.eval()
        for param in self.utonia.parameters():
            param.requires_grad_(False)
        self._utonia_label = label
        self.utonia_scale = float(pc_cfg.get("utonia_scale", 4.0))
        self._utonia_cache_handles: Dict[Tuple[str, str], dict] = {}
        self._utonia_cache_miss_warn_count = 0
        self._utonia_cache_live_warn_count = 0
        self._utonia_cache_batches = 0
        self._utonia_cache_hits = 0
        self._utonia_cache_row_misses = 0
        self._utonia_cache_whole_batch_live = 0
        self._utonia_shuffle_orders_forced = False
        self._utonia_transform_random = _build_utonia_default_transform(
            utonia,
            scale=self.utonia_scale,
            deterministic=False,
        )
        self._utonia_transform_deterministic = (
            _build_utonia_default_transform(utonia, scale=self.utonia_scale, deterministic=True)
            if self._utonia_cache_dir is not None
            else None
        )
        self._utonia_transform = (
            self._utonia_transform_deterministic
            if self._utonia_transform_deterministic is not None
            else self._utonia_transform_random
        )
        if self._utonia_cache_dir is not None:
            self._force_utonia_deterministic_serialization()
        self._utonia_feature_dim = int(pc_cfg.get("utonia_feature_dim", UTONIA_FEATURE_DIM))
        self._utonia_geometry_logged = False
        self._utonia_invalid_warn_count = 0

    def _force_utonia_deterministic_serialization(self) -> None:
        found = 0
        for module in self.utonia.modules():
            if hasattr(module, "shuffle_orders"):
                found += 1
                module.shuffle_orders = False
        bad = [
            type(module).__name__
            for module in self.utonia.modules()
            if hasattr(module, "shuffle_orders") and bool(module.shuffle_orders)
        ]
        if bad:
            raise RuntimeError(f"{self._utonia_label} failed to force shuffle_orders=False: {bad[:8]}")
        if found == 0:
            raise RuntimeError(
                f"{self._utonia_label} could not find any Utonia shuffle_orders flags; "
                "cannot prove serialization/GridPooling randperm is disabled"
            )
        self._utonia_shuffle_orders_forced = True

    def _utonia_cache_enabled(self, pc_cfg) -> bool:
        return _maybe_cache_dir(pc_cfg.get("utonia_cache_dir", self._utonia_cache_dir)) is not None

    def _resolve_utonia_deterministic(self, pc_cfg, deterministic: Optional[bool]) -> bool:
        cache_mode = self._utonia_cache_enabled(pc_cfg)
        return cache_mode or bool(deterministic)

    def _select_utonia_transform(self, deterministic: bool):
        if not deterministic:
            return self._utonia_transform_random
        self._force_utonia_deterministic_serialization()
        if self._utonia_transform_deterministic is None:
            import utonia

            self._utonia_transform_deterministic = _build_utonia_default_transform(
                utonia,
                scale=self.utonia_scale,
                deterministic=True,
            )
        return self._utonia_transform_deterministic

    def _imgs_to_rgb_255_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        device = next(self.parameters()).device
        out = []
        for example_imgs in batch_images:
            if len(example_imgs) != int(self.num_cameras):
                raise ValueError(
                    f"{self._utonia_label} expected exactly {self.num_cameras} single-frame "
                    f"images per sample, got {len(example_imgs)}"
                )
            out.append(_to_rgb_tensor_255(example_imgs[view_idx], self.ffs_image_size))
        return torch.stack(out, dim=0).to(device=device, dtype=torch.float32)

    def _build_pointcloud_samples(self, batch_images: List, pc_cfg) -> Tuple[torch.Tensor, List[PointCloudSample]]:
        disparity = self.compute_ffs_disparity(batch_images)
        rgb = self._imgs_to_rgb_255_tensor(batch_images, self.ffs_image1_idx)
        samples = backproject_disparity_to_pointcloud(
            disparity=disparity,
            left_rgb_255=rgb,
            fovy_degrees=float(pc_cfg.get("fovy_degrees", 45.0)),
            baseline_m=float(pc_cfg.get("baseline_m", 0.06)),
            image_width=int(pc_cfg.get("image_width", self.ffs_image_size)),
            image_height=int(pc_cfg.get("image_height", self.ffs_image_size)),
            backproject_stride=int(pc_cfg.get("backproject_stride", 4)),
            depth_min=float(pc_cfg.get("depth_min", 0.05)),
            depth_max=float(pc_cfg.get("depth_max", 3.0)),
            disp_eps=float(pc_cfg.get("disp_eps", 1e-3)),
        )
        self._warn_invalid_pointcloud_batch(samples, pc_cfg)
        self._log_geometry_diagnostics_once(disparity, samples, pc_cfg)
        return disparity, samples

    def _warn_invalid_pointcloud_batch(self, samples: Sequence[PointCloudSample], pc_cfg) -> None:
        if not samples:
            return
        threshold = float(pc_cfg.get("invalid_ratio_warn_threshold", 0.9))
        mean_invalid = sum(float(s.invalid_ratio) for s in samples) / float(len(samples))
        all_empty = all(int(s.coord.shape[0]) == 0 for s in samples)
        if mean_invalid < threshold and not all_empty:
            return
        count = int(getattr(self, "_utonia_invalid_warn_count", 0))
        if count < 3 or count % 100 == 0:
            logger.warning(
                "%s high point-cloud invalid_ratio=%.4f (threshold=%.4f, point_counts=%s); "
                "check FFS disparity / stereo pairing / depth range",
                self._utonia_label,
                mean_invalid,
                threshold,
                [int(s.coord.shape[0]) for s in samples],
            )
        self._utonia_invalid_warn_count = count + 1

    def _log_geometry_diagnostics_once(
        self,
        disparity: torch.Tensor,
        samples: Sequence[PointCloudSample],
        pc_cfg,
    ) -> None:
        if self._utonia_geometry_logged:
            return
        self._utonia_geometry_logged = True
        with torch.no_grad():
            disp = disparity.detach().float()
            valid = disp > float(pc_cfg.get("disp_eps", 1e-3))
            valid_disp = disp[valid]
            if valid_disp.numel():
                disp_q = torch.quantile(valid_disp.cpu(), torch.tensor([0.05, 0.5, 0.95])).tolist()
            else:
                disp_q = []
            all_depth = (
                torch.cat([s.coord[:, 2].detach().cpu() for s in samples if int(s.coord.shape[0]) > 0], dim=0)
                if any(int(s.coord.shape[0]) > 0 for s in samples)
                else torch.empty(0)
            )
            depth_q = (
                torch.quantile(all_depth, torch.tensor([0.05, 0.5, 0.95])).tolist()
                if all_depth.numel()
                else []
            )
            point_counts = [int(s.coord.shape[0]) for s in samples]
            invalid = [round(float(s.invalid_ratio), 4) for s in samples]
            clamp = [round(float(s.clamp_ratio), 4) for s in samples]
            preview = samples[0].coord[:5].detach().cpu().tolist() if samples else []
        logger.info(
            "%s geometry diag once: disp_q05/50/95=%s depth_q05/50/95_m=%s "
            "point_counts=%s invalid_ratio=%s clamp_ratio=%s first_points=%s "
            "sim_gt_depth_compare=not_provided",
            self._utonia_label,
            disp_q,
            depth_q,
            point_counts,
            invalid,
            clamp,
            preview,
        )

    def _utonia_forward_transformed(self, point) -> torch.Tensor:
        """Frozen Utonia forward + multi-scale pooling-chain walk + inverse on ONE already-transformed
        point dict -> per-point features. Factored out so the per-sample loop AND the equality smoke
        share it (the transform's GridSample mode='train' is RANDOM, so a valid loop-vs-batched
        equality test must feed BOTH paths the SAME pre-transformed points)."""
        # Frozen Utonia: DeepSpeed bf16 recasts its weights every step, but the point-cloud input
        # is fp32 -> "mat1 and mat2 must have the same dtype (Float vs BFloat16)" crash at step 0.
        # Force fp32 each forward (proven FFS pattern in FFSCommon._compute_ffs_disparity).
        for key in list(point.keys()):
            if torch.is_tensor(point[key]):
                point[key] = point[key].to(next(self.utonia.parameters()).device)
        if next(self.utonia.parameters()).dtype != torch.float32:
            self.utonia.float()
        self.utonia.eval()
        if self.utonia.training:
            raise RuntimeError(f"{self._utonia_label} Utonia must stay in eval() for deterministic frozen features")
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            out = self.utonia(point)
        o = out
        while "pooling_parent" in o.keys():
            parent = o.pop("pooling_parent")
            inv = o.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, o.feat[inv]], dim=-1)
            o = parent
        per_point = o.feat[point["inverse"]] if "inverse" in point.keys() else o.feat
        if int(per_point.shape[-1]) != self._utonia_feature_dim:
            raise RuntimeError(
                f"{self._utonia_label} Utonia feature dim {int(per_point.shape[-1])} "
                f"!= expected {self._utonia_feature_dim}"
            )
        return per_point.detach().clone().float()

    def _extract_utonia_features_one(
        self,
        sample: PointCloudSample,
        deterministic: Optional[bool] = None,
    ) -> torch.Tensor:
        device = next(self.utonia.parameters()).device
        if int(sample.coord.shape[0]) == 0:
            return torch.zeros(0, self._utonia_feature_dim, device=device, dtype=torch.float32)
        point_np = {
            "coord": sample.coord.detach().cpu().numpy().astype(np.float32),
            "color": sample.color.detach().cpu().numpy().astype(np.float32),
            "normal": sample.normal.detach().cpu().numpy().astype(np.float32),
        }
        transform = self._select_utonia_transform(bool(deterministic))
        return self._utonia_forward_transformed(transform(point_np))

    def _utonia_forward_transformed_batched(self, points: List) -> List[torch.Tensor]:
        """Batched-offset Utonia forward over a list of already-transformed point dicts: ONE forward
        for the whole batch (vs per-sample loop). Returns per-dict per-point features in input order.
        Recipe = official demo/3_batch_forward: collate_fn batches via an offset tensor; after the
        same multi-scale walk, split back per sample by grid offset + each sample's own inverse."""
        import utonia
        device = next(self.utonia.parameters()).device
        inverses = [p["inverse"] for p in points]  # capture before collate consumes the dicts
        batched = utonia.data.collate_fn(points)
        for key in list(batched.keys()):
            if torch.is_tensor(batched[key]):
                batched[key] = batched[key].to(device)
        if next(self.utonia.parameters()).dtype != torch.float32:
            self.utonia.float()
        self.utonia.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            out = self.utonia(batched)
        o = out
        while "pooling_parent" in o.keys():
            parent = o.pop("pooling_parent")
            inv = o.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, o.feat[inv]], dim=-1)
            o = parent
        grid_feat = o.feat  # [total_grid_points, feat_dim] in collate (input) order
        if int(grid_feat.shape[-1]) != self._utonia_feature_dim:
            raise RuntimeError(
                f"{self._utonia_label} batched Utonia feature dim {int(grid_feat.shape[-1])} "
                f"!= expected {self._utonia_feature_dim}"
            )
        starts = [0] + o.offset.detach().cpu().tolist()  # cumulative grid-point counts per sample
        return [
            grid_feat[int(starts[j]):int(starts[j + 1])][inverses[j].to(grid_feat.device)].detach().clone().float()
            for j in range(len(points))
        ]

    def _extract_utonia_features_batched(self, samples: Sequence[PointCloudSample]) -> List[torch.Tensor]:
        """R9 batched path: ONE Utonia forward for the batch instead of the per-sample loop (32x).
        Splits back per sample. MUST match the loop ON IDENTICAL points (equality smoke gates it;
        the transform's GridSample mode='train' is random, so production augments per step = intended)."""
        device = next(self.utonia.parameters()).device
        results: List[torch.Tensor] = [
            torch.zeros(0, self._utonia_feature_dim, device=device, dtype=torch.float32) for _ in samples
        ]
        points = []
        out_pos: List[int] = []
        for i, sample in enumerate(samples):
            if int(sample.coord.shape[0]) == 0:
                continue
            point_np = {
                "coord": sample.coord.detach().cpu().numpy().astype(np.float32),
                "color": sample.color.detach().cpu().numpy().astype(np.float32),
                "normal": sample.normal.detach().cpu().numpy().astype(np.float32),
            }
            p = self._utonia_transform(point_np)
            if "inverse" not in p:
                raise RuntimeError(f"{self._utonia_label} transform has no 'inverse'; batched split needs it")
            points.append(p)
            out_pos.append(i)
        if not points:
            return results
        feats = self._utonia_forward_transformed_batched(points)
        for j, orig_i in enumerate(out_pos):
            results[orig_i] = feats[j]
        return results

    def compute_utonia_point_features(
        self,
        batch_images: List,
        pc_cfg,
        deterministic: Optional[bool] = None,
    ) -> UtoniaPointCloudBatch:
        deterministic = self._resolve_utonia_deterministic(pc_cfg, deterministic)
        disparity, samples = self._build_pointcloud_samples(batch_images, pc_cfg)
        if bool(pc_cfg.get("utonia_batched", False)):
            # R9 batched-offset: one forward for the batch. MUST match the loop (equality smoke gates it).
            point_features = self._extract_utonia_features_batched(samples)
        else:
            # Per-sample loop is the correctness oracle for Method #10 R9 (default-safe).
            point_features = [
                self._extract_utonia_features_one(sample, deterministic=deterministic)
                for sample in samples
            ]
        return UtoniaPointCloudBatch(samples=samples, point_features=point_features, disparity=disparity)

    def compute_utonia_grid(
        self,
        batch_images: List,
        pc_cfg,
        grid_hw: Tuple[int, int] = (8, 8),
        sample_ids: Optional[Sequence[Optional[Tuple[str, object, object]]]] = None,
        deterministic: Optional[bool] = None,
    ) -> torch.Tensor:
        cache_dir = _maybe_cache_dir(pc_cfg.get("utonia_cache_dir", self._utonia_cache_dir))
        deterministic = self._resolve_utonia_deterministic(pc_cfg, deterministic)
        if cache_dir is not None:
            self._force_utonia_deterministic_serialization()
            self._utonia_cache_batches += 1
        if cache_dir is not None and sample_ids is not None:
            if len(sample_ids) != len(batch_images):
                raise ValueError(
                    f"{self._utonia_label} sample_ids length {len(sample_ids)} "
                    f"!= batch size {len(batch_images)}"
                )
            normalized_ids = [_normalize_sample_id(sample_id) for sample_id in sample_ids]
            if all(sample_id is not None for sample_id in normalized_ids):
                rows: List[Optional[torch.Tensor]] = []
                missing: List[int] = []
                for i, sample_id in enumerate(normalized_ids):
                    cached = self._read_utonia_cache_row(
                        cache_dir,
                        sample_id,
                        grid_hw,
                        pc_cfg,
                    )
                    if cached is None:
                        rows.append(None)
                        missing.append(i)
                        self._utonia_cache_row_misses += 1
                    else:
                        rows.append(cached)
                        self._utonia_cache_hits += 1
                for i in missing:
                    rows[i] = self._compute_utonia_grid_live(
                        [batch_images[i]],
                        pc_cfg,
                        grid_hw=grid_hw,
                        deterministic=True,
                    )[0]
                assert all(row is not None for row in rows), "Utonia cache fill left an unfilled batch slot"
                self._log_utonia_cache_telemetry()
                return torch.stack(rows, dim=0)

            self._utonia_cache_whole_batch_live += len(batch_images)
            self._info_utonia_cache_live_batch(
                "partial_sample_ids" if any(sample_id is not None for sample_id in normalized_ids) else "missing_sample_ids"
            )
            self._log_utonia_cache_telemetry()
        elif cache_dir is not None:
            self._utonia_cache_whole_batch_live += len(batch_images)
            self._info_utonia_cache_live_batch("sample_ids_none")
            self._log_utonia_cache_telemetry()

        return self._compute_utonia_grid_live(
            batch_images,
            pc_cfg,
            grid_hw=grid_hw,
            deterministic=deterministic,
        )

    def _compute_utonia_grid_live(
        self,
        batch_images: List,
        pc_cfg,
        grid_hw: Tuple[int, int] = (8, 8),
        deterministic: Optional[bool] = None,
    ) -> torch.Tensor:
        batch = self.compute_utonia_point_features(batch_images, pc_cfg, deterministic=deterministic)
        grid, counts = pool_point_features_to_grid(
            samples=batch.samples,
            point_features=batch.point_features,
            grid_hw=grid_hw,
            image_width=int(pc_cfg.get("image_width", self.ffs_image_size)),
            image_height=int(pc_cfg.get("image_height", self.ffs_image_size)),
            feature_dim=self._utonia_feature_dim,
        )
        empty = counts == 0
        empty_ratio = float(empty.float().mean().detach().cpu()) if empty.numel() else 0.0
        self._utonia_last_empty_cell_ratio = empty_ratio
        logger.debug("%s Utonia grid empty_cell_ratio=%.4f", self._utonia_label, empty_ratio)
        return grid

    def _info_utonia_cache_live_batch(self, reason: str) -> None:
        count = int(getattr(self, "_utonia_cache_live_warn_count", 0))
        if count < 3 or count % 100 == 0:
            logger.info(
                "%s Utonia cache configured but serving whole batch live reason=%s; "
                "this is expected for eval/predict_action without dataset ids",
                self._utonia_label,
                reason,
            )
        self._utonia_cache_live_warn_count = count + 1

    def _log_utonia_cache_telemetry(self) -> None:
        batches = int(getattr(self, "_utonia_cache_batches", 0))
        if batches <= 3 or batches % 100 == 0:
            hits = int(getattr(self, "_utonia_cache_hits", 0))
            misses = int(getattr(self, "_utonia_cache_row_misses", 0))
            live = int(getattr(self, "_utonia_cache_whole_batch_live", 0))
            total_lookup = hits + misses
            hit_rate = float(hits) / float(total_lookup) if total_lookup else 0.0
            logger.info(
                "%s Utonia cache telemetry batches=%d hits=%d row_misses=%d "
                "whole_batch_live_samples=%d hit_rate=%.4f",
                self._utonia_label,
                batches,
                hits,
                misses,
                live,
                hit_rate,
            )

    def _load_utonia_cache_suite(
        self,
        cache_dir: str,
        suite: str,
        grid_hw: Tuple[int, int],
        pc_cfg,
    ) -> dict:
        cache_root = Path(cache_dir).expanduser()
        handle_key = (str(cache_root), str(suite))
        if handle_key in self._utonia_cache_handles:
            return self._utonia_cache_handles[handle_key]

        handles = validate_utonia_cache(
            cache_root,
            [str(suite)],
            dataset_or_none=None,
            ffs_sha256=getattr(self, "_ffs_actual_sha256", None),
            utonia_sha256=getattr(self, "_utonia_actual_sha256", None),
            pc_cfg=pc_cfg,
            grid_hw=grid_hw,
            require_all_done=True,
        )
        handle = handles[str(suite)]
        self._utonia_cache_handles[handle_key] = handle
        logger.info(
            "%s opened validated Utonia cache suite=%s rows=%d path=%s",
            self._utonia_label,
            suite,
            int(handle["grid"].shape[0]),
            cache_root / str(suite) / "grid.f16.npy",
        )
        return handle

    def _warn_utonia_cache_miss(self, sample_id, reason: str) -> None:
        count = int(getattr(self, "_utonia_cache_miss_warn_count", 0))
        if count < 3 or count % 100 == 0:
            logger.warning(
                "%s Utonia cache miss sample_id=%s reason=%s; falling back to live deterministic compute",
                self._utonia_label,
                sample_id,
                reason,
            )
        self._utonia_cache_miss_warn_count = count + 1

    def _read_utonia_cache_row(
        self,
        cache_dir: str,
        sample_id,
        grid_hw: Tuple[int, int],
        pc_cfg,
    ) -> Optional[torch.Tensor]:
        suite, traj_id, base_index = sample_id
        handle = self._load_utonia_cache_suite(cache_dir, suite, grid_hw, pc_cfg)
        key = (traj_id, base_index)
        row = handle["index"].get(key)
        if row is None:
            self._warn_utonia_cache_miss(sample_id, "key_not_found")
            return None
        grid = handle["grid"]
        row = int(row)
        if row < 0 or row >= int(grid.shape[0]):
            raise RuntimeError(
                f"{self._utonia_label} cache row out of range for sample_id={sample_id}: "
                f"row={row} rows={int(grid.shape[0])}"
            )
        done = handle["done"]
        if not bool(done[row]):
            self._warn_utonia_cache_miss(sample_id, "row_not_done")
            return None
        arr = np.array(grid[row], copy=True)
        device = next(self.utonia.parameters()).device
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def compute_utonia_padded_features(self, batch_images: List, pc_cfg) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.compute_utonia_point_features(batch_images, pc_cfg)
        return pad_point_features(batch.point_features, feature_dim=self._utonia_feature_dim)


class _BenchModel(UtoniaPointCloudMixin, QwenGR00TNet0FFSMixin, nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.num_cameras = 2
        self._init_frozen_ffs_for_disparity(cfg, "[UtoniaCostBench]")
        self._init_frozen_utonia(cfg, "[UtoniaCostBench]")
        self.cfg = cfg

    def forward(self, batch_images: List) -> UtoniaPointCloudBatch:
        return self.compute_utonia_point_features(batch_images, self.cfg)


def _synthetic_pair(seed: int, size: int = 256):
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    left = np.zeros((size, size, 3), dtype=np.uint8)
    left[..., 0] = (xx * 3 + yy + rng.integers(0, 11, size=(size, size))) % 256
    left[..., 1] = (xx + yy * 2 + 41) % 256
    left[..., 2] = (((xx // 9) * 29 + (yy // 7) * 31) % 256).astype(np.uint8)
    right = np.roll(left, shift=-6, axis=1)
    return [Image.fromarray(right, mode="RGB"), Image.fromarray(left, mode="RGB")]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def run_cost_benchmark(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("UTONIA_COST_BENCH_FAIL reason=no_cuda")
        return 2
    cfg = {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256 or None,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.image_size,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "utonia_ckpt_path": args.utonia_ckpt_path,
        "utonia_expected_sha256": args.utonia_expected_sha256 or None,
        "utonia_scale": args.utonia_scale,
        "utonia_enable_flash": bool(args.utonia_enable_flash),
        "fovy_degrees": args.fovy_degrees,
        "baseline_m": args.baseline_m,
        "image_width": args.image_size,
        "image_height": args.image_size,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "disp_eps": args.disp_eps,
    }
    batch_sizes = sorted(set(_parse_csv_ints(args.batch_sizes)))
    strides = [args.backproject_stride, *_parse_csv_ints(args.fallback_strides)]
    strides = list(dict.fromkeys(max(int(s), 1) for s in strides))
    examples = [_synthetic_pair(1000 + i, args.image_size) for i in range(max(batch_sizes))]

    for stride in strides:
        cfg["backproject_stride"] = stride
        try:
            model = _BenchModel(cfg).cuda().eval()
            results = []
            ok = True
            for bs in batch_sizes:
                batch = examples[:bs]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    model(batch)
                torch.cuda.synchronize()
                start = time.perf_counter()
                counts: List[int] = []
                for _ in range(max(int(args.steps), 1)):
                    out = model(batch)
                    counts = [int(f.shape[0]) for f in out.point_features]
                torch.cuda.synchronize()
                sec = (time.perf_counter() - start) / float(max(int(args.steps), 1))
                peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                results.append((bs, sec, peak_gb, counts))
                print(
                    f"UTONIA_COST_BENCH stride={stride} bs={bs} sec_per_step={sec:.4f} "
                    f"peak_mem_gb={peak_gb:.3f} point_count_min={min(counts)} point_count_max={max(counts)}"
                )
                if args.max_sec_per_step > 0 and sec > float(args.max_sec_per_step):
                    ok = False
            if ok:
                print(f"UTONIA_COST_BENCH_SELECTED_STRIDE={stride}")
                print("UTONIA_COST_BENCH_OK")
                return 0
            print(
                f"UTONIA_COST_BENCH stride={stride} exceeded max_sec_per_step={args.max_sec_per_step}; "
                "trying fallback stride"
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"UTONIA_COST_BENCH stride={stride} oom; trying fallback stride")
            torch.cuda.empty_cache()
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
    print("UTONIA_COST_BENCH_FAIL reason=no_stride_met_gate")
    return 3


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("benchmark", help="R5 GPU pre-train cost gate")
    bench.add_argument("--ffs-model-path", required=True)
    bench.add_argument("--utonia-ckpt-path", required=True)
    bench.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", ""))
    bench.add_argument("--ffs-expected-sha256", default="")
    bench.add_argument("--utonia-expected-sha256", default="")
    bench.add_argument("--utonia-scale", type=float, default=4.0)
    bench.add_argument("--utonia-enable-flash", action="store_true")
    bench.add_argument("--image-size", type=int, default=256)
    bench.add_argument("--fovy-degrees", type=float, default=45.0)
    bench.add_argument("--baseline-m", type=float, default=0.06)
    bench.add_argument("--depth-min", type=float, default=0.05)
    bench.add_argument("--depth-max", type=float, default=3.0)
    bench.add_argument("--disp-eps", type=float, default=1e-3)
    bench.add_argument("--backproject-stride", type=int, default=4)
    bench.add_argument("--fallback-strides", default="8,16")
    bench.add_argument("--batch-sizes", default="1,4,8")
    bench.add_argument("--steps", type=int, default=2)
    bench.add_argument("--max-sec-per-step", type=float, default=10.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if getattr(args, "ffs_repo_dir", "") and args.ffs_repo_dir not in sys.path:
        sys.path.insert(0, args.ffs_repo_dir)
    if args.cmd == "benchmark":
        return run_cost_benchmark(args)
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
