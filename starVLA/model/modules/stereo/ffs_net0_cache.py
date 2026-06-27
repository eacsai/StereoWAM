"""Pooled Fast-FoundationStereo net[0] cache for depth-token training."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

FFS_NET0_CACHE_REPRESENTATION = "ffs_net0_pooled"
FFS_NET0_CACHE_DTYPE = "float32"
FFS_NET0_CACHE_FILENAME = "net0_pooled.f32.npy"
FFS_NET0_CACHE_IMAGE_SHA_FILENAME = "image_sha256.s64.npy"
FFS_NET0_CACHE_SCHEMA_VERSION = 4
FFS_STARTUP_CANDIDATE_ROWS_PER_SUITE = 32


@contextlib.contextmanager
def ffs_tf32_disabled():
    """Run FFS fp32 forwards with TF32 disabled, restoring caller state after."""
    old_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    old_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        if old_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
        if old_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = old_cudnn


def _maybe_cache_dir(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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


def _image_array(image) -> np.ndarray:
    if torch.is_tensor(image):
        return image.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(image))


def _image_sha(image) -> str:
    import hashlib

    arr = _image_array(image)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _image_shape_key(images) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    return tuple((tuple(_image_array(img).shape), str(_image_array(img).dtype)) for img in images)


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _decode_image_sha_cell(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("ascii")
    return str(value)


def _decode_image_sha_row(
    image_sha,
    row: int,
    *,
    suite: str,
    key: Tuple[object, object],
    required: bool,
) -> Optional[List[str]]:
    if image_sha is None:
        if required:
            raise RuntimeError(
                f"FFS net0 cache missing per-row image SHA sidecar for suite={suite} key={key!r}"
            )
        return None
    values = [_decode_image_sha_cell(x) for x in np.asarray(image_sha[int(row)]).reshape(-1).tolist()]
    if all(value == "" for value in values):
        if required:
            raise RuntimeError(
                f"FFS net0 cache missing per-row image SHA for suite={suite} row={row} key={key!r}"
            )
        return None
    if not values or not all(_is_sha256(value) for value in values):
        raise RuntimeError(
            f"FFS net0 cache invalid per-row image SHA for suite={suite} row={row} key={key!r}: {values!r}"
        )
    return values


def _validate_image_sha_array(
    image_sha,
    *,
    suite: str,
    index: dict,
    count: int,
    done,
    require_all_done: bool,
    expected_num_images: int,
) -> None:
    if image_sha is None:
        if require_all_done or bool(np.asarray(done, dtype=np.bool_).any()):
            raise RuntimeError(
                f"FFS net0 cache missing required {FFS_NET0_CACHE_IMAGE_SHA_FILENAME} for suite={suite}"
            )
        return
    if image_sha.dtype.kind != "S" or int(image_sha.dtype.itemsize) != 64:
        raise RuntimeError(
            f"FFS net0 cache image SHA sidecar dtype must be S64 for suite={suite}; got {image_sha.dtype}"
        )
    expected_shape = (int(count), int(expected_num_images))
    if tuple(int(x) for x in image_sha.shape) != expected_shape:
        raise RuntimeError(
            f"FFS net0 cache image SHA sidecar shape {tuple(image_sha.shape)} != expected {expected_shape} "
            f"for suite={suite}"
        )
    for key, row in index.items():
        row = int(row)
        required = bool(require_all_done) or bool(done[row])
        _decode_image_sha_row(image_sha, row, suite=suite, key=key, required=required)


def _all_steps_sha256(all_steps: Sequence[Tuple[object, object]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for traj_id, base_index in all_steps:
        h.update(repr((_cache_key_part(traj_id), _cache_key_part(base_index))).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _all_steps_from_index(index: dict, count: int) -> List[Tuple[object, object]]:
    rows: List[Optional[Tuple[object, object]]] = [None] * int(count)
    for key, row in index.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise RuntimeError(f"FFS net0 cache index key must be (traj_id,base_index), got {key!r}")
        row = int(row)
        if row < 0 or row >= int(count):
            raise RuntimeError(f"FFS net0 cache index row out of range: key={key!r} row={row} count={count}")
        if rows[row] is not None:
            raise RuntimeError(f"FFS net0 cache duplicate row {row}: {rows[row]!r} and {key!r}")
        rows[row] = (_cache_key_part(key[0]), _cache_key_part(key[1]))
    if any(row is None for row in rows):
        missing = [i for i, row in enumerate(rows) if row is None][:8]
        raise RuntimeError(f"FFS net0 cache index has holes at rows={missing}")
    return [row for row in rows if row is not None]


def _all_steps_from_dataset(dataset) -> List[Tuple[object, object]]:
    return [
        (_cache_key_part(traj_id), _cache_key_part(base_index))
        for traj_id, base_index in dataset.all_steps
    ]


def _require_dataset_steps_covered(
    *,
    suite: str,
    dataset_steps: Sequence[Tuple[object, object]],
    index_steps: Sequence[Tuple[object, object]],
) -> None:
    """Require the cache index to cover every training row, while allowing extra rows."""
    dataset_set = set(dataset_steps)
    index_set = set(index_steps)
    missing = sorted(dataset_set - index_set, key=repr)
    if missing:
        raise RuntimeError(
            f"FFS net0 cache coverage gap for suite={suite}: "
            f"missing_training_keys_first={missing[:8]} "
            f"missing={len(missing)} dataset_unique={len(dataset_set)} cache_unique={len(index_set)}"
        )


def _dataset_map(dataset_or_none) -> Dict[str, object]:
    if dataset_or_none is None:
        return {}
    if isinstance(dataset_or_none, Mapping):
        return {str(k): v for k, v in dataset_or_none.items() if v is not None}
    datasets = getattr(dataset_or_none, "datasets", None)
    if datasets is not None:
        return {str(single.dataset_name): single for single in datasets}
    name = getattr(dataset_or_none, "dataset_name", None)
    if name is not None:
        return {str(name): dataset_or_none}
    return {}


def _dataset_for_cache_suite(dataset_or_none, suite: str):
    return _dataset_map(dataset_or_none).get(str(suite))


def _require_cache_meta_value(meta: dict, suite: str, key: str, expected) -> None:
    if expected is None:
        raise RuntimeError(
            f"FFS net0 cache cannot validate meta {key} for suite={suite}; "
            "the loaded checkpoint SHA was not recorded"
        )
    cached = meta.get(key)
    if cached is None:
        raise RuntimeError(f"FFS net0 cache meta missing {key} for suite={suite}")
    if str(cached) != str(expected):
        raise RuntimeError(
            f"FFS net0 cache meta {key} mismatch for suite={suite}: cached={cached} expected={expected}"
        )


def _require_optional_meta_value(meta: dict, suite: str, key: str, expected) -> None:
    if expected is None:
        return
    cached = meta.get(key)
    if cached is None:
        raise RuntimeError(f"FFS net0 cache meta missing {key} for suite={suite}")
    if str(cached) != str(expected):
        raise RuntimeError(
            f"FFS net0 cache meta {key} mismatch for suite={suite}: cached={cached} expected={expected}"
        )


def _expected_view_constants(ffs_cfg) -> dict:
    if any(_cfg_get(ffs_cfg, key, None) is not None for key in ("primary_idx", "right_view_idx", "primary_cam_id")):
        return {
            "num_cameras": int(_cfg_get(ffs_cfg, "num_cameras", 2)),
            "primary_idx": int(_cfg_get(ffs_cfg, "primary_idx", 1)),
            "right_view_idx": int(_cfg_get(ffs_cfg, "right_view_idx", 0)),
            "primary_cam_id": int(_cfg_get(ffs_cfg, "primary_cam_id", 1)),
            "ffs_feature_source": str(_cfg_get(ffs_cfg, "ffs_feature_source", "gru_hidden")),
            "gru_hidden_dim": int(_cfg_get(ffs_cfg, "gru_hidden_dim", 16)),
        }
    return {
        "num_cameras": int(_cfg_get(ffs_cfg, "num_cameras", 2)),
        "left_ref_idx": int(_cfg_get(ffs_cfg, "left_ref_idx", 1)),
        "primary_view_idx": int(_cfg_get(ffs_cfg, "primary_view_idx", 0)),
        "inject_cam_id": int(_cfg_get(ffs_cfg, "inject_cam_id", 1)),
        "ffs_feature_source": str(_cfg_get(ffs_cfg, "ffs_feature_source", "gru_hidden")),
        "gru_hidden_dim": int(_cfg_get(ffs_cfg, "gru_hidden_dim", 16)),
    }


def _view_convention_fingerprint(ffs_cfg) -> dict:
    legacy = any(_cfg_get(ffs_cfg, key, None) is not None for key in ("primary_idx", "right_view_idx", "primary_cam_id"))
    if legacy:
        return {
            "view_order": ["right_view", "primary"],
            "reference_view": "legacy_primary_after_unrotate",
            "net0_frame": "legacy_primary_rotated",
            "unrotate": True,
        }
    return {
        "view_order": ["primary", "left_view"],
        "reference_view": "left_view",
        "net0_frame": "left_view",
        "unrotate": False,
    }


def _image_preprocessing_meta(ffs_cfg) -> dict:
    meta = {
        "image_size": int(_cfg_get(ffs_cfg, "ffs_image_size", 256)),
        "resize": "torchvision.transforms.Resize",
        "interpolation": "bilinear",
        "input_scale": "image_tensor_float32_times_255",
        "dtype": "float32",
    }
    meta.update(_view_convention_fingerprint(ffs_cfg))
    return meta


def collect_ffs_runtime_fingerprint(*, ffs_model_sha256: Optional[str], ffs_cfg) -> dict:
    gpu_name = None
    gpu_cc = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_cc = ".".join(str(x) for x in torch.cuda.get_device_capability(idx))
    return {
        "schema_version": FFS_NET0_CACHE_SCHEMA_VERSION,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": None if torch.backends.cudnn.version() is None else str(torch.backends.cudnn.version()),
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_cc,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "ffs_model_sha256": None if ffs_model_sha256 is None else str(ffs_model_sha256),
        "image_preprocessing": _image_preprocessing_meta(ffs_cfg),
        "cache_dtype": FFS_NET0_CACHE_DTYPE,
    }


def _warn_runtime_fingerprint_mismatch(meta: dict, suite: str, ffs_cfg) -> None:
    cached = meta.get("runtime_fingerprint")
    if not isinstance(cached, dict):
        logger.warning("FFS net0 cache suite=%s has no runtime_fingerprint meta", suite)
        return
    current = collect_ffs_runtime_fingerprint(
        ffs_model_sha256=meta.get("ffs_model_sha256"),
        ffs_cfg=ffs_cfg,
    )
    warn_keys = (
        "schema_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "gpu_name",
        "gpu_compute_capability",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_deterministic",
        "torch_deterministic_algorithms",
        "ffs_model_sha256",
        "image_preprocessing",
        "cache_dtype",
    )
    mismatches = {
        key: (cached.get(key), current.get(key))
        for key in warn_keys
        if cached.get(key) != current.get(key)
    }
    if mismatches:
        logger.warning(
            "FFS net0 cache runtime fingerprint mismatch for suite=%s; V1 equality check is authoritative. mismatches=%s",
            suite,
            mismatches,
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
            f"FFS net0 cache meta image_sha256_samples must be a non-empty list for suite={suite}"
        )

    image_sha_by_key: Dict[Tuple[object, object], List[str]] = {}
    for pos, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError(f"FFS net0 cache image SHA sample #{pos} must be a dict for suite={suite}")
        if "traj_id" not in sample or "base_index" not in sample:
            raise RuntimeError(f"FFS net0 cache image SHA sample #{pos} missing traj_id/base_index for suite={suite}")
        key = (_cache_key_part(sample["traj_id"]), _cache_key_part(sample["base_index"]))
        if key not in index:
            raise RuntimeError(f"FFS net0 cache image SHA sample key={key!r} missing from index for suite={suite}")

        row = sample.get("row", index[key])
        try:
            row = int(row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"FFS net0 cache image SHA sample key={key!r} has invalid row={row!r}") from exc
        if row < 0 or row >= int(count):
            raise RuntimeError(f"FFS net0 cache image SHA sample key={key!r} row={row} out of range count={count}")
        if int(index[key]) != row:
            raise RuntimeError(
                f"FFS net0 cache image SHA sample key={key!r} row mismatch: meta={row} index={index[key]}"
            )

        shas = sample.get("image_sha256")
        if not isinstance(shas, list) or not shas or not all(_is_sha256(x) for x in shas):
            raise RuntimeError(
                f"FFS net0 cache image SHA sample key={key!r} must contain non-empty 64-hex image_sha256 list"
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
                    f"FFS net0 cache image SHA sample row={row} key mismatch for suite={suite}: "
                    f"meta={key!r} dataset={live_key!r}"
                )
            actual = [_image_sha(img) for img in live_sample["image"]]
            if actual != shas:
                raise RuntimeError(
                    f"FFS net0 cache image SHA mismatch for suite={suite} row={row} key={key!r}: "
                    f"cached={shas} live={actual}"
                )
    return image_sha_by_key


def build_ffs_net0_cache_meta(
    *,
    suite: str,
    count: int,
    all_steps_sha256: str,
    image_sha256_samples: list[dict],
    raw_observed_shape: Sequence[int],
    pooled_shape: Sequence[int],
    pool_hw: int,
    num_depth_tokens: int,
    ffs_cfg,
    ffs_model_path: str,
    ffs_model_sha256: str,
    data_root_dir: str,
    data_mix: str,
    video_backend: str,
    delete_pause_frame: Optional[bool] = None,
) -> dict:
    """Build the fail-closed metadata record shared by precompute and smoke."""
    runtime_fingerprint = collect_ffs_runtime_fingerprint(
        ffs_model_sha256=ffs_model_sha256,
        ffs_cfg=ffs_cfg,
    )
    view_fingerprint = _view_convention_fingerprint(ffs_cfg)
    return {
        "schema_version": FFS_NET0_CACHE_SCHEMA_VERSION,
        "suite": str(suite),
        "representation": FFS_NET0_CACHE_REPRESENTATION,
        "dtype": FFS_NET0_CACHE_DTYPE,
        **view_fingerprint,
        "dims": [int(x) for x in pooled_shape],
        "raw_observed_shape": [int(x) for x in raw_observed_shape],
        "pooled_shape": [int(x) for x in pooled_shape],
        "pool_hw": int(pool_hw),
        "num_depth_tokens": int(num_depth_tokens),
        "count": int(count),
        "all_steps_sha256": str(all_steps_sha256),
        "image_size": int(_cfg_get(ffs_cfg, "ffs_image_size", 256)),
        "image_preprocessing": _image_preprocessing_meta(ffs_cfg),
        "data_root_dir": str(data_root_dir),
        "data_mix": str(data_mix),
        "video_backend": str(video_backend),
        "delete_pause_frame": None if delete_pause_frame is None else bool(delete_pause_frame),
        "ffs_model_path": str(ffs_model_path),
        "ffs_model_sha256": str(ffs_model_sha256),
        "runtime_fingerprint": runtime_fingerprint,
        "view_constants": _expected_view_constants(ffs_cfg),
        "image_sha256_samples": image_sha256_samples,
    }


def validate_ffs_net0_cache(
    cache_dir: str | Path,
    suites: Sequence[str],
    dataset_or_none=None,
    ffs_sha256: Optional[str] = None,
    ffs_cfg=None,
    pool_hw: Optional[int] = None,
    num_depth_tokens: Optional[int] = None,
    require_all_done: bool = True,
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
    expected_delete_pause_frame: Optional[bool] = None,
) -> Dict[str, dict]:
    """Validate and open pooled FFS net[0] cache suites.

    This intentionally fails closed. Cache hits are allowed only when the stored
    representation, dtype, geometry, checkpoint SHA, dataset coverage, index hash,
    and per-row image-SHA sidecar are compatible with the running configuration.
    Hit/miss byte-equivalence assumes precompute and training both use CUDA pooling.
    """
    if ffs_cfg is None:
        raise RuntimeError("validate_ffs_net0_cache requires ffs_cfg")

    cache_root = Path(cache_dir).expanduser()
    expected_c = int(_cfg_get(ffs_cfg, "gru_hidden_dim", 16))
    expected_pool_hw = int(pool_hw if pool_hw is not None else _cfg_get(ffs_cfg, "pool_hw", 4))
    expected_num_depth_tokens = int(
        num_depth_tokens if num_depth_tokens is not None else _cfg_get(
            ffs_cfg, "num_depth_tokens", expected_pool_hw * expected_pool_hw
        )
    )
    expected_shape = (expected_c, expected_pool_hw, expected_pool_hw)
    expected_view_constants = _expected_view_constants(ffs_cfg)
    expected_view_fingerprint = _view_convention_fingerprint(ffs_cfg)
    expected_image_size = int(_cfg_get(ffs_cfg, "ffs_image_size", 256))
    expected_num_images = int(_cfg_get(ffs_cfg, "num_cameras", 2))

    handles: Dict[str, dict] = {}
    for suite in suites:
        suite = str(suite)
        suite_dir = cache_root / suite
        net0_path = suite_dir / FFS_NET0_CACHE_FILENAME
        image_sha_path = suite_dir / FFS_NET0_CACHE_IMAGE_SHA_FILENAME
        index_path = suite_dir / "index.pkl"
        meta_path = suite_dir / "meta.json"
        done_path = suite_dir / "done.npy"
        missing_files = []
        if not suite_dir.is_dir():
            missing_files.append(str(suite_dir))
        for path in (net0_path, image_sha_path, index_path, meta_path, done_path):
            if not path.is_file():
                missing_files.append(str(path))
        if missing_files:
            raise FileNotFoundError(f"FFS net0 cache structurally absent for suite={suite}: {missing_files}")

        with open(index_path, "rb") as fh:
            index = pickle.load(fh)
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        net0 = np.load(net0_path, mmap_mode="r")
        done = np.load(done_path, mmap_mode="r")
        image_sha = np.load(image_sha_path, mmap_mode="r")

        if tuple(net0.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"FFS net0 cache shape tail {tuple(net0.shape[1:])} != expected {expected_shape} "
                f"for {net0_path}"
            )
        count = int(net0.shape[0])
        if tuple(done.shape) != (count,):
            raise RuntimeError(
                f"FFS net0 cache done.npy shape {tuple(done.shape)} != rows {(count,)} for {done_path}"
            )
        if done.dtype != np.dtype(np.bool_):
            raise RuntimeError(f"FFS net0 cache done.npy dtype must be bool for suite={suite}; got {done.dtype}")
        _validate_image_sha_array(
            image_sha,
            suite=suite,
            index=index,
            count=count,
            done=done,
            require_all_done=require_all_done,
            expected_num_images=expected_num_images,
        )
        if require_all_done and not bool(np.all(done)):
            incomplete = int(count - int(np.asarray(done, dtype=np.bool_).sum()))
            raise RuntimeError(f"FFS net0 cache suite={suite} is incomplete: {incomplete}/{count} rows done=False")

        meta_count = meta.get("count")
        try:
            meta_count_int = int(meta_count)
        except (TypeError, ValueError):
            meta_count_int = -1
        if meta_count_int != count or len(index) != count:
            raise RuntimeError(
                f"FFS net0 cache count mismatch for suite={suite}: "
                f"meta_count={meta_count} index={len(index)} net0={count}"
            )
        if int(meta.get("schema_version", -1)) != FFS_NET0_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"FFS net0 cache schema_version mismatch for suite={suite}: "
                f"meta={meta.get('schema_version')} expected={FFS_NET0_CACHE_SCHEMA_VERSION}"
            )
        if str(meta.get("representation")) != FFS_NET0_CACHE_REPRESENTATION:
            raise RuntimeError(
                f"FFS net0 cache representation mismatch for suite={suite}: "
                f"meta={meta.get('representation')} expected={FFS_NET0_CACHE_REPRESENTATION}"
            )
        if str(meta.get("dtype")) != FFS_NET0_CACHE_DTYPE or net0.dtype != np.float32:
            raise RuntimeError(
                f"FFS net0 cache dtype mismatch for suite={suite}: meta={meta.get('dtype')} net0={net0.dtype}"
            )
        if meta.get("dims") != list(expected_shape) or meta.get("pooled_shape") != list(expected_shape):
            raise RuntimeError(
                f"FFS net0 cache pooled shape mismatch for suite={suite}: "
                f"dims={meta.get('dims')} pooled_shape={meta.get('pooled_shape')} expected={list(expected_shape)}"
            )
        raw_shape = meta.get("raw_observed_shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 3
            or int(raw_shape[0]) != expected_c
            or int(raw_shape[1]) <= 0
            or int(raw_shape[2]) <= 0
        ):
            raise RuntimeError(
                f"FFS net0 cache raw_observed_shape invalid for suite={suite}: "
                f"{raw_shape!r}; expected [C,H,W] with C={expected_c}"
            )
        if int(meta.get("pool_hw", -1)) != expected_pool_hw:
            raise RuntimeError(
                f"FFS net0 cache pool_hw mismatch for suite={suite}: "
                f"meta={meta.get('pool_hw')} expected={expected_pool_hw}"
            )
        if int(meta.get("num_depth_tokens", -1)) != expected_num_depth_tokens:
            raise RuntimeError(
                f"FFS net0 cache num_depth_tokens mismatch for suite={suite}: "
                f"meta={meta.get('num_depth_tokens')} expected={expected_num_depth_tokens}"
            )
        if int(meta.get("image_size", -1)) != expected_image_size:
            raise RuntimeError(
                f"FFS net0 cache image_size mismatch for suite={suite}: "
                f"meta={meta.get('image_size')} expected={expected_image_size}"
            )
        if meta.get("image_preprocessing") != _image_preprocessing_meta(ffs_cfg):
            raise RuntimeError(
                f"FFS net0 cache image_preprocessing mismatch for suite={suite}: "
                f"meta={meta.get('image_preprocessing')} expected={_image_preprocessing_meta(ffs_cfg)}"
            )
        for key, expected_value in expected_view_fingerprint.items():
            if meta.get(key) != expected_value:
                raise RuntimeError(
                    f"FFS net0 cache {key} mismatch for suite={suite}: "
                    f"meta={meta.get(key)!r} expected={expected_value!r}"
                )
        if meta.get("view_constants") != expected_view_constants:
            keys = sorted(set((meta.get("view_constants") or {}).keys()) | set(expected_view_constants.keys()))
            cached_view = meta.get("view_constants") or {}
            mismatches = {
                key: (cached_view.get(key), expected_view_constants.get(key))
                for key in keys
                if cached_view.get(key) != expected_view_constants.get(key)
            }
            raise RuntimeError(f"FFS net0 cache view_constants mismatch for suite={suite}: {mismatches}")
        if str(meta.get("suite")) != suite:
            raise RuntimeError(f"FFS net0 cache suite meta mismatch: dir={suite} meta={meta.get('suite')}")

        for required_key in ("data_root_dir", "data_mix", "video_backend", "delete_pause_frame", "all_steps_sha256"):
            if required_key not in meta:
                raise RuntimeError(f"FFS net0 cache meta missing {required_key} for suite={suite}")
        if meta.get("delete_pause_frame") is not None and not isinstance(meta.get("delete_pause_frame"), bool):
            raise RuntimeError(
                f"FFS net0 cache meta delete_pause_frame must be a bool/null for suite={suite}; "
                f"got {meta.get('delete_pause_frame')!r}"
            )

        index_all_steps = _all_steps_from_index(index, count)
        index_sha = _all_steps_sha256(index_all_steps)
        if meta.get("all_steps_sha256") != index_sha:
            raise RuntimeError(
                f"FFS net0 cache all_steps_sha256 mismatch for suite={suite}: "
                f"meta={meta.get('all_steps_sha256')} index={index_sha}"
            )

        dataset = _dataset_for_cache_suite(dataset_or_none, suite)
        if dataset is not None:
            dataset_steps = _all_steps_from_dataset(dataset)
            # Precompute may intentionally cover the full unfiltered suite, while
            # scene-flow / gt_only training sees a filtered subset. Row lookup is by
            # (traj_id, base_index), so ordering and cache extras are harmless; only
            # missing training keys are real coverage failures. The strict
            # meta-vs-index hash above still guards the cache index contents.
            _require_dataset_steps_covered(
                suite=suite,
                dataset_steps=dataset_steps,
                index_steps=index_all_steps,
            )
            _require_optional_meta_value(meta, suite, "video_backend", getattr(dataset, "video_backend", None))
            _require_optional_meta_value(meta, suite, "delete_pause_frame", getattr(dataset, "delete_pause_frame", None))

        _require_cache_meta_value(meta, suite, "ffs_model_sha256", ffs_sha256)
        _require_optional_meta_value(meta, suite, "data_root_dir", expected_data_root)
        _require_optional_meta_value(meta, suite, "data_mix", expected_data_mix)
        _require_optional_meta_value(meta, suite, "video_backend", expected_video_backend)
        _require_optional_meta_value(meta, suite, "delete_pause_frame", expected_delete_pause_frame)
        _warn_runtime_fingerprint_mismatch(meta, suite, ffs_cfg)

        image_sha_samples_by_key = _validate_image_sha_samples(
            suite=suite,
            meta=meta,
            index=index,
            count=count,
            dataset=dataset,
        )
        for key, sample_shas in image_sha_samples_by_key.items():
            row = int(index[key])
            row_shas = _decode_image_sha_row(
                image_sha,
                row,
                suite=suite,
                key=key,
                required=bool(require_all_done) or bool(done[row]),
            )
            if row_shas is not None and row_shas != sample_shas:
                raise RuntimeError(
                    f"FFS net0 cache image SHA sidecar/meta mismatch for suite={suite} row={row} key={key!r}: "
                    f"sidecar={row_shas} meta={sample_shas}"
                )
        handles[suite] = {
            "net0": net0,
            "index": index,
            "meta": meta,
            "done": done,
            "image_sha": image_sha,
            "image_sha_samples_by_key": image_sha_samples_by_key,
        }
    return handles


@dataclass
class FFSNet0CacheBatchStats:
    hits: int = 0
    row_misses: int = 0
    whole_batch_live: int = 0
    live_reason: Optional[str] = None
    miss_details: List[Tuple[int, Tuple[str, object, object], str]] = field(default_factory=list)


def _load_ffs_net0_cache_suite(
    *,
    cache_dir: str,
    suite: str,
    handles: Dict[Tuple[str, str, int, int], dict],
    ffs_sha256: Optional[str],
    ffs_cfg,
    pool_hw: int,
    num_depth_tokens: int,
    dataset_or_none=None,
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
    expected_delete_pause_frame: Optional[bool] = None,
) -> dict:
    cache_root = Path(cache_dir).expanduser()
    handle_key = (str(cache_root), str(suite), int(pool_hw), int(num_depth_tokens))
    if handle_key in handles:
        return handles[handle_key]
    suite_handles = validate_ffs_net0_cache(
        cache_root,
        [str(suite)],
        dataset_or_none=dataset_or_none,
        ffs_sha256=ffs_sha256,
        ffs_cfg=ffs_cfg,
        pool_hw=pool_hw,
        num_depth_tokens=num_depth_tokens,
        require_all_done=True,
        expected_data_root=expected_data_root,
        expected_data_mix=expected_data_mix,
        expected_video_backend=expected_video_backend,
        expected_delete_pause_frame=expected_delete_pause_frame,
    )
    handle = suite_handles[str(suite)]
    handles[handle_key] = handle
    logger.info(
        "opened validated FFS net0 cache suite=%s rows=%d path=%s",
        suite,
        int(handle["net0"].shape[0]),
        cache_root / str(suite) / FFS_NET0_CACHE_FILENAME,
    )
    return handle


def _read_ffs_net0_cache_row(
    *,
    cache_dir: str,
    sample_id,
    handles: Dict[Tuple[str, str, int, int], dict],
    ffs_sha256: Optional[str],
    ffs_cfg,
    pool_hw: int,
    num_depth_tokens: int,
    device,
    live_images,
    dataset_or_none=None,
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
    expected_delete_pause_frame: Optional[bool] = None,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if live_images is None:
        raise RuntimeError("FFS net0 cache row read requires live_images for mandatory image-SHA validation")
    suite, traj_id, base_index = sample_id
    handle = _load_ffs_net0_cache_suite(
        cache_dir=cache_dir,
        suite=str(suite),
        handles=handles,
        ffs_sha256=ffs_sha256,
        ffs_cfg=ffs_cfg,
        pool_hw=pool_hw,
        num_depth_tokens=num_depth_tokens,
        dataset_or_none=dataset_or_none,
        expected_data_root=expected_data_root,
        expected_data_mix=expected_data_mix,
        expected_video_backend=expected_video_backend,
        expected_delete_pause_frame=expected_delete_pause_frame,
    )
    key = (traj_id, base_index)
    row = handle["index"].get(key)
    if row is None:
        return None, "key_not_found"
    net0 = handle["net0"]
    row = int(row)
    if row < 0 or row >= int(net0.shape[0]):
        raise RuntimeError(
            f"FFS net0 cache row out of range for sample_id={sample_id}: row={row} rows={int(net0.shape[0])}"
        )
    done = handle["done"]
    if not bool(done[row]):
        return None, "row_not_done"
    cached_shas = _decode_image_sha_row(
        handle.get("image_sha"),
        row,
        suite=str(suite),
        key=key,
        required=True,
    )
    live_shas = [_image_sha(img) for img in live_images]
    if live_shas != cached_shas:
        return None, "image_sha_mismatch"
    arr = np.array(net0[row], copy=True)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32), None


def _validate_live_pooled_batch(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    expected_tail: Tuple[int, int, int],
    device,
    source: str,
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"FFS net0 {source} must return a Tensor, got {type(tensor).__name__}")
    expected = (int(batch_size), *expected_tail)
    if tuple(int(x) for x in tensor.shape) != expected:
        raise RuntimeError(f"FFS net0 {source} shape {tuple(tensor.shape)} != expected {expected}")
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"FFS net0 {source} dtype {tensor.dtype} != expected torch.float32")
    return tensor.to(device=device, dtype=torch.float32)


def read_ffs_net0_pooled_cache_batch(
    *,
    cache_dir: str,
    batch_images: List,
    sample_ids: Optional[Sequence[Optional[Tuple[str, object, object]]]],
    handles: Dict[Tuple[str, str, int, int], dict],
    ffs_sha256: Optional[str],
    ffs_cfg,
    pool_hw: int,
    num_depth_tokens: int,
    device,
    live_fallback_fn: Callable[[List], torch.Tensor],
    dataset_or_none=None,
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
    expected_delete_pause_frame: Optional[bool] = None,
) -> Tuple[torch.Tensor, FFSNet0CacheBatchStats]:
    """Read a pooled net0 batch from cache, falling back live per missing row."""
    stats = FFSNet0CacheBatchStats()
    expected_tail = (
        int(_cfg_get(ffs_cfg, "gru_hidden_dim", 16)),
        int(pool_hw),
        int(pool_hw),
    )
    if sample_ids is None:
        stats.whole_batch_live = len(batch_images)
        stats.live_reason = "sample_ids_none"
        return _validate_live_pooled_batch(
            live_fallback_fn(batch_images),
            batch_size=len(batch_images),
            expected_tail=expected_tail,
            device=device,
            source="whole-batch live fallback",
        ), stats
    if len(sample_ids) != len(batch_images):
        raise ValueError(f"FFS net0 cache sample_ids length {len(sample_ids)} != batch size {len(batch_images)}")

    normalized_ids = [_normalize_sample_id(sample_id) for sample_id in sample_ids]
    if not all(sample_id is not None for sample_id in normalized_ids):
        stats.whole_batch_live = len(batch_images)
        stats.live_reason = (
            "partial_sample_ids" if any(sample_id is not None for sample_id in normalized_ids) else "missing_sample_ids"
        )
        return _validate_live_pooled_batch(
            live_fallback_fn(batch_images),
            batch_size=len(batch_images),
            expected_tail=expected_tail,
            device=device,
            source="whole-batch live fallback",
        ), stats

    rows: List[Optional[torch.Tensor]] = []
    missing: List[int] = []
    for i, sample_id in enumerate(normalized_ids):
        cached, reason = _read_ffs_net0_cache_row(
            cache_dir=cache_dir,
            sample_id=sample_id,
            handles=handles,
            ffs_sha256=ffs_sha256,
            ffs_cfg=ffs_cfg,
            pool_hw=pool_hw,
            num_depth_tokens=num_depth_tokens,
            device=device,
            live_images=batch_images[i],
            dataset_or_none=dataset_or_none,
            expected_data_root=expected_data_root,
            expected_data_mix=expected_data_mix,
            expected_video_backend=expected_video_backend,
            expected_delete_pause_frame=expected_delete_pause_frame,
        )
        if cached is None:
            rows.append(None)
            missing.append(i)
            stats.row_misses += 1
            stats.miss_details.append((i, sample_id, str(reason)))
        else:
            rows.append(cached)
            stats.hits += 1

    for i in missing:
        live = _validate_live_pooled_batch(
            live_fallback_fn([batch_images[i]]),
            batch_size=1,
            expected_tail=expected_tail,
            device=device,
            source="row live fallback",
        )
        rows[i] = live[0].to(device=device, dtype=torch.float32)

    if not all(row is not None for row in rows):
        raise RuntimeError("FFS net0 cache fill left an unfilled batch slot")
    return torch.stack([row for row in rows if row is not None], dim=0), stats


@dataclass
class FFSNet0StartupCheckResult:
    checked: int
    max_abs_diff: float
    max_rel_diff: float


def _startup_candidate_rows(count: int, seed: int, limit: int = FFS_STARTUP_CANDIDATE_ROWS_PER_SUITE) -> List[int]:
    if count <= 0:
        return []
    limit = max(0, int(limit))
    rows = {0, count // 2, count - 1}
    if limit <= 0:
        return []
    if len(rows) > limit:
        rows = set(sorted(rows)[:limit])
    rng = random.Random(seed)
    remaining = max(0, limit - len(rows))
    if remaining > 0:
        pool = [row for row in range(count) if row not in rows]
        rows.update(rng.sample(pool, min(len(pool), remaining)))
    return sorted(row for row in rows if 0 <= row < count)


def _startup_check_plan(dataset_or_none, *, min_total: int, per_stratum: int, seed: int) -> List[Tuple[str, object, int]]:
    datasets = _dataset_map(dataset_or_none)
    if not datasets:
        raise RuntimeError("FFS cache startup equality check requires the real training dataset")

    strata: Dict[Tuple[object, ...], List[Tuple[str, object, int]]] = {}
    shape_preprocess_strata = set()
    for suite, dataset in sorted(datasets.items()):
        count = int(len(getattr(dataset, "all_steps", [])))
        for row in _startup_candidate_rows(count, seed + sum(ord(c) for c in suite)):
            sample = dataset[row]
            shape_preprocess_key = (
                _image_shape_key(sample["image"]),
                getattr(dataset, "video_backend", None),
                getattr(dataset, "delete_pause_frame", None),
            )
            shape_preprocess_strata.add(shape_preprocess_key)
            stratum_key = (
                suite,
                *shape_preprocess_key,
            )
            strata.setdefault(stratum_key, [])
            if len(strata[stratum_key]) < per_stratum:
                strata[stratum_key].append((suite, dataset, row))

    if len(shape_preprocess_strata) > 1:
        logger.warning(
            "FFS cache startup equality check observed %d image shape/preprocess strata within "
            "the first %d candidate rows per suite. The candidate cap may need raising for full "
            "stratum coverage on non-uniform datasets. observed_strata=%s",
            len(shape_preprocess_strata),
            FFS_STARTUP_CANDIDATE_ROWS_PER_SUITE,
            sorted(shape_preprocess_strata, key=repr),
        )

    plan: List[Tuple[str, object, int]] = []
    seen = set()
    for key in sorted(strata, key=repr):
        for item in strata[key][:per_stratum]:
            ident = (item[0], item[2])
            if ident not in seen:
                seen.add(ident)
                plan.append(item)

    rng = random.Random(seed)
    suites = sorted(datasets)
    attempts = 0
    while len(plan) < int(min_total) and suites and attempts < int(min_total) * 100:
        suite = suites[attempts % len(suites)]
        dataset = datasets[suite]
        count = int(len(getattr(dataset, "all_steps", [])))
        if count > 0:
            row = rng.randrange(count)
            ident = (suite, row)
            if ident not in seen:
                seen.add(ident)
                plan.append((suite, dataset, row))
        attempts += 1
    if not plan:
        raise RuntimeError("FFS cache startup equality check found no dataset rows to sample")
    return plan


def run_ffs_net0_cache_startup_check(
    *,
    cache_dir: str,
    dataset_or_none,
    handles: Dict[Tuple[str, str, int, int], dict],
    ffs_sha256: Optional[str],
    ffs_cfg,
    pool_hw: int,
    num_depth_tokens: int,
    device,
    compute_pooled_fn: Callable[[List], torch.Tensor],
    expected_data_root: Optional[str] = None,
    expected_data_mix: Optional[str] = None,
    expected_video_backend: Optional[str] = None,
    expected_delete_pause_frame: Optional[bool] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    seed: int = 1729,
) -> FFSNet0StartupCheckResult:
    dataset_map = _dataset_map(dataset_or_none)
    if not dataset_map:
        raise RuntimeError("FFS cache startup equality check requires dataset_or_none")

    suites = sorted(dataset_map)
    suite_handles = validate_ffs_net0_cache(
        cache_dir,
        suites,
        dataset_or_none=dataset_map,
        ffs_sha256=ffs_sha256,
        ffs_cfg=ffs_cfg,
        pool_hw=pool_hw,
        num_depth_tokens=num_depth_tokens,
        require_all_done=True,
        expected_data_root=expected_data_root,
        expected_data_mix=expected_data_mix,
        expected_video_backend=expected_video_backend,
        expected_delete_pause_frame=expected_delete_pause_frame,
    )
    cache_root = str(Path(cache_dir).expanduser())
    for suite, handle in suite_handles.items():
        handles[(cache_root, str(suite), int(pool_hw), int(num_depth_tokens))] = handle

    per_stratum = 2
    min_total = max(16, per_stratum * max(1, len(suites)))
    plan = _startup_check_plan(dataset_map, min_total=min_total, per_stratum=per_stratum, seed=seed)
    # Tolerance note: the FROZEN FFS forward is NOT bit-deterministic even same-GPU
    # (custom cost-volume/correlation kernels) — measured same-env run-to-run diff ~1.5e-5 abs.
    # So this is a "within-FFS-forward-noise" no-op check (catches wrong image / wrong weights /
    # TF32 or cross-env drift, which are >>1e-4), NOT a bit-exact equality. atol set ~6x above the
    # measured noise floor; env-overridable.
    atol = float(os.environ.get("FFS_CACHE_EQ_ATOL", atol if atol is not None else 1e-4))
    rtol = float(os.environ.get("FFS_CACHE_EQ_RTOL", rtol if rtol is not None else 1e-3))

    global_max_abs = 0.0
    global_max_rel = 0.0
    for suite, dataset, row in plan:
        sample = dataset[row]
        sample_id = (
            str(suite),
            _cache_key_part(sample["traj_id"]),
            _cache_key_part(sample["base_index"]),
        )
        cached, reason = _read_ffs_net0_cache_row(
            cache_dir=cache_dir,
            sample_id=sample_id,
            handles=handles,
            ffs_sha256=ffs_sha256,
            ffs_cfg=ffs_cfg,
            pool_hw=pool_hw,
            num_depth_tokens=num_depth_tokens,
            device=device,
            live_images=sample["image"],
            dataset_or_none=dataset_map,
            expected_data_root=expected_data_root,
            expected_data_mix=expected_data_mix,
            expected_video_backend=expected_video_backend,
            expected_delete_pause_frame=expected_delete_pause_frame,
        )
        if cached is None:
            raise RuntimeError(f"startup equality check cache miss for sample_id={sample_id}: reason={reason}")

        fresh = _validate_live_pooled_batch(
            compute_pooled_fn([sample["image"]]),
            batch_size=1,
            expected_tail=(
                int(_cfg_get(ffs_cfg, "gru_hidden_dim", 16)),
                int(pool_hw),
                int(pool_hw),
            ),
            device=device,
            source="startup equality live FFS",
        )[0]
        cached = cached.to(device=device, dtype=torch.float32)
        diff = (fresh - cached).abs()
        max_abs = float(diff.max().detach().cpu().item())
        denom = torch.maximum(cached.abs(), torch.full_like(cached, 1e-12))
        max_rel = float((diff / denom).max().detach().cpu().item())
        global_max_abs = max(global_max_abs, max_abs)
        global_max_rel = max(global_max_rel, max_rel)
        logger.info(
            "FFS cache startup equality sample=%s row=%d max_abs=%.8g max_rel=%.8g atol=%.3g rtol=%.3g",
            sample_id,
            row,
            max_abs,
            max_rel,
            atol,
            rtol,
        )
        if not torch.allclose(fresh, cached, atol=atol, rtol=rtol):
            raise RuntimeError(
                "FFS cache invalid — precompute env != training env: "
                f"key={sample_id} row={row} max_abs={max_abs:.8g} max_rel={max_rel:.8g} "
                f"atol={atol} rtol={rtol}"
            )

    logger.info(
        "FFS cache startup equality check passed: checked=%d max_abs=%.8g max_rel=%.8g",
        len(plan),
        global_max_abs,
        global_max_rel,
    )
    return FFSNet0StartupCheckResult(
        checked=len(plan),
        max_abs_diff=global_max_abs,
        max_rel_diff=global_max_rel,
    )
