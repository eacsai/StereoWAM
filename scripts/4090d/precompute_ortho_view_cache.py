#!/usr/bin/env python3
"""Precompute LIBERO leftprimary stereo -> orthogonal VLM image cache."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
PROBE_DIR = ROOT / "scripts" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

import probe_orthogonal_multiview_render as probe  # noqa: E402


DEFAULT_DATA_ROOT = "playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW"
DEFAULT_DATA_MIX = "libero_all_sfstereo_leftprimary"
DEFAULT_FFS_REPO_DIR = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo"
DEFAULT_FFS_MODEL = f"{DEFAULT_FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth"


def _cache_key_part(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _all_steps_sha256(all_steps: list[tuple[object, object]]) -> str:
    payload = json.dumps(all_steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _parse_sample_indices(value: str) -> list[int] | None:
    value = str(value or "").strip()
    if not value:
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _load_cfg(args: argparse.Namespace):
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "datasets": {
                    "vla_data": {
                        "data_root_dir": args.data_root,
                        "data_mix": args.data_mix,
                        "video_backend": args.video_backend,
                    }
                }
            }
        ),
    )
    return apply_config_compat(cfg)


def _normalized_all_steps(dataset) -> list[tuple[object, object]]:
    return [(_cache_key_part(t), _cache_key_part(b)) for t, b in dataset.all_steps]


def _open_done(path: Path, count: int, resume: bool):
    if path.is_file() and resume:
        done = np.load(path, mmap_mode="r+")
        if tuple(done.shape) != (count,) or done.dtype != np.dtype(np.bool_):
            raise RuntimeError(f"existing done.npy has shape={done.shape} dtype={done.dtype}, expected {(count,)} bool")
        return done
    done = np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(count,))
    done[:] = False
    done.flush()
    return done


def _png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    iio.imwrite(buf, img.astype(np.uint8), format="png")
    return buf.getvalue()


def _to_np_256(img: Image.Image) -> np.ndarray:
    return probe.to_np(img.resize((probe.W, probe.H), Image.BILINEAR))


def _probe_suite_name(dataset_name: str) -> str:
    for suite in probe.OURRENDER_SUITE_ORDER:
        if str(dataset_name).startswith(suite):
            return suite
    raise RuntimeError(f"cannot map dataset_name={dataset_name!r} to probe LIBERO suite")


class OrthoRenderer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._ffs = None

    def _ensure_ffs(self):
        if self._ffs is not None:
            return self._ffs
        ffs_root = str(Path(self.args.ffs_repo_dir).resolve())
        if ffs_root not in sys.path:
            sys.path.insert(0, ffs_root)
        from core.utils.utils import InputPadder  # noqa: WPS433
        from Utils import AMP_DTYPE  # noqa: WPS433

        model = torch.load(self.args.ffs_model_path, map_location="cpu", weights_only=False)
        model.args.valid_iters = int(self.args.valid_iters)
        model.args.max_disp = int(self.args.max_disp)
        model.cuda().eval()
        torch.autograd.set_grad_enabled(False)
        self._ffs = (model, InputPadder, AMP_DTYPE)
        return self._ffs

    def run_ffs_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        model, InputPadder, AMP_DTYPE = self._ensure_ffs()
        a = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
        b = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(
                a,
                b,
                iters=int(self.args.valid_iters),
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(img_left.shape[0], img_left.shape[1])
        return disp.clip(0, None).astype(np.float32)

    def render(self, sample: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        images = sample["image"]
        if len(images) != 2:
            raise RuntimeError(f"expected stereo [primary,left_view], got {len(images)} images")
        primary = _to_np_256(images[0])
        left_view = _to_np_256(images[1])
        suite_name = sample["dataset_name"]
        right_view_pos_w, right_view_R_mjc_to_w = probe.load_right_view_pose(_probe_suite_name(suite_name))

        disp = self.run_ffs_disparity(left_view, primary)
        depth, valid = probe.disparity_to_depth(disp, probe.FX, probe.BASELINE_M)
        pts, cols = probe.unproject(depth, left_view, valid, probe.VALID_LO, probe.VALID_HI)
        pts_w = probe.camera_to_world(pts, right_view_pos_w, right_view_R_mjc_to_w)
        pts_w, table_deg, table_inliers, leveled, level_reason = probe.maybe_level_table(
            pts_w,
            cols,
            self.args.level_table,
        )
        origin_w = pts_w.mean(0)
        pts_rel, cols_rel = probe.workspace_crop(pts_w, cols, origin_w)
        if len(pts_rel) == 0:
            raise RuntimeError("workspace crop produced 0 points")

        top = probe.render_world_view(pts_rel, cols_rel, "top")
        front = probe.render_world_view(pts_rel, cols_rel, "front")
        side = probe.render_world_view(pts_rel, cols_rel, "side")
        legend = probe.make_legend()
        grid = probe.compose_grid(top, front, side, legend)
        axo = np.zeros((int(self.args.axo_size), int(self.args.axo_size), 3), dtype=np.uint8)
        info = {
            "num_points": int(len(pts_rel)),
            "table_deg": None if np.isnan(table_deg) else float(table_deg),
            "table_inliers": int(table_inliers),
            "leveled": bool(leveled),
            "level_reason": str(level_reason),
        }
        return grid, axo, info


def _ensure_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ortho_views ("
        "row INTEGER PRIMARY KEY, "
        "traj_id TEXT NOT NULL, "
        "base_index TEXT NOT NULL, "
        "grid_png BLOB NOT NULL, "
        "axo_png BLOB NOT NULL, "
        "render_info TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _write_meta(path: Path, *, dataset, cfg, args: argparse.Namespace, ffs_sha256: str, count: int) -> None:
    meta = {
        "schema_version": 1,
        "suite": dataset.dataset_name,
        "format": "sqlite_png_blobs",
        "db": "ortho_views.sqlite",
        "columns": {"grid": "grid_png", "axo": "axo_png"},
        "count": int(count),
        "all_steps_sha256": _all_steps_sha256(_normalized_all_steps(dataset)),
        "data_mix": str(cfg.datasets.vla_data.data_mix),
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "video_backend": str(dataset.video_backend),
        "delete_pause_frame": bool(dataset.delete_pause_frame),
        "view_order": ["primary", "left_view", "ortho_cache"],
        "stereo_convention": "leftprimary: primary=geometric_right, left_view=geometric_left",
        "ffs_call": "ffs(left_view, primary)",
        "reference_view": "left_view",
        "depth_frame": "left_view",
        "render_functions": {
            "module": str(PROBE_DIR / "probe_orthogonal_multiview_render.py"),
            "reuse": [
                "load_right_view_pose",
                "camera_to_world",
                "maybe_level_table",
                "render_world_view",
                "render_axonometric",
                "compose_grid",
            ],
        },
        "geometry": {
            "fovy_degrees": float(probe.FOVY_DEG),
            "baseline_m": float(probe.BASELINE_M),
            "fx": float(probe.FX),
            "image_hw": [int(probe.H), int(probe.W)],
            "workspace_size_m": float(probe.WORKSPACE_SIZE),
            "grid_image_size": [int(probe.IMG_SIZE * 2 + 6), int(probe.IMG_SIZE * 2 + 6)],
            "axo_image_size": [int(args.axo_size), int(args.axo_size)],
            "meters_per_px": float(probe.METERS_PER_PX),
            "level_table": str(args.level_table),
        },
        "ffs_model_path": str(args.ffs_model_path),
        "ffs_model_sha256": ffs_sha256,
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def _rows_from_sample_indices(dataset_mixture, sample_indices: list[int] | None, selected_suites: set[str]) -> dict[str, set[int]]:
    out = {dataset.dataset_name: set() for dataset in dataset_mixture.datasets if dataset.dataset_name in selected_suites}
    if sample_indices is None:
        for dataset in dataset_mixture.datasets:
            if dataset.dataset_name in selected_suites:
                out[dataset.dataset_name] = set(range(len(dataset.all_steps)))
        return out
    by_name = {dataset.dataset_name: dataset for dataset in dataset_mixture.datasets}
    for index in sample_indices:
        dataset, trajectory_id, step = dataset_mixture.sample_step(int(index))
        if dataset.dataset_name not in selected_suites:
            continue
        normalized = [(_cache_key_part(t), _cache_key_part(b)) for t, b in dataset.all_steps]
        row = {key: i for i, key in enumerate(normalized)}[(_cache_key_part(trajectory_id), _cache_key_part(step))]
        out[dataset.dataset_name].add(row)
    return out


def precompute_suite(renderer: OrthoRenderer, dataset, rows: set[int], suite_dir: Path, args: argparse.Namespace, cfg) -> None:
    count = len(dataset.all_steps)
    suite_dir.mkdir(parents=True, exist_ok=True)
    done_path = suite_dir / "done.npy"
    index_path = suite_dir / "index.pkl"
    meta_path = suite_dir / "meta.json"
    db_path = suite_dir / "ortho_views.sqlite"

    all_steps = _normalized_all_steps(dataset)
    if args.resume and index_path.exists():
        with open(index_path, "rb") as fh:
            existing = pickle.load(fh)
        expected = {key: row for row, key in enumerate(all_steps)}
        if existing != expected:
            raise RuntimeError(f"existing index.pkl does not match dataset all_steps for {dataset.dataset_name}")
    with open(index_path, "wb") as fh:
        pickle.dump({key: row for row, key in enumerate(all_steps)}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    done = _open_done(done_path, count, resume=args.resume)
    ffs_sha256 = _sha256_file(args.ffs_model_path)
    _write_meta(meta_path, dataset=dataset, cfg=cfg, args=args, ffs_sha256=ffs_sha256, count=count)
    conn = _ensure_db(db_path)

    todo = sorted(int(r) for r in rows)
    start = time.time()
    written = 0
    skipped = 0
    print(f"ORTHO_SUITE_START suite={dataset.dataset_name} rows={len(todo)} count={count}")
    for row in todo:
        if row < 0 or row >= count:
            raise IndexError(f"row {row} out of range for {dataset.dataset_name} count={count}")
        if bool(done[row]) and args.resume:
            skipped += 1
            continue
        sample = dataset[row]
        grid, axo, info = renderer.render(sample)
        conn.execute(
            "INSERT OR REPLACE INTO ortho_views(row,traj_id,base_index,grid_png,axo_png,render_info) "
            "VALUES (?,?,?,?,?,?)",
            (
                int(row),
                json.dumps(_cache_key_part(sample["traj_id"]), sort_keys=True),
                json.dumps(_cache_key_part(sample["base_index"]), sort_keys=True),
                sqlite3.Binary(_png_bytes(grid)),
                sqlite3.Binary(_png_bytes(axo)),
                json.dumps(info, sort_keys=True),
            ),
        )
        done[row] = True
        written += 1
        if written % int(args.log_every) == 0 or written == len(todo):
            conn.commit()
            done.flush()
            elapsed = max(time.time() - start, 1e-6)
            print(
                "ORTHO_PROGRESS "
                f"suite={dataset.dataset_name} written={written} skipped={skipped} "
                f"todo={len(todo)} rate={written / elapsed:.3f}_rows_per_sec"
            )
    conn.commit()
    done.flush()
    conn.close()
    print(f"ORTHO_SUITE_DONE suite={dataset.dataset_name} written={written} skipped={skipped}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--suites", default="all")
    parser.add_argument("--sample-indices", default="", help="Comma-separated mixture DataLoader indices for smoke precompute")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-model-path", default=os.environ.get("FFS_MODEL_PATH", DEFAULT_FFS_MODEL))
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--axo-size", type=int, default=448)
    parser.add_argument("--level-table", choices=["off", "auto", "always"], default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ["FFS_REPO_DIR"] = args.ffs_repo_dir
    if not torch.cuda.is_available():
        raise RuntimeError("precompute_ortho_view_cache.py requires CUDA")

    cfg = _load_cfg(args)
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    mixture_spec = DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]
    available = [name for name, _weight, robot_type in mixture_spec]
    bad = [(name, robot_type) for name, _weight, robot_type in mixture_spec if robot_type != "libero_franka_sfstereo_leftprimary"]
    if str(cfg.datasets.vla_data.data_mix) != DEFAULT_DATA_MIX or bad:
        raise ValueError(f"ortho cache supports {DEFAULT_DATA_MIX} only; bad={bad}")
    suites = _parse_suites(args.suites, available)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
    rows_by_suite = _rows_from_sample_indices(dataset, _parse_sample_indices(args.sample_indices), set(suites))
    by_name = {single.dataset_name: single for single in dataset.datasets}
    renderer = OrthoRenderer(args)
    for suite in suites:
        precompute_suite(renderer, by_name[suite], rows_by_suite[suite], Path(args.cache_dir) / suite, args, cfg)
    print("ORTHO_PRECOMPUTE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
