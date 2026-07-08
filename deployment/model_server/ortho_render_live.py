# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Eval-time LIVE orthogonal 2x2 grid renderer for orthogrid VLA checkpoints.

Training-time the third "orthogonal grid" view (top/front/side orthographic
projection of the stereo point cloud + legend) is read from a precomputed
sqlite cache keyed by (trajectory_id, base_index). At eval the policy server
sees LIVE simulator observations with no trajectory_id -> the cache is
unusable, so the grid must be rendered live inside the server.

This module reuses the EXACT offline renderer
(``scripts/4090d/precompute_ortho_view_cache.OrthoRenderer``) byte-for-byte —
the same code path that produced the cache the checkpoint trained on — so the
live grid is faithful to training (verified: re-rendering cached frames
reproduces the cache grid pixel-for-pixel, see
``scripts/tools/verify_renderer_vs_cache.py``). We do NOT re-implement any
render math here; we only adapt the entry point from a dataset ``sample`` dict
to live ``(primary, left_view, suite)`` numpy inputs.

FFS import-time note: ``probe_orthogonal_multiview_render`` (imported by
``precompute_ortho_view_cache``) loads the Fast-FoundationStereo core at import
time, so ``FFS_REPO_DIR`` MUST be valid before importing this module's renderer.
We set it in ``__init__`` before the import.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# starVLA repo root = parents[2] of deployment/model_server/ortho_render_live.py
_STARVLA_ROOT = str(Path(__file__).resolve().parents[2])

# Defaults mirror scripts/4090d/precompute_ortho_view_cache.py so the live path
# uses the SAME FFS weights / repo the cache was generated with.
DEFAULT_FFS_REPO_DIR = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo"


def _ffs_model_path_for(repo_dir: str) -> str:
    return os.path.join(repo_dir, "weights", "20-30-48", "model_best_bp2_serialize.pth")


class LiveOrthoRenderer:
    """Live orthogonal-grid renderer that reuses the offline OrthoRenderer.

    Construction is cheap (imports the reusable renderer + FFS core modules).
    The FFS network itself is loaded lazily on the first ``render_grid`` call
    (``OrthoRenderer._ensure_ffs``), so server startup stays fast.

    Args:
        ffs_repo_dir: Fast-FoundationStereo repo path. Defaults to
            ``$FFS_REPO_DIR`` then :data:`DEFAULT_FFS_REPO_DIR`.
        ffs_model_path: FFS weights path. Defaults to
            ``$FFS_MODEL_PATH`` then ``<repo>/weights/20-30-48/model_best_bp2_serialize.pth``.
        valid_iters / max_disp / axo_size / level_table: passed through to the
            OrthoRenderer; MUST match the values used to generate the cache
            (valid_iters=8, max_disp=192, axo_size=448, level_table="auto").
    """

    def __init__(
        self,
        ffs_repo_dir: Optional[str] = None,
        ffs_model_path: Optional[str] = None,
        valid_iters: int = 8,
        max_disp: int = 192,
        axo_size: int = 448,
        level_table: str = "auto",
    ) -> None:
        ffs_repo_dir = ffs_repo_dir or os.environ.get("FFS_REPO_DIR") or DEFAULT_FFS_REPO_DIR
        ffs_repo_dir = str(Path(ffs_repo_dir).resolve())
        if not Path(ffs_repo_dir).is_dir():
            raise FileNotFoundError(
                f"LiveOrthoRenderer: FFS repo dir not found: {ffs_repo_dir}. "
                f"Set FFS_REPO_DIR (eval launcher exports it)."
            )
        ffs_model_path = (
            ffs_model_path or os.environ.get("FFS_MODEL_PATH") or _ffs_model_path_for(ffs_repo_dir)
        )
        if not Path(ffs_model_path).is_file():
            raise FileNotFoundError(
                f"LiveOrthoRenderer: FFS weights not found: {ffs_model_path}"
            )

        # probe_orthogonal_multiview_render imports FFS core at import time; make
        # sure the env is set BEFORE importing the reusable renderer.
        os.environ["FFS_REPO_DIR"] = ffs_repo_dir
        os.environ.setdefault("STARVLA_ROOT", _STARVLA_ROOT)

        scripts_4090d = os.path.join(_STARVLA_ROOT, "scripts", "4090d")
        if scripts_4090d not in sys.path:
            sys.path.insert(0, scripts_4090d)

        # precompute_ortho_view_cache also inserts scripts/tools onto sys.path and
        # imports probe (which loads FFS). Import it AFTER FFS_REPO_DIR is set.
        import precompute_ortho_view_cache as pc  # noqa: WPS433

        self._pc = pc
        self._args = SimpleNamespace(
            ffs_repo_dir=ffs_repo_dir,
            ffs_model_path=ffs_model_path,
            valid_iters=int(valid_iters),
            max_disp=int(max_disp),
            axo_size=int(axo_size),
            level_table=str(level_table),
        )
        # Same OrthoRenderer the offline cache used (verified byte-identical).
        self._renderer = pc.OrthoRenderer(self._args)
        self._calls = 0
        logging.info(
            "LiveOrthoRenderer ready: ffs_repo=%s valid_iters=%d max_disp=%d level_table=%s",
            ffs_repo_dir,
            self._args.valid_iters,
            self._args.max_disp,
            self._args.level_table,
        )

    def render_grid(
        self, primary_256: np.ndarray, left_256: np.ndarray, suite_name: str
    ) -> Tuple[np.ndarray, dict]:
        """Render the orthogonal 2x2 grid from a live stereo pair.

        Args:
            primary_256: primary camera frame, HxWx3 uint8 (geometric RIGHT eye
                under leftprimary convention). MUST be native 256px (the offline
                cache re-samples to 256 via ``_to_np_256``; passing 256 keeps the
                FFS input identical to training).
            left_256: left_view frame, HxWx3 uint8 (geometric LEFT eye; at eval
                this is the rightview render, the logical ``left_view`` alias).
            suite_name: LIBERO suite (libero_object|libero_goal|libero_spatial|
                libero_10) — selects the per-suite static rightview pose used to
                lift the point cloud to world.

        Returns:
            (grid, info) where ``grid`` is the composed 454x454x3 uint8 grid
            image (same as the training cache) and ``info`` is the renderer's
            per-frame diagnostics (num_points, table_deg, ...).
        """
        primary = np.ascontiguousarray(np.asarray(primary_256, dtype=np.uint8))
        left = np.ascontiguousarray(np.asarray(left_256, dtype=np.uint8))
        for _nm, _im in (("primary", primary), ("left", left)):
            if _im.ndim != 3 or _im.shape[-1] != 3:
                raise ValueError(
                    f"render_grid {_nm} must be HxWx3 uint8, got shape {_im.shape}"
                )
        # CRITICAL train/eval parity: the offline ortho cache fed OrthoRenderer the
        # dataloader's per-camera image, which datasets._pack_sample ALWAYS resizes to
        # 224 via Image.fromarray(frame).resize((224,224)) (PIL default resample).
        # OrthoRenderer._to_np_256 then upsamples that 224 PIL back to 256 for FFS.
        # So the cache's FFS input is 224->256, NOT native 256. We replicate the exact
        # same 256->224 resize here so the live grid geometry matches the training grid
        # distribution (feeding native 256 would give a sharper, out-of-distribution
        # grid). Sources differ (live sim vs offline render) — that residual is covered
        # by the final eval numbers, but the resize pipeline is now byte-identical.
        sample = {
            "image": [
                Image.fromarray(primary).resize((224, 224)),
                Image.fromarray(left).resize((224, 224)),
            ],
            "dataset_name": str(suite_name),
        }
        grid, _axo, info = self._renderer.render(sample)
        self._calls += 1
        _dump_dir = os.environ.get("ORTHO_DUMP_DIR")
        if _dump_dir:
            try:
                self._dump_grid(_dump_dir, str(suite_name), grid, primary, left)
            except Exception as _e:  # diagnostic-only; never break eval
                logging.warning("[ortho-dump] save failed: %s", _e)
        return np.ascontiguousarray(np.asarray(grid, dtype=np.uint8)), info

    def _dump_grid(self, dump_dir, suite, grid, primary, left):
        """Diagnostic-only (ORTHO_DUMP_DIR env): save the live eval-time grid plus
        the t=0 native-256 input frames as PNG, bounded by ORTHO_DUMP_CAP grids per
        suite (default 60). Unset in normal eval -> zero cost. Lets us visualize
        exactly what the model consumed at test time vs the training cache grid."""
        counts = self.__dict__.setdefault("_dump_counts", {})
        n = counts.get(suite, 0)
        if n >= int(os.environ.get("ORTHO_DUMP_CAP", "60")):
            return
        d = Path(dump_dir) / suite
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(grid, dtype=np.uint8)).save(d / f"grid_{n:04d}.png")
        if n == 0:  # source stereo pair once per suite (native 256, as client sent)
            Image.fromarray(np.asarray(primary, dtype=np.uint8)).save(d / "input_primary_256.png")
            Image.fromarray(np.asarray(left, dtype=np.uint8)).save(d / "input_left_256.png")
        counts[suite] = n + 1
