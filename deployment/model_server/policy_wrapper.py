# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        # --- orthogrid live-render detection (eval-time third view) ---
        # Orthogrid checkpoints consume a third "orthogonal 2x2 grid" image that
        # training read from a precomputed cache keyed by (traj_id, frame). At
        # eval there is no such key, so the server renders the grid live and
        # injects it here (deployment/model_server/ortho_render_live.py).
        # Detected from config (ortho_cache.enabled + view==grid), never run_id.
        self._ortho_enabled = False
        self._ortho_position = "last"
        self._live_ortho = None
        self._ortho_debug_n = 0
        _vla = (model_cfg.get("datasets", {}) or {}).get("vla_data", {}) or {}
        _oc = _vla.get("ortho_cache", {}) or {}
        if bool(_oc.get("enabled", False)) and str(_oc.get("view", "")) == "grid":
            self._ortho_enabled = True
            self._ortho_position = str(_oc.get("position", "last"))
            if self._ortho_position not in ("last", "first"):
                raise ValueError(
                    f"PolicyServerWrapper: ortho_cache.position must be 'last' or "
                    f"'first', got {self._ortho_position!r}"
                )
            from deployment.model_server.ortho_render_live import LiveOrthoRenderer
            self._live_ortho = LiveOrthoRenderer()
            # fail-closed: orthogrid ckpts were trained with a specific
            # ORTHO_PROMPT_TEXT; QWen3_5 silently falls back to the bare
            # instruction if the env var is unset (prompt contract broken).
            # The eval launcher exports it before starting the server; refuse
            # to run if a bypass/stale launcher left it unset.
            if not (os.environ.get("ORTHO_PROMPT_TEXT") or "").strip():
                raise ValueError(
                    "orthogrid checkpoint but ORTHO_PROMPT_TEXT env is unset/empty; "
                    "the eval launcher must export the byte-exact training prompt "
                    "before starting server_policy.py (fail-closed)."
                )
            logging.info(
                "PolicyServerWrapper: ORTHOGRID live-render ENABLED (position=%s)",
                self._ortho_position,
            )

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def _inject_ortho_grid(
        self,
        examples: List[dict],
        suite_name: Optional[str],
        raw_stereo_256: Optional[list],
    ) -> List[dict]:
        """Render the live orthogonal grid and inject it as the third image.

        Fail-closed: an orthogrid checkpoint MUST receive ``suite_name`` (for
        the per-suite camera pose) and ``raw_stereo_256`` (the un-resized 256px
        stereo pair) from the client; missing either is a hard error rather
        than a silent wrong-geometry render.
        """
        if suite_name is None or raw_stereo_256 is None:
            raise ValueError(
                "orthogrid checkpoint requires suite_name and raw_stereo_256 in "
                "the request (client must send them); refusing to render with "
                "missing geometry inputs (fail-closed)."
            )
        if len(examples) != 1:
            raise ValueError(
                f"orthogrid live-render is request-level (one raw_stereo_256 per "
                f"request), but got a batch of {len(examples)} examples; refusing to "
                f"apply one grid to a batch (fail-closed). Send B=1 requests."
            )
        if not (isinstance(raw_stereo_256, (list, tuple)) and len(raw_stereo_256) == 2):
            raise ValueError(
                f"raw_stereo_256 must be [primary_256, left_256]; got "
                f"{type(raw_stereo_256).__name__}"
            )
        grid, info = self._live_ortho.render_grid(
            raw_stereo_256[0], raw_stereo_256[1], suite_name
        )
        for ex in examples:
            imgs = list(ex.get("image", []))
            # single-frame only: [primary, left_view] (R7 multi-frame unsupported)
            if len(imgs) != 2:
                raise ValueError(
                    f"orthogrid eval expects exactly 2 camera images "
                    f"[primary,left_view] per example, got {len(imgs)} "
                    f"(multi-frame orthogrid unsupported; fail-closed)."
                )
            if self._ortho_position == "first":
                imgs.insert(0, grid)
            else:
                imgs.append(grid)
            ex["image"] = imgs
        if self._ortho_debug_n < 4:
            self._ortho_debug_n += 1
            _prompt = os.environ.get("ORTHO_PROMPT_TEXT") or ""
            _phash = hashlib.md5(_prompt.encode()).hexdigest()[:8] if _prompt else "MISSING"
            _sizes = [tuple(np.asarray(im).shape[:2]) for im in examples[0]["image"]]
            logging.info(
                "[ortho-eval-inject] suite=%s position=%s n_images=%d sizes=%s "
                "num_points=%s prompt_md5=%s prompt_len=%d",
                suite_name,
                self._ortho_position,
                len(examples[0]["image"]),
                _sizes,
                info.get("num_points"),
                _phash,
                len(_prompt),
            )
        return examples

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        suite_name: Optional[str] = None,
        raw_stereo_256: Optional[list] = None,
        episode_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        if self._ortho_enabled:
            examples = self._inject_ortho_grid(examples, suite_name, raw_stereo_256)

        # Past-flow rollout FIFO: reset when the episode changes (server holds ONE persistent
        # framework across all episodes/tasks/suites, else past flow leaks between episodes).
        if episode_id is not None and episode_id != getattr(self, "_pf_last_episode_id", None):
            if hasattr(self._framework, "reset_past_flow"):
                self._framework.reset_past_flow()
            self._pf_last_episode_id = episode_id

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm}
