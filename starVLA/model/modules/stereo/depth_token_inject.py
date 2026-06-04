"""Depth-token sequence insertion for QwenPI + frozen FFS.

This module turns the frozen FoundationStereo net[0] feature into a short run
of learned depth tokens, then inserts those tokens into the Qwen VLM sequence
right after the primary-view image-token run.

The framework computes and stashes the depth tokens before the VLM forward.
Two pre-hooks then handle sequence bookkeeping:
  1. top HF model pre-hook: records per-token camera ids and the insertion
     point for each sample.
  2. language_model pre-hook: inserts the depth tokens into inputs_embeds,
     position_ids, attention_mask, and the shared cam-rope state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .cam_rope_hook import compute_per_token_cam_id

logger = logging.getLogger(__name__)


class DepthTokenProjector(nn.Module):
    """Pool FFS net[0] to 4x4 tokens and project each token to VLM width."""

    def __init__(
        self,
        in_ch: int = 16,
        llm_dim: int = 1024,
        num_tokens: int = 16,
        pool_hw: int = 4,
    ) -> None:
        super().__init__()
        if pool_hw * pool_hw != num_tokens:
            raise ValueError(
                f"pool_hw*pool_hw must equal num_tokens; got {pool_hw}x{pool_hw} "
                f"vs {num_tokens}"
            )
        self.in_ch = int(in_ch)
        self.llm_dim = int(llm_dim)
        self.num_tokens = int(num_tokens)
        self.pool_hw = int(pool_hw)
        self.pool = nn.AdaptiveAvgPool2d((pool_hw, pool_hw))
        self.proj = nn.Linear(in_ch, llm_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, net0: torch.Tensor) -> torch.Tensor:
        if net0.dim() != 4:
            raise ValueError(f"DepthTokenProjector expects (B,C,H,W), got {tuple(net0.shape)}")
        if int(net0.shape[1]) != self.in_ch:
            raise ValueError(
                f"DepthTokenProjector expected {self.in_ch} channels, got {int(net0.shape[1])}"
            )
        x = self.pool(net0)                    # (B, C, 4, 4)
        x = x.flatten(2).transpose(1, 2)       # (B, 16, C)
        return self.proj(x.to(self.proj.weight.dtype))                    # (B, 16, D)


@dataclass
class DepthTokenInjectState:
    depth_tokens: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    insert_idx: Optional[torch.Tensor] = None
    keep_mask: Optional[torch.Tensor] = None
    primary_cam_id: int = 0


_STATE = DepthTokenInjectState()


def set_depth_tokens(depth_tokens: torch.Tensor) -> None:
    _STATE.depth_tokens = depth_tokens


def clear_state() -> None:
    _STATE.depth_tokens = None
    _STATE.per_token_cam_id = None
    _STATE.insert_idx = None
    _STATE.keep_mask = None


def get_state() -> DepthTokenInjectState:
    return _STATE


def _extract_input_ids(args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and args:
        input_ids = args[0]
    return input_ids


def _locate_language_model(hf_model):
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError(
            "[depth_token_inject] could not locate model.language_model with .layers "
            "on the Qwen VLM model"
        )
    return lm


def _insert_batched_2d(
    x: torch.Tensor,
    insert_idx: torch.Tensor,
    inserted: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for b in range(x.shape[0]):
        idx = int(insert_idx[b].item())
        rows.append(torch.cat([x[b, :idx], inserted[b], x[b, idx:]], dim=0))
    return torch.stack(rows, dim=0)


def _insert_batched_3d(
    x: torch.Tensor,
    insert_idx: torch.Tensor,
    inserted: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for b in range(x.shape[0]):
        idx = int(insert_idx[b].item())
        rows.append(torch.cat([x[b, :idx], inserted[b], x[b, idx:]], dim=0))
    return torch.stack(rows, dim=0)


def _make_keep_mask(
    batch_size: int,
    seq_len: int,
    insert_idx: torch.Tensor,
    num_insert: int,
    device,
) -> torch.Tensor:
    rows = []
    for b in range(batch_size):
        idx = int(insert_idx[b].item())
        before = torch.ones(idx, dtype=torch.bool, device=device)
        middle = torch.zeros(num_insert, dtype=torch.bool, device=device)
        after = torch.ones(seq_len - idx, dtype=torch.bool, device=device)
        rows.append(torch.cat([before, middle, after], dim=0))
    return torch.stack(rows, dim=0)


def _depth_position_columns(
    position_ids: torch.Tensor,
    cam: torch.Tensor,
    insert_idx: torch.Tensor,
    num_insert: int,
    primary_cam_id: int,
) -> torch.Tensor:
    """Insert per-sample position columns along the last axis.

    position_ids is treated as (..., B, S). This covers Qwen M-RoPE tensors
    shaped (3, B, S) or (4, B, S), while also working for plain (B, S).
    """
    samples = []
    for b in range(cam.shape[0]):
        pos_b = position_ids[..., b, :]  # (..., S)
        idx = int(insert_idx[b].item())
        primary_pos = (cam[b] == primary_cam_id).nonzero(as_tuple=True)[0]
        n_prim = int(primary_pos.numel())
        if n_prim > 0:
            k = int(round(num_insert ** 0.5))   # pooled-grid side (16 tokens -> 4x4)
            g = int(round(n_prim ** 0.5))        # primary token grid side (square images)
            if k * k == num_insert and g * g == n_prim and g >= k:
                # MED-4 fix (2026-06-03, codex review): pick the 4x4 CELL-CENTER tokens of
                # the primary image's g×g token grid, matching where the pooled FFS tokens
                # come from. (The old even-linspace over the flattened run traced a
                # row-major stripe, not faithful 2D cell centers.) Square-grid only;
                # non-square falls back to linspace below.
                sel = []
                for i in range(k):
                    r = min(int((i + 0.5) * g / k), g - 1)
                    for j in range(k):
                        c = min(int((j + 0.5) * g / k), g - 1)
                        sel.append(r * g + c)
                source_idx = primary_pos.index_select(
                    0, torch.tensor(sel, dtype=torch.long, device=primary_pos.device)
                )
            elif n_prim == 1:
                source_idx = primary_pos.expand(num_insert)
            else:
                lin = torch.linspace(
                    0,
                    n_prim - 1,
                    steps=num_insert,
                    device=primary_pos.device,
                )
                source_idx = primary_pos.index_select(0, lin.round().long())
            insert_pos = pos_b.index_select(-1, source_idx.to(position_ids.device))
        else:
            fallback = max(idx - 1, 0)
            insert_pos = pos_b[..., fallback : fallback + 1].expand(
                *pos_b.shape[:-1], num_insert
            )
        samples.append(torch.cat([pos_b[..., :idx], insert_pos, pos_b[..., idx:]], dim=-1))
    return torch.stack(samples, dim=-2)


def _compute_insert_idx(cam: torch.Tensor, primary_cam_id: int) -> torch.Tensor:
    # HIGH-2 fix (2026-06-03, codex review): insert depth tokens BEFORE all image
    # tokens, so that under Qwen's CAUSAL attention every image + text token (all of
    # which come AFTER the insertion point) can attend back to the depth tokens.
    # The old "after the primary image run" placement left the primary image tokens
    # (the main visual signal) unable to see depth at all — only later right-view /
    # text tokens could. `primary_cam_id` is kept for signature compatibility; the
    # anchor is now the FIRST image token of ANY view (cam >= 0).
    idxs = []
    for b in range(cam.shape[0]):
        image_pos = (cam[b] >= 0).nonzero(as_tuple=True)[0]
        if int(image_pos.numel()) == 0:
            raise RuntimeError(
                f"[depth_token_inject] sample {b}: no image tokens (cam>=0) to anchor insertion"
            )
        idx = int(image_pos.min().item())  # right BEFORE the first image token
        idxs.append(idx)
    return torch.tensor(idxs, dtype=torch.long, device=cam.device)


def reinstall_depth_token_outer_hook(
    top_module,
    outer_hook_fn: Callable,
) -> torch.utils.hooks.RemovableHandle:
    """Attach the depth-token outer bookkeeping hook to a top-level wrapper."""
    handle = top_module.register_forward_pre_hook(outer_hook_fn, with_kwargs=True)
    setattr(handle, "depth_token_outer_hook_fn", outer_hook_fn)
    setattr(
        handle,
        "reinstall_outer_hook",
        lambda new_top_module: reinstall_depth_token_outer_hook(new_top_module, outer_hook_fn),
    )
    return handle


def install_depth_token_hooks(
    hf_model,
    lm=None,
    *,
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
    primary_cam_id: int = 0,
    cam_rope_state=None,
    image_token_id: Optional[int] = None,
) -> Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
    """Install the outer bookkeeping hook and inner sequence-insertion hook."""
    if image_token_id is None:
        image_token_id = int(hf_model.config.image_token_id)
    lm = lm if lm is not None else _locate_language_model(hf_model)
    state = _STATE
    state.primary_cam_id = int(primary_cam_id)

    def _outer_pre_hook(module, args, kwargs):
        input_ids = _extract_input_ids(args, kwargs)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if input_ids is None:
            state.per_token_cam_id = None
            state.insert_idx = None
            state.keep_mask = None
            return None

        cam = None
        if cam_rope_state is not None:
            cam = getattr(cam_rope_state, "per_token_cam_id", None)
            if cam is not None and tuple(cam.shape) != tuple(input_ids.shape):
                cam = None
        if cam is None:
            cam = compute_per_token_cam_id(
                input_ids=input_ids,
                image_token_id=image_token_id,
                image_grid_thw=image_grid_thw,
                num_cameras=num_cameras,
                spatial_merge_size=spatial_merge_size,
            )

        state.per_token_cam_id = cam
        state.insert_idx = _compute_insert_idx(cam, int(primary_cam_id))
        state.keep_mask = None
        return None

    def _inner_pre_hook(module, args, kwargs):
        depth_tokens = state.depth_tokens
        cam = state.per_token_cam_id
        insert_idx = state.insert_idx
        if depth_tokens is None or cam is None or insert_idx is None:
            return None

        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            return None

        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        if depth_tokens.shape[0] != batch_size:
            raise RuntimeError(
                f"[depth_token_inject] depth batch {depth_tokens.shape[0]} != VLM batch {batch_size}"
            )
        if depth_tokens.shape[2] != hidden_dim:
            raise RuntimeError(
                f"[depth_token_inject] depth hidden dim {depth_tokens.shape[2]} != VLM dim {hidden_dim}"
            )
        if cam.shape != (batch_size, seq_len):
            raise RuntimeError(
                f"[depth_token_inject] cam-id shape {tuple(cam.shape)} does not match "
                f"inputs_embeds {(batch_size, seq_len)}"
            )

        depth = depth_tokens.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        num_insert = int(depth.shape[1])
        insert_idx_local = insert_idx.to(device=inputs_embeds.device)

        kwargs["inputs_embeds"] = _insert_batched_3d(inputs_embeds, insert_idx_local, depth)
        state.keep_mask = _make_keep_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            insert_idx=insert_idx_local,
            num_insert=num_insert,
            device=inputs_embeds.device,
        )

        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None:
            if int(position_ids.shape[-1]) != seq_len:
                raise RuntimeError(
                    f"[depth_token_inject] position_ids last dim {position_ids.shape[-1]} "
                    f"!= original seq_len {seq_len}"
                )
            if int(position_ids.shape[-2]) != batch_size:
                raise RuntimeError(
                    f"[depth_token_inject] position_ids batch dim {position_ids.shape[-2]} "
                    f"!= batch_size {batch_size}"
                )
            kwargs["position_ids"] = _depth_position_columns(
                position_ids=position_ids,
                cam=cam.to(position_ids.device),
                insert_idx=insert_idx.to(position_ids.device),
                num_insert=num_insert,
                primary_cam_id=int(primary_cam_id),
            )

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise RuntimeError(
                    f"[depth_token_inject] attention_mask shape {tuple(attention_mask.shape)} "
                    f"!= {(batch_size, seq_len)}"
                )
            valid = attention_mask.new_ones((batch_size, num_insert))
            kwargs["attention_mask"] = _insert_batched_2d(
                attention_mask,
                insert_idx.to(attention_mask.device),
                valid,
            )

        inserted_cam = cam.new_full((batch_size, num_insert), -1)
        new_cam = _insert_batched_2d(cam, insert_idx.to(cam.device), inserted_cam)
        state.per_token_cam_id = new_cam

        if cam_rope_state is not None:
            cam_rope_state.per_token_cam_id = new_cam
            if bool(getattr(cam_rope_state, "epipolar_mask_enabled", False)):
                row_id = getattr(cam_rope_state, "per_token_row_id", None)
                if row_id is not None:
                    inserted_row = row_id.new_full((batch_size, num_insert), -1)
                    cam_rope_state.per_token_row_id = _insert_batched_2d(
                        row_id,
                        insert_idx.to(row_id.device),
                        inserted_row,
                    )

        return args, kwargs

    outer_handle = reinstall_depth_token_outer_hook(hf_model, _outer_pre_hook)
    inner_handle = lm.register_forward_pre_hook(_inner_pre_hook, with_kwargs=True)
    logger.info(
        "[depth_token_inject] installed outer(insert-index) + inner(sequence insertion) "
        "hooks (image_token_id=%s, num_cameras=%s, primary_cam_id=%s)",
        image_token_id,
        num_cameras,
        primary_cam_id,
    )
    return outer_handle, inner_handle
