"""Depth-token sequence insertion for QwenPI + frozen FFS.

This module turns the frozen FoundationStereo net[0] feature into a short run
of learned depth tokens, then inserts those tokens into the Qwen VLM sequence
immediately BEFORE the primary-view (left) image-token run (see the insertion
point in the language_model pre-hook below), reusing the primary cell's
position id.

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

    @staticmethod
    def build_pool(pool_hw: int) -> nn.Module:
        pool_hw = int(pool_hw)
        return nn.AdaptiveAvgPool2d((pool_hw, pool_hw))

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
        self.pool = self.build_pool(pool_hw)
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
        x = self.pool(net0)                    # (B, C, pool_hw, pool_hw)
        return self.project_pooled(x)

    def project_pooled(self, pooled_net0: torch.Tensor) -> torch.Tensor:
        if pooled_net0.dim() != 4:
            raise ValueError(
                f"DepthTokenProjector expects pooled (B,C,H,W), got {tuple(pooled_net0.shape)}"
            )
        expected = (self.in_ch, self.pool_hw, self.pool_hw)
        if tuple(int(x) for x in pooled_net0.shape[1:]) != expected:
            raise ValueError(
                f"DepthTokenProjector expected pooled tail {expected}, got "
                f"{tuple(int(x) for x in pooled_net0.shape[1:])}"
            )
        x = pooled_net0
        x = x.flatten(2).transpose(1, 2)       # (B, 16, C)
        return self.proj(x.to(self.proj.weight.dtype))                    # (B, 16, D)


@dataclass
class DepthTokenInjectState:
    depth_tokens: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    original_per_token_cam_id: Optional[torch.Tensor] = None
    insert_idx: Optional[torch.Tensor] = None
    keep_mask: Optional[torch.Tensor] = None
    position_ids_before: Optional[torch.Tensor] = None
    position_ids_after: Optional[torch.Tensor] = None
    inserted_depth_embeds: Optional[torch.Tensor] = None
    original_seq_len: Optional[int] = None
    primary_cam_id: int = 0


_STATE = DepthTokenInjectState()


def set_depth_tokens(depth_tokens: torch.Tensor) -> None:
    _STATE.depth_tokens = depth_tokens


def clear_state() -> None:
    _STATE.depth_tokens = None
    _STATE.per_token_cam_id = None
    _STATE.original_per_token_cam_id = None
    _STATE.insert_idx = None
    _STATE.keep_mask = None
    _STATE.position_ids_before = None
    _STATE.position_ids_after = None
    _STATE.inserted_depth_embeds = None
    _STATE.original_seq_len = None


def get_state() -> DepthTokenInjectState:
    return _STATE


def _extract_input_ids(args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and args:
        input_ids = args[0]
    return input_ids


def _locate_language_model(hf_model):
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
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
    batch_size, seq_len = x.shape
    num_insert = int(inserted.shape[1])
    insert_idx = insert_idx.to(device=x.device, dtype=torch.long)
    src_pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
    src_pos = src_pos.expand(batch_size, seq_len)
    out_pos = src_pos + (src_pos >= insert_idx.unsqueeze(1)).to(torch.long) * num_insert
    ins_pos = insert_idx.unsqueeze(1) + torch.arange(num_insert, device=x.device, dtype=torch.long).unsqueeze(0)

    out = x.new_empty((batch_size, seq_len + num_insert))
    out.scatter_(1, out_pos, x)
    out.scatter_(1, ins_pos, inserted)
    return out


def _insert_batched_3d(
    x: torch.Tensor,
    insert_idx: torch.Tensor,
    inserted: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, hidden_dim = x.shape
    num_insert = int(inserted.shape[1])
    insert_idx = insert_idx.to(device=x.device, dtype=torch.long)
    src_pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
    src_pos = src_pos.expand(batch_size, seq_len)
    out_pos = src_pos + (src_pos >= insert_idx.unsqueeze(1)).to(torch.long) * num_insert
    ins_pos = insert_idx.unsqueeze(1) + torch.arange(num_insert, device=x.device, dtype=torch.long).unsqueeze(0)

    out = x.new_empty((batch_size, seq_len + num_insert, hidden_dim))
    out.scatter_(1, out_pos.unsqueeze(-1).expand(-1, -1, hidden_dim), x)
    out.scatter_(1, ins_pos.unsqueeze(-1).expand(-1, -1, hidden_dim), inserted)
    return out


def _make_keep_mask(
    batch_size: int,
    seq_len: int,
    insert_idx: torch.Tensor,
    num_insert: int,
    device,
) -> torch.Tensor:
    pos = torch.arange(seq_len + num_insert, device=device, dtype=torch.long).unsqueeze(0)
    idx = insert_idx.to(device=device, dtype=torch.long).unsqueeze(1)
    inserted = (pos >= idx) & (pos < idx + num_insert)
    return ~inserted.expand(batch_size, seq_len + num_insert)


def _insert_position_columns(
    position_ids: torch.Tensor,
    insert_idx: torch.Tensor,
    inserted: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(position_ids.shape[-2])
    seq_len = int(position_ids.shape[-1])
    num_insert = int(inserted.shape[-1])
    leading = tuple(position_ids.shape[:-2])
    insert_idx = insert_idx.to(device=position_ids.device, dtype=torch.long)

    src_pos = torch.arange(seq_len, device=position_ids.device, dtype=torch.long).unsqueeze(0)
    src_pos = src_pos.expand(batch_size, seq_len)
    out_pos = src_pos + (src_pos >= insert_idx.unsqueeze(1)).to(torch.long) * num_insert
    ins_pos = insert_idx.unsqueeze(1) + torch.arange(
        num_insert, device=position_ids.device, dtype=torch.long
    ).unsqueeze(0)

    out = position_ids.new_empty((*leading, batch_size, seq_len + num_insert))
    view_prefix = (1,) * len(leading)
    out.scatter_(
        -1,
        out_pos.reshape(*view_prefix, batch_size, seq_len).expand(*leading, batch_size, seq_len),
        position_ids,
    )
    out.scatter_(
        -1,
        ins_pos.reshape(*view_prefix, batch_size, num_insert).expand(*leading, batch_size, num_insert),
        inserted,
    )
    return out


def _depth_source_indices(
    cam: torch.Tensor,
    insert_idx: torch.Tensor,
    num_insert: int,
    primary_cam_id: int,
) -> torch.Tensor:
    batch_size, seq_len = cam.shape
    device = cam.device
    primary = cam == int(primary_cam_id)
    counts = primary.sum(dim=1)
    seq_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    seq_pos = seq_pos.expand(batch_size, seq_len)
    primary_pos = torch.where(primary, seq_pos, torch.full_like(seq_pos, seq_len)).sort(dim=1).values

    if num_insert == 1:
        offsets = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    else:
        steps = torch.arange(num_insert, device=device, dtype=torch.float32).unsqueeze(0)
        scale = (counts.clamp_min(1) - 1).to(torch.float32).unsqueeze(1) / float(num_insert - 1)
        offsets = torch.round(steps * scale).to(torch.long)

    one_primary = counts == 1
    offsets = torch.where(one_primary.unsqueeze(1), torch.zeros_like(offsets), offsets)

    k = int(round(num_insert ** 0.5))
    if k * k == num_insert:
        g = torch.round(torch.sqrt(counts.to(torch.float32))).to(torch.long)
        square_ok = (counts > 0) & (g * g == counts) & (g >= k)
        grid = torch.arange(k, device=device, dtype=torch.long)
        r = torch.minimum(((2 * grid + 1).unsqueeze(0) * g.unsqueeze(1)) // (2 * k), g.unsqueeze(1) - 1)
        c = torch.minimum(((2 * grid + 1).unsqueeze(0) * g.unsqueeze(1)) // (2 * k), g.unsqueeze(1) - 1)
        square_offsets = (r.unsqueeze(2) * g.view(batch_size, 1, 1) + c.unsqueeze(1)).reshape(
            batch_size, num_insert
        )
        offsets = torch.where(square_ok.unsqueeze(1), square_offsets, offsets)

    source_idx = primary_pos.gather(1, offsets.clamp(min=0, max=max(seq_len - 1, 0)))
    fallback = (insert_idx.to(device=device, dtype=torch.long) - 1).clamp(min=0, max=max(seq_len - 1, 0))
    return torch.where((counts > 0).unsqueeze(1), source_idx, fallback.unsqueeze(1).expand(-1, num_insert))


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
    source_idx = _depth_source_indices(
        cam=cam,
        insert_idx=insert_idx.to(cam.device),
        num_insert=num_insert,
        primary_cam_id=primary_cam_id,
    ).to(position_ids.device)
    leading = tuple(position_ids.shape[:-2])
    batch_size = int(position_ids.shape[-2])
    view_prefix = (1,) * len(leading)
    gather_idx = source_idx.reshape(*view_prefix, batch_size, num_insert).expand(
        *leading, batch_size, num_insert
    )
    insert_pos = torch.gather(position_ids, dim=-1, index=gather_idx)
    return _insert_position_columns(position_ids, insert_idx.to(position_ids.device), insert_pos)


def _neutral_position_columns(
    position_ids: torch.Tensor,
    insert_idx: torch.Tensor,
    num_insert: int,
) -> torch.Tensor:
    """Insert neutral columns by repeating the insertion anchor position.

    This is for unordered global summary tokens. Unlike depth tokens, these do
    not borrow a spatial grid of primary image positions.
    """
    seq_len = int(position_ids.shape[-1])
    leading = tuple(position_ids.shape[:-2])
    batch_size = int(position_ids.shape[-2])
    anchor = (insert_idx.to(device=position_ids.device, dtype=torch.long) - 1).clamp(
        min=0,
        max=max(seq_len - 1, 0),
    )
    view_prefix = (1,) * len(leading)
    gather_idx = anchor.reshape(*view_prefix, batch_size, 1).expand(
        *leading,
        batch_size,
        num_insert,
    )
    insert_pos = torch.gather(position_ids, dim=-1, index=gather_idx)
    return _insert_position_columns(position_ids, insert_idx.to(position_ids.device), insert_pos)


def _compute_insert_idx(cam: torch.Tensor, primary_cam_id: int) -> torch.Tensor:
    # FFS #4 GR00T spec (2026-06-08): the prompt is real text in the message
    # layer; the hook inserts only depth tokens immediately before the primary
    # (left, cam_id=1 for right-first LIBERO stereo) image-token block.
    batch_size, seq_len = cam.shape
    primary = cam == int(primary_cam_id)
    pos = torch.arange(seq_len, device=cam.device, dtype=torch.long).unsqueeze(0)
    pos = pos.expand(batch_size, seq_len)
    missing = torch.full_like(pos, seq_len + 1)
    idx = torch.where(primary, pos, missing).min(dim=1).values

    # Preserve the precise fail-closed message for CPU/debug runs. On CUDA, the
    # valid training path stays fully on-device; an invalid sample will fail at
    # the first scatter instead of synchronizing every step just to format text.
    if not cam.is_cuda:
        missing_rows = (idx == seq_len + 1).nonzero(as_tuple=True)[0].tolist()
        if missing_rows:
            b = int(missing_rows[0])
            raise RuntimeError(
                f"[depth_token_inject] sample {b}: no primary_cam_id={primary_cam_id} "
                "image tokens to anchor depth-token insertion"
            )
    return idx


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
    position_mode: str = "primary_grid",
) -> Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
    """Install the outer bookkeeping hook and inner sequence-insertion hook."""
    position_mode = str(position_mode)
    if position_mode not in {"primary_grid", "neutral"}:
        raise ValueError(
            "install_depth_token_hooks position_mode must be 'primary_grid' or 'neutral', "
            f"got {position_mode!r}"
        )
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

        state.original_per_token_cam_id = cam
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
            # N2 fail-closed (review 2026-06-08): outer hook 已备好插入(insert_idx 等非 None),
            # 但 inputs_embeds 不在 kwargs(被位置参数传?) → 无法插 depth token。静默跳过会让
            # 训练退化成 baseline 且无报错 → 直接 raise。
            raise RuntimeError(
                "[depth_token_inject] insertion prepared but inputs_embeds not in kwargs "
                "(passed positionally?). Cannot inject depth tokens → fail-closed."
            )

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
        state.original_seq_len = int(seq_len)
        state.inserted_depth_embeds = depth.detach()

        kwargs["inputs_embeds"] = _insert_batched_3d(inputs_embeds, insert_idx_local, depth)
        state.keep_mask = _make_keep_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            insert_idx=insert_idx_local,
            num_insert=num_insert,
            device=inputs_embeds.device,
        )

        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # N1 fail-closed (review 2026-06-08): inputs_embeds 已扩了 depth token, 但 lm 没把
            # M-RoPE position_ids 当 kwarg 传(传 None 让 lm 内部算) → embeds 与 position 长度不一致
            # 会静默错位/学坏。直接 raise: 模型必须把 position_ids 传进 language_model.forward。
            raise RuntimeError(
                "[depth_token_inject] inputs_embeds expanded with depth tokens but position_ids "
                "is None (not a kwarg to language_model.forward) → would desync, fail-closed."
            )
        if position_ids is not None:
            state.position_ids_before = position_ids.detach()
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
            if position_mode == "primary_grid":
                kwargs["position_ids"] = _depth_position_columns(
                    position_ids=position_ids,
                    cam=cam.to(position_ids.device),
                    insert_idx=insert_idx.to(position_ids.device),
                    num_insert=num_insert,
                    primary_cam_id=int(primary_cam_id),
                )
            else:
                kwargs["position_ids"] = _neutral_position_columns(
                    position_ids=position_ids,
                    insert_idx=insert_idx.to(position_ids.device),
                    num_insert=num_insert,
                )
            state.position_ids_after = kwargs["position_ids"].detach()

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
        "hooks (image_token_id=%s, num_cameras=%s, primary_cam_id=%s, position_mode=%s)",
        image_token_id,
        num_cameras,
        primary_cam_id,
        position_mode,
    )
    return outer_handle, inner_handle
