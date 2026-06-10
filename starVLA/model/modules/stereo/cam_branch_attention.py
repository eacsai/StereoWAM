# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Vendored and adapted from StereoWorld
#   models/transformers/prope_utils.py
# and its PropeSelfAttention use in
#   models/transformers/wan_transformer_3d.py
"""Parallel PRoPE camera-geometry attention branch for Qwen3.5-GR00T.

The branch is deliberately separate from the main Qwen attention path:
it gathers image tokens only, applies zero-parameter PRoPE transforms inside a
narrow self-attention branch, zero-initializes the branch exit projection, and
scatters the result back only to image rows.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import MethodType
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_rope_hook import _PATCHED_FORWARD_DISPATCH, compute_per_token_cam_id

logger = logging.getLogger(__name__)

DEFAULT_CAM_BRANCH_LAYERS = [3, 7, 11, 15, 19, 23]


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) transform with the explicit rigid-body formula."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"expected [...,4,4] SE(3) transforms, got {tuple(transforms.shape)}")
    r_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", r_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 intrinsics to homogeneous 4x4 matrices."""
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3] intrinsics, got {tuple(Ks.shape)}")
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert no-skew 3x3 intrinsics with the explicit diagonal formula."""
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3] intrinsics, got {tuple(Ks.shape)}")
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=Ks.dtype)


def _apply_tiled_projmat(
    feats: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-token or per-camera projection matrix to 4-wide feature tiles."""
    batch, num_heads, seqlen, feat_dim = feats.shape
    dim = int(matrix.shape[-1])
    if feat_dim % dim != 0:
        raise ValueError(f"feat_dim={feat_dim} must be divisible by matrix dim={dim}")

    if matrix.shape[1] == seqlen:
        feats_4 = feats.view(batch, num_heads, seqlen, feat_dim // dim, dim)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats_4)
        return out.reshape(feats.shape)

    cameras = matrix.shape[1]
    if seqlen <= cameras or seqlen % cameras != 0:
        raise ValueError(
            f"PRoPE per-camera path needs seqlen > cameras and divisible; "
            f"got feats={tuple(feats.shape)} matrix={tuple(matrix.shape)}"
        )
    if tuple(matrix.shape) != (batch, cameras, dim, dim):
        raise ValueError(f"unexpected matrix shape {tuple(matrix.shape)}")
    out = torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape(batch, num_heads, cameras, -1, feat_dim // dim, dim),
    )
    return out.reshape(feats.shape)


def build_prope_matrix_triple(
    K_norm: torch.Tensor,
    T_cam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build PRoPE P, P transpose, and P inverse from separate K and SE(3) parts."""
    K_norm = K_norm.to(dtype=torch.float32)
    T_cam = T_cam.to(dtype=torch.float32)
    P = torch.einsum("...ij,...jk->...ik", _lift_K(K_norm), T_cam)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_SE3(T_cam),
        _lift_K(_invert_K(K_norm)),
    )
    P_T = P.transpose(-1, -2)
    return P, P_T, P_inv


def _normalize_libero_intrinsic(
    fovy_degrees: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    f = (float(image_height) / 2.0) / math.tan(math.radians(float(fovy_degrees) / 2.0))
    K = torch.tensor(
        [
            [f, 0.0, float(image_width) / 2.0],
            [0.0, f, float(image_height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    K_norm = torch.zeros_like(K)
    K_norm[0, 0] = K[0, 0] / float(image_width)
    K_norm[1, 1] = K[1, 1] / float(image_height)
    K_norm[0, 2] = K[0, 2] / float(image_width) - 0.5
    K_norm[1, 2] = K[1, 2] / float(image_height) - 0.5
    K_norm[2, 2] = 1.0
    return K_norm


def compute_libero_camera_prope_triples(
    *,
    num_cameras: int = 2,
    baseline_m: float = 0.06,
    fovy_degrees: float = 45.0,
    image_width: int = 256,
    image_height: int = 256,
    right_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct right/left LIBERO PRoPE matrix triples in token camera order."""
    if int(num_cameras) != 2:
        raise ValueError(f"cam_branch currently expects two LIBERO cameras, got {num_cameras}")

    K_norm = _normalize_libero_intrinsic(
        fovy_degrees=fovy_degrees,
        image_width=image_width,
        image_height=image_height,
    )
    T_left = torch.eye(4, dtype=torch.float32)
    T_right = torch.eye(4, dtype=torch.float32)
    # Sign convention: +baseline here is BYTE-IDENTICAL to the legacy
    # compute_libero_camera_P_stack (cam_rope.py:103), which was verified against
    # renderer ground truth (memory: feedback_stereo_view_order). Under PRoPE's
    # strict camera<-world viewmat reading a right camera at +X would carry
    # t_x=-baseline; the discrepancy is absorbed by the learned q/k/v projections
    # and only the RELATIVE transform between the two cameras matters for the
    # attention metric. Do not "fix" the sign in isolation — it would break
    # comparability with every cam_rope-era P_stack artifact.
    T_right[0, 3] = float(baseline_m)
    T_order = [T_right, T_left] if right_first else [T_left, T_right]
    T_stack = torch.stack(T_order, dim=0)
    K_stack = K_norm.unsqueeze(0).repeat(int(num_cameras), 1, 1)
    return build_prope_matrix_triple(K_stack, T_stack)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class CamBranchAttention(nn.Module):
    """Image-token-only parallel PRoPE self-attention branch."""

    def __init__(
        self,
        hidden_dim: int,
        branch_heads: int = 4,
        branch_head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_heads = int(branch_heads)
        self.branch_head_dim = int(branch_head_dim)
        self.branch_dim = self.branch_heads * self.branch_head_dim
        if self.branch_heads <= 0:
            raise ValueError(f"branch_heads must be positive, got {branch_heads}")
        if self.branch_head_dim % 4 != 0:
            raise ValueError(f"branch_head_dim must be divisible by 4, got {branch_head_dim}")

        self.q_proj = nn.Linear(self.hidden_dim, self.branch_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.branch_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.branch_dim, bias=False)
        self.norm_q = _RMSNorm(self.branch_head_dim, eps=rms_norm_eps)
        self.norm_k = _RMSNorm(self.branch_head_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.branch_dim, self.hidden_dim, bias=True)
        self.capture_raw_output = False
        self.last_raw_output: Optional[torch.Tensor] = None

        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _project_heads(self, proj: nn.Linear, hidden_img: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        batch, n_img, _ = hidden_img.shape
        x = proj(hidden_img).view(batch, n_img, self.branch_heads, self.branch_head_dim)
        x = norm(x)
        return x.transpose(1, 2).contiguous()

    def forward_raw(
        self,
        hidden_img: torch.Tensor,
        P_img: torch.Tensor,
        P_T_img: torch.Tensor,
        P_inv_img: torch.Tensor,
    ) -> torch.Tensor:
        """Return merged branch features before the zero-initialized out projection."""
        if hidden_img.ndim != 3:
            raise ValueError(f"hidden_img must be [B,N,H], got {tuple(hidden_img.shape)}")
        if hidden_img.shape[1] == 0:
            return hidden_img.new_zeros(hidden_img.shape[0], 0, self.branch_dim)

        matrix_dtype = hidden_img.dtype
        P_img = P_img.to(device=hidden_img.device, dtype=matrix_dtype)
        P_T_img = P_T_img.to(device=hidden_img.device, dtype=matrix_dtype)
        P_inv_img = P_inv_img.to(device=hidden_img.device, dtype=matrix_dtype)

        q = self._project_heads(self.q_proj, hidden_img, self.norm_q)
        k = self._project_heads(self.k_proj, hidden_img, self.norm_k)
        v = self.v_proj(hidden_img).view(
            hidden_img.shape[0],
            hidden_img.shape[1],
            self.branch_heads,
            self.branch_head_dim,
        ).transpose(1, 2).contiguous()

        q = _apply_tiled_projmat(q, P_T_img)
        k = _apply_tiled_projmat(k, P_inv_img)
        v = _apply_tiled_projmat(v, P_inv_img)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = _apply_tiled_projmat(out, P_img)
        return out.transpose(1, 2).reshape(hidden_img.shape[0], hidden_img.shape[1], self.branch_dim)

    def forward(
        self,
        hidden_img: torch.Tensor,
        P_img: torch.Tensor,
        P_T_img: torch.Tensor,
        P_inv_img: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.forward_raw(hidden_img, P_img, P_T_img, P_inv_img)
        if self.capture_raw_output:
            self.last_raw_output = raw.detach()
        return self.out_proj(raw)


@dataclass
class CamBranchState:
    image_positions: Optional[torch.Tensor] = None
    P_img: Optional[torch.Tensor] = None
    P_T_img: Optional[torch.Tensor] = None
    P_inv_img: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    forward_step: int = 0
    logging_frequency: int = 20
    # Per-layer memo of the last logged forward_step: kills duplicate log lines (and
    # the extra GPU sync) when gradient checkpointing re-runs the wrapped forward
    # during backward recompute.
    last_logged_step: dict = None
    top_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    top_pre_hook_fn: Optional[Callable] = None

    def clear(self) -> None:
        self.image_positions = None
        self.P_img = None
        self.P_T_img = None
        self.P_inv_img = None
        self.per_token_cam_id = None


def _language_model_layers(hf_model) -> nn.ModuleList:
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("[cam_branch] could not locate language_model.layers")
    return lm.layers


def _layer_attention(layer: nn.Module) -> Optional[nn.Module]:
    for child_name in ("self_attn", "attention", "attn"):
        if hasattr(layer, child_name):
            return getattr(layer, child_name)
    return None


def _parse_layers(layers: Optional[Sequence[int] | str | int]) -> List[int]:
    if layers is None:
        return list(DEFAULT_CAM_BRANCH_LAYERS)
    if isinstance(layers, int):
        # OmegaConf YAML-parses a single-layer CLI value like "11" into a bare int.
        return [layers]
    if isinstance(layers, str):
        if not layers.strip():
            return list(DEFAULT_CAM_BRANCH_LAYERS)
        return [int(part.strip()) for part in layers.split(",") if part.strip()]
    return [int(layer_idx) for layer_idx in layers]


def _discover_supported_attention_layers(hf_model) -> List[Tuple[int, nn.Module]]:
    out: List[Tuple[int, nn.Module]] = []
    # Contract note: _PATCHED_FORWARD_DISPATCH is borrowed from cam_rope_hook purely
    # as "the set of standard softmax attention classes whose forward returns a
    # post-o_proj residual-ready (attn_output[B,S,H], ...) tuple". If a new class is
    # ever added there for cam_rope purposes, re-verify that contract holds before
    # letting cam_branch install on it (the runtime shape check catches structural
    # but not semantic drift).
    supported = set(_PATCHED_FORWARD_DISPATCH.keys())
    for layer_idx, layer in enumerate(_language_model_layers(hf_model)):
        attn = _layer_attention(layer)
        if attn is not None and type(attn).__name__ in supported:
            out.append((layer_idx, attn))
    return out


def _make_cam_branch_forward(
    *,
    orig_forward: Callable,
    branch: CamBranchAttention,
    state: CamBranchState,
    layer_idx: int,
) -> Callable:
    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        output = orig_forward(hidden_states, *args, **kwargs)
        if state.image_positions is None or state.P_img is None:
            return output

        attn_output = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(attn_output) or attn_output.ndim != 3:
            raise RuntimeError(
                f"[cam_branch] layer {layer_idx} expected attention output [B,S,H], "
                f"got {type(attn_output).__name__}"
            )

        positions = state.image_positions.to(device=hidden_states.device)
        if positions.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                f"[cam_branch] cached batch={positions.shape[0]} but hidden batch={hidden_states.shape[0]}"
            )
        gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hidden_img = hidden_states.gather(dim=1, index=gather_index)

        branch_out = branch(
            hidden_img,
            state.P_img,
            state.P_T_img,
            state.P_inv_img,
        ).to(device=attn_output.device, dtype=attn_output.dtype)

        if (
            state.logging_frequency > 0
            and state.forward_step % state.logging_frequency == 0
            and (state.last_logged_step or {}).get(layer_idx) != state.forward_step
        ):
            if state.last_logged_step is None:
                state.last_logged_step = {}
            state.last_logged_step[layer_idx] = state.forward_step
            norm = float(branch_out.detach().float().norm().cpu())
            logger.info(
                "[cam_branch] fwd=%d layer=%d branch_out_l2=%.6g",
                state.forward_step,
                layer_idx,
                norm,
            )

        out = attn_output.clone()
        out.scatter_add_(dim=1, index=gather_index, src=branch_out)
        if isinstance(output, tuple):
            return (out, *output[1:])
        return out

    return _forward


def install_cam_branch(
    hf_model,
    *,
    branch_heads: int = 4,
    branch_head_dim: int = 128,
    layers: Optional[Sequence[int] | str] = None,
    num_cameras: int = 2,
    baseline_m: float = 0.06,
    fovy_degrees: float = 45.0,
    image_width: int = 256,
    image_height: int = 256,
    spatial_merge_size: int = 2,
    right_first: bool = True,
    extra_image_cam_id: Optional[int] = None,
    logging_frequency: int = 20,
) -> Tuple[CamBranchState, List[CamBranchAttention]]:
    """Install image-token-only cam_branch wrappers on selected softmax layers.

    Constraint: the per-batch bookkeeping requires a UNIFORM image-token count per
    sample (it hard-raises otherwise). This holds for all VLA stereo batches (every
    sample = 2 same-resolution images). Co-train batches with variable image sizes
    (train_starvla_cotrain.py + sharegpt4v) are NOT supported with cam_branch on.
    """
    supported = _discover_supported_attention_layers(hf_model)
    supported_by_idx = {idx: attn for idx, attn in supported}
    target_layers = _parse_layers(layers)
    missing = [idx for idx in target_layers if idx not in supported_by_idx]
    if missing:
        raise RuntimeError(
            f"[cam_branch] requested layers {missing} are not supported softmax attention layers; "
            f"supported={sorted(supported_by_idx)}"
        )

    text_cfg = hf_model.config.text_config if hasattr(hf_model.config, "text_config") else hf_model.config
    hidden_dim = int(text_cfg.hidden_size)
    image_token_id = int(hf_model.config.image_token_id)
    P_stack, P_T_stack, P_inv_stack = compute_libero_camera_prope_triples(
        num_cameras=num_cameras,
        baseline_m=baseline_m,
        fovy_degrees=fovy_degrees,
        image_width=image_width,
        image_height=image_height,
        right_first=right_first,
    )
    state = CamBranchState(logging_frequency=max(int(logging_frequency), 0))
    branches: List[CamBranchAttention] = []

    for layer_idx in target_layers:
        attn = supported_by_idx[layer_idx]
        if getattr(attn, "_stereo_cam_branch_installed", False):
            raise RuntimeError(f"[cam_branch] layer {layer_idx} already has cam_branch installed")
        branch = CamBranchAttention(
            hidden_dim=hidden_dim,
            branch_heads=branch_heads,
            branch_head_dim=branch_head_dim,
        )
        base_dtype = attn.q_proj.weight.dtype if hasattr(attn, "q_proj") else next(attn.parameters()).dtype
        branch = branch.to(dtype=base_dtype)
        orig_forward = attn.forward
        attn._stereo_cam_branch_installed = True
        attn._stereo_cam_branch_layer_idx = int(layer_idx)
        attn._stereo_cam_branch_orig_forward = orig_forward
        attn.forward = MethodType(
            _make_cam_branch_forward(
                orig_forward=orig_forward,
                branch=branch,
                state=state,
                layer_idx=int(layer_idx),
            ),
            attn,
        )
        branches.append(branch)

    def _pre_forward_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.forward_step += 1
        if input_ids is None:
            # Fail-open by design (matches the cam_rope hook convention), but warn
            # once: a future inputs_embeds-only call path would otherwise silently
            # disable a TRAINED branch — the exact silent-inert failure class this
            # project was burned by.
            if not getattr(state, "_warned_no_input_ids", False):
                state._warned_no_input_ids = True
                logger.warning(
                    "[cam_branch] forward received no input_ids — branch SKIPPED for "
                    "such calls. If this appears during training, the branch is not "
                    "being exercised; investigate the call path."
                )
            state.clear()
            return None

        per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=int(num_cameras),
            spatial_merge_size=int(spatial_merge_size),
            extra_image_cam_id=extra_image_cam_id,
        )
        # Hot path: the code BELOW adds only 2 host-device syncs per step (the old
        # per-sample nonzero loop + per-check .item() readbacks cost ~B+6). NOTE:
        # compute_per_token_cam_id above retains its own pre-existing host readbacks
        # (cam_rope_hook.py tolist/nonzero) — that cost is shared with every live
        # depth-token/FFS arm and is part of their measured healthy baseline; it
        # moves to a vectorized neutral module at cam_rope soft-retirement.
        image_mask = per_token_cam_id >= 0
        bsz = int(input_ids.shape[0])
        flat_idx = image_mask.nonzero(as_tuple=False)  # sync 1; row-major order
        n_total = int(flat_idx.shape[0])
        if n_total == 0:
            state.clear()
            state.per_token_cam_id = per_token_cam_id
            return None
        if n_total % bsz != 0:
            raise RuntimeError(
                f"[cam_branch] image token total {n_total} not divisible by batch {bsz} — "
                "per-sample image-token counts differ; uniform counts are required."
            )
        n_img = n_total // bsz

        # Fused fail-closed validation, single boolean readback (sync 2):
        # (a) completeness — compute_per_token_cam_id silently leaves mismatched image
        #     runs at -1; partial coverage must refuse, not shrink the branch.
        # (b) uniformity — the .view(bsz, n_img) reshape below would silently MISALIGN
        #     positions if counts were e.g. [96,160] (sum still divisible by bsz).
        raw_image_counts = (input_ids == image_token_id).sum(dim=1)
        assigned_counts = image_mask.sum(dim=1)
        bad = (raw_image_counts != assigned_counts).any() | (assigned_counts != n_img).any()
        if bool(bad):
            raise RuntimeError(
                "[cam_branch] cam-id assignment incomplete or non-uniform: raw image tokens "
                f"{raw_image_counts.tolist()} vs assigned {assigned_counts.tolist()} "
                f"(expected uniform {n_img}/sample). Refusing."
            )

        positions = flat_idx[:, 1].view(bsz, n_img)
        cam_ids = per_token_cam_id.gather(dim=1, index=positions)
        # cam-id range validation only while warming up: violations can only come from
        # compute_per_token_cam_id logic (layout-independent), so the first forwards
        # prove it; skipping the .item() readbacks afterwards keeps the path async.
        if state.forward_step <= 3:
            min_cam = int(cam_ids.min().item())
            max_cam = int(cam_ids.max().item())
            if min_cam < 0 or max_cam >= int(num_cameras):
                raise RuntimeError(
                    f"[cam_branch] invalid image cam ids: min={min_cam} max={max_cam}"
                )

        P = P_stack.to(device=input_ids.device, dtype=torch.float32)
        P_T = P_T_stack.to(device=input_ids.device, dtype=torch.float32)
        P_inv = P_inv_stack.to(device=input_ids.device, dtype=torch.float32)
        state.per_token_cam_id = per_token_cam_id
        state.image_positions = positions
        state.P_img = P[cam_ids]
        state.P_T_img = P_T[cam_ids]
        state.P_inv_img = P_inv[cam_ids]
        return None

    state.top_pre_hook_fn = _pre_forward_hook
    state.top_pre_hook_handle = hf_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    logger.info(
        "[cam_branch] installed %d branches on language_model.layers%s "
        "(heads=%d, head_dim=%d, num_cameras=%d, baseline_m=%.4f, right_first=%s)",
        len(branches),
        target_layers,
        int(branch_heads),
        int(branch_head_dim),
        int(num_cameras),
        float(baseline_m),
        bool(right_first),
    )
    return state, branches
