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
from .visual_attention import (
    parse_exact_bool,
    prepare_visual_attention_call,
    resolve_visual_attention_profile,
)

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


def build_same_row_opposite_indices(
    camera_ids: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    topk: int = 4,
) -> Tuple[torch.Tensor, int, int]:
    """Build sparse opposite-camera/same-row candidate indices.

    ``camera_ids`` and ``row_ids`` are in the compressed image-token order used
    by ``CamBranchAttention``. The function never constructs an N-by-N mask:
    each query gets the contiguous row of the other camera directly. The
    returned indices have shape ``[B, N, W]`` and ``effective_k=min(topk,W)``.
    """
    if camera_ids.ndim != 2 or row_ids.shape != camera_ids.shape:
        raise ValueError(
            f"camera_ids and row_ids must both be [B,N], got "
            f"{tuple(camera_ids.shape)} and {tuple(row_ids.shape)}"
        )
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    batch, n_tokens = camera_ids.shape
    if n_tokens == 0 or n_tokens % 2 != 0:
        raise ValueError(f"expected two equal camera grids, got N={n_tokens}")
    n_per_camera = n_tokens // 2
    candidate_batches = []
    widths = []
    for b in range(batch):
        cams = camera_ids[b]
        rows = row_ids[b]
        unique_cams = torch.unique(cams).tolist()
        if sorted(int(x) for x in unique_cams) != [0, 1]:
            raise ValueError(f"expected camera ids {{0,1}}, sample {b} has {unique_cams}")
        if int((cams == 0).sum().item()) != n_per_camera or int((cams == 1).sum().item()) != n_per_camera:
            raise ValueError(f"camera token counts are not equal in sample {b}")
        row_counts = []
        for cam in (0, 1):
            cam_rows = rows[cams == cam]
            values = torch.unique(cam_rows, sorted=True)
            if values.numel() == 0 or int(values.min().item()) < 0:
                raise ValueError(f"invalid row ids for sample {b}, camera {cam}")
            counts = torch.stack([(cam_rows == row).sum() for row in values])
            if not bool(torch.all(counts == counts[0])):
                raise ValueError(f"non-uniform row width for sample {b}, camera {cam}")
            row_counts.append(int(counts[0].item()))
        if row_counts[0] != row_counts[1] or row_counts[0] <= 0:
            raise ValueError(f"camera row widths disagree or are empty: {row_counts}")
        width = row_counts[0]
        widths.append(width)
        col = torch.arange(width, device=camera_ids.device, dtype=torch.long)
        other_base = torch.where(cams == 0, n_per_camera, 0).to(dtype=torch.long)
        candidate_batches.append(other_base[:, None] + rows.to(dtype=torch.long)[:, None] * width + col[None, :])
    if len(set(widths)) != 1:
        raise ValueError(f"all samples must have the same merged row width, got {widths}")
    width = widths[0]
    return torch.stack(candidate_batches, dim=0), min(int(topk), width), width


def _build_stereo_grid_metadata(
    *,
    per_token_cam_id: torch.Tensor,
    image_positions: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    num_cameras: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compressed camera, row and column ids with strict stereo guards."""
    if int(num_cameras) != 2:
        raise ValueError(f"E5 same-row routing currently requires 2 cameras, got {num_cameras}")
    batch, n_img = image_positions.shape
    if image_grid_thw is None or image_grid_thw.ndim != 2 or image_grid_thw.shape[0] != batch * 2:
        raise ValueError(
            f"E5 requires exactly 2 image_grid_thw rows per sample; got "
            f"shape={None if image_grid_thw is None else tuple(image_grid_thw.shape)}, B={batch}"
        )
    s2 = max(int(spatial_merge_size), 1)
    grid = image_grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
    hw = []
    for record_index, (t, h_raw, w_raw) in enumerate(grid):
        if int(t) != 1 or int(h_raw) % s2 != 0 or int(w_raw) % s2 != 0:
            raise ValueError(f"E5 requires t=1 and divisible grids, record_index={record_index}, grid={(t,h_raw,w_raw)}")
        hw.append((int(h_raw) // s2, int(w_raw) // s2))
    if len(set(hw)) != 1:
        raise ValueError(f"E5 requires identical merged H/W across all cameras, got {hw}")
    merged_h, merged_w = hw[0]
    n_per_image = merged_h * merged_w
    if n_img != 2 * n_per_image:
        raise ValueError(f"E5 image token count {n_img} != 2*H*W={2*n_per_image}")

    rows_one = torch.arange(merged_h, device=per_token_cam_id.device, dtype=torch.long).repeat_interleave(merged_w)
    cols_one = torch.arange(merged_w, device=per_token_cam_id.device, dtype=torch.long).repeat(merged_h)
    expected_rows = torch.cat((rows_one, rows_one), dim=0).expand(batch, -1)
    expected_cols = torch.cat((cols_one, cols_one), dim=0).expand(batch, -1)
    expected_cams = torch.cat(
        (
            torch.zeros(n_per_image, device=per_token_cam_id.device, dtype=torch.long),
            torch.ones(n_per_image, device=per_token_cam_id.device, dtype=torch.long),
        ),
        dim=0,
    ).expand(batch, -1)
    actual_cams = per_token_cam_id.gather(dim=1, index=image_positions)
    if not bool(torch.equal(actual_cams, expected_cams)):
        raise ValueError(
            "E5 camera binding mismatch: expected record_index=2*b+run_slot with "
            "run_slot->camera_id [0,1]; refusing possible left/right swap or non-camera-major input"
        )
    return actual_cams, expected_rows, expected_cols


class CrossViewCamBranchAttention(nn.Module):
    """Sparse opposite-camera, same-row PRoPE attention used by E5."""

    def __init__(
        self,
        hidden_dim: int,
        branch_heads: int,
        branch_head_dim: int,
        *,
        out_proj_init_std: float,
        init_seed: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_heads = int(branch_heads)
        self.branch_head_dim = int(branch_head_dim)
        self.branch_dim = self.branch_heads * self.branch_head_dim
        self.q_proj = nn.Linear(hidden_dim, self.branch_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.branch_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.branch_dim, bias=False)
        self.norm_q = _RMSNorm(self.branch_head_dim)
        self.norm_k = _RMSNorm(self.branch_head_dim)
        self.out_proj = nn.Linear(self.branch_dim, hidden_dim, bias=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.xavier_uniform_(proj.weight)
            if float(out_proj_init_std) == 0.0:
                nn.init.zeros_(self.out_proj.weight)
            else:
                nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(out_proj_init_std))
            nn.init.zeros_(self.out_proj.bias)

    def _project_heads(self, proj: nn.Linear, hidden_img: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        batch, n_img, _ = hidden_img.shape
        x = proj(hidden_img).view(batch, n_img, self.branch_heads, self.branch_head_dim)
        return norm(x).transpose(1, 2).contiguous()

    @staticmethod
    def _gather_candidates(values: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
        # Per-batch advanced indexing avoids materialising a dense [Nq,Nimage]
        # gather view. The only candidate axis is the merged row width W.
        return torch.stack(
            [values[b][:, candidate_indices[b], :] for b in range(values.shape[0])], dim=0
        )

    def forward(
        self,
        hidden_img: torch.Tensor,
        P_img: torch.Tensor,
        P_T_img: torch.Tensor,
        P_inv_img: torch.Tensor,
        candidate_indices: torch.Tensor,
        effective_k: int,
    ) -> torch.Tensor:
        if candidate_indices.ndim != 3 or int(effective_k) <= 0:
            raise ValueError("E5 candidate route is missing or empty")
        q = self._project_heads(self.q_proj, hidden_img, self.norm_q)
        k = self._project_heads(self.k_proj, hidden_img, self.norm_k)
        v = self.v_proj(hidden_img).view(
            hidden_img.shape[0], hidden_img.shape[1], self.branch_heads, self.branch_head_dim
        ).transpose(1, 2).contiguous()
        q = _apply_tiled_projmat(q, P_T_img)
        k = _apply_tiled_projmat(k, P_inv_img)
        v = _apply_tiled_projmat(v, P_inv_img)
        k_candidates = self._gather_candidates(k, candidate_indices)
        v_candidates = self._gather_candidates(v, candidate_indices)
        scores = (q.unsqueeze(3) * k_candidates).sum(dim=-1) * (self.branch_head_dim ** -0.5)
        # topk is materially faster than a full argsort. A tiny rank-dependent
        # float32 tie-break preserves the documented ascending-candidate order
        # for exact score ties without changing the gathered attention scores.
        rank_bias = torch.arange(
            scores.shape[-1], device=scores.device, dtype=torch.float32
        ).mul_(1e-7)
        select_scores = scores.float() - rank_bias.view(1, 1, 1, -1)
        _, selected = torch.topk(select_scores, k=int(effective_k), dim=-1, largest=True, sorted=True)
        selected_scores = torch.gather(scores, dim=-1, index=selected)
        selected_values = torch.gather(
            v_candidates,
            dim=3,
            index=selected.unsqueeze(-1).expand(-1, -1, -1, -1, self.branch_head_dim),
        )
        weights = torch.softmax(selected_scores.float(), dim=-1).to(dtype=selected_values.dtype)
        out = (weights.unsqueeze(-1) * selected_values).sum(dim=3)
        out = _apply_tiled_projmat(out, P_img)
        return self.out_proj(out.transpose(1, 2).reshape(hidden_img.shape[0], hidden_img.shape[1], self.branch_dim))


class CamBranchAttention(nn.Module):
    """Image-token-only parallel PRoPE self-attention branch."""

    def __init__(
        self,
        hidden_dim: int,
        branch_heads: int = 4,
        branch_head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        out_proj_init_std: float = 0.0,
        e5_enabled: bool = False,
        e5_topk: int = 4,
        e5_gate_init: float = 0.1,
        e5_out_proj_init_std: float = 0.02,
        e5_init_seed: int = 42,
        *,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.is_causal = parse_exact_bool(is_causal, name="stereo_cam_branch_is_causal")
        self.hidden_dim = int(hidden_dim)
        self.branch_heads = int(branch_heads)
        self.branch_head_dim = int(branch_head_dim)
        self.branch_dim = self.branch_heads * self.branch_head_dim
        self.out_proj_init_std = float(out_proj_init_std)
        if not math.isfinite(self.out_proj_init_std) or self.out_proj_init_std < 0.0:
            raise ValueError(
                f"out_proj_init_std must be finite and non-negative, got {out_proj_init_std}"
            )
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
        if self.out_proj_init_std == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            # Keep the optional E3 draw out of the model-wide initialization stream.
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.out_proj_init_std)
        nn.init.zeros_(self.out_proj.bias)
        self.e5_branch: Optional[CrossViewCamBranchAttention] = None
        self.e5_gate: Optional[nn.Parameter] = None
        self.e5_topk = 0
        if bool(e5_enabled):
            self.e5_topk = int(e5_topk)
            with torch.random.fork_rng(devices=[]):
                self.e5_branch = CrossViewCamBranchAttention(
                    hidden_dim=self.hidden_dim,
                    branch_heads=self.branch_heads,
                    branch_head_dim=self.branch_head_dim,
                    out_proj_init_std=float(e5_out_proj_init_std),
                    init_seed=int(e5_init_seed),
                )
                self.e5_gate = nn.Parameter(torch.tensor(float(e5_gate_init), dtype=torch.float32))

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
        # The matched causal arm uses the same SDPA primitive with an explicit
        # main-backbone mask; the branch direction is controlled independently.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
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
        if self.e5_branch is not None:
            raise RuntimeError("E5 branch requires route metadata; call forward_composite")
        return self.out_proj(raw)

    def forward_composite(
        self,
        hidden_img: torch.Tensor,
        P_img: torch.Tensor,
        P_T_img: torch.Tensor,
        P_inv_img: torch.Tensor,
        candidate_indices: Optional[torch.Tensor],
        effective_k: Optional[int],
    ) -> torch.Tensor:
        raw = self.forward_raw(hidden_img, P_img, P_T_img, P_inv_img)
        if self.capture_raw_output:
            self.last_raw_output = raw.detach()
        out = self.out_proj(raw)
        if self.e5_branch is not None:
            if candidate_indices is None or effective_k is None:
                raise RuntimeError("E5 enabled but same-row candidate route is unavailable")
            out = out + self.e5_gate.to(dtype=out.dtype) * self.e5_branch(
                hidden_img, P_img, P_T_img, P_inv_img, candidate_indices, effective_k
            )
        return out


@dataclass
class CamBranchState:
    image_positions: Optional[torch.Tensor] = None
    P_img: Optional[torch.Tensor] = None
    P_T_img: Optional[torch.Tensor] = None
    P_inv_img: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    per_token_row_id: Optional[torch.Tensor] = None
    per_token_col_id: Optional[torch.Tensor] = None
    e5_candidate_indices: Optional[torch.Tensor] = None
    e5_effective_k: Optional[int] = None
    image_grid_thw: Optional[torch.Tensor] = None
    forward_step: int = 0
    logging_frequency: int = 20
    # Per-layer memo of the last logged forward_step: kills duplicate log lines (and
    # the extra GPU sync) when gradient checkpointing re-runs the wrapped forward
    # during backward recompute.
    last_logged_step: dict = None
    top_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    top_pre_hook_fn: Optional[Callable] = None
    top_post_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    backward_finalize_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    # FFS/Utonia compatibility: when token insertion happens inside the
    # language_model pre-hook, the top-level cam_branch hook sees pre-insert
    # sequence positions. In defer mode, it only records raw cam ids/counts;
    # depth_token_inject calls refresh_from_per_token_cam_id after inserting
    # the 64 prompt/depth rows, so image_positions match the real hidden states.
    defer_position_cache_until_token_insert: bool = False
    refresh_from_per_token_cam_id: Optional[Callable] = None
    raw_image_token_counts: Optional[torch.Tensor] = None
    # Fail-closed sentinel: set True when the top hook DEFERS caching (it expects
    # depth_token_inject to call refresh after token insertion); cleared by refresh
    # or clear(). If a cam_branch layer runs while still True, insertion never
    # refreshed -> raise instead of silently degrading to base attention.
    deferred_refresh_pending: bool = False
    # Gradient-checkpointed decoder recomputation re-enters the wrapped decoder
    # layers without re-entering the top-level model hook. Keep the route alive
    # until the next top-level pre-hook clears it and rebuilds the next route.
    retain_for_checkpoint_recompute: bool = False

    def clear(self) -> None:
        self.image_positions = None
        self.P_img = None
        self.P_T_img = None
        self.P_inv_img = None
        self.per_token_cam_id = None
        self.per_token_row_id = None
        self.per_token_col_id = None
        self.e5_candidate_indices = None
        self.e5_effective_k = None
        self.raw_image_token_counts = None
        self.image_grid_thw = None
        self.deferred_refresh_pending = False
        self.retain_for_checkpoint_recompute = False


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
    visual_attention_mode: str = "legacy_mixed",
) -> Callable:
    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if state.deferred_refresh_pending:
            raise RuntimeError(
                f"[cam_branch] layer {layer_idx} reached before deferred position refresh"
            )
        call_args, call_kwargs = prepare_visual_attention_call(
            hidden_states,
            args,
            kwargs,
            image_positions=state.image_positions,
            mode=visual_attention_mode,
            layer_idx=layer_idx,
        )
        output = orig_forward(hidden_states, *call_args, **call_kwargs)
        if state.deferred_refresh_pending:
            raise RuntimeError(
                f"[cam_branch] layer {layer_idx} reached with deferred position cache still "
                "PENDING: the token-insertion path (depth_token_inject) did not call "
                "refresh_from_per_token_cam_id after inserting tokens, so a TRAINED branch "
                "would silently degrade to base attention. Refusing (fail-closed)."
            )
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

        branch_out = branch.forward_composite(
            hidden_img,
            state.P_img,
            state.P_T_img,
            state.P_inv_img,
            state.e5_candidate_indices,
            state.e5_effective_k,
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


def _make_visual_attention_forward(
    *,
    orig_forward: Callable,
    state: CamBranchState,
    layer_idx: int,
    visual_attention_mode: str,
) -> Callable:
    """Wrap a non-branch full-attention layer with the same visual mask path."""
    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if state.deferred_refresh_pending:
            raise RuntimeError(
                f"[cam_branch] layer {layer_idx} reached before deferred position refresh"
            )
        call_args, call_kwargs = prepare_visual_attention_call(
            hidden_states,
            args,
            kwargs,
            image_positions=state.image_positions,
            mode=visual_attention_mode,
            layer_idx=layer_idx,
        )
        return orig_forward(hidden_states, *call_args, **call_kwargs)
    return _forward

def install_cam_branch(
    hf_model,
    *,
    branch_heads: int = 4,
    branch_head_dim: int = 128,
    out_proj_init_std: float = 0.0,
    visual_attention_mode: str = "legacy_mixed",
    stereo_cam_branch_is_causal: bool = False,
    attention_implementation: str = "flash_attention_2",
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
    defer_position_cache_until_token_insert: bool = False,
    e5_enabled: bool = False,
    e5_layers: Optional[Sequence[int] | str] = None,
    e5_topk: int = 4,
    e5_gate_init: float = 0.1,
    e5_out_proj_init_std: float = 0.02,
    e5_init_seed: int = 42,
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
    if len(target_layers) != len(set(target_layers)):
        raise ValueError(f"duplicate Camera Branch layer indices: {target_layers}")
    visual_attention_mode, stereo_cam_branch_is_causal, attention_implementation = (
        resolve_visual_attention_profile(
            visual_attention_mode,
            stereo_cam_branch_is_causal,
            attention_implementation,
            cam_branch_enabled=True,
        )
    )
    visual_attention_enabled = visual_attention_mode != "legacy_mixed"
    e5_target_layers = _parse_layers(e5_layers) if e5_layers is not None else [11, 15, 19, 23]
    if len(e5_target_layers) != len(set(e5_target_layers)):
        raise ValueError(f"duplicate E5 layer indices: {e5_target_layers}")
    if bool(e5_enabled):
        if not set(e5_target_layers).issubset(set(target_layers)):
            raise ValueError(
                f"E5 layers {e5_target_layers} must be a subset of E3 layers {target_layers}"
            )
        unsupported_e5 = [idx for idx in e5_target_layers if idx not in supported_by_idx]
        if unsupported_e5:
            raise RuntimeError(f"E5 requested non-full/unsupported layers: {unsupported_e5}")
    missing = [idx for idx in target_layers if idx not in supported_by_idx]
    if missing:
        raise RuntimeError(
            f"[cam_branch] requested layers {missing} are not supported softmax attention layers; "
            f"supported={sorted(supported_by_idx)}"
        )

    target_set = set(target_layers)
    for layer_idx, attn in supported:
        if visual_attention_enabled:
            if getattr(attn, "_stereo_cam_branch_installed", False):
                raise RuntimeError(f"[cam_branch] layer {layer_idx} already has cam_branch installed")
            if getattr(attn, "_stereo_visual_attention_installed", False):
                raise RuntimeError(f"[cam_branch] layer {layer_idx} already has visual attention installed")
        elif layer_idx in target_set and getattr(attn, "_stereo_cam_branch_installed", False):
            raise RuntimeError(f"[cam_branch] layer {layer_idx} already has cam_branch installed")
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
    state.defer_position_cache_until_token_insert = bool(defer_position_cache_until_token_insert)
    branches: List[CamBranchAttention] = []

    for layer_idx in target_layers:
        attn = supported_by_idx[layer_idx]
        if getattr(attn, "_stereo_cam_branch_installed", False):
            raise RuntimeError(f"[cam_branch] layer {layer_idx} already has cam_branch installed")
        if visual_attention_enabled and getattr(attn, "_stereo_visual_attention_installed", False):
            raise RuntimeError(f"[cam_branch] layer {layer_idx} already has visual attention installed")
        branch = CamBranchAttention(
            hidden_dim=hidden_dim,
            branch_heads=branch_heads,
            branch_head_dim=branch_head_dim,
            out_proj_init_std=out_proj_init_std,
            is_causal=stereo_cam_branch_is_causal,
            e5_enabled=bool(e5_enabled and layer_idx in e5_target_layers),
            e5_topk=int(e5_topk),
            e5_gate_init=float(e5_gate_init),
            e5_out_proj_init_std=float(e5_out_proj_init_std),
            e5_init_seed=int(e5_init_seed) + int(layer_idx),
        )
        base_dtype = attn.q_proj.weight.dtype if hasattr(attn, "q_proj") else next(attn.parameters()).dtype
        branch = branch.to(dtype=base_dtype)
        orig_forward = attn.forward
        attn._stereo_cam_branch_installed = True
        attn._stereo_cam_branch_layer_idx = int(layer_idx)
        attn._stereo_cam_branch_orig_forward = orig_forward
        if visual_attention_enabled:
            attn._stereo_visual_attention_installed = True
        attn.forward = MethodType(
            _make_cam_branch_forward(
                orig_forward=orig_forward,
                branch=branch,
                state=state,
                layer_idx=int(layer_idx),
                visual_attention_mode=visual_attention_mode,
            ),
            attn,
        )
        branches.append(branch)

    if visual_attention_enabled:
        target_set = set(target_layers)
        for layer_idx, attn in supported:
            if layer_idx in target_set:
                continue
            if getattr(attn, "_stereo_cam_branch_installed", False):
                raise RuntimeError(f"[cam_branch] non-target layer {layer_idx} already has cam_branch installed")
            if getattr(attn, "_stereo_visual_attention_installed", False):
                raise RuntimeError(f"[cam_branch] layer {layer_idx} already has visual attention installed")
            orig_forward = attn.forward
            attn._stereo_visual_attention_installed = True
            attn._stereo_visual_attention_layer_idx = int(layer_idx)
            attn._stereo_visual_attention_orig_forward = orig_forward
            attn.forward = MethodType(
                _make_visual_attention_forward(
                    orig_forward=orig_forward,
                    state=state,
                    layer_idx=int(layer_idx),
                    visual_attention_mode=visual_attention_mode,
                ),
                attn,
            )

    def _cache_positions_from_per_token_cam_id(
        per_token_cam_id: torch.Tensor,
        raw_image_counts: Optional[torch.Tensor] = None,
    ):
        # Hot path: this adds only 2 host-device syncs per step. It works for both
        # normal cam_branch (pre-insert sequence) and token-inserting FFS paths after
        # depth_token_inject has expanded cam ids to the post-insert sequence.
        # Any invocation satisfies the deferred-refresh obligation (fail-closed sentinel).
        state.deferred_refresh_pending = False
        image_mask = per_token_cam_id >= 0
        bsz = int(per_token_cam_id.shape[0])
        flat_idx = image_mask.nonzero(as_tuple=False)  # sync 1; row-major order
        n_total = int(flat_idx.shape[0])
        if n_total == 0:
            state.clear()
            state.per_token_cam_id = per_token_cam_id
            state.raw_image_token_counts = raw_image_counts
            return None
        if n_total % bsz != 0:
            raise RuntimeError(
                f"[cam_branch] image token total {n_total} not divisible by batch {bsz} — "
                "per-sample image-token counts differ; uniform counts are required."
            )
        n_img = n_total // bsz

        assigned_counts = image_mask.sum(dim=1)
        if raw_image_counts is None:
            raw_image_counts = assigned_counts
        bad = (raw_image_counts.to(assigned_counts.device) != assigned_counts).any() | (assigned_counts != n_img).any()
        if bool(bad):
            raise RuntimeError(
                "[cam_branch] cam-id assignment incomplete or non-uniform: raw image tokens "
                f"{raw_image_counts.tolist()} vs assigned {assigned_counts.tolist()} "
                f"(expected uniform {n_img}/sample). Refusing."
            )

        positions = flat_idx[:, 1].view(bsz, n_img)
        cam_ids = per_token_cam_id.gather(dim=1, index=positions)
        if state.forward_step <= 3:
            min_cam = int(cam_ids.min().item())
            max_cam = int(cam_ids.max().item())
            if min_cam < 0 or max_cam >= int(num_cameras):
                raise RuntimeError(
                    f"[cam_branch] invalid image cam ids: min={min_cam} max={max_cam}"
                )

        P = P_stack.to(device=per_token_cam_id.device, dtype=torch.float32)
        P_T = P_T_stack.to(device=per_token_cam_id.device, dtype=torch.float32)
        P_inv = P_inv_stack.to(device=per_token_cam_id.device, dtype=torch.float32)
        state.per_token_cam_id = per_token_cam_id
        state.raw_image_token_counts = raw_image_counts
        state.image_positions = positions
        state.P_img = P[cam_ids]
        state.P_T_img = P_T[cam_ids]
        state.P_inv_img = P_inv[cam_ids]
        if bool(e5_enabled):
            if state.image_grid_thw is None:
                raise RuntimeError("E5 enabled but image_grid_thw metadata is unavailable")
            actual_cams, row_ids, col_ids = _build_stereo_grid_metadata(
                per_token_cam_id=per_token_cam_id,
                image_positions=positions,
                image_grid_thw=state.image_grid_thw,
                spatial_merge_size=int(spatial_merge_size),
                num_cameras=int(num_cameras),
            )
            candidate_indices, effective_k, _ = build_same_row_opposite_indices(
                actual_cams, row_ids, topk=int(e5_topk)
            )
            state.per_token_row_id = row_ids
            state.per_token_col_id = col_ids
            state.e5_candidate_indices = candidate_indices
            state.e5_effective_k = int(effective_k)
        return None

    state.refresh_from_per_token_cam_id = _cache_positions_from_per_token_cam_id

    def _pre_forward_hook(module, args, kwargs):
        if state.retain_for_checkpoint_recompute:
            raise RuntimeError(
                "[cam_branch] a checkpointed visual forward is still in flight; "
                "refusing an overlapping top-level forward before backward completes"
            )
        state.clear()
        # Clear stale state from the previous top-level forward. This is also
        # the lifecycle boundary after a checkpointed forward retained context
        # for backward recomputation.
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.forward_step += 1
        state.image_grid_thw = image_grid_thw
        if input_ids is None:
            if visual_attention_enabled:
                state.clear()
                raise RuntimeError(
                    "[cam_branch] visual attention mode requires input_ids so image-token "
                    "positions cannot be guessed from inputs_embeds alone"
                )
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
        raw_image_counts = (input_ids == image_token_id).sum(dim=1)
        if state.defer_position_cache_until_token_insert:
            # Token-inserting FFS paths (e.g. Utonia prompt tokens) will shift the
            # image rows inside the language_model pre-hook. Do NOT cache positions
            # here; depth_token_inject refreshes them after insertion using post-insert
            # cam ids, while retaining raw_image_counts for fail-closed validation.
            state.clear()
            state.image_grid_thw = image_grid_thw
            state.per_token_cam_id = per_token_cam_id
            state.raw_image_token_counts = raw_image_counts
            state.deferred_refresh_pending = True  # depth_token_inject MUST refresh post-insert
            return None

        return state.refresh_from_per_token_cam_id(per_token_cam_id, raw_image_counts)

    state.top_pre_hook_fn = _pre_forward_hook
    state.top_pre_hook_handle = hf_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    def _gradient_checkpointing_active() -> bool:
        candidates = [
            hf_model,
            getattr(hf_model, "model", None),
            getattr(getattr(hf_model, "model", None), "language_model", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            for attr in ("is_gradient_checkpointing", "gradient_checkpointing"):
                value = getattr(candidate, attr, False)
                if callable(value):
                    value = value()
                if bool(value):
                    return True
        return False

    def _post_forward_hook(module, args, kwargs, output):
        # Decoder-layer checkpoint recomputation does not re-enter this top-level
        # hook. Retain the latest image route until the next top-level pre-hook;
        # that pre-hook clears it before building the next route.
        if (
            bool(getattr(module, "training", False))
            and torch.is_grad_enabled()
            and _gradient_checkpointing_active()
            and (state.image_positions is not None or state.deferred_refresh_pending)
        ):
            state.retain_for_checkpoint_recompute = True
            return None
        state.clear()
        return None

    state.top_post_hook_handle = hf_model.register_forward_hook(
        _post_forward_hook, with_kwargs=True, always_call=True
    )

    def _backward_finalize_hook(grad):
        state.clear()
        return grad
    finalize_parameter = None
    decoder_layers = getattr(hf_model, "layers", None)
    if decoder_layers is not None and len(decoder_layers) > 0:
        finalize_parameter = next(
            (parameter for parameter in decoder_layers[0].parameters() if parameter.requires_grad),
            None,
        )
    if finalize_parameter is None:
        finalize_parameter = next(
            (parameter for parameter in hf_model.parameters() if parameter.requires_grad),
            None,
        )
    if finalize_parameter is not None:
        if hasattr(finalize_parameter, "register_post_accumulate_grad_hook"):
            state.backward_finalize_handle = finalize_parameter.register_post_accumulate_grad_hook(
                lambda _parameter: state.clear()
            )
        else:
            state.backward_finalize_handle = finalize_parameter.register_hook(_backward_finalize_hook)
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
