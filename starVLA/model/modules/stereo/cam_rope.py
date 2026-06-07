"""Stereo Camera-Frame RoPE — dim-expansion variant (StereoWorld 2603.17375 Eq 6-8).

Phase 3 Level B. Adds learnable q_cam / k_cam projections of dimension d_c to
SELECTED standard attention layers, so the attention dot product gets an extra
'camera-aware' contribution:

    q̃_i = [q_i | q_cam_i] ∈ ℝ^(d + d_c)
    k̃_j = [k_j | k_cam_j] ∈ ℝ^(d + d_c)
    v_j  ∈ ℝ^d          (zero-padded to d + d_c only for Flash/SDPA shape compat)

    attn_score(i, j) = (q̃_i · k̃_j) / sqrt(d)
                     = (q_i · k_j + q_cam_i · k_cam_j) / sqrt(d)
                                            ^^^^^^^^^^^^^^^^^^
                          camera-conditioned via I_{d_c/4} ⊗ P_t rotation

P_t = lift(K) @ T_t                                              (Eq 5 / 20)
For LIBERO stereo:
    T_left  = I_4
    T_right = translate(+baseline, 0, 0)        # default baseline = 6 cm

Init: **Zero** (W_q_cam = W_k_cam = 0). At step 0, q_cam = k_cam = 0 so the
extended attention output is byte-identical to the unpatched baseline (verified
by smoke test). Paper's Copy-Init-from-temporal-axis is not applicable here:
Qwen3.5-0.8B's standard attention layers use 1D RoPE (no M-RoPE temporal
subspace to copy from); Zero Init is paper Eq's other variant.

Scope: only the 6 of 24 layers that use Qwen3_5Attention (positions
3, 7, 11, 15, 19, 23). The other 18 are Qwen3_5GatedDeltaNet (linear
attention) — paper's softmax-attention math does not apply there.

Geometric helper (apply_camera_rope) mirrors PRoPE's _apply_tiled_projmat
function (NeurIPS 2025 arxiv 2507.10496); we re-implement to avoid an extra
dependency. The math is identical.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Camera projection: K, T -> 4x4 P matrix (one per camera).
# --------------------------------------------------------------------------- #

def _normalize_intrinsic(K: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """PRoPE-style intrinsic normalization: scale focal by image dim, shift principal."""
    K_norm = torch.zeros_like(K)
    K_norm[0, 0] = K[0, 0] / image_width
    K_norm[1, 1] = K[1, 1] / image_height
    K_norm[0, 2] = K[0, 2] / image_width - 0.5
    K_norm[1, 2] = K[1, 2] / image_height - 0.5
    K_norm[2, 2] = 1.0
    return K_norm


def _lift_K_to_4x4(K: torch.Tensor) -> torch.Tensor:
    """Embed a 3x3 intrinsic into a homogeneous 4x4 (image<-camera) transform."""
    out = torch.zeros(4, 4, device=K.device, dtype=K.dtype)
    out[:3, :3] = K
    out[3, 3] = 1.0
    return out


def compute_libero_camera_P_stack(
    fovy_degrees: float = 45.0,
    image_width: int = 256,
    image_height: int = 256,
    baseline_m: float = 0.06,
    right_first: bool = True,
) -> torch.Tensor:
    """Compute the per-camera P_t matrices for LIBERO 2-camera stereo.

    Returns
    -------
    P_stack : (num_cameras=2, 4, 4) float32 tensor
        P_stack[0] = P_right = lift(K_norm) @ T_right (right_view, +baseline_m in X) -- RIGHT-FIRST
        P_stack[1] = P_left  = lift(K_norm) @ T_left  (primary, origin)

    Notes
    -----
    LIBERO agentview's default fovy is 45 degrees; render resolution 256x256.
    These are configurable (in case future renders use a different camera).

    right_first (default True): P_stack[0]=right_view(+baseline), [1]=primary(origin),
    matching the right-first convention (video_keys=[right_view, primary]). Set
    right_first=False for OLD left-first runs (video_keys=[primary, right_view]).
    """
    f = (image_height / 2.0) / math.tan(math.radians(fovy_degrees / 2.0))
    K = torch.tensor([
        [f, 0.0, image_width / 2.0],
        [0.0, f, image_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    K_norm = _normalize_intrinsic(K, image_width=image_width, image_height=image_height)
    K_4 = _lift_K_to_4x4(K_norm)

    T_left = torch.eye(4, dtype=torch.float32)
    T_right = torch.eye(4, dtype=torch.float32)
    T_right[0, 3] = baseline_m  # +baseline meters in camera-X (right of left camera)

    P_left = K_4 @ T_left
    P_right = K_4 @ T_right
    order = [P_right, P_left] if right_first else [P_left, P_right]
    return torch.stack(order, dim=0)  # right_first(default): cam0=right_view(+baseline); else left-first (old): cam0=primary(origin)


# --------------------------------------------------------------------------- #
# Apply I_{d_c/4} ⊗ P_t rotation on a (B, H, S, d_c) tensor — mirrors PRoPE.
# --------------------------------------------------------------------------- #

def apply_camera_rope(
    q_cam: torch.Tensor,
    per_token_cam_id: torch.Tensor,
    P_stack: torch.Tensor,
) -> torch.Tensor:
    """Apply I_{d_c/4} ⊗ P_{cam_id} on each token's d_c-dim vector.

    Parameters
    ----------
    q_cam : (B, H, S, d_c) float tensor
        Camera-RoPE branch features (already post-projection).
    per_token_cam_id : (B, S) long tensor
        Camera id per token: -1 for text (zero out), 0..num_cameras-1 for image tokens.
        Out-of-range ids are treated as 0 because the corresponding token gets
        zero-masked anyway (mask applied first).
    P_stack : (num_cameras, 4, 4) float tensor
        Pre-computed camera projection matrices.

    Returns
    -------
    Tensor with same shape (B, H, S, d_c). Text positions are exactly zero.

    Math reference (StereoWorld Eq 7):
        R̃^cam_t(d + d_c) =  block-diag(  R_orig(d),  I_{d_c/4} ⊗ P_t  )
        The d block is preserved; only the new d_c block gets P_t rotation.

    Implementation:
        Reshape (B, H, S, d_c) -> (B, H, S, d_c/4, 4); apply 4x4 P_t along the
        last axis using einsum. Equivalent to mat-mul with I_{d_c/4} ⊗ P_t.
    """
    B, H, S, d_c = q_cam.shape
    if d_c % 4 != 0:
        raise ValueError(f'd_c must be divisible by 4, got {d_c}')
    k = d_c // 4

    # Cast P to q_cam's dtype/device
    P = P_stack.to(device=q_cam.device, dtype=q_cam.dtype)

    # Zero text positions BEFORE looking up P (so cam_id=-1 doesn't index P out of range).
    is_image = (per_token_cam_id >= 0)                # (B, S) bool
    mask = is_image.to(q_cam.dtype).view(B, 1, S, 1)  # (B, 1, S, 1) -> broadcast over H, d_c
    q_cam = q_cam * mask

    # Per-token P_t lookup. cam_id clipped to non-negative for indexing.
    cam_id_clipped = per_token_cam_id.clamp(min=0)    # (B, S)
    P_per_token = P[cam_id_clipped]                   # (B, S, 4, 4)

    # Reshape to k blocks of 4: (B, H, S, k, 4)
    q_cam_4 = q_cam.view(B, H, S, k, 4)

    # einsum: out[b, h, s, kk, i] = sum_j P[b, s, i, j] * q_cam_4[b, h, s, kk, j]
    q_cam_rot = torch.einsum('bsij,bhskj->bhski', P_per_token, q_cam_4)

    return q_cam_rot.reshape(B, H, S, d_c)


# --------------------------------------------------------------------------- #
# Per-attention-layer state holder: q_cam_proj + k_cam_proj projections.
# --------------------------------------------------------------------------- #

class StereoCamRoPELayer(nn.Module):
    """Per-standard-attention-layer extra projections for the camera-RoPE branch.

    Owns the learnable q_cam_proj and k_cam_proj weights for ONE attention layer.
    Holds (by reference, not ownership) the per-batch state and the constant P_stack.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads_q: int,
        n_heads_kv: int,
        d_c: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.d_c = d_c

        # Q: hidden_dim -> n_heads_q * d_c
        # K: hidden_dim -> n_heads_kv * d_c (GQA: fewer KV heads)
        self.q_cam_proj = nn.Linear(hidden_dim, n_heads_q * d_c, bias=False)
        self.k_cam_proj = nn.Linear(hidden_dim, n_heads_kv * d_c, bias=False)

        # === Zero Init ===
        # W = 0 at init -> q_cam = k_cam = 0 for all inputs -> attention score
        # in the extended d_c block is exactly 0. With scale=1/sqrt(head_dim) we
        # preserve byte-identical step-0 behavior (verified via smoke test).
        # The model gradually activates the camera signal as gradients accumulate.
        nn.init.zeros_(self.q_cam_proj.weight)
        nn.init.zeros_(self.k_cam_proj.weight)

        # These are set by the framework after construction.
        self.state_holder = None       # StereoCamRoPEState (per-batch)
        # camera_P_stack set via register_buffer by install_stereo_cam_rope_hooks
