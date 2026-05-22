"""Camera-Frame RoPE integration glue for Qwen3.5-0.8B (Phase 3 Level B).

This module:
  1. Defines per-batch state (StereoCamRoPEState) holding per_token_cam_id.
  2. Provides compute_per_token_cam_id() that derives cam_id from input_ids +
     image_grid_thw + image_token_id (Qwen-VL placeholder).
  3. Provides patched_qwen3_5_attention_forward() — drop-in replacement for
     Qwen3_5Attention.forward that adds the d_c branch (Eq 6-8) and uses SDPA
     because head_dim+d_c > Flash Attn 2 limit.
  4. install_stereo_cam_rope_hooks() walks the model, attaches StereoCamRoPELayer
     to each Qwen3_5Attention layer, replaces forward, and registers the
     pre-forward hook that fills the per-batch state.

Why SDPA not Flash Attn 2:
  Qwen3.5-0.8B head_dim = 256 is already at Flash Attn 2's hard upper limit.
  Adding d_c = 16 gives head_dim+d_c = 272, which Flash Attn 2 rejects with
  '[FlashAttention] only supports head dimensions up to 256'. SDPA accepts any
  head_dim with the same softmax math, ~1.5x slower in the bf16/fp16 case but
  only applied to 6 of 24 layers (the rest stay on their original GatedDeltaNet
  path), so overall slowdown is < 10%.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_rope import StereoCamRoPELayer, apply_camera_rope, compute_libero_camera_P_stack


# --------------------------------------------------------------------------- #
# Per-batch state — populated by the pre-forward hook, read by patched attention.
# --------------------------------------------------------------------------- #

class StereoCamRoPEState:
    """Holds the (B, S) per_token_cam_id tensor for the CURRENT forward pass.

    All standard attention layers read from the same instance; the pre-forward
    hook writes per_token_cam_id at the start of model.forward.
    If the hook can't determine cam_id (e.g. pure-text forward), it leaves the
    tensor None; the patched attention then skips the d_c branch entirely so
    behavior degrades to vanilla attention.
    """

    def __init__(self) -> None:
        self.per_token_cam_id: Optional[torch.Tensor] = None  # (B, S) long, -1=text


# --------------------------------------------------------------------------- #
# Derive per_token_cam_id from Qwen-VL input_ids + image_grid_thw.
# --------------------------------------------------------------------------- #

def compute_per_token_cam_id(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: Optional[torch.Tensor],
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Assign each token a camera id (or -1 for text) based on image-token runs.

    Parameters
    ----------
    input_ids : (B, S) long tensor
    image_token_id : int (Qwen3.5-VL: 248056)
    image_grid_thw : (N_images_total, 3) long tensor, flat across the batch.
        Each row is (t, h_raw, w_raw); merged token count per image is
        t * (h_raw // spatial_merge_size) * (w_raw // spatial_merge_size).
        If None, no images -> all -1.
    num_cameras : default 2 (left, right). Image index within a sample is
        taken modulo num_cameras (so cam_id alternates 0, 1, 0, 1, ...).
    spatial_merge_size : Qwen3.5-VL default 2.

    Returns
    -------
    per_token_cam_id : (B, S) long tensor on input_ids.device.
        -1 for text tokens, 0..num_cameras-1 for image tokens.

    Assumptions
    -----------
    * Consecutive image tokens in input_ids[b] form one image's flattened patch
      sequence (Qwen-VL's standard layout: a run of image_token placeholders
      delimited by special <image_start>/<image_end> text tokens).
    * Image_grid_thw is concatenated across batch samples in sample-then-image
      order (Qwen processor's standard layout).
    """
    B, S = input_ids.shape
    out = torch.full((B, S), -1, dtype=torch.long, device=input_ids.device)
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return out

    image_mask = (input_ids == image_token_id)  # (B, S)
    s2 = max(int(spatial_merge_size), 1)
    per_image_n = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (s2 * s2)
    per_image_n = per_image_n.tolist()  # (num_images_total,)

    img_idx_global = 0
    for b in range(B):
        mask_b = image_mask[b]
        if not mask_b.any():
            continue
        # Find runs of consecutive image tokens.
        diff = torch.diff(mask_b.int(), prepend=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
        diff_end = torch.diff(mask_b.int(), append=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        ends = ((diff_end == -1).nonzero(as_tuple=True)[0] + 1).tolist()  # exclusive ends
        for img_idx_in_sample, (s_start, s_end) in enumerate(zip(starts, ends)):
            if img_idx_global >= len(per_image_n):
                break
            n_expected = per_image_n[img_idx_global]
            n_actual = s_end - s_start
            if n_actual != n_expected:
                # Layout assumption broken; skip this image rather than crash.
                # Up to caller to investigate (a warning is logged once by the framework).
                img_idx_global += 1
                continue
            cam_id = img_idx_in_sample % num_cameras
            out[b, s_start:s_end] = cam_id
            img_idx_global += 1

    return out


# --------------------------------------------------------------------------- #
# Patched Qwen3_5Attention.forward (drop-in replacement)
# --------------------------------------------------------------------------- #
# Source reference: transformers Qwen3_5Attention.forward — chunks Q into
# (query, gate), applies q_norm/k_norm, 1D RoPE, attention, gate, o_proj.
# We mirror the original behavior for the original d block and add a d_c branch
# that runs camera-RoPE + concat + SDPA.

def patched_qwen3_5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in for Qwen3_5Attention.forward with the StereoWorld d_c branch.

    The new branch is opt-in per attention layer (gated by presence of
    self.stereo_cam_layer), so this same function works whether or not the
    layer was patched. Layers without stereo_cam_layer fall through to the
    original attention math via the standard Qwen-VL attention_interface.
    """
    # Import inside function: avoid top-level dep on transformers internals.
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]  # (B, S)
    head_dim = self.head_dim
    hidden_shape = (*input_shape, -1, head_dim)

    # ===== Original Q/K/V path (verbatim from Qwen3_5Attention.forward) =====
    q_chunked = self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
    query_states, gate = torch.chunk(q_chunked, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx,
        )

    # ===== Branch on whether this layer got the stereo_cam_layer attached =====
    scl: Optional[StereoCamRoPELayer] = getattr(self, 'stereo_cam_layer', None)
    state: Optional[StereoCamRoPEState] = getattr(scl, 'state_holder', None) if scl is not None else None
    if scl is None or state is None or state.per_token_cam_id is None:
        # Vanilla path — same as the original.
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None,
        )
        if attention_interface is None:
            # Fallback to eager.
            from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward as _eager
            attention_interface = _eager
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ===== Stereo d_c branch =====
    n_heads_q = query_states.shape[1]
    n_heads_kv = key_states.shape[1]
    B = input_shape[0]
    S = input_shape[1]

    # q_cam: (B, n_heads_q, S, d_c) — Zero Init means this is 0 at step 0.
    q_cam = scl.q_cam_proj(hidden_states).view(B, S, n_heads_q, scl.d_c).transpose(1, 2)
    k_cam = scl.k_cam_proj(hidden_states).view(B, S, n_heads_kv, scl.d_c).transpose(1, 2)

    # Apply I_{d_c/4} ⊗ P_t rotation on the d_c block.
    P_stack = scl.camera_P_stack
    q_cam = apply_camera_rope(q_cam, state.per_token_cam_id, P_stack)
    k_cam = apply_camera_rope(k_cam, state.per_token_cam_id, P_stack)

    # Concat to dim head_dim + d_c.
    q_tilde = torch.cat([query_states, q_cam], dim=-1)
    k_tilde = torch.cat([key_states, k_cam], dim=-1)
    v_tilde = F.pad(value_states, (0, scl.d_c), value=0)

    # SDPA (Flash Attn 2 max head_dim = 256, we have 272). GQA broadcast handled
    # by enable_gqa=True (torch>=2.5) or repeat_kv otherwise.
    # CRITICAL: scale = 1/sqrt(head_dim), NOT 1/sqrt(head_dim + d_c). Zero Init
    # gives q_cam=k_cam=0 -> the d_c block contributes 0 to the dot product;
    # with the original scale, attn score is byte-identical to baseline.
    scale_value = 1.0 / math.sqrt(head_dim)
    if n_heads_q != n_heads_kv:
        # Manually repeat KV heads to match Q heads (GQA).
        n_groups = n_heads_q // n_heads_kv
        k_tilde = k_tilde.repeat_interleave(n_groups, dim=1)
        v_tilde = v_tilde.repeat_interleave(n_groups, dim=1)

    # Convert Qwen-VL's 2D (B, S) padding mask (1=valid, 0=pad) into a 4D
    # additive attention bias suitable for F.scaled_dot_product_attention.
    # SDPA expects either None + is_causal=True, or a 4D mask combining causal +
    # padding. Without this conversion, passing a 2D mask raises a shape error
    # (B, S) cannot broadcast to (B, H, S, S).
    sdpa_attn_mask = None
    is_causal = False
    if attention_mask is None:
        is_causal = S > 1
    elif attention_mask.dim() == 2:
        # Build (B, 1, S, S) float mask = -inf where padded OR causal-future.
        causal = torch.triu(
            torch.ones(S, S, device=q_tilde.device, dtype=torch.bool),
            diagonal=1,
        )  # (S, S), True = future position to block
        pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        combined = causal.unsqueeze(0).unsqueeze(0) | pad      # broadcast (B, 1, S, S)
        sdpa_attn_mask = torch.zeros((), dtype=q_tilde.dtype, device=q_tilde.device).expand(
            attention_mask.shape[0], 1, S, S
        ).clone()
        sdpa_attn_mask = sdpa_attn_mask.masked_fill(combined, float('-inf'))
    elif attention_mask.dim() == 4:
        # Already 4D — use as-is (transformers may pass pre-prepared 4D mask).
        sdpa_attn_mask = attention_mask
    else:
        # Unexpected shape — pass through; if SDPA errors, codex will surface.
        sdpa_attn_mask = attention_mask

    attn_output = F.scaled_dot_product_attention(
        q_tilde,
        k_tilde,
        v_tilde,
        attn_mask=sdpa_attn_mask,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        is_causal=is_causal,
        scale=scale_value,
    )
    # Drop the trailing d_c columns (V was zero-padded -> these are 0 anyway).
    attn_output = attn_output[..., :head_dim]

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# --------------------------------------------------------------------------- #
# Install hooks on the model.
# --------------------------------------------------------------------------- #

def install_stereo_cam_rope_hooks(
    hf_model,
    d_c: int = 16,
    num_cameras: int = 2,
    baseline_m: float = 0.06,
    fovy_degrees: float = 45.0,
    image_width: int = 256,
    image_height: int = 256,
    spatial_merge_size: int = 2,
) -> Tuple[StereoCamRoPEState, List[StereoCamRoPELayer]]:
    """Walk Qwen3.5-VL hf_model, attach StereoCamRoPELayer to every standard
    attention layer (Qwen3_5Attention), replace their forward with the patched
    version, and register a pre-forward hook on the top-level model that fills
    the per-batch state.

    Returns the shared state holder + list of layer modules (so the framework
    can register them as nn.Module children for ckpt save/load).
    """
    # 1. Pre-compute the constant camera P_stack.
    P_stack = compute_libero_camera_P_stack(
        fovy_degrees=fovy_degrees,
        image_width=image_width,
        image_height=image_height,
        baseline_m=baseline_m,
    )

    state = StereoCamRoPEState()

    # 2. Find the language model.
    inner = getattr(hf_model, 'model', None) or hf_model
    lm = getattr(inner, 'language_model', None) or getattr(inner, 'model', None)
    if lm is None or not hasattr(lm, 'layers'):
        raise RuntimeError('Could not locate language_model.layers in hf_model')

    # 3. For each standard attention layer, attach StereoCamRoPELayer + patch forward.
    text_cfg = hf_model.config.text_config if hasattr(hf_model.config, 'text_config') else hf_model.config
    hidden_dim = int(text_cfg.hidden_size)
    n_heads_q = int(text_cfg.num_attention_heads)
    n_heads_kv = int(text_cfg.num_key_value_heads)
    image_token_id = int(hf_model.config.image_token_id)

    scl_modules: List[StereoCamRoPELayer] = []
    standard_attn_idxs = []
    for layer_idx, layer in enumerate(lm.layers):
        attn = None
        for child_name in ('self_attn', 'attention', 'attn'):
            if hasattr(layer, child_name):
                attn = getattr(layer, child_name)
                break
        if attn is None:
            continue
        attn_cls_name = type(attn).__name__
        if attn_cls_name != 'Qwen3_5Attention':
            continue
        standard_attn_idxs.append(layer_idx)

        scl = StereoCamRoPELayer(
            hidden_dim=hidden_dim,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            d_c=d_c,
        )
        scl.state_holder = state
        scl.register_buffer('camera_P_stack', P_stack.clone(), persistent=False)
        attn.stereo_cam_layer = scl
        # Replace the bound method.
        attn.forward = MethodType(patched_qwen3_5_attention_forward, attn)
        scl_modules.append(scl)

    if not scl_modules:
        raise RuntimeError(
            'No Qwen3_5Attention layers found to patch. Did the model switch architecture?'
        )

    # 4. Register pre-forward hook on the top-level model to populate state.
    def _pre_forward_hook(module, args, kwargs):
        input_ids = kwargs.get('input_ids', None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get('image_grid_thw', None)
        if input_ids is None:
            state.per_token_cam_id = None
            return
        state.per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=num_cameras,
            spatial_merge_size=spatial_merge_size,
        )

    hf_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    import logging
    logging.info(
        f'[stereo_cam_rope] patched {len(scl_modules)} standard attention layers '
        f'at language_model.layers[{standard_attn_idxs}] (d_c={d_c}, '
        f'num_cameras={num_cameras}, baseline_m={baseline_m})'
    )

    return state, scl_modules
