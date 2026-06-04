"""Camera-Frame RoPE integration glue for Qwen3.5-0.8B (Phase 3 B) and
Qwen3-VL-4B-Instruct (Phase 3 C, M-RoPE Copy Init ablation).

This module:
  1. Defines per-batch state (StereoCamRoPEState) holding per_token_cam_id.
  2. Provides compute_per_token_cam_id() that derives cam_id from input_ids +
     image_grid_thw + image_token_id (Qwen-VL placeholder).
  3. Provides patched_qwen3_5_attention_forward() — drop-in replacement for
     Qwen3_5Attention.forward (Qwen3.5-0.8B) that adds the d_c branch (Eq 6-8).
     branch.  Differs from the Qwen3.5 patch: no gated tail, no q_chunked
     split, multimodal 3D position_embeddings (post-apply_interleaved_mrope).
  5. install_stereo_cam_rope_hooks() walks the model, dispatches to the correct
     patched forward per attention class, optionally Copy-Init's q_cam_proj /
     k_cam_proj from the M-RoPE temporal subspace slice of q_proj / k_proj,
     and registers the pre-forward hook that fills per-batch state.

Why SDPA not Flash Attn 2:
  Qwen3.5-0.8B head_dim = 256 is already at Flash Attn 2's hard upper limit.
  Adding d_c = 16 gives head_dim+d_c = 272, which Flash Attn 2 rejects with
  '[FlashAttention] only supports head dimensions up to 256'. SDPA accepts any
  head_dim with the same softmax math. For Qwen3-VL-4B (head_dim=128, d_c=48
  -> 176) FA2 would technically fit, but we keep SDPA across both backbones
  for code symmetry (the d_c-block math is the same).
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Callable, List, Optional, Tuple

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
        # === Phase 4 epipolar mask support (backward compat: disabled by default) ===
        self.per_token_row_id: Optional[torch.Tensor] = None  # (B, S) long, -1=text/non-image
        self.epipolar_mask_enabled: bool = False
        self.top_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.top_pre_hook_fn: Optional[Callable] = None
        self.reinstall_top_pre_hook: Optional[Callable] = None


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
# --------------------------------------------------------------------------- #
# Derive per_token_row_id for epipolar mask.
# --------------------------------------------------------------------------- #

def compute_per_token_row_id(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Assign each image token a row index within its image (0-indexed). Text/non-image = -1."""
    B, S = input_ids.shape
    out = torch.full((B, S), -1, dtype=torch.long, device=input_ids.device)
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return out

    image_mask = (input_ids == image_token_id)
    s2 = max(int(spatial_merge_size), 1)
    per_image_n = ((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (s2 * s2)).tolist()
    per_image_w = (image_grid_thw[:, 2] // s2).tolist()

    img_idx_global = 0
    for b in range(B):
        mask_b = image_mask[b]
        diff = torch.diff(mask_b.int(), prepend=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        starts = torch.nonzero(diff == 1, as_tuple=True)[0].tolist()
        diff_end = torch.diff(mask_b.int(), append=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        ends = torch.nonzero(diff_end == -1, as_tuple=True)[0].tolist()
        for img_idx_in_sample, (s_start, s_end) in enumerate(zip(starts, ends)):
            if img_idx_global >= len(per_image_n):
                continue
            n_expected = per_image_n[img_idx_global]
            w_image = per_image_w[img_idx_global]
            actual_len = s_end - s_start + 1
            if actual_len != n_expected or w_image <= 0:
                img_idx_global += 1
                continue
            for i in range(actual_len):
                out[b, s_start + i] = i // w_image
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

    # === Phase 4 Epipolar attention mask (StereoWorld paper Sec 3.3) ===
    # Block cross-view image-image attention except where same horizontal row.
    if state.epipolar_mask_enabled and state.per_token_row_id is not None:
        per_cam = state.per_token_cam_id  # (B, S)
        per_row = state.per_token_row_id  # (B, S)
        i_cam = per_cam.unsqueeze(-1)
        j_cam = per_cam.unsqueeze(-2)
        i_row = per_row.unsqueeze(-1)
        j_row = per_row.unsqueeze(-2)
        epi_block = (i_cam != j_cam) & (i_cam >= 0) & (j_cam >= 0) & (i_row != j_row)  # (B, S, S) True=block
        if sdpa_attn_mask is None:
            # Need explicit 4D mask now. Build from causal (if is_causal) or empty.
            sdpa_attn_mask = torch.zeros((per_cam.shape[0], 1, S, S), dtype=q_tilde.dtype, device=q_tilde.device)
            if is_causal:
                _causal_only = torch.triu(torch.ones(S, S, device=q_tilde.device, dtype=torch.bool), diagonal=1)
                sdpa_attn_mask = sdpa_attn_mask.masked_fill(_causal_only.unsqueeze(0).unsqueeze(0), float("-inf"))
                is_causal = False
        sdpa_attn_mask = sdpa_attn_mask.masked_fill(epi_block.unsqueeze(1), float("-inf"))

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





# Dispatch table: attention class name -> patched forward function.
# Extend this when adding support for new VLM backbones.
_PATCHED_FORWARD_DISPATCH = {
    'Qwen3_5Attention': patched_qwen3_5_attention_forward,
}


def reinstall_stereo_cam_rope_outer_hook(
    top_module,
    outer_hook_fn: Callable,
) -> torch.utils.hooks.RemovableHandle:
    """Attach the cam-rope outer bookkeeping hook to a top-level wrapper."""
    handle = top_module.register_forward_pre_hook(outer_hook_fn, with_kwargs=True)
    setattr(handle, "stereo_cam_rope_outer_hook_fn", outer_hook_fn)
    setattr(
        handle,
        "reinstall_outer_hook",
        lambda new_top_module: reinstall_stereo_cam_rope_outer_hook(new_top_module, outer_hook_fn),
    )
    return handle


def install_stereo_cam_rope_hooks(
    hf_model,
    d_c: int = 16,
    num_cameras: int = 2,
    baseline_m: float = 0.06,
    fovy_degrees: float = 45.0,
    image_width: int = 256,
    image_height: int = 256,
    spatial_merge_size: int = 2,
    init_mode: str = 'zero',
    epipolar_mask_enabled: bool = False,
) -> Tuple[StereoCamRoPEState, List[StereoCamRoPELayer]]:
    """Walk hf_model's language_model.layers, attach StereoCamRoPELayer to each
    supported standard attention layer (Qwen3_5Attention),
    replace its forward with the matching patched version, optionally Copy-Init
    the d_c projections from the M-RoPE temporal subspace, and register a
    pre-forward hook on the top-level model that fills the per-batch state.

    init_mode:
      * 'zero'  — Zero-init q_cam_proj/k_cam_proj so step-0 attention is
        byte-identical to the baseline backbone. (Currently the only mode;
        copy_temporal_mrope variant was removed after 4B+GR00T proved unusable.)

    Returns the shared state holder + list of layer modules (so the framework
    can register them as nn.Module children for ckpt save/load).
    """
    if init_mode != 'zero':
        raise RuntimeError(
            f"init_mode must be 'zero' (copy_temporal_mrope removed); got {init_mode!r}."
        )

    # 1. Pre-compute the constant camera P_stack.
    P_stack = compute_libero_camera_P_stack(
        fovy_degrees=fovy_degrees,
        image_width=image_width,
        image_height=image_height,
        baseline_m=baseline_m,
    )

    state = StereoCamRoPEState()
    state.epipolar_mask_enabled = bool(epipolar_mask_enabled)

    # 2. Find the language model.
    inner = getattr(hf_model, 'model', None) or hf_model
    lm = getattr(inner, 'language_model', None) or getattr(inner, 'model', None)
    if lm is None or not hasattr(lm, 'layers'):
        raise RuntimeError('Could not locate language_model.layers in hf_model')

    # 3. For each supported attention layer, attach StereoCamRoPELayer + patch forward.
    text_cfg = hf_model.config.text_config if hasattr(hf_model.config, 'text_config') else hf_model.config
    hidden_dim = int(text_cfg.hidden_size)
    n_heads_q = int(text_cfg.num_attention_heads)
    n_heads_kv = int(text_cfg.num_key_value_heads)
    image_token_id = int(hf_model.config.image_token_id)

    scl_modules: List[StereoCamRoPELayer] = []
    standard_attn_idxs = []
    patched_attn_classes: List[str] = []
    for layer_idx, layer in enumerate(lm.layers):
        attn = None
        for child_name in ('self_attn', 'attention', 'attn'):
            if hasattr(layer, child_name):
                attn = getattr(layer, child_name)
                break
        if attn is None:
            continue
        attn_cls_name = type(attn).__name__
        patched_forward = _PATCHED_FORWARD_DISPATCH.get(attn_cls_name)
        if patched_forward is None:
            continue
        standard_attn_idxs.append(layer_idx)
        patched_attn_classes.append(attn_cls_name)

        scl = StereoCamRoPELayer(
            hidden_dim=hidden_dim,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            d_c=d_c,
        )

        # Match base model dtype so DeepSpeed ZeRO-3 defragment doesn't choke on
        # mixed-dtype param groups (q_cam_proj/k_cam_proj are nn.Linear so default
        # fp32; base Qwen3-VL is bf16).
        base_dtype = attn.q_proj.weight.dtype
        scl = scl.to(base_dtype)

        scl.state_holder = state
        scl.register_buffer('camera_P_stack', P_stack.clone(), persistent=False)
        attn.stereo_cam_layer = scl
        # Replace the bound method (per-class dispatch).
        attn.forward = MethodType(patched_forward, attn)
        scl_modules.append(scl)

    if not scl_modules:
        raise RuntimeError(
            'No supported attention layers found to patch.  Looked for: '
            f'{sorted(_PATCHED_FORWARD_DISPATCH.keys())}.  '
            'Did the model switch architecture?'
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
        if state.epipolar_mask_enabled:
            state.per_token_row_id = compute_per_token_row_id(
                input_ids=input_ids,
                image_token_id=image_token_id,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
        else:
            state.per_token_row_id = None

    top_hook_handle = reinstall_stereo_cam_rope_outer_hook(hf_model, _pre_forward_hook)
    state.top_pre_hook_handle = top_hook_handle
    state.top_pre_hook_fn = _pre_forward_hook
    state.reinstall_top_pre_hook = getattr(top_hook_handle, "reinstall_outer_hook")

    import logging
    unique_cls = sorted(set(patched_attn_classes))
    logging.info(
        f'[stereo_cam_rope] patched {len(scl_modules)} attention layers '
        f'({unique_cls}) at language_model.layers[{standard_attn_idxs}] '
        f'(d_c={d_c}, num_cameras={num_cameras}, baseline_m={baseline_m}, '
        f'init_mode={init_mode}, epipolar={state.epipolar_mask_enabled})'
    )

    return state, scl_modules
