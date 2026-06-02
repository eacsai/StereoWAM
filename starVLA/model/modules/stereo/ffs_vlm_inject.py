"""FFS -> VLM-input stereo injection (ControlNet-style, but into the VLM input).

Motivation (2026-05-30)
-----------------------
StereoPolicy (Fei-Fei / Stanford) and StereoVLA (Wang He) both add stereo
features at the **VLM input** so the geometry flows through the *whole* VLM and
fuses with the language instruction. Our existing FFS frameworks do NOT do this:

  * QwenPI_ControlNet_FFS  -> adds a residual onto the VLM *output* hidden states
                              (the per-layer condition the action DiT reads).
  * QwenPI_ControlVLA_FFS  -> parallel K/V branch inside the action DiT cross-attn.

Both inject on the *action-condition* side; the FFS signal never passes through a
single VLM transformer layer. This module is the missing variant: a zero-gated
residual added to the merged `inputs_embeds` at the **image-token positions**,
i.e. before the VLM language-model stack runs. Our differentiator vs the two
papers is the *ControlNet-style zero-gated* injection that keeps the pretrained
VLM byte-identical at step 0 (warm-start safe).

Where it hooks (verified against transformers Qwen3-VL source)
-------------------------------------------------------------
  Qwen3VLModel.forward      builds inputs_embeds, masked_scatter's the image
                            embeds in, then calls
                            self.language_model(inputs_embeds=<merged>, ...).
  Qwen3VLTextModel.forward  receives the merged inputs_embeds as a kwarg and
                            sets hidden_states = inputs_embeds.

So we use TWO pre-hooks (same split as cam_rope):
  (1) OUTER pre-hook on the top HF model: reads input_ids + image_grid_thw,
      derives per_token_cam_id (which tokens are the primary/left view) and the
      primary-view token grid (h_tok, w_tok). Stored in a per-batch state.
  (2) INNER pre-hook on language_model (with_kwargs): reads the merged
      inputs_embeds kwarg + state + the stashed FFS feature, builds a zero-gated
      residual, and scatter-adds it onto the primary-view image-token positions.
      Returns (args, kwargs) with the modified inputs_embeds.

The FFS feature itself is computed by the framework BEFORE the VLM forward
(in its _encode_vl_hidden_states override) and stashed on the module-level state
via set_ffs_feature(); the inner hook reads it.

Anti-drift design (the lesson from the FFS-injection post-mortem)
----------------------------------------------------------------
The residual is `gate * spatial_proj(ffs)` where:
  * spatial_proj is NORMAL-initialised (NOT zero).
  * gate is a SINGLE learnable scalar initialised to 0.
At step 0, residual = 0 * proj(...) = 0  -> inputs_embeds byte-identical to the
baseline VLM. But d(residual)/d(gate) = proj(...) != 0, so the gate gets a
non-zero gradient and ramps the injection magnitude up SLOWLY (one scalar knob),
acting as a gentle global brake. (Zero-initialising BOTH gate and proj would be a
dead saddle: each one's gradient is proportional to the other, so neither ever
leaves zero. We zero exactly one.)

Only image tokens of the PRIMARY (left) view are touched; text/system tokens and
the right-view tokens are never modified -- the FFS net[0] disparity map lives in
the left-camera frame, so the left-view token grid is the geometrically correct
target, and language grounding is left untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_rope_hook import compute_per_token_cam_id


# --------------------------------------------------------------------------- #
# Per-batch state — written by the outer hook + the framework, read by the
# inner (language_model) hook. Module-level singleton, mirroring controlvla.
# --------------------------------------------------------------------------- #


@dataclass
class FFSVLMInjectState:
    # FFS feature map for the batch, (B, C_ffs, h, w). Set by the framework
    # right before the VLM forward; cleared after.
    ffs_feat: Optional[torch.Tensor] = None
    # (B, S) long: -1 text, 0 = primary/left image token, 1 = right image token.
    per_token_cam_id: Optional[torch.Tensor] = None
    # Per-sample primary-view token grid (h_tok, w_tok), len B. Used to fold the
    # FFS feature onto the primary token positions in the right 2D order.
    primary_grid: Optional[List[Tuple[int, int]]] = None
    enabled: bool = True


_STATE = FFSVLMInjectState()


def set_ffs_feature(ffs_feat: torch.Tensor) -> None:
    _STATE.ffs_feat = ffs_feat


def clear_ffs_state() -> None:
    _STATE.ffs_feat = None
    _STATE.per_token_cam_id = None
    _STATE.primary_grid = None


def set_enabled(flag: bool) -> None:
    _STATE.enabled = bool(flag)


def get_state() -> FFSVLMInjectState:
    return _STATE


def _raise_primary_grid_mismatch(
    sample_idx: int,
    h_tok: int,
    w_tok: int,
    n_primary: int,
    primary_cam_id: int,
) -> None:
    expected = h_tok * w_tok
    raise RuntimeError(
        f"[ffs_vlm_inject] sample {sample_idx}: primary_cam_id={primary_cam_id} "
        f"expected primary token grid {h_tok}x{w_tok} = {expected}, "
        f"got {n_primary} assigned primary image tokens. "
        f"image_grid_thw / image-token layout mismatch; refusing to silently "
        f"disable FFS VLM-input injection."
    )


# --------------------------------------------------------------------------- #
# Injector module: holds the trainable params (spatial_proj + scalar gate).
# Registered as an nn.Module child of the framework so it trains / saves / shards.
# --------------------------------------------------------------------------- #


class FFSVLMInjector(nn.Module):
    """FFS feature map -> per-image-token residual in VLM hidden space.

    spatial_proj : Conv 1x1 stack  (C_ffs -> hidden -> llm_dim), NORMAL init.
    gate         : single learnable scalar, init 0  (the slow global brake).

    forward(ffs_feat_b, h_tok, w_tok) -> (n_tok = h_tok*w_tok, llm_dim) residual
    for ONE sample, already gated. At step 0 (gate=0) the residual is exactly 0.
    """

    def __init__(self, in_ch: int, llm_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.in_ch = in_ch
        self.llm_dim = llm_dim
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, llm_dim, kernel_size=1),
        )
        # SINGLE scalar gate, init 0. This (NOT a zero-init final linear) is the
        # step-0 no-op + slow-ramp brake. spatial_proj stays NORMAL-init so the
        # gate has a non-zero gradient and actually learns (no dead saddle).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, ffs_feat_b: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
        # ffs_feat_b: (1, C_ffs, h, w)
        x = self.spatial_proj(ffs_feat_b)                       # (1, llm_dim, h, w)
        # Real 2D spatial interpolation onto the primary-view token grid — NOT
        # the bogus 1D token-axis interpolation the old ControlNet variant used.
        x = F.interpolate(x, size=(h_tok, w_tok), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2).contiguous()           # (1, h_tok*w_tok, llm_dim)
        x = x.squeeze(0)                                         # (n_tok, llm_dim)
        return self.gate * x                                    # gate=0 -> 0 at step 0


# --------------------------------------------------------------------------- #
# Install: outer hook (cam-id + grid) on the HF model, inner hook (inject) on
# the language model. Returns (state, []) — no per-layer modules; the single
# injector is owned by the framework.
# --------------------------------------------------------------------------- #


def install_ffs_vlm_input_hooks(
    hf_model,
    injector: FFSVLMInjector,
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
    primary_cam_id: int = 0,
) -> FFSVLMInjectState:
    """Register the two pre-hooks. `hf_model` is the Qwen3VLForConditionalGeneration
    instance (self.qwen_vl_interface.model). The injector's params must already be
    a registered child of the framework.
    """
    image_token_id = int(hf_model.config.image_token_id)

    # Locate language_model (Qwen3VLTextModel): hf_model.model.language_model.
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError(
            "[ffs_vlm_inject] could not locate model.language_model with .layers "
            "on the Qwen3-VL model — did the architecture change?"
        )

    state = _STATE

    # ---- OUTER hook: derive per_token_cam_id + primary token grid ------------ #
    def _outer_pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if input_ids is None or image_grid_thw is None:
            state.per_token_cam_id = None
            state.primary_grid = None
            return
        state.per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=num_cameras,
            spatial_merge_size=spatial_merge_size,
        )
        # Primary-view token grid per sample. image_grid_thw rows are concatenated
        # sample-then-image; with num_cameras images per sample, the primary image
        # of sample b is row b*num_cameras (primary_cam_id picks which image index
        # within the sample is "primary"; default 0 = the first/left image).
        B = input_ids.shape[0]
        s2 = max(int(spatial_merge_size), 1)
        thw = image_grid_thw
        grid: List[Tuple[int, int]] = []
        for b in range(B):
            row = b * num_cameras + primary_cam_id
            if row >= thw.shape[0]:
                grid.append((0, 0))
                continue
            h_tok = int(thw[row, 1]) // s2
            w_tok = int(thw[row, 2]) // s2
            grid.append((h_tok, w_tok))
        state.primary_grid = grid
        if state.ffs_feat is not None:
            image_mask = (input_ids == image_token_id)
            for b, (h_tok, w_tok) in enumerate(grid):
                if not bool(image_mask[b].any()):
                    continue
                n_primary = int((state.per_token_cam_id[b] == primary_cam_id).sum().item())
                if n_primary == 0 or h_tok * w_tok != n_primary:
                    _raise_primary_grid_mismatch(
                        sample_idx=b,
                        h_tok=h_tok,
                        w_tok=w_tok,
                        n_primary=n_primary,
                        primary_cam_id=primary_cam_id,
                    )

    # ---- INNER hook: inject zero-gated residual into merged inputs_embeds ----- #
    def _inner_pre_hook(module, args, kwargs):
        if not state.enabled:
            return None
        if state.ffs_feat is None or state.per_token_cam_id is None or state.primary_grid is None:
            return None  # no-op (e.g. pure-text forward, or feature not stashed)

        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            # language_model is always called with inputs_embeds in this pipeline;
            # if it ever isn't, skip rather than crash.
            return None

        cam = state.per_token_cam_id              # (B, S)
        ffs = state.ffs_feat                      # (B, C_ffs, h, w)
        B, S, D = inputs_embeds.shape
        add = torch.zeros_like(inputs_embeds)     # fresh tensor; final add stays out-of-place

        for b in range(B):
            primary_pos = (cam[b] == primary_cam_id).nonzero(as_tuple=True)[0]  # primary token idxs
            n_primary = int(primary_pos.numel())
            h_tok, w_tok = state.primary_grid[b]
            # Fail loud if the grid bookkeeping and the actual token run disagree —
            # this is exactly the kind of silent geometry mismatch that bit us before.
            if h_tok * w_tok != n_primary:
                _raise_primary_grid_mismatch(
                    sample_idx=b,
                    h_tok=h_tok,
                    w_tok=w_tok,
                    n_primary=n_primary,
                    primary_cam_id=primary_cam_id,
                )
            if n_primary == 0:
                continue
            residual = injector(ffs[b : b + 1], h_tok, w_tok)      # (n_primary, D), gated
            add[b, primary_pos] = residual.to(add.dtype)

        kwargs["inputs_embeds"] = inputs_embeds + add
        return args, kwargs

    hf_model.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True)
    lm.register_forward_pre_hook(_inner_pre_hook, with_kwargs=True)

    import logging
    logging.info(
        f"[ffs_vlm_inject] installed outer(cam-id+grid) + inner(inputs_embeds inject) "
        f"hooks (image_token_id={image_token_id}, num_cameras={num_cameras}, "
        f"primary_cam_id={primary_cam_id}, gate init 0)"
    )
    return state
