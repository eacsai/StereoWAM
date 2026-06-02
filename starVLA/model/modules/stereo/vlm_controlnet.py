"""True VLM ControlNet for injecting frozen FFS stereo features into Qwen3.5.

The trainable path is:
  inputs_embeds + projected FFS hint on primary-view image tokens
  -> six copied Qwen3_5Attention decoder layers
  -> per-depth zero-init Linear residuals
  -> frozen trunk hidden states at the matching six cam_rope depths

Only the zero-convs are zero-initialized. The hint projection stays normally
initialized so the zero-conv weights receive a useful first gradient.
"""
from __future__ import annotations

import copy
import io
import logging
from dataclasses import dataclass
from types import MethodType
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_rope import StereoCamRoPELayer
from .cam_rope_hook import (
    StereoCamRoPEState,
    _PATCHED_FORWARD_DISPATCH,
    compute_per_token_cam_id,
)

logger = logging.getLogger(__name__)


@dataclass
class VLMControlNetState:
    ffs_feat: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    primary_grid: Optional[List[Tuple[int, int]]] = None
    ctrl_h: Optional[torch.Tensor] = None
    injections: Optional[List[torch.Tensor]] = None
    enabled: bool = True


_STATE = VLMControlNetState()


def set_ffs_feature(ffs_feat: torch.Tensor) -> None:
    _STATE.ffs_feat = ffs_feat


def clear_state() -> None:
    _STATE.ffs_feat = None
    _STATE.per_token_cam_id = None
    _STATE.primary_grid = None
    _STATE.ctrl_h = None
    _STATE.injections = None


def set_enabled(flag: bool) -> None:
    _STATE.enabled = bool(flag)


def get_state() -> VLMControlNetState:
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
        f"[vlm_controlnet] sample {sample_idx}: primary_cam_id={primary_cam_id} "
        f"expected primary token grid {h_tok}x{w_tok} = {expected}, "
        f"got {n_primary} assigned primary image tokens. "
        f"image_grid_thw / image-token layout mismatch; refusing to silently "
        f"disable VLM ControlNet FFS injection."
    )


def _get_self_attn(layer: nn.Module) -> nn.Module:
    for child_name in ("self_attn", "attention", "attn"):
        if hasattr(layer, child_name):
            return getattr(layer, child_name)
    raise RuntimeError(f"[vlm_controlnet] could not find attention child on {type(layer).__name__}")


def _strip_cam_rope_from_layer(layer: nn.Module) -> nn.Module:
    """Return a clean decoder-layer copy, without instance monkeypatch state."""
    copied = copy.deepcopy(layer)
    attn = _get_self_attn(copied)
    if hasattr(attn, "stereo_cam_layer"):
        delattr(attn, "stereo_cam_layer")
    if "forward" in getattr(attn, "__dict__", {}):
        delattr(attn, "forward")
    return copied


class FFSControlNetHint(nn.Module):
    """FFS feature map -> primary-image-token hint in VLM hidden space."""

    def __init__(self, in_ch: int = 16, llm_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.in_ch = int(in_ch)
        self.llm_dim = int(llm_dim)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, llm_dim, kernel_size=1),
        )

    def forward(self, ffs_feat_b: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
        x = self.spatial_proj(ffs_feat_b)
        x = F.interpolate(x, size=(h_tok, w_tok), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x.squeeze(0)
        return x


class VLMControlNetBranch(nn.Module):
    """Copied six-layer VLM ControlNet branch plus per-depth zero-convs."""

    def __init__(
        self,
        source_layers: Sequence[nn.Module],
        inject_depths: Sequence[int],
        llm_dim: int,
        base_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if len(source_layers) != len(inject_depths):
            raise ValueError(
                f"[vlm_controlnet] source_layers len {len(source_layers)} != "
                f"inject_depths len {len(inject_depths)}"
            )
        self.inject_depths = [int(x) for x in inject_depths]
        self.llm_dim = int(llm_dim)
        self.branch_layers = nn.ModuleList(
            [_strip_cam_rope_from_layer(layer) for layer in source_layers]
        )
        self.zero_convs = nn.ModuleList(
            [nn.Linear(self.llm_dim, self.llm_dim) for _ in self.inject_depths]
        )
        for zero in self.zero_convs:
            nn.init.zeros_(zero.weight)
            nn.init.zeros_(zero.bias)
        self.to(dtype=base_dtype)
        for name, param in self.named_parameters():
            param.requires_grad = not (".stereo_cam_layer." in name)

    def compute_injections(self, hidden_states: torch.Tensor, layer_kwargs: dict) -> List[torch.Tensor]:
        h = hidden_states
        injections: List[torch.Tensor] = []
        passthrough = {
            k: v for k, v in layer_kwargs.items()
            if k not in {"hidden_states", "past_key_values", "use_cache"}
        }
        for layer, zero in zip(self.branch_layers, self.zero_convs):
            out = layer(h, **passthrough)
            h = out[0] if isinstance(out, tuple) else out
            injections.append(zero(h))
        return injections

    def assert_zero_convs_zero(self) -> None:
        for idx, zero in enumerate(self.zero_convs):
            if torch.count_nonzero(zero.weight).item() != 0:
                raise RuntimeError(f"[vlm_controlnet] zero_conv[{idx}].weight is not all-zero")
            if torch.count_nonzero(zero.bias).item() != 0:
                raise RuntimeError(f"[vlm_controlnet] zero_conv[{idx}].bias is not all-zero")


def attach_branch_cam_rope(
    branch_layers: Sequence[nn.Module],
    trunk_state: StereoCamRoPEState,
    trunk_scl_modules: Sequence[StereoCamRoPELayer],
    base_dtype: torch.dtype,
) -> None:
    """Patch branch attention layers with cam_rope that shares the trunk state."""
    if len(branch_layers) != len(trunk_scl_modules):
        raise RuntimeError(
            f"[vlm_controlnet] branch/trunk cam_rope count mismatch: "
            f"{len(branch_layers)} vs {len(trunk_scl_modules)}"
        )

    for idx, (layer, trunk_scl) in enumerate(zip(branch_layers, trunk_scl_modules)):
        attn = _get_self_attn(layer)
        attn_cls_name = type(attn).__name__
        patched_forward = _PATCHED_FORWARD_DISPATCH.get(attn_cls_name)
        if patched_forward is None:
            raise RuntimeError(
                f"[vlm_controlnet] branch layer {idx} attention class {attn_cls_name} "
                "is not supported by cam_rope dispatch"
            )

        if hasattr(attn, "stereo_cam_layer"):
            delattr(attn, "stereo_cam_layer")
        if "forward" in getattr(attn, "__dict__", {}):
            delattr(attn, "forward")

        d_c = int(trunk_scl.d_c)
        q_out = int(trunk_scl.q_cam_proj.out_features)
        k_out = int(trunk_scl.k_cam_proj.out_features)
        scl = StereoCamRoPELayer(
            hidden_dim=int(trunk_scl.q_cam_proj.in_features),
            n_heads_q=q_out // d_c,
            n_heads_kv=k_out // d_c,
            d_c=d_c,
        ).to(dtype=base_dtype)
        scl.load_state_dict(trunk_scl.state_dict(), strict=True)
        scl.state_holder = trunk_state
        if hasattr(trunk_scl, "camera_P_stack"):
            scl.register_buffer(
                "camera_P_stack",
                trunk_scl.camera_P_stack.detach().clone().to(dtype=base_dtype),
                persistent=False,
            )
        for param in scl.parameters():
            param.requires_grad = False
        attn.stereo_cam_layer = scl
        attn.forward = MethodType(patched_forward, attn)

    logger.info(
        "[vlm_controlnet] branch cam_rope mode: shared trunk StereoCamRoPEState, "
        "trunk stereo_cam_layer weights copied and frozen"
    )


def copy_trunk_non_cam_params_to_branch(
    trunk_layers: Sequence[nn.Module],
    branch_layers: Sequence[nn.Module],
) -> None:
    """Copy all non-cam_rope decoder-layer params from trunk to branch."""
    if len(trunk_layers) != len(branch_layers):
        raise RuntimeError(
            f"[vlm_controlnet] copy-init count mismatch: {len(trunk_layers)} vs {len(branch_layers)}"
        )
    for layer_idx, (trunk, branch) in enumerate(zip(trunk_layers, branch_layers)):
        trunk_params = {
            name: param for name, param in trunk.named_parameters()
            if ".stereo_cam_layer." not in name
        }
        branch_params = {
            name: param for name, param in branch.named_parameters()
            if ".stereo_cam_layer." not in name
        }
        if set(trunk_params) != set(branch_params):
            missing = sorted(set(trunk_params) - set(branch_params))[:10]
            extra = sorted(set(branch_params) - set(trunk_params))[:10]
            raise RuntimeError(
                f"[vlm_controlnet] copy-init param-name mismatch at layer {layer_idx}; "
                f"missing={missing}, extra={extra}"
            )
        for name, trunk_param in trunk_params.items():
            branch_param = branch_params[name]
            with torch.no_grad():
                branch_param.copy_(trunk_param.to(dtype=branch_param.dtype))
            expected = trunk_param.detach().to(dtype=branch_param.dtype, device=branch_param.device)
            if not torch.equal(branch_param.detach(), expected):
                raise RuntimeError(
                    f"[vlm_controlnet] copy-init equality audit failed for layer {layer_idx} param {name}"
                )
            if branch_param.data_ptr() == trunk_param.data_ptr():
                raise RuntimeError(
                    f"[vlm_controlnet] copy-init storage audit failed for layer {layer_idx} param {name}"
                )


def assert_branch_roundtrip_equal(branch: VLMControlNetBranch) -> None:
    payload = io.BytesIO()
    torch.save(branch.state_dict(), payload)
    payload.seek(0)
    restored = torch.load(payload, map_location="cpu")
    current = branch.state_dict()
    if set(current) != set(restored):
        raise RuntimeError("[vlm_controlnet] branch round-trip key mismatch")
    for key, value in current.items():
        if not torch.equal(value.detach().cpu(), restored[key].detach().cpu()):
            raise RuntimeError(f"[vlm_controlnet] branch round-trip tensor mismatch at {key}")


def install_vlm_controlnet_hooks(
    hf_model,
    branch: VLMControlNetBranch,
    hint: FFSControlNetHint,
    inject_depths: Sequence[int],
    num_cameras: int,
    spatial_merge_size: int,
    primary_cam_id: int,
    image_token_id: Optional[int] = None,
) -> VLMControlNetState:
    """Install one outer hook, one LM hook, one branch-run hook, and six add hooks."""
    image_token_id = int(
        image_token_id if image_token_id is not None else hf_model.config.image_token_id
    )
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("[vlm_controlnet] could not locate language_model.layers")
    layers = lm.layers
    inject_depths = [int(x) for x in inject_depths]
    if not inject_depths:
        raise RuntimeError("[vlm_controlnet] inject_depths is empty")
    for depth in inject_depths:
        if depth < 0 or depth >= len(layers):
            raise RuntimeError(f"[vlm_controlnet] invalid inject depth {depth}")

    state = _STATE

    def _outer_pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.ctrl_h = None
        state.injections = None
        if input_ids is None or image_grid_thw is None:
            state.per_token_cam_id = None
            state.primary_grid = None
            return None
        state.per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=num_cameras,
            spatial_merge_size=spatial_merge_size,
        )
        B = input_ids.shape[0]
        s2 = max(int(spatial_merge_size), 1)
        grid: List[Tuple[int, int]] = []
        for b in range(B):
            row = b * num_cameras + primary_cam_id
            if row >= image_grid_thw.shape[0]:
                grid.append((0, 0))
                continue
            h_tok = int(image_grid_thw[row, 1]) // s2
            w_tok = int(image_grid_thw[row, 2]) // s2
            grid.append((h_tok, w_tok))
        state.primary_grid = grid
        if state.ffs_feat is not None:
            image_mask = input_ids == image_token_id
            for b, (h_tok, w_tok) in enumerate(grid):
                if not bool(image_mask[b].any()):
                    continue
                n_primary = int((state.per_token_cam_id[b] == primary_cam_id).sum().item())
                if n_primary == 0 or h_tok * w_tok != n_primary:
                    _raise_primary_grid_mismatch(b, h_tok, w_tok, n_primary, primary_cam_id)
        return None

    def _lm_pre_hook(module, args, kwargs):
        if not state.enabled:
            return None
        if state.ffs_feat is None or state.per_token_cam_id is None or state.primary_grid is None:
            return None
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            return None

        cam = state.per_token_cam_id
        ffs = state.ffs_feat
        add = torch.zeros_like(inputs_embeds)
        B = inputs_embeds.shape[0]
        for b in range(B):
            primary_pos = (cam[b] == primary_cam_id).nonzero(as_tuple=True)[0]
            n_primary = int(primary_pos.numel())
            h_tok, w_tok = state.primary_grid[b]
            if h_tok * w_tok != n_primary:
                _raise_primary_grid_mismatch(b, h_tok, w_tok, n_primary, primary_cam_id)
            if n_primary == 0:
                continue
            hint_dtype = next(hint.parameters()).dtype
            residual = hint(ffs[b : b + 1].to(dtype=hint_dtype), h_tok, w_tok)
            add[b, primary_pos] = residual.to(dtype=add.dtype)
        state.ctrl_h = inputs_embeds.clone() + add
        return None

    def _first_layer_pre_hook(module, args, kwargs):
        if not state.enabled or state.ffs_feat is None or state.ctrl_h is None:
            state.injections = None
            return None
        layer_kwargs = dict(kwargs)
        branch_dtype = next(branch.parameters()).dtype
        state.injections = branch.compute_injections(state.ctrl_h.to(dtype=branch_dtype), layer_kwargs)
        return None

    def _make_add_hook(k: int):
        def _add_hook(module, args, output):
            if not state.enabled or state.injections is None:
                return None
            inj = state.injections[k]
            if isinstance(output, tuple):
                hidden = output[0]
                out_hidden = hidden + inj.to(dtype=hidden.dtype, device=hidden.device)
                return (out_hidden, *output[1:])
            out_hidden = output + inj.to(dtype=output.dtype, device=output.device)
            return out_hidden
        return _add_hook

    hf_model.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True)
    lm.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True)
    layers[inject_depths[0]].register_forward_pre_hook(_first_layer_pre_hook, with_kwargs=True)
    for k, depth in enumerate(inject_depths):
        layers[depth].register_forward_hook(_make_add_hook(k))

    logger.info(
        "[vlm_controlnet] installed hooks: outer grid, LM control-input capture, "
        f"branch-run at depth {inject_depths[0]}, residual adds at {inject_depths}"
    )
    return state
