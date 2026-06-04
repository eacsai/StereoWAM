"""Multi-layer gated FFS residual injection for QwenPI LLaMA-Adapter style runs.

This module is intentionally separate from the VLM-ControlNet branch. The trainable
path is one shared FFS hint projection plus one zero-initialized Linear gate at each
cam_rope softmax layer. Each gate output is added only to primary-view image tokens.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .cam_rope_hook import compute_per_token_cam_id
from .vlm_controlnet import FFSControlNetHint

logger = logging.getLogger(__name__)


@dataclass
class MultiLayerGatedFFSState:
    """Per-forward state shared by the outer hook and six layer hooks."""

    enabled: bool = True
    ffs_feat: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    primary_grid: Optional[List[Tuple[int, int]]] = None
    hints: Optional[List[Optional[torch.Tensor]]] = None
    last_residual_token_mask: Optional[List[Optional[torch.Tensor]]] = None
    view_order: List[int] = field(default_factory=list)
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)

    def clear_runtime(self) -> None:
        self.ffs_feat = None
        self.per_token_cam_id = None
        self.primary_grid = None
        self.hints = None
        self.last_residual_token_mask = None

    def set_ffs_feature(self, ffs_feat: torch.Tensor) -> None:
        self.ffs_feat = ffs_feat
        self.per_token_cam_id = None
        self.primary_grid = None
        self.hints = None
        self.last_residual_token_mask = None


def assert_gates_zero(gates: nn.ModuleList) -> None:
    """Fail if any zero-conv Linear has a non-zero weight or bias."""
    for idx, gate in enumerate(gates):
        if not isinstance(gate, nn.Linear):
            raise TypeError(f"[llama_adapter_ffs] gate[{idx}] is {type(gate).__name__}, expected nn.Linear")
        if int(torch.count_nonzero(gate.weight.detach()).item()) != 0:
            raise RuntimeError(f"[llama_adapter_ffs] gate[{idx}].weight is not zero-initialized")
        if gate.bias is not None and int(torch.count_nonzero(gate.bias.detach()).item()) != 0:
            raise RuntimeError(f"[llama_adapter_ffs] gate[{idx}].bias is not zero-initialized")


def set_enabled(state: MultiLayerGatedFFSState, flag: bool) -> None:
    state.enabled = bool(flag)


def clear_state(state: MultiLayerGatedFFSState) -> None:
    state.clear_runtime()


def set_ffs_feature(state: MultiLayerGatedFFSState, ffs_feat: torch.Tensor) -> None:
    state.set_ffs_feature(ffs_feat)


def _locate_language_model(hf_model):
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("[llama_adapter_ffs] could not locate language_model.layers")
    return lm


def _extract_input_ids(args, kwargs):
    input_ids = kwargs.get("input_ids", None)
    if input_ids is None and args:
        input_ids = args[0]
    return input_ids


def _build_view_order(
    *,
    num_cameras: int,
    primary_cam_id: int,
    right_view_idx: int,
    reverse_image_order: bool,
) -> List[int]:
    if not (0 <= primary_cam_id < num_cameras):
        raise ValueError(f"[llama_adapter_ffs] primary_cam_id={primary_cam_id} out of range")
    if not (0 <= right_view_idx < num_cameras):
        raise ValueError(f"[llama_adapter_ffs] right_view_idx={right_view_idx} out of range")
    if primary_cam_id == right_view_idx:
        raise ValueError("[llama_adapter_ffs] primary_cam_id and right_view_idx must differ")
    if reverse_image_order:
        if num_cameras != 2:
            raise ValueError("[llama_adapter_ffs] reverse_image_order currently requires num_cameras=2")
        return [int(right_view_idx), int(primary_cam_id)]
    return list(range(int(num_cameras)))


def _map_run_cam_to_view_cam(cam: torch.Tensor, view_order: Sequence[int]) -> torch.Tensor:
    """Map run-order camera ids from compute_per_token_cam_id to physical view ids."""
    out = cam.clone()
    for run_idx, view_id in enumerate(view_order):
        out[cam == int(run_idx)] = int(view_id)
    return out


def _primary_grid_from_image_grid(
    image_grid_thw: torch.Tensor,
    *,
    batch_size: int,
    num_cameras: int,
    primary_cam_id: int,
    spatial_merge_size: int,
    view_order: Sequence[int],
) -> List[Tuple[int, int]]:
    s2 = max(int(spatial_merge_size), 1)
    try:
        primary_run_idx = list(view_order).index(int(primary_cam_id))
    except ValueError as exc:
        raise RuntimeError(
            f"[llama_adapter_ffs] view_order={list(view_order)} does not contain primary_cam_id={primary_cam_id}"
        ) from exc

    grid: List[Tuple[int, int]] = []
    for b in range(batch_size):
        row = b * int(num_cameras) + primary_run_idx
        if image_grid_thw is None or row >= int(image_grid_thw.shape[0]):
            grid.append((0, 0))
            continue
        h_tok = int(image_grid_thw[row, 1].item()) // s2
        w_tok = int(image_grid_thw[row, 2].item()) // s2
        grid.append((h_tok, w_tok))
    return grid


def _raise_primary_grid_mismatch(
    sample_idx: int,
    h_tok: int,
    w_tok: int,
    n_primary: int,
    primary_cam_id: int,
) -> None:
    raise RuntimeError(
        "[llama_adapter_ffs] primary image-token grid mismatch: "
        f"sample={sample_idx}, h_tok={h_tok}, w_tok={w_tok}, "
        f"h*w={h_tok * w_tok}, primary_cam_id={primary_cam_id}, n_primary_tokens={n_primary}. "
        "Refusing to silently misalign FFS residuals."
    )


def _compute_hints(
    *,
    state: MultiLayerGatedFFSState,
    hint: FFSControlNetHint,
    primary_cam_id: int,
    right_view_idx: int,
) -> None:
    if state.ffs_feat is None or state.per_token_cam_id is None or state.primary_grid is None:
        state.hints = None
        return

    hint_param = next(hint.parameters())
    hint_dtype = hint_param.dtype
    hint_device = hint_param.device
    hints: List[Optional[torch.Tensor]] = []
    for b, (h_tok, w_tok) in enumerate(state.primary_grid):
        primary_pos = (state.per_token_cam_id[b] == int(primary_cam_id)).nonzero(as_tuple=True)[0]
        n_primary = int(primary_pos.numel())
        if n_primary == 0:
            raise RuntimeError(
                f"[llama_adapter_ffs] sample {b} has no primary-view image tokens "
                f"(primary_cam_id={primary_cam_id}); refusing to silently disable FFS residuals."
            )
        right_pos = (state.per_token_cam_id[b] == int(right_view_idx)).nonzero(as_tuple=True)[0]
        if int(right_pos.numel()) == 0:
            raise RuntimeError(
                f"[llama_adapter_ffs] sample {b} has no right-view image tokens "
                f"(right_view_idx={right_view_idx}); refusing to run with broken stereo cam_id layout."
            )
        if h_tok * w_tok != n_primary:
            _raise_primary_grid_mismatch(b, h_tok, w_tok, n_primary, primary_cam_id)
        residual_hint = hint(
            state.ffs_feat[b : b + 1].to(device=hint_device, dtype=hint_dtype),
            h_tok,
            w_tok,
        )
        if tuple(residual_hint.shape) != (n_primary, hint.llm_dim):
            raise RuntimeError(
                f"[llama_adapter_ffs] hint shape {tuple(residual_hint.shape)} != "
                f"expected {(n_primary, hint.llm_dim)} for sample {b}"
            )
        hints.append(residual_hint)
    state.hints = hints


def install_multilayer_gated_ffs_hooks(
    hf_model,
    *,
    hint: FFSControlNetHint,
    gates: nn.ModuleList,
    inject_depths: Sequence[int],
    num_cameras: int,
    spatial_merge_size: int,
    primary_cam_id: int,
    right_view_idx: int,
    reverse_image_order: bool,
    cam_rope_state=None,
    image_token_id: Optional[int] = None,
) -> MultiLayerGatedFFSState:
    """Install the outer bookkeeping hook and one residual hook per target layer."""
    image_token_id = int(image_token_id if image_token_id is not None else hf_model.config.image_token_id)
    lm = _locate_language_model(hf_model)
    layers = lm.layers
    inject_depths = [int(x) for x in inject_depths]
    if len(inject_depths) != len(gates):
        raise RuntimeError(
            f"[llama_adapter_ffs] inject_depth count {len(inject_depths)} != gate count {len(gates)}"
        )
    if not inject_depths:
        raise RuntimeError("[llama_adapter_ffs] inject_depths is empty")
    for depth in inject_depths:
        if depth < 0 or depth >= len(layers):
            raise RuntimeError(f"[llama_adapter_ffs] invalid inject depth {depth}")

    view_order = _build_view_order(
        num_cameras=int(num_cameras),
        primary_cam_id=int(primary_cam_id),
        right_view_idx=int(right_view_idx),
        reverse_image_order=bool(reverse_image_order),
    )
    state = MultiLayerGatedFFSState(view_order=view_order)

    def _outer_pre_hook(module, args, kwargs):
        input_ids = _extract_input_ids(args, kwargs)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.per_token_cam_id = None
        state.primary_grid = None
        state.hints = None
        state.last_residual_token_mask = [None for _ in inject_depths]

        if input_ids is None or image_grid_thw is None or image_grid_thw.numel() == 0:
            if cam_rope_state is not None:
                cam_rope_state.per_token_cam_id = None
            return None

        run_cam = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=int(num_cameras),
            spatial_merge_size=int(spatial_merge_size),
        )
        view_cam = _map_run_cam_to_view_cam(run_cam, view_order)
        state.per_token_cam_id = view_cam
        if cam_rope_state is not None:
            cam_rope_state.per_token_cam_id = view_cam

        batch_size = int(input_ids.shape[0])
        state.primary_grid = _primary_grid_from_image_grid(
            image_grid_thw,
            batch_size=batch_size,
            num_cameras=int(num_cameras),
            primary_cam_id=int(primary_cam_id),
            spatial_merge_size=int(spatial_merge_size),
            view_order=view_order,
        )
        if state.ffs_feat is not None:
            _compute_hints(
                state=state,
                hint=hint,
                primary_cam_id=int(primary_cam_id),
                right_view_idx=int(right_view_idx),
            )
        return None

    def _make_add_hook(k: int):
        def _add_hook(module, args, output):
            if not state.enabled or state.hints is None or state.per_token_cam_id is None:
                return None
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if hidden is None:
                return None
            if tuple(state.per_token_cam_id.shape) != tuple(hidden.shape[:2]):
                raise RuntimeError(
                    f"[llama_adapter_ffs] cam-id shape {tuple(state.per_token_cam_id.shape)} "
                    f"does not match hidden shape {tuple(hidden.shape[:2])}"
                )

            out_hidden = hidden.clone()
            direct_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            gate = gates[k]
            gate_param = next(gate.parameters())
            gate_dtype = gate_param.dtype
            gate_device = gate_param.device
            cam = state.per_token_cam_id.to(device=hidden.device)

            for b, hint_b in enumerate(state.hints):
                left_pos = (cam[b] == int(primary_cam_id)).nonzero(as_tuple=True)[0]
                n_left = int(left_pos.numel())
                if n_left == 0:
                    continue
                if hint_b is None:
                    raise RuntimeError(
                        f"[llama_adapter_ffs] sample {b} has {n_left} primary tokens but no cached hint"
                    )
                residual = gate(hint_b.to(device=gate_device, dtype=gate_dtype))
                if int(residual.shape[0]) != n_left or int(residual.shape[1]) != int(hidden.shape[-1]):
                    raise RuntimeError(
                        f"[llama_adapter_ffs] gate[{k}] residual shape {tuple(residual.shape)} "
                        f"does not match left tokens {(n_left, int(hidden.shape[-1]))}"
                    )
                residual = residual.to(device=hidden.device, dtype=hidden.dtype)
                out_hidden[b, left_pos] = out_hidden[b, left_pos] + residual
                changed = residual.detach().abs().sum(dim=-1) > 0
                if bool(changed.any()):
                    direct_mask[b, left_pos[changed]] = True

            if state.last_residual_token_mask is None:
                state.last_residual_token_mask = [None for _ in inject_depths]
            state.last_residual_token_mask[k] = direct_mask.detach().cpu()
            if isinstance(output, tuple):
                return (out_hidden, *output[1:])
            return out_hidden

        return _add_hook

    state.handles.append(hf_model.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True))
    for k, depth in enumerate(inject_depths):
        state.handles.append(layers[depth].register_forward_hook(_make_add_hook(k)))

    logger.info(
        "[llama_adapter_ffs] installed gated residual hooks: depths=%s, view_order=%s, "
        "primary_cam_id=%s, reverse_image_order=%s",
        inject_depths,
        view_order,
        primary_cam_id,
        reverse_image_order,
    )
    return state
