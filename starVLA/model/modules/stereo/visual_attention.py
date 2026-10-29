"""Pure helpers for the matched Qwen3.5 visual-attention ablation.

The helpers in this module intentionally do not know about Camera Branch
modules or model hooks.  They turn the upstream Qwen mask into an explicit
additive SDPA mask for a full-attention prefill and rewrite a bound attention
call without mutating module-level ``is_causal`` state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


VISUAL_ATTENTION_MODES = frozenset(("legacy_mixed", "causal", "image_bidir"))


def parse_exact_bool(value: Any, *, name: str) -> bool:
    """Parse a config boolean without Python's dangerous ``bool("false")``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{name} must be a boolean or exact 'true'/'false', got {value!r}")


def resolve_visual_attention_profile(
    mode: Any,
    branch_is_causal: Any,
    attention_implementation: Any,
    *,
    cam_branch_enabled: bool,
) -> Tuple[str, bool, str]:
    """Validate and normalize the public visual-ablation configuration."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in VISUAL_ATTENTION_MODES:
        raise ValueError(
            "stereo_visual_attention_mode must be one of "
            f"{sorted(VISUAL_ATTENTION_MODES)}, got {mode!r}"
        )
    normalized_branch = parse_exact_bool(
        branch_is_causal,
        name="stereo_cam_branch_is_causal",
    )
    backend = str(attention_implementation).strip().lower()
    if normalized_mode == "legacy_mixed":
        if normalized_branch:
            raise ValueError(
                "legacy_mixed is the historical non-causal Camera Branch profile; "
                "use visual_attention_mode=causal with stereo_cam_branch_is_causal=true"
            )
    else:
        if not cam_branch_enabled:
            raise ValueError(
                f"{normalized_mode} requires stereo_cam_branch_enabled=true"
            )
        expected = normalized_mode == "causal"
        if normalized_branch is not expected:
            raise ValueError(
                "only matched visual profiles are allowed: "
                "(causal,true) or (image_bidir,false)"
            )
        if backend != "sdpa":
            raise ValueError(
                f"{normalized_mode} requires attn_implementation=sdpa; got {attention_implementation!r}"
            )
    return normalized_mode, normalized_branch, backend


def _negative_sentinel(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError(f"visual attention masks require a floating dtype, got {dtype}")
    return float("-inf")


def _expand_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.expand(batch_size, *mask.shape[1:])
    raise ValueError(
        f"attention mask batch={mask.shape[0]} is incompatible with hidden batch={batch_size}"
    )


def _as_additive_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize None/2-D/4-D Qwen masks to additive ``[B,1|H,Q,K]``."""
    sentinel = _negative_sentinel(dtype)
    if attention_mask is None:
        if q_len != k_len:
            raise ValueError("an omitted mask is only supported for square prefill attention")
        out = torch.zeros((batch_size, 1, q_len, k_len), device=device, dtype=dtype)
        upper = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
        return out.masked_fill(upper.view(1, 1, q_len, k_len), sentinel)

    if not torch.is_tensor(attention_mask):
        raise TypeError(f"unsupported attention mask type: {type(attention_mask).__name__}")
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        if tuple(mask.shape[-1:]) != (k_len,):
            raise ValueError(
                f"2-D padding mask must be [B,{k_len}], got {tuple(mask.shape)}"
            )
        mask = _expand_batch(mask, batch_size).to(dtype=torch.bool)
        out = torch.zeros((batch_size, 1, q_len, k_len), device=device, dtype=dtype)
        out = out.masked_fill((~mask).view(batch_size, 1, 1, k_len), sentinel)
        if q_len == k_len:
            upper = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
            out = out.masked_fill(upper.view(1, 1, q_len, k_len), sentinel)
        return out

    if mask.ndim != 4:
        raise ValueError(
            "visual attention accepts only None, [B,K] padding, or additive/boolean "
            f"[B,1|H,Q,K] masks; got shape={tuple(mask.shape)}"
        )
    if mask.shape[-2:] != (q_len, k_len):
        raise ValueError(
            f"4-D mask must end in [{q_len},{k_len}], got {tuple(mask.shape)}"
        )
    mask = _expand_batch(mask, batch_size)
    if mask.dtype == torch.bool:
        out = torch.zeros(mask.shape, device=device, dtype=dtype)
        return out.masked_fill(~mask, sentinel)
    if not mask.dtype.is_floating_point:
        raise TypeError(f"4-D visual attention mask must be bool/float, got {mask.dtype}")
    out = mask.to(dtype=dtype).clone()
    return out.masked_fill(_blocked_cells(out), sentinel)


def _blocked_cells(mask: torch.Tensor) -> torch.Tensor:
    sentinel = torch.finfo(mask.dtype).min
    return torch.isneginf(mask) | (mask <= sentinel / 2.0)


def _image_mask(
    image_positions: torch.Tensor,
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(image_positions) or image_positions.ndim != 2:
        raise ValueError(
            "image_positions must be a [B,N_img] integer tensor, "
            f"got {type(image_positions).__name__} {getattr(image_positions, 'shape', None)}"
        )
    if image_positions.shape[0] != batch_size:
        raise ValueError(
            f"image_positions batch={image_positions.shape[0]} != hidden batch={batch_size}"
        )
    positions = image_positions.to(device=device, dtype=torch.long)
    if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= min(q_len, k_len)):
        raise ValueError(
            f"image_positions must be in [0,{min(q_len, k_len)}), got "
            f"min={int(positions.min())} max={int(positions.max())}"
        )
    if positions.shape[1] > 1:
        ordered = positions.sort(dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("image_positions contains duplicate positions within a sample")
    image = torch.zeros((batch_size, q_len), device=device, dtype=torch.bool)
    if positions.numel():
        image.scatter_(1, positions, True)
    return image


def build_visual_attention_mask(
    attention_mask: Optional[torch.Tensor],
    image_positions: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the explicit mask used by both matched visual attention arms.

    The image tokens are assumed to precede the instruction tokens, as in the
    pure stereo data route.  The explicit image-to-non-image block is kept in
    place as a safety invariant even if a malformed token order is supplied.
    """
    if mode not in ("causal", "image_bidir"):
        raise ValueError(f"explicit visual mask is unavailable for mode={mode!r}")
    mask = _as_additive_mask(
        attention_mask,
        batch_size=batch_size,
        q_len=q_len,
        k_len=k_len,
        device=device,
        dtype=dtype,
    )
    image = _image_mask(
        image_positions,
        batch_size=batch_size,
        q_len=q_len,
        k_len=k_len,
        device=device,
    )
    blocked = _blocked_cells(mask)
    key_padding = blocked.all(dim=-2).all(dim=1)
    query_padding = blocked.all(dim=-1).all(dim=1)
    valid_query_image = image & ~query_padding
    valid_key_image = image & ~key_padding
    sentinel = _negative_sentinel(dtype)

    image_to_non_image = valid_query_image.unsqueeze(-1) & ~image.unsqueeze(-2)
    mask = mask.masked_fill(image_to_non_image.unsqueeze(1), sentinel)
    if mode == "image_bidir":
        image_pairs = valid_query_image.unsqueeze(-1) & valid_key_image.unsqueeze(-2)
        mask = mask.masked_fill(image_pairs.unsqueeze(1), 0.0)
    return mask


def _get_call_argument(
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    *,
    name: str,
    positional_index: int,
) -> Tuple[Any, Optional[Tuple[str, Any]]]:
    if name in kwargs and len(args) > positional_index:
        raise TypeError(f"attention call supplied {name!r} both positionally and by keyword")
    if name in kwargs:
        return kwargs[name], ("kwargs", name)
    if len(args) > positional_index:
        return args[positional_index], ("args", positional_index)
    return None, None


def _replace_call_argument(
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    location: Optional[Tuple[str, Any]],
    *,
    name: str,
    value: Any,
) -> Tuple[List[Any], Dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if location is None or location[0] == "kwargs":
        new_kwargs[name] = value
    else:
        new_args[int(location[1])] = value
    return new_args, new_kwargs


def _cache_has_history(
    past_key_values: Any,
    cache_position: Any,
    *,
    layer_idx: Optional[int] = None,
) -> bool:
    if past_key_values is None:
        return False
    try:
        if layer_idx is None:
            sequence_length = past_key_values.get_seq_length()
        else:
            try:
                sequence_length = past_key_values.get_seq_length(layer_idx)
            except TypeError:
                sequence_length = past_key_values.get_seq_length()
        if int(sequence_length) > 0:
            return True
    except (AttributeError, TypeError, ValueError):
        return True
    if torch.is_tensor(cache_position) and cache_position.numel():
        return int(cache_position.min()) > 0
    return False


def prepare_visual_attention_call(
    hidden_states: torch.Tensor,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    *,
    image_positions: Optional[torch.Tensor],
    mode: str,
    layer_idx: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Rewrite one Qwen3.5 full-attention call for a matched visual arm."""
    if mode == "legacy_mixed":
        return list(args), dict(kwargs)
    if mode not in ("causal", "image_bidir"):
        raise ValueError(f"unknown visual attention mode={mode!r}")
    if image_positions is None or image_positions.numel() == 0:
        return list(args), dict(kwargs)

    attention_mask, mask_location = _get_call_argument(
        args, kwargs, name="attention_mask", positional_index=1
    )
    past_key_values, _ = _get_call_argument(
        args, kwargs, name="past_key_values", positional_index=2
    )
    cache_position, _ = _get_call_argument(
        args, kwargs, name="cache_position", positional_index=3
    )
    q_len = int(hidden_states.shape[1])
    if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
        k_len = int(attention_mask.shape[-1])
    else:
        k_len = q_len
    active_cache = _cache_has_history(past_key_values, cache_position, layer_idx=layer_idx)
    if active_cache or q_len != k_len:
        raise RuntimeError(
            f"[cam_branch] visual mode={mode} cannot rewrite cached/image attention "
            f"at layer {layer_idx}: q_len={q_len} k_len={k_len}; refusing ambiguous image query"
        )
    if q_len != k_len:
        raise RuntimeError(f"[cam_branch] visual prefill requires Q==K at layer {layer_idx}")

    rewritten_mask = build_visual_attention_mask(
        attention_mask,
        image_positions,
        mode=mode,
        batch_size=int(hidden_states.shape[0]),
        q_len=q_len,
        k_len=k_len,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    new_args, new_kwargs = _replace_call_argument(
        args,
        kwargs,
        mask_location,
        name="attention_mask",
        value=rewritten_mask,
    )
    # Qwen3_5Attention forwards this kwarg to sdpa_attention_forward, where it
    # takes precedence over module.is_causal without mutating the module.
    new_kwargs["is_causal"] = False
    return new_args, new_kwargs
