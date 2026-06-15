"""Neutral-position point-token insertion wrapper.

Method #10 resampler tokens are unordered global point-cloud summaries, so they
reuse the depth-token sequence bookkeeping with neutral position columns rather
than spatial primary-image position columns.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .depth_token_inject import (
    DepthTokenInjectState as PointTokenInjectState,
    clear_state as clear_point_state,
    get_state as get_point_state,
    install_depth_token_hooks,
    set_depth_tokens,
)


def set_point_tokens(point_tokens: torch.Tensor) -> None:
    set_depth_tokens(point_tokens)


def install_point_token_hooks(
    hf_model,
    lm=None,
    *,
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
    primary_cam_id: int = 0,
    cam_rope_state=None,
    image_token_id: Optional[int] = None,
) -> Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
    return install_depth_token_hooks(
        hf_model=hf_model,
        lm=lm,
        num_cameras=num_cameras,
        spatial_merge_size=spatial_merge_size,
        primary_cam_id=primary_cam_id,
        cam_rope_state=cam_rope_state,
        image_token_id=image_token_id,
        position_mode="neutral",
    )


__all__ = [
    "PointTokenInjectState",
    "clear_point_state",
    "get_point_state",
    "install_point_token_hooks",
    "set_point_tokens",
]
