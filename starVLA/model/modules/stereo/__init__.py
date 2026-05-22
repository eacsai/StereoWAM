# Phase 3 Level B (Camera-Frame RoPE dim expansion). cam_embed.py (Level A2)
# lives on a sibling branch (phase3a_stereo_camembed); on this branch only
# cam_rope is present.
from .cam_rope import (
    StereoCamRoPELayer,
    apply_camera_rope,
    compute_libero_camera_P_stack,
)
from .cam_rope_hook import (
    StereoCamRoPEState,
    compute_per_token_cam_id,
    install_stereo_cam_rope_hooks,
    patched_qwen3_5_attention_forward,
)

__all__ = [
    'StereoCamRoPELayer',
    'StereoCamRoPEState',
    'apply_camera_rope',
    'compute_libero_camera_P_stack',
    'compute_per_token_cam_id',
    'install_stereo_cam_rope_hooks',
    'patched_qwen3_5_attention_forward',
]
