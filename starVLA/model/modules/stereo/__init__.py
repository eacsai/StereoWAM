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
from .cam_branch_attention import (
    CamBranchAttention,
    CamBranchState,
    build_prope_matrix_triple,
    install_cam_branch,
)

__all__ = [
    'CamBranchAttention',
    'CamBranchState',
    'StereoCamRoPELayer',
    'StereoCamRoPEState',
    'apply_camera_rope',
    'build_prope_matrix_triple',
    'compute_libero_camera_P_stack',
    'compute_per_token_cam_id',
    'install_cam_branch',
    'install_stereo_cam_rope_hooks',
    'patched_qwen3_5_attention_forward',
]
