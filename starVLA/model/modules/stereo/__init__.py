# Stereo modules. cam_rope (StereoCamRoPELayer dim-expansion d_c branch) was
# removed — it was a provably-inert double-zero no-op. cam_branch (the parallel
# PRoPE replacement) and the shared compute_per_token_cam_id util remain.
from .cam_rope_hook import compute_per_token_cam_id
from .cam_branch_attention import (
    CamBranchAttention,
    CamBranchState,
    build_prope_matrix_triple,
    install_cam_branch,
)

__all__ = [
    'CamBranchAttention',
    'CamBranchState',
    'build_prope_matrix_triple',
    'compute_per_token_cam_id',
    'install_cam_branch',
]
