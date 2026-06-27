"""Shared helpers for per-token camera-id assignment (stereo input layout).

Originally this module also hosted the cam_rope d_c-branch patched attention
forward + install hooks (StereoCamRoPELayer). That cam_rope path was REMOVED —
it was a provably-inert double-zero no-op (q_cam/k_cam both zero-init -> zero
gradient deadlock) and it pushed head_dim past Flash-Attn-2's 256 cap (~4x
slower SDPA fallback). The StereoWorld-correct replacement is cam_branch
(cam_branch_attention.py).

What remains here is convention-agnostic and shared by the surviving paths:
  - compute_per_token_cam_id(): derives a per-token camera id from input_ids +
    image_grid_thw (used by every FFS injection module + cam_branch).
  - _PATCHED_FORWARD_DISPATCH: cam_branch_attention.py and
    llama_adapter_prefix_inject.py read its KEY SET (the supported full-softmax
    attention class names); the values are unused now (hence None).
"""
from __future__ import annotations

from typing import Optional

import torch


def compute_per_token_cam_id(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: Optional[torch.Tensor],
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
    extra_image_cam_id: Optional[int] = None,
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
    num_cameras : default 2 (left, right). Images are laid out CAMERA-MAJOR
        (outer camera, inner frame), so cam_id = img_idx // num_frames where
        num_frames = images_per_sample // num_cameras (single-frame stereo -> [0,1];
        3-frame stereo -> [0,0,0,1,1,1]). NOT modulo (that assumes interleaving).
    spatial_merge_size : Qwen3.5-VL default 2.
    extra_image_cam_id : optional gated extra-image cam id. When set, each
        sample must contain exactly one trailing extra image after the normal
        camera-major images; that final run is assigned this cam id.

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
    _nc = max(int(num_cameras), 1)
    if extra_image_cam_id is not None:
        extra_image_cam_id = int(extra_image_cam_id)
        if extra_image_cam_id < 0 or extra_image_cam_id >= _nc:
            raise ValueError(
                f"[cam_rope] extra_image_cam_id={extra_image_cam_id} out of range for num_cameras={_nc}"
            )
        expected_total_images = B * (_nc + 1)
        if int(image_grid_thw.shape[0]) != expected_total_images:
            raise ValueError(
                "[cam_rope] stereo_extra_image_cam_id is set, so each sample must have exactly "
                f"{_nc + 1} image_grid_thw rows ({_nc} normal camera images + 1 trailing extra); "
                f"got total rows={int(image_grid_thw.shape[0])} for batch_size={B}"
            )

    img_idx_global = 0
    for b in range(B):
        mask_b = image_mask[b]
        if not mask_b.any():
            if extra_image_cam_id is not None:
                raise ValueError(
                    f"[cam_rope] stereo_extra_image_cam_id set but sample {b} has no image tokens "
                    f"(expected {_nc + 1} image runs)"
                )
            continue
        # Find runs of consecutive image tokens.
        diff = torch.diff(mask_b.int(), prepend=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
        diff_end = torch.diff(mask_b.int(), append=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        ends = ((diff_end == -1).nonzero(as_tuple=True)[0] + 1).tolist()  # exclusive ends
        # camera-major layout (outer camera, inner frame): images_per_camera = num_frames.
        # cam_id = img_idx // num_frames (NOT % num_cameras, which assumes interleaving).
        if extra_image_cam_id is None:
            if len(starts) % _nc != 0:
                raise ValueError(f"[cam_rope] {len(starts)} images not a multiple of num_cameras={_nc}; camera-major layout broken (mono+cam_rope? set num_cameras=1)")
            num_frames_b = max(len(starts) // _nc, 1)
        else:
            expected_runs = _nc + 1
            if len(starts) != expected_runs:
                raise ValueError(
                    "[cam_rope] stereo_extra_image_cam_id is set, so each sample must have exactly "
                    f"{expected_runs} image token runs ({_nc} normal camera images + 1 trailing extra); "
                    f"sample={b} has {len(starts)}"
                )
            num_frames_b = 1
        for img_idx_in_sample, (s_start, s_end) in enumerate(zip(starts, ends)):
            if img_idx_global >= len(per_image_n):
                if extra_image_cam_id is not None:
                    raise ValueError(
                        "[cam_rope] image_grid_thw ended before all image token runs were assigned "
                        f"with stereo_extra_image_cam_id set (sample={b}, image={img_idx_in_sample})"
                    )
                break
            n_expected = per_image_n[img_idx_global]
            n_actual = s_end - s_start
            if n_actual != n_expected:
                if extra_image_cam_id is not None:
                    raise ValueError(
                        f"[cam_rope] image token run length mismatch with stereo_extra_image_cam_id set: "
                        f"sample={b} image={img_idx_in_sample} actual={n_actual} expected={n_expected}"
                    )
                # Layout assumption broken; skip this image rather than crash.
                # Up to caller to investigate (a warning is logged once by the framework).
                img_idx_global += 1
                continue
            if extra_image_cam_id is not None and img_idx_in_sample == len(starts) - 1:
                cam_id = extra_image_cam_id
            else:
                cam_id = (img_idx_in_sample // num_frames_b) % max(int(num_cameras), 1)
            out[b, s_start:s_end] = cam_id
            img_idx_global += 1

    return out


# Registry of standard (full-softmax) attention class names whose forward returns
# a post-o_proj, residual-ready (attn_output[B,S,H], ...) tuple. This was a
# class->patched-forward dispatch for the (removed) cam_rope d_c branch;
# cam_branch_attention.py and llama_adapter_prefix_inject.py only read .keys()
# (the supported-class set), so the values are unused (None).
_PATCHED_FORWARD_DISPATCH = {
    'Qwen3_5Attention': None,
}
