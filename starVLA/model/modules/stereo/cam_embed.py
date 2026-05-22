"""Stereo cam_id additive embedding (Phase 3 Level A2).

Adds learnable cam_id (left/right) + 2D patch position embeddings to image
token features inside the VLM, BEFORE language transformer self-attention.

Design (StereoWorld 2603.17375 Section 3 idea, light additive version):
  image_feat[cam, row, col] += cam_id_embed[cam] + pos_x_embed[col] + pos_y_embed[row]

Init: zeros — at training-start the model behavior is byte-identical to the
mono baseline; the model can incrementally activate the camera signal during
fine-tune. This is the safest possible introduction.

Hook target: register forward_hook on the Qwen3.5-VL visual module
(). Visual outputs are (sum_thw, llm_hidden) flat across
all images in batch; we split by image_grid_thw and add per-image embeddings.

Opt-in: controlled by config.framework.qwenvl.stereo_cam_embed_enabled
(default False -> module not constructed, hook not registered).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StereoCamEmbedding(nn.Module):
    """Learnable additive embedding tagging image patches with (cam_id, row, col).

    Parameters
    ----------
    hidden_dim : int
        Dimension of the image token embeddings (LLM hidden size after visual merger).
    num_cameras : int
        Number of stereo views, default 2 (left/right). Future extension: 3+ cameras.
    max_patches_x, max_patches_y : int
        Upper bound on patch grid dimensions. Embedding table size (cheap, kept
        large for safety; only the actually-used positions get gradients).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_cameras: int = 2,
        max_patches_x: int = 64,
        max_patches_y: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cameras = num_cameras
        self.cam_id_embed = nn.Embedding(num_cameras, hidden_dim)
        self.pos_x_embed = nn.Embedding(max_patches_x, hidden_dim)
        self.pos_y_embed = nn.Embedding(max_patches_y, hidden_dim)

        # Zero init: model output unchanged at step 0; signal activates during fine-tune.
        nn.init.zeros_(self.cam_id_embed.weight)
        nn.init.zeros_(self.pos_x_embed.weight)
        nn.init.zeros_(self.pos_y_embed.weight)

    def forward(
        self,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 1,
    ) -> torch.Tensor:
        """Add (cam_id, row, col) embeddings to image features in-place semantically.

        Parameters
        ----------
        image_features : (sum_thw, hidden_dim) tensor
            Concatenated image token features for all images in batch.
        image_grid_thw : (num_images, 3) tensor
            Per-image (t, h, w) patch grid sizes (Qwen-VL standard layout).
            We assume t == 1 for static images; h * w gives patches per image.

        Returns
        -------
        Tensor with same shape as image_features, with embeddings added.

        Notes
        -----
        * For each sample in the batch we assume images appear in [primary, right_view]
          order. cam_id is assigned as image_index % num_cameras within a sample.
        * If the actual image count per sample diverges from num_cameras, the cam_id
          assignment still cycles modulo num_cameras (safe but not meaningful).
        """
        if image_features.numel() == 0 or image_grid_thw.numel() == 0:
            return image_features

        device = image_features.device
        total_tokens = image_features.shape[0]

        # Split image_features by per-image MERGED patch count.
        # Each image contributes (t * h * w) // (s * s) tokens after spatial merge
        # (Qwen3.5-VL uses spatial_merge_size=2, so 4 raw patches become 1 LLM token).
        s = max(int(spatial_merge_size), 1)
        per_image_n = ((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (s * s)).tolist()
        assert sum(per_image_n) == total_tokens, (
            f'image_grid_thw merged counts sum {sum(per_image_n)} != image_features rows {total_tokens} (s={s})'
        )

        # Build the additive embedding for every image, then concat.
        chunks = []
        for img_idx, (n, thw) in enumerate(zip(per_image_n, image_grid_thw.tolist())):
            t, h_raw, w_raw = thw
            h, w = h_raw // s, w_raw // s
            assert t * h * w == n, f'merged grid ({t},{h},{w}) != n={n} for image {img_idx} '
            cam_id = img_idx % self.num_cameras
            cam_id_tensor = torch.tensor(cam_id, device=device, dtype=torch.long)
            # cam_id_embed: (hidden_dim,) -> broadcast to (n, hidden_dim)
            cam_e = self.cam_id_embed(cam_id_tensor)  # (hidden_dim,)
            cam_e = cam_e.unsqueeze(0).expand(n, -1)

            # 2D pos embed: rows then cols.
            # The patch grid for this image is t=1 x h x w, flattened row-major.
            rows = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            cols = torch.arange(w, device=device).repeat(h).repeat(t)
            assert rows.shape[0] == n and cols.shape[0] == n
            pos_e = self.pos_y_embed(rows) + self.pos_x_embed(cols)  # (n, hidden_dim)

            chunks.append(cam_e + pos_e)

        delta = torch.cat(chunks, dim=0).to(image_features.dtype)
        assert delta.shape == image_features.shape
        return image_features + delta
