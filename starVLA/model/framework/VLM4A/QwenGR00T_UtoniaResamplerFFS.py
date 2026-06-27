from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import (
    DEFAULT_UTONIA_POINTCLOUD_CFG,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.point_token_inject import (
    clear_point_state,
    install_point_token_hooks,
    set_point_tokens,
)
from starVLA.model.modules.stereo.utonia_pointcloud import UTONIA_FEATURE_DIM, UtoniaPointCloudMixin
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


class UtoniaResamplerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, queries: torch.Tensor, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~point_mask
        attn, _ = self.cross_attn(
            query=self.q_norm(queries),
            key=self.kv_norm(points),
            value=self.kv_norm(points),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn
        queries = queries + self.ffn(queries)
        return queries


class UtoniaResampler(nn.Module):
    """Global unordered Perceiver-style resampler from N point feats to K VLM tokens."""

    def __init__(
        self,
        num_queries: int = 64,
        point_dim: int = UTONIA_FEATURE_DIM,
        d_model: int = 1024,
        n_layers: int = 1,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.point_dim = int(point_dim)
        self.d_model = int(d_model)
        self.queries = nn.Parameter(torch.randn(self.num_queries, self.d_model) * 0.02)
        self.point_proj = nn.Linear(self.point_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [UtoniaResamplerBlock(self.d_model, int(n_heads)) for _ in range(int(n_layers))]
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, point_feats: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if point_feats.ndim != 3 or int(point_feats.shape[-1]) != self.point_dim:
            raise ValueError(
                f"UtoniaResampler expected (B,N,{self.point_dim}), got {tuple(point_feats.shape)}"
            )
        B, N, _ = point_feats.shape
        if point_mask is None:
            point_mask = torch.ones(B, N, device=point_feats.device, dtype=torch.bool)
        if point_mask.shape != (B, N):
            raise ValueError(f"point_mask shape {tuple(point_mask.shape)} != {(B, N)}")
        point_mask = point_mask.to(device=point_feats.device, dtype=torch.bool)
        empty = ~point_mask.any(dim=1)
        if bool(empty.any()):
            point_mask = point_mask.clone()
            point_mask[empty, 0] = True
        points = self.point_proj(point_feats.to(dtype=self.point_proj.weight.dtype))
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).to(dtype=points.dtype, device=points.device)
        for block in self.blocks:
            queries = block(queries, points, point_mask)
        return self.out_proj(queries)


@dataclass
class QwenGR00TUtoniaResamplerFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_UtoniaResamplerFFS"
    utonia_pointcloud: dict = field(default_factory=lambda: dict(DEFAULT_UTONIA_POINTCLOUD_CFG))


@FRAMEWORK_REGISTRY.register("QwenGR00T_UtoniaResamplerFFS")
class QwenGR00T_UtoniaResamplerFFS(UtoniaPointCloudMixin, QwenGR00TFFSBase):
    """Method #10 Version B: insert neutral-position Utonia resampler tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TUtoniaResamplerFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        pc_cfg = self.config.framework.get("utonia_pointcloud", {})
        self._utonia_pc_cfg = pc_cfg
        self._init_frozen_ffs_for_disparity(pc_cfg, "[GR00T-UtoniaResampler-FFS]")
        self._init_frozen_utonia(pc_cfg, "[GR00T-UtoniaResampler-FFS]")

        self.num_point_tokens = int(pc_cfg.get("num_point_tokens", 64))
        self.point_token_prompt = str(pc_cfg.get("point_prompt", "Left view point cloud features:"))
        self.utonia_resampler = UtoniaResampler(
            num_queries=self.num_point_tokens,
            point_dim=UTONIA_FEATURE_DIM,
            d_model=llm_dim,
            n_layers=int(pc_cfg.get("resampler_layers", 1)),
            n_heads=int(pc_cfg.get("resampler_heads", 8)),
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        hf_model = self.qwen_vl_interface.model
        self._point_token_hook_handles = install_point_token_hooks(
            hf_model=hf_model,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
            cam_rope_state=getattr(self, "_stereo_cam_rope_state", None),
            image_token_id=int(hf_model.config.image_token_id),
        )

    def _build_pointtoken_qwenvl_inputs(self, batch_images, instructions):
        """Build [primary image][point prompt text][left_view image][instruction] messages."""

        assert len(batch_images) == len(instructions), "Images and instructions must have the same length"
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            if len(imgs) <= max(self.primary_view_idx, self.left_ref_idx):
                raise ValueError(
                    "[GR00T-UtoniaResampler-FFS] expected primary,left_view stereo images with "
                    f"at least {max(self.primary_view_idx, self.left_ref_idx) + 1} views, got {len(imgs)}"
                )
            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                task_prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                task_prompt = instruction
            content = [
                {"type": "image", "image": imgs[self.primary_view_idx]},
                {"type": "text", "text": self.point_token_prompt},
                {"type": "image", "image": imgs[self.left_ref_idx]},
                {"type": "text", "text": task_prompt},
            ]
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.qwen_vl_interface.model, "device", None)
        if model_device is None:
            model_device = next(self.qwen_vl_interface.model.parameters()).device
        return batch_inputs.to(model_device)

    def _encode_last_hidden_with_ffs(self, batch_images, instructions) -> torch.Tensor:
        qwen_inputs = self._build_pointtoken_qwenvl_inputs(batch_images, instructions)
        self._prepare_ffs_for_vlm(batch_images)
        try:
            return self._run_qwenvl_forward(qwen_inputs)
        finally:
            self._cleanup_ffs_after_vlm()

    def _prepare_ffs_for_vlm(self, batch_images):
        clear_point_state()
        point_feats, point_mask = self.compute_utonia_padded_features(batch_images, self._utonia_pc_cfg)
        point_tokens = self.utonia_resampler(point_feats, point_mask)
        set_point_tokens(point_tokens)

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "utonia.", "utonia_resampler.")

    def _frozen_encoder_key_prefixes(self):
        return ("ffs.", "utonia.")
