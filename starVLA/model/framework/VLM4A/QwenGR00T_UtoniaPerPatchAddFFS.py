from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    clear_layer_hook_state,
    install_ffs_vlm_layer_residual_hooks,
    language_model_layers,
    layer_attention,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.utonia_pointcloud import UTONIA_FEATURE_DIM, UtoniaPointCloudMixin
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


DEFAULT_UTONIA_POINTCLOUD_CFG = {
    "utonia_ckpt_path": "./playground/Pretrained_models/Utonia/utonia.pth",
    "utonia_expected_sha256": None,
    "utonia_scale": 4.0,
    "utonia_enable_flash": False,
    "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
    "ffs_expected_sha256": None,
    "ffs_feature_source": "gru_hidden",
    "gru_hidden_dim": 16,
    "ffs_image_size": 256,
    "fovy_degrees": 45.0,
    "baseline_m": 0.06,
    "image_width": 256,
    "image_height": 256,
    "backproject_stride": 4,
    "depth_min": 0.05,
    "depth_max": 3.0,
    "disp_eps": 1e-3,
    "num_cameras": 2,
    "left_ref_idx": 1,
    "primary_view_idx": 0,
    "inject_cam_id": 1,
    "utonia_cache_dir": None,
    "num_point_tokens": 64,
    "resampler_layers": 1,
    "resampler_heads": 8,
    "gate_init": "zero",
    "inject_hidden_dim": 256,
    "inject_depths": [3, 7, 11, 15, 19, 23],
    "expected_vlm_layers": 24,
}


class MaskAwareSpatialNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        mask = occupancy.to(device=x.device, dtype=x.dtype)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=(2, 3), keepdim=True) / denom
        var = ((x - mean).square() * mask).sum(dim=(2, 3), keepdim=True) / denom
        out = (x - mean) * torch.rsqrt(var + self.eps)
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out * mask


class UtoniaPerTokenProjector(nn.Module):
    """Utonia point-grid map plus occupancy -> VLM-token residual."""

    def __init__(self, in_ch: int, llm_dim: int, hidden_dim: int = 256, gate_init: str = "zero") -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.llm_dim = int(llm_dim)
        self.conv1 = nn.Conv2d(self.in_ch, int(hidden_dim), kernel_size=1)
        self.norm = MaskAwareSpatialNorm(int(hidden_dim))
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(int(hidden_dim), self.llm_dim, kernel_size=1)
        self.zero_proj = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        gate_init = str(gate_init).lower()
        if gate_init != "zero":
            raise ValueError(f"UtoniaPerTokenProjector only supports gate_init='zero', got {gate_init!r}")
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)
        self.gate_init = gate_init

    def forward(self, point_grid_b: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
        if point_grid_b.ndim != 4 or int(point_grid_b.shape[1]) != self.in_ch:
            raise ValueError(
                f"UtoniaPerTokenProjector expected (B,{self.in_ch},H,W), got {tuple(point_grid_b.shape)}"
            )
        if int(h_tok) != int(w_tok):
            raise RuntimeError(
                "Version A Utonia per-patch injection requires a square primary token grid; "
                f"got h_tok={int(h_tok)} w_tok={int(w_tok)}"
            )
        proj_dtype = next(self.parameters()).dtype
        x = point_grid_b.to(dtype=proj_dtype)
        occupancy = x[:, -1:, :, :].clamp(0.0, 1.0)
        x = self.conv1(x)
        x = self.norm(x, occupancy)
        x = self.act(x)
        x = self.conv2(x)
        if (int(x.shape[-2]), int(x.shape[-1])) != (int(h_tok), int(w_tok)):
            x = F.interpolate(x, size=(h_tok, w_tok), mode="bilinear", align_corners=False)
            occupancy = F.interpolate(occupancy, size=(h_tok, w_tok), mode="nearest")
        x = x * occupancy
        x = x.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        return self.zero_proj(x)


@dataclass
class QwenGR00TUtoniaPerPatchAddFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_UtoniaPerPatchAddFFS"
    utonia_pointcloud: dict = field(default_factory=lambda: dict(DEFAULT_UTONIA_POINTCLOUD_CFG))


@FRAMEWORK_REGISTRY.register("QwenGR00T_UtoniaPerPatchAddFFS")
class QwenGR00T_UtoniaPerPatchAddFFS(UtoniaPointCloudMixin, QwenGR00TFFSBase):
    """Method #10 Version A: Utonia per-patch residuals into primary image tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TUtoniaPerPatchAddFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        pc_cfg = self.config.framework.get("utonia_pointcloud", {})
        self._utonia_pc_cfg = pc_cfg
        self._init_frozen_ffs_for_disparity(pc_cfg, "[GR00T-UtoniaPerPatch-FFS]")
        self._init_frozen_utonia(pc_cfg, "[GR00T-UtoniaPerPatch-FFS]")

        target_layers = [int(x) for x in pc_cfg.get("inject_depths", [3, 7, 11, 15, 19, 23])]
        if target_layers != [3, 7, 11, 15, 19, 23]:
            raise ValueError(
                "[GR00T-UtoniaPerPatch-FFS] inject_depths must be fixed [3,7,11,15,19,23], "
                f"got {target_layers}"
            )
        layers = language_model_layers(self.qwen_vl_interface.model)
        expected_layers = int(pc_cfg.get("expected_vlm_layers", 24))
        if len(layers) != expected_layers:
            raise RuntimeError(
                f"[GR00T-UtoniaPerPatch-FFS] expected {expected_layers} VLM layers, got {len(layers)}"
            )
        bad_attn = []
        for idx in target_layers:
            attn_name = type(layer_attention(layers[idx])).__name__
            if attn_name != "Qwen3_5Attention":
                bad_attn.append((idx, attn_name))
        if bad_attn:
            raise RuntimeError(
                "[GR00T-UtoniaPerPatch-FFS] target layers must be Qwen3_5Attention softmax layers; "
                f"bad={bad_attn}"
            )
        self._utonia_inject_depths = target_layers

        self.utonia_pertoken_projectors = nn.ModuleList(
            [
                UtoniaPerTokenProjector(
                    in_ch=UTONIA_FEATURE_DIM + 1,
                    llm_dim=llm_dim,
                    hidden_dim=int(pc_cfg.get("inject_hidden_dim", 256)),
                    gate_init=str(pc_cfg.get("gate_init", "zero")),
                )
                for _ in target_layers
            ]
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._utonia_layer_hook_state = install_ffs_vlm_layer_residual_hooks(
            self.qwen_vl_interface.model,
            projectors=self.utonia_pertoken_projectors,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
            target_layer_indices=target_layers,
            label="GR00T-UtoniaPerPatch-FFS",
        )

    def _pool_grid_hw(self) -> Tuple[int, int]:
        n = int(self._utonia_pc_cfg.get("num_point_tokens", 64))
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"Version A num_point_tokens must be a square grid budget, got {n}")
        return side, side

    def _prepare_ffs_for_vlm(self, batch_images, sample_ids=None):
        clear_layer_hook_state(self._utonia_layer_hook_state)
        self._utonia_layer_hook_state.ffs_feat = self.compute_utonia_grid(
            batch_images,
            self._utonia_pc_cfg,
            grid_hw=self._pool_grid_hw(),
            sample_ids=sample_ids,
        )

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "utonia.", "utonia_pertoken_projectors.")

    def _frozen_encoder_key_prefixes(self):
        return ("ffs.", "utonia.")
