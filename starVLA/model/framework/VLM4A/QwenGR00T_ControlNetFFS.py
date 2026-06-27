from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    FFSPerTokenProjector,
    QwenGR00TFFSBase,
    clear_layer_hook_state,
    install_ffs_vlm_layer_residual_hooks,
    language_model_layers,
    layer_attention,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenGR00TControlNetFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_ControlNetFFS"
    ffs_controlnet: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "inject_hidden_dim": 256,
            "num_cameras": 2,
            "left_ref_idx": 1,
            "primary_view_idx": 0,
            "inject_cam_id": 1,
            "expected_vlm_layers": 24,
            "inject_depths": [3, 7, 11, 15, 19, 23],
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_ControlNetFFS")
class QwenGR00T_ControlNetFFS(QwenGR00TFFSBase):
    """GR00T + FFS #2: zero-init residuals at the 6 cam_rope SOFTMAX layers only."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TControlNetFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_controlnet", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-ControlNet-FFS]")

        layers = language_model_layers(self.qwen_vl_interface.model)
        expected_layers = int(ffs_cfg.get("expected_vlm_layers", 24))
        if len(layers) != expected_layers:
            raise RuntimeError(
                f"[GR00T-ControlNet-FFS] expected {expected_layers} VLM layers, got {len(layers)}"
            )

        # Inject ONLY at the cam_rope softmax layers (where the geometric attention
        # lives); the linear-attention layers carry no RoPE/cam_rope, so the FFS
        # disparity is skipped there — matching #3's softmax-only coverage. cam_rope is
        # attached only to softmax layers, so discovering cam_rope depths == finding the
        # softmax layers (same mechanism as QwenGR00T_VLMControlNetFFS).
        inject_depths = []
        for idx, layer in enumerate(layers):
            att = layer_attention(layer, required=False)
            if att is not None and hasattr(att, "stereo_cam_layer"):
                inject_depths.append(idx)
        expected_depths = [int(x) for x in ffs_cfg.get("inject_depths", [3, 7, 11, 15, 19, 23])]
        if inject_depths != expected_depths:
            raise RuntimeError(
                f"[GR00T-ControlNet-FFS] cam_rope softmax depth mismatch: "
                f"expected {expected_depths}, got {inject_depths}"
            )
        self._ffs_inject_depths = inject_depths

        self.ffs_layer_projectors = nn.ModuleList(
            [
                FFSPerTokenProjector(
                    in_ch=self.ffs_feat_dim,
                    llm_dim=llm_dim,
                    hidden_dim=int(ffs_cfg.get("inject_hidden_dim", 256)),
                )
                for _ in range(len(inject_depths))
            ]
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._ffs_layer_hook_state = install_ffs_vlm_layer_residual_hooks(
            self.qwen_vl_interface.model,
            projectors=self.ffs_layer_projectors,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
            target_layer_indices=inject_depths,
            label="GR00T-ControlNet-FFS",
        )

    def _prepare_ffs_for_vlm(self, batch_images):
        clear_layer_hook_state(self._ffs_layer_hook_state)
        self._ffs_layer_hook_state.ffs_feat = self._compute_ffs_feature(batch_images)

    def _cleanup_ffs_after_vlm(self) -> None:
        # Keep state through backward: checkpointed VLM layers can rerun hooks
        # while gradients flow into the top-level projectors.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "ffs_layer_projectors.")
