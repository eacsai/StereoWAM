from dataclasses import dataclass, field
from typing import Optional

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    FFSPerTokenProjector,
    QwenGR00TFFSBase,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.ffs_vlm_inject import (
    clear_ffs_state,
    install_ffs_vlm_input_hooks,
    set_ffs_feature,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenGR00TVLMInputFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_VLMInputFFS"
    ffs_vlm_input: dict = field(
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
            "inject_gate_init": "zero",
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_VLMInputFFS")
class QwenGR00T_VLMInputFFS(QwenGR00TFFSBase):
    """GR00T + FFS #1: zero-init net[0] residual before the VLM trunk."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TVLMInputFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_vlm_input", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-VLMInput-FFS]")

        self.ffs_vlm_injector = FFSPerTokenProjector(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            hidden_dim=int(ffs_cfg.get("inject_hidden_dim", 256)),
            gate_init=str(ffs_cfg.get("inject_gate_init", "zero")),
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._ffs_vlm_state = install_ffs_vlm_input_hooks(
            self.qwen_vl_interface.model,
            injector=self.ffs_vlm_injector,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
        )

    def _prepare_ffs_for_vlm(self, batch_images):
        clear_ffs_state()
        set_ffs_feature(self._compute_ffs_feature(batch_images))

    def _cleanup_ffs_after_vlm(self) -> None:
        # Keep hook state through backward: VLM gradient checkpointing may rerun
        # the input hook while computing gradients for the top-level injector.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "ffs_vlm_injector.")
