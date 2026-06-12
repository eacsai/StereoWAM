from dataclasses import dataclass, field
from typing import Optional

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.llama_adapter_prefix_inject import install_llama_adapter_prefix
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenGR00TLlamaAdapterPrefixFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_LlamaAdapterPrefixFFS"
    ffs_llama_adapter_prefix: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "num_cameras": 2,
            "primary_idx": 1,
            "right_view_idx": 0,
            "primary_cam_id": 1,
            "n_prompts": 10,
            "absorb_dim": 256,
            "gate_per_head": False,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_LlamaAdapterPrefixFFS")
class QwenGR00T_LlamaAdapterPrefixFFS(QwenGR00TFFSBase):
    """GR00T + FFS #5: per-layer LLaMA-Adapter style prefix attention."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TLlamaAdapterPrefixFFSDefaultConfig, self.config)
        self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_llama_adapter_prefix", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-LlamaAdapterPrefix-FFS]")

        trainer_cfg = getattr(self.config, "trainer", {})
        logging_frequency = (
            int(trainer_cfg.get("logging_frequency", 20))
            if hasattr(trainer_cfg, "get")
            else 20
        )
        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self.ffs_prefix_adapter = install_llama_adapter_prefix(
            self.qwen_vl_interface.model,
            in_ch=self.ffs_feat_dim,
            n_prompts=int(ffs_cfg.get("n_prompts", 10)),
            absorb_dim=int(ffs_cfg.get("absorb_dim", 256)),
            gate_per_head=bool(ffs_cfg.get("gate_per_head", False)),
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            extra_image_cam_id=self.config.framework.qwenvl.get("stereo_extra_image_cam_id", None),
            logging_frequency=logging_frequency,
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))
        self._ffs_prefix_state = self.ffs_prefix_adapter.state
        self._assert_key_prefixes_have_state()

    def _assert_key_prefixes_have_state(self) -> None:
        keys = set(self.state_dict().keys())
        missing = []
        for prefix in self._ffs_key_prefixes():
            if prefix == "ffs.":
                continue
            if not any(key.startswith(prefix) for key in keys):
                missing.append(prefix)
        if missing:
            raise RuntimeError(
                "[GR00T-LlamaAdapterPrefix-FFS audit] configured FFS key prefixes "
                f"have no state_dict keys: {missing}"
            )

    def _prepare_ffs_for_vlm(self, batch_images):
        self.ffs_prefix_adapter.clear_summary()
        net0 = self._compute_ffs_feature(batch_images)
        self.ffs_prefix_adapter.prepare_from_net0(net0)

    def _cleanup_ffs_after_vlm(self) -> None:
        # Keep the summary through backward: checkpointed VLM layers may rerun the
        # wrapped attention while gradients flow into ffs_prefix_adapter.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "ffs_prefix_adapter.")

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )
        if init_from_baseline:
            self.ffs_prefix_adapter.assert_all_gates_zero("[GR00T-LlamaAdapterPrefix-FFS audit]")
        return result
