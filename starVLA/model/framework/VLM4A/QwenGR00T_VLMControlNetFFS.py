from dataclasses import dataclass, field
from typing import List, Optional

import torch

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    language_model_layers,
    layer_attention,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.vlm_controlnet import (
    FFSControlNetHint,
    VLMControlNetBranch,
    assert_branch_roundtrip_equal,
    attach_branch_cam_rope,
    clear_state,
    copy_trunk_non_cam_params_to_branch,
    install_vlm_controlnet_hooks,
    set_ffs_feature,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenGR00TVLMControlNetFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_VLMControlNetFFS"
    ffs_vlm_controlnet: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "hint_hidden_dim": 256,
            "num_cameras": 2,
            "primary_idx": 1,
            "right_view_idx": 0,
            "primary_cam_id": 1,
            "inject_depths": [3, 7, 11, 15, 19, 23],
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_VLMControlNetFFS")
class QwenGR00T_VLMControlNetFFS(QwenGR00TFFSBase):
    """GR00T + FFS #3: six-layer VLM ControlNet branch on cam_rope softmax depths."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TVLMControlNetFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_vlm_controlnet", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-VLMControlNet-FFS]")

        expected_depths = [int(x) for x in ffs_cfg.get("inject_depths", [3, 7, 11, 15, 19, 23])]
        if expected_depths != [3, 7, 11, 15, 19, 23]:
            raise ValueError(
                f"[GR00T-VLMControlNet-FFS] inject_depths must be [3, 7, 11, 15, 19, 23], "
                f"got {expected_depths}"
            )
        self._vlm_controlnet_inject_depths = self._discover_cam_rope_depths()
        if self._vlm_controlnet_inject_depths != expected_depths:
            raise RuntimeError(
                "[GR00T-VLMControlNet-FFS] cam_rope softmax depth mismatch: "
                f"expected {expected_depths}, got {self._vlm_controlnet_inject_depths}"
            )

        trunk_layers = self._vlm_controlnet_trunk_layers()
        trunk_scls = self._vlm_controlnet_trunk_scls()
        self._assert_softmax_layers(trunk_layers)

        base_dtype = qwen_vlm_base_dtype(self.qwen_vl_interface)
        self._vlm_controlnet_base_dtype = base_dtype
        self.ffs_controlnet_hint = FFSControlNetHint(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            hidden_dim=int(ffs_cfg.get("hint_hidden_dim", 256)),
        ).to(dtype=base_dtype)
        self.ffs_controlnet_branch = VLMControlNetBranch(
            source_layers=trunk_layers,
            inject_depths=self._vlm_controlnet_inject_depths,
            llm_dim=llm_dim,
            base_dtype=base_dtype,
        )
        attach_branch_cam_rope(
            self.ffs_controlnet_branch.branch_layers,
            self._stereo_cam_rope_state,
            trunk_scls,
            base_dtype,
        )
        self.ffs_controlnet_branch.assert_zero_convs_zero()

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._vlm_controlnet_state = install_vlm_controlnet_hooks(
            self.qwen_vl_interface.model,
            branch=self.ffs_controlnet_branch,
            hint=self.ffs_controlnet_hint,
            inject_depths=self._vlm_controlnet_inject_depths,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.primary_cam_id,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
        )

    def _discover_cam_rope_depths(self) -> List[int]:
        depths: List[int] = []
        for idx, layer in enumerate(language_model_layers(self.qwen_vl_interface.model)):
            attn = layer_attention(layer, required=False)
            if attn is not None and hasattr(attn, "stereo_cam_layer"):
                depths.append(idx)
        return depths

    def _vlm_controlnet_trunk_layers(self) -> List[torch.nn.Module]:
        layers = language_model_layers(self.qwen_vl_interface.model)
        return [layers[idx] for idx in self._vlm_controlnet_inject_depths]

    def _vlm_controlnet_trunk_scls(self) -> List[torch.nn.Module]:
        out = []
        for layer in self._vlm_controlnet_trunk_layers():
            attn = layer_attention(layer)
            scl = getattr(attn, "stereo_cam_layer", None)
            if scl is None:
                raise RuntimeError("[GR00T-VLMControlNet-FFS] trunk cam_rope layer missing")
            out.append(scl)
        return out

    def _assert_softmax_layers(self, layers: List[torch.nn.Module]) -> None:
        bad = []
        for depth, layer in zip(self._vlm_controlnet_inject_depths, layers):
            attn_name = type(layer_attention(layer)).__name__
            if attn_name != "Qwen3_5Attention":
                bad.append((depth, attn_name))
        if bad:
            raise RuntimeError(f"[GR00T-VLMControlNet-FFS] non-softmax layers found: {bad}")

    def _prepare_ffs_for_vlm(self, batch_images):
        clear_state()
        set_ffs_feature(self._compute_ffs_feature(batch_images))

    def _cleanup_ffs_after_vlm(self) -> None:
        # Do not clear here: gradient checkpointing may rerun VLM hooks during backward.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "ffs_controlnet_hint.", "ffs_controlnet_branch.")

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        # Default False (strict): a standalone / eval / resume load MUST contain the
        # FFS-adapter keys. ONLY the trainer's warm-start path passes
        # init_from_baseline=True (via signature inspection) to allow the adapter
        # keys to be fresh-initialised off a non-FFS baseline (e.g. B).
        own_keys = set(self.state_dict().keys())
        provided = set(state_dict.keys())

        def _is_controlnet_param_key(key: str) -> bool:
            return key.startswith("ffs_controlnet_hint.") or key.startswith("ffs_controlnet_branch.")

        expected_controlnet = {key for key in own_keys if _is_controlnet_param_key(key)}
        provided_controlnet = {key for key in provided if _is_controlnet_param_key(key)}
        branch_keys_absent = not provided_controlnet
        if provided_controlnet and provided_controlnet != expected_controlnet:
            missing = sorted(expected_controlnet - provided_controlnet)[:20]
            extra = sorted(provided_controlnet - expected_controlnet)[:20]
            raise RuntimeError(
                "[GR00T-VLMControlNet-FFS audit] partial ControlNet checkpoint; "
                f"missing_first={missing}, unexpected_controlnet_first={extra}"
            )

        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )

        if init_from_baseline and branch_keys_absent:
            trunk_layers = self._vlm_controlnet_trunk_layers()
            trunk_scls = self._vlm_controlnet_trunk_scls()
            copy_trunk_non_cam_params_to_branch(trunk_layers, self.ffs_controlnet_branch.branch_layers)
            attach_branch_cam_rope(
                self.ffs_controlnet_branch.branch_layers,
                self._stereo_cam_rope_state,
                trunk_scls,
                self._vlm_controlnet_base_dtype,
            )
            self.ffs_controlnet_branch.assert_zero_convs_zero()
            assert_branch_roundtrip_equal(self.ffs_controlnet_branch)
            self._audit_fresh_branch_roundtrip()
        return result

    def _audit_fresh_branch_roundtrip(self) -> None:
        fresh = VLMControlNetBranch(
            source_layers=self._vlm_controlnet_trunk_layers(),
            inject_depths=self._vlm_controlnet_inject_depths,
            llm_dim=int(self.config.framework.qwenvl.vl_hidden_dim),
            base_dtype=self._vlm_controlnet_base_dtype,
        )
        attach_branch_cam_rope(
            fresh.branch_layers,
            self._stereo_cam_rope_state,
            self._vlm_controlnet_trunk_scls(),
            self._vlm_controlnet_base_dtype,
        )
        fresh.load_state_dict(self.ffs_controlnet_branch.state_dict(), strict=True)
        ref = self.ffs_controlnet_branch.state_dict()
        got = fresh.state_dict()
        if set(ref) != set(got):
            raise RuntimeError("[GR00T-VLMControlNet-FFS audit] fresh branch key mismatch")
        for key, value in ref.items():
            if not torch.equal(value.detach().cpu(), got[key].detach().cpu()):
                raise RuntimeError(f"[GR00T-VLMControlNet-FFS audit] fresh branch mismatch at {key}")
