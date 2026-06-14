from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TFFSBase
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.falcon_spatial_inject import install_falcon_spatial_inject
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TFalconSpatialInjectFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_FalconSpatialInjectFFS"
    ffs_falcon_spatial_inject: dict = field(
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
            "pooling": "amax",
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_FalconSpatialInjectFFS")
class QwenGR00T_FalconSpatialInjectFFS(QwenGR00TFFSBase):
    """GR00T + FFS #9: FALCON-style global spatial adapter at future_tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TFalconSpatialInjectFFSDefaultConfig, self.config)
        self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_falcon_spatial_inject", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-FalconSpatialInject-FFS]")

        trainer_cfg = getattr(self.config, "trainer", {})
        logging_frequency = (
            int(trainer_cfg.get("logging_frequency", 20))
            if hasattr(trainer_cfg, "get")
            else 20
        )
        self.ffs_spatial_inject = install_falcon_spatial_inject(
            self.action_model,
            in_ch=self.ffs_feat_dim,
            pooling=str(ffs_cfg.get("pooling", "amax")),
            logging_frequency=logging_frequency,
            label="[GR00T-FalconSpatialInject-FFS]",
        )
        self.ffs_spatial_inject.assert_spatial_gate_zero("[GR00T-FalconSpatialInject-FFS audit]")
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
                "[GR00T-FalconSpatialInject-FFS audit] configured FFS key prefixes "
                f"have no state_dict keys: {missing}"
            )

    def _prepare_ffs_for_vlm(self, batch_images: List) -> None:
        # This method injects FFS only at the action head. The VLM path stays
        # byte-identical to plain GR00T.
        return None

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _encode_last_hidden_plain(self, batch_images: List, instructions: List[str]) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        return self._run_qwenvl_forward(qwen_inputs)

    def _compute_spatial_vec(self, batch_images: List) -> torch.Tensor:
        net0 = self._compute_ffs_feature(batch_images)
        return self.ffs_spatial_inject.compute_spatial_vec(net0)

    def _repeated_diffusion_steps(self) -> int:
        return int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            if self.config and hasattr(self.config, "framework")
            else 4
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden = self._encode_last_hidden_plain(batch_images, instructions)
        spatial_vec = self._compute_spatial_vec(batch_images)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            repeated_diffusion_steps = self._repeated_diffusion_steps()
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            spatial_bias = self.ffs_spatial_inject.make_spatial_bias(
                spatial_vec,
                repeated_diffusion_steps=repeated_diffusion_steps,
            )
            self.ffs_spatial_inject.set_head_bias(spatial_bias)
            try:
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                )
            finally:
                self.ffs_spatial_inject.clear_head_bias()

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: str):
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden = self._encode_last_hidden_plain(batch_images, instructions)
        spatial_vec = self._compute_spatial_vec(batch_images)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            spatial_bias = self.ffs_spatial_inject.make_spatial_bias(
                spatial_vec,
                repeated_diffusion_steps=1,
            )
            self.ffs_spatial_inject.set_head_bias(spatial_bias)
            try:
                pred_actions = self.action_model.predict_action(last_hidden, state)
            finally:
                self.ffs_spatial_inject.clear_head_bias()
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _ffs_key_prefixes(self):
        return ("ffs.", "ffs_spatial_inject.")

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )
        if init_from_baseline:
            self.ffs_spatial_inject.assert_spatial_gate_zero("[GR00T-FalconSpatialInject-FFS audit]")
        return result
