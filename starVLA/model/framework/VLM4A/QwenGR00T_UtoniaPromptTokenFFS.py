from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    qwen_vlm_base_dtype,
)
from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import (
    DEFAULT_UTONIA_POINTCLOUD_CFG,
    UtoniaPerTokenProjector,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id
from starVLA.model.modules.stereo.depth_token_inject import (
    clear_state as clear_depth_state,
    get_state as get_depth_state,
    install_depth_token_hooks,
    set_depth_tokens,
)
from starVLA.model.modules.stereo.utonia_pointcloud import UTONIA_FEATURE_DIM, UtoniaPointCloudMixin
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


def _default_prompttoken_cfg() -> dict:
    cfg = dict(DEFAULT_UTONIA_POINTCLOUD_CFG)
    cfg.update(
        {
            "num_point_tokens": 64,
            "point_prompt": "Left image point-cloud features:",
            "gate_init": "zero",
            "inject_hidden_dim": 256,
        }
    )
    return cfg


@dataclass
class QwenGR00TUtoniaPromptTokenFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_UtoniaPromptTokenFFS"
    utonia_pointcloud: dict = field(default_factory=_default_prompttoken_cfg)


@FRAMEWORK_REGISTRY.register("QwenGR00T_UtoniaPromptTokenFFS")
class QwenGR00T_UtoniaPromptTokenFFS(UtoniaPointCloudMixin, QwenGR00TFFSBase):
    """Cached Utonia grid source with input prompt-token insertion.

    Version A uses the same cached grid source but adds per-layer residuals onto
    primary image tokens. This variant changes the injection topology: one
    zero-initialized projector creates 64 prompt rows that are inserted before
    the primary image token run and kept in the sequence for the action head.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TUtoniaPromptTokenFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        pc_cfg = self.config.framework.get("utonia_pointcloud", {})
        self._utonia_pc_cfg = pc_cfg
        self._init_frozen_ffs_for_disparity(pc_cfg, "[GR00T-UtoniaPromptToken-FFS]")
        self._init_frozen_utonia(pc_cfg, "[GR00T-UtoniaPromptToken-FFS]")

        self.num_point_tokens = int(pc_cfg.get("num_point_tokens", 64))
        if self.num_point_tokens != 64:
            raise ValueError(
                "[GR00T-UtoniaPromptToken-FFS] num_point_tokens must stay fixed at 64 "
                f"for the no-pooling Utonia grid contract, got {self.num_point_tokens}"
            )
        self.point_token_prompt = str(pc_cfg.get("point_prompt", "Left image point-cloud features:"))
        self.utonia_prompt_projector = UtoniaPerTokenProjector(
            in_ch=UTONIA_FEATURE_DIM + 1,
            llm_dim=llm_dim,
            hidden_dim=int(pc_cfg.get("inject_hidden_dim", 256)),
            gate_init=str(pc_cfg.get("gate_init", "zero")),
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        hf_model = self.qwen_vl_interface.model
        self._depth_token_hook_handles = install_depth_token_hooks(
            hf_model=hf_model,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
            cam_rope_state=getattr(self, "_stereo_cam_rope_state", None),
            cam_branch_state=getattr(self, "_stereo_cam_branch_state", None),
            image_token_id=int(hf_model.config.image_token_id),
            position_mode="primary_grid",
        )

    def _pool_grid_hw(self) -> Tuple[int, int]:
        side = int(round(self.num_point_tokens ** 0.5))
        if side * side != self.num_point_tokens:
            raise ValueError(
                "[GR00T-UtoniaPromptToken-FFS] num_point_tokens must be a square grid budget, "
                f"got {self.num_point_tokens}"
            )
        return side, side

    def _build_utonia_prompttoken_qwenvl_inputs(self, batch_images, instructions):
        """Build [primary image][point-cloud prompt text][left_view image][instruction] messages.

        The depth-token insertion hook injects the actual 64 Utonia rows
        immediately before the left image token run.
        """

        assert len(batch_images) == len(instructions), "Images and instructions must have the same length"
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            if len(imgs) <= max(self.primary_view_idx, self.left_ref_idx):
                raise ValueError(
                    "[GR00T-UtoniaPromptToken-FFS] expected primary,left_view stereo images with "
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

    def _assert_primary_token_count_64(self, qwen_inputs) -> None:
        input_ids = qwen_inputs.get("input_ids", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        if input_ids is None or image_grid_thw is None:
            raise RuntimeError(
                "[GR00T-UtoniaPromptToken-FFS] cannot verify primary token count: "
                "qwen_inputs missing input_ids or image_grid_thw"
            )
        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        cam = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
            image_grid_thw=image_grid_thw,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
        )
        counts = (cam == int(self.inject_cam_id)).sum(dim=1)
        expected = torch.full_like(counts, self.num_point_tokens)
        if not torch.equal(counts, expected):
            raise RuntimeError(
                "[GR00T-UtoniaPromptToken-FFS] left_view token count must be 64 "
                f"before inserting Utonia prompt tokens; got {counts.detach().cpu().tolist()} "
                f"for inject_cam_id={self.inject_cam_id}"
            )

    def _encode_last_hidden_with_ffs(
        self,
        batch_images: List,
        instructions: List[str],
        sample_ids=None,
    ) -> torch.Tensor:
        qwen_inputs = self._build_utonia_prompttoken_qwenvl_inputs(batch_images, instructions)
        self._assert_primary_token_count_64(qwen_inputs)
        self._prepare_ffs_for_vlm(batch_images, sample_ids=sample_ids)
        try:
            last_hidden = self._run_qwenvl_forward(qwen_inputs)
            # Fix#1: 64 Utonia point tokens stay inserted mid-sequence ->
            # extend the action-head mask by num_point_tokens.
            self._stash_pending_mask(qwen_inputs, num_insert=self.num_point_tokens)
            return last_hidden
        finally:
            self._cleanup_ffs_after_vlm()

    def _prepare_ffs_for_vlm(self, batch_images, sample_ids=None):
        clear_depth_state()
        point_grid = self.compute_utonia_grid(
            batch_images,
            self._utonia_pc_cfg,
            grid_hw=self._pool_grid_hw(),
            sample_ids=sample_ids,
        )
        point_tokens = self.utonia_prompt_projector(point_grid, *self._pool_grid_hw())
        if point_tokens.ndim == 2:
            point_tokens = point_tokens.unsqueeze(0)
        expected_shape = (
            len(batch_images),
            self.num_point_tokens,
            int(self.config.framework.qwenvl.vl_hidden_dim),
        )
        if tuple(point_tokens.shape) != expected_shape:
            raise RuntimeError(
                "[GR00T-UtoniaPromptToken-FFS] Utonia prompt projector must return "
                f"{expected_shape}, got {tuple(point_tokens.shape)}"
            )
        set_depth_tokens(point_tokens)

    def _cleanup_ffs_after_vlm(self) -> None:
        # Keep the module-global insert state through backward because gradient
        # checkpointing can replay the language-model pre-hook.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "utonia.", "utonia_prompt_projector.")

    def _frozen_encoder_key_prefixes(self):
        return ("ffs.", "utonia.")

    def get_utonia_prompt_insert_state(self):
        return get_depth_state()
