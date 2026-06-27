from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TFFSBase
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.controlvla_branch import (
    clear_ffs_tokens,
    install_controlvla_branches,
    set_ffs_tokens,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = logging.getLogger(__name__)


def _remap_legacy_attn1_keys(state_dict, own_keys=None):
    remapped = {}
    n_legacy = 0
    for key, value in state_dict.items():
        if (
            ".attn1." in key
            and ".attn1.base." not in key
            and any(part in key for part in (".to_q.", ".to_k.", ".to_v.", ".to_out."))
        ):
            new_key = key.replace(".attn1.", ".attn1.base.", 1)
            # With GR00T interleave_self_attention=True, self-attn-only blocks are
            # not wrapped. Remap only keys whose wrapped destination actually
            # exists; leave self-attn block keys at their original attn1.* names.
            if own_keys is None or new_key in own_keys:
                remapped[new_key] = value
                n_legacy += 1
                continue
        remapped[key] = value
    return remapped, n_legacy


@dataclass
class QwenGR00TControlVLAFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_ControlVLAFFS"
    ffs_controlvla: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "ffs_pool_size": 8,
            "num_cameras": 2,
            "left_ref_idx": 1,
            "primary_view_idx": 0,
            "inject_cam_id": 1,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_ControlVLAFFS")
class QwenGR00T_ControlVLAFFS(QwenGR00TFFSBase):
    """GR00T + FFS #7: ControlVLA-style parallel K/V branch in action DiT cross-attn."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TControlVLAFFSDefaultConfig, self.config)
        self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_controlvla", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-ControlVLA-FFS]")
        self.ffs_pool_size = int(ffs_cfg.get("ffs_pool_size", 8))
        if self.ffs_pool_size <= 0:
            raise ValueError(
                f"[GR00T-ControlVLA-FFS] ffs_pool_size must be positive, got {self.ffs_pool_size}"
            )

        n_ffs_tokens = self.ffs_pool_size * self.ffs_pool_size
        self.ffs_pos_emb = nn.Parameter(
            torch.randn(1, n_ffs_tokens, self.ffs_feat_dim) * 0.02
        )

        dit_inner = self.action_model.model
        diffusion_cfg = self.config.framework.action_model.diffusion_model_cfg
        configured_num_layers = int(
            diffusion_cfg.get("num_layers", len(dit_inner.transformer_blocks))
            if hasattr(diffusion_cfg, "get")
            else len(dit_inner.transformer_blocks)
        )
        self._n_controlvla_blocks = install_controlvla_branches(
            action_dit_model=dit_inner,
            ffs_token_dim=self.ffs_feat_dim,
            expected_n_total_blocks=configured_num_layers,
            require_all_cross_attn=False,
        )
        self._assert_controlvla_coverage()
        self._assert_key_prefixes_have_state()
        logger.info(
            "[GR00T-ControlVLA-FFS] net[0] tokens=%dx%d dim=%d patched_cross_attn=%d",
            self.ffs_pool_size,
            self.ffs_pool_size,
            self.ffs_feat_dim,
            self._n_controlvla_blocks,
        )

    def _assert_controlvla_coverage(self) -> None:
        blocks = self.action_model.model.transformer_blocks
        cross_attn_indices = [idx for idx, block in enumerate(blocks) if block.cross_attention_dim is not None]
        patched_indices = [idx for idx in cross_attn_indices if hasattr(blocks[idx].attn1, "to_k_z")]
        if patched_indices != cross_attn_indices:
            raise RuntimeError(
                "[GR00T-ControlVLA-FFS audit] patched cross-attn blocks "
                f"{patched_indices} != expected {cross_attn_indices}"
            )
        if self._n_controlvla_blocks != len(cross_attn_indices):
            raise RuntimeError(
                "[GR00T-ControlVLA-FFS audit] install returned "
                f"{self._n_controlvla_blocks} but expected {len(cross_attn_indices)} cross-attn blocks"
            )

    def _assert_key_prefixes_have_state(self) -> None:
        keys = set(self.state_dict().keys())
        missing = []
        for prefix in self._ffs_key_prefixes():
            if prefix == "ffs.":
                continue
            if not any(key == prefix or key.startswith(prefix) for key in keys):
                missing.append(prefix)
        if missing:
            raise RuntimeError(
                "[GR00T-ControlVLA-FFS audit] configured FFS key prefixes "
                f"have no state_dict keys: {missing}"
            )

    def _prepare_ffs_for_vlm(self, batch_images: List) -> None:
        # #7 injects only into the action head. The VLM path stays unchanged.
        return None

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _encode_last_hidden_plain(self, batch_images: List, instructions: List[str]) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        return self._run_qwenvl_forward(qwen_inputs)

    def _controlvla_branch_dtype(self) -> torch.dtype:
        for block in self.action_model.model.transformer_blocks:
            attn1 = getattr(block, "attn1", None)
            if hasattr(attn1, "to_k_z"):
                return attn1.to_k_z.weight.dtype
        raise RuntimeError("[GR00T-ControlVLA-FFS] no patched ControlVLA attn1 branch found")

    def _compute_ffs_tokens(self, batch_images: List) -> torch.Tensor:
        net0 = self._compute_ffs_feature(batch_images)
        ffs_pooled = F.adaptive_avg_pool2d(
            net0, (self.ffs_pool_size, self.ffs_pool_size)
        )
        ffs_tokens = ffs_pooled.flatten(2).transpose(1, 2).contiguous().detach()
        ffs_tokens = ffs_tokens + self.ffs_pos_emb.to(
            device=ffs_tokens.device,
            dtype=ffs_tokens.dtype,
        )
        return ffs_tokens.to(dtype=self._controlvla_branch_dtype())

    def _repeated_diffusion_steps(self) -> int:
        return int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            if self.config and hasattr(self.config, "framework")
            else 4
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        clear_ffs_tokens()
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden = self._encode_last_hidden_plain(batch_images, instructions)
        ffs_tokens = self._compute_ffs_tokens(batch_images)

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

            set_ffs_tokens(ffs_tokens)
            try:
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                )
            finally:
                clear_ffs_tokens()

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: str):
        clear_ffs_tokens()
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden = self._encode_last_hidden_plain(batch_images, instructions)
        ffs_tokens = self._compute_ffs_tokens(batch_images)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            set_ffs_tokens(ffs_tokens)
            try:
                pred_actions = self.action_model.predict_action(last_hidden, state)
            finally:
                clear_ffs_tokens()
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _ffs_key_prefixes(self) -> Tuple[str, ...]:
        prefixes = ["ffs.", "ffs_pos_emb"]
        for idx, block in enumerate(self.action_model.model.transformer_blocks):
            if getattr(block, "cross_attention_dim", None) is None:
                continue
            prefixes.extend(
                [
                    f"action_model.model.transformer_blocks.{idx}.attn1.to_k_z.",
                    f"action_model.model.transformer_blocks.{idx}.attn1.to_v_z.",
                ]
            )
        return tuple(prefixes)

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        remapped, n_legacy = _remap_legacy_attn1_keys(
            state_dict,
            own_keys=set(self.state_dict().keys()),
        )
        if n_legacy:
            logger.info(
                "[GR00T-ControlVLA-FFS audit] remapped %d legacy attn1.* trunk keys to attn1.base.*",
                n_legacy,
            )
        if init_from_baseline:
            own_keys = set(self.state_dict().keys())
            provided_keys = set(remapped.keys())
            own_controlvla_keys = {
                key
                for key in own_keys
                if key == "ffs_pos_emb"
                or ".attn1.to_k_z." in key
                or ".attn1.to_v_z." in key
            }
            provided_controlvla_keys = own_controlvla_keys & provided_keys
            missing_controlvla_keys = own_controlvla_keys - provided_controlvla_keys
            if provided_controlvla_keys and missing_controlvla_keys:
                raise RuntimeError(
                    "[GR00T-ControlVLA-FFS audit] checkpoint contains a partial "
                    "ControlVLA branch family: "
                    f"{len(provided_controlvla_keys)} present, "
                    f"{len(missing_controlvla_keys)} missing; first missing="
                    f"{sorted(missing_controlvla_keys)[:10]}"
                )
        return super().load_state_dict(
            remapped,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )
