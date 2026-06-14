from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    _raise_primary_grid_mismatch,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id
from starVLA.model.modules.stereo.value_residual_branch import (
    clear_aligned_ffs,
    install_value_residual_branches,
    set_aligned_ffs,
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
            # not wrapped. Remap only keys whose wrapped destination exists.
            if own_keys is None or new_key in own_keys:
                remapped[new_key] = value
                n_legacy += 1
                continue
        remapped[key] = value
    return remapped, n_legacy


def build_primary_aligned_ffs(
    *,
    net0: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    last_hidden: torch.Tensor,
    image_token_id: int,
    num_cameras: int,
    spatial_merge_size: int,
    primary_cam_id: int,
    label: str = "GR00T-ValueResidual-FFS",
) -> torch.Tensor:
    """Scatter raw FFS net[0] features onto primary image-token memory slots."""

    if net0.ndim != 4:
        raise RuntimeError(f"[{label}] net0 must be [B,C,H,W], got {tuple(net0.shape)}")
    if input_ids.ndim != 2 or last_hidden.ndim != 3:
        raise RuntimeError(
            f"[{label}] expected input_ids [B,T] and last_hidden [B,T,D], got "
            f"{tuple(input_ids.shape)} and {tuple(last_hidden.shape)}"
        )
    B, T_vl = input_ids.shape
    if int(last_hidden.shape[0]) != B or int(last_hidden.shape[1]) != T_vl:
        raise RuntimeError(
            f"[{label}] qwen input/hidden shape mismatch: input_ids={tuple(input_ids.shape)} "
            f"last_hidden={tuple(last_hidden.shape)}"
        )
    if int(net0.shape[0]) != B:
        raise RuntimeError(f"[{label}] FFS batch {net0.shape[0]} != VLM batch {B}")
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        raise RuntimeError(f"[{label}] image_grid_thw is required for primary-token alignment")
    min_grid_rows = B * int(num_cameras)
    if int(image_grid_thw.shape[0]) < min_grid_rows:
        raise RuntimeError(
            f"[{label}] image_grid_thw has {int(image_grid_thw.shape[0])} rows, "
            f"expected at least {min_grid_rows} rows for B={B}, num_cameras={int(num_cameras)}"
        )

    cam_id = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=int(image_token_id),
        image_grid_thw=image_grid_thw,
        num_cameras=int(num_cameras),
        spatial_merge_size=int(spatial_merge_size),
    )
    C = int(net0.shape[1])
    device = last_hidden.device
    dtype = last_hidden.dtype
    aligned = torch.zeros(B, T_vl, C, device=device, dtype=dtype)
    s2 = max(int(spatial_merge_size), 1)

    for b in range(B):
        row = b * int(num_cameras) + int(primary_cam_id)
        h_tok = int(image_grid_thw[row, 1].item()) // s2
        w_tok = int(image_grid_thw[row, 2].item()) // s2

        primary_pos = (cam_id[b] == int(primary_cam_id)).nonzero(as_tuple=True)[0]
        n_primary = int(primary_pos.numel())
        if h_tok * w_tok != n_primary:
            _raise_primary_grid_mismatch(label, b, h_tok, w_tok, n_primary, int(primary_cam_id))
        if n_primary == 0:
            continue

        feat_b = net0[b : b + 1].detach().to(device=device)
        interp_dtype = dtype
        if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            interp_dtype = torch.float32
        residual = F.interpolate(
            feat_b.to(dtype=interp_dtype),
            size=(h_tok, w_tok),
            mode="bilinear",
            align_corners=False,
        )
        residual = residual.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        if int(residual.shape[0]) != n_primary:
            _raise_primary_grid_mismatch(label, b, h_tok, w_tok, n_primary, int(primary_cam_id))
        aligned[b, primary_pos.to(device=device)] = residual.to(device=device, dtype=dtype)

    return aligned.detach()


@dataclass
class QwenGR00TValueResidualFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_ValueResidualFFS"
    ffs_value_residual: dict = field(
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
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_ValueResidualFFS")
class QwenGR00T_ValueResidualFFS(QwenGR00TFFSBase):
    """GR00T + FFS #8: aligned FFS residuals into action cross-attn Values."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TValueResidualFFSDefaultConfig, self.config)
        self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_value_residual", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-ValueResidual-FFS]")

        self.spatial_merge_size = int(
            self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2)
        )
        self._image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)

        dit_inner = self.action_model.model
        diffusion_cfg = self.config.framework.action_model.diffusion_model_cfg
        configured_num_layers = int(
            diffusion_cfg.get("num_layers", len(dit_inner.transformer_blocks))
            if hasattr(diffusion_cfg, "get")
            else len(dit_inner.transformer_blocks)
        )
        self._n_value_residual_blocks = install_value_residual_branches(
            action_dit_model=dit_inner,
            ffs_feat_dim=self.ffs_feat_dim,
            expected_n_total_blocks=configured_num_layers,
            require_all_cross_attn=False,
        )
        self._assert_value_residual_coverage()
        self._assert_key_prefixes_have_state()
        logger.info(
            "[GR00T-ValueResidual-FFS] net[0] dim=%d patched_cross_attn=%d spatial_merge=%d",
            self.ffs_feat_dim,
            self._n_value_residual_blocks,
            self.spatial_merge_size,
        )

    def _assert_value_residual_coverage(self) -> None:
        blocks = self.action_model.model.transformer_blocks
        cross_attn_indices = [idx for idx, block in enumerate(blocks) if block.cross_attention_dim is not None]
        patched_indices = [idx for idx in cross_attn_indices if hasattr(blocks[idx].attn1, "to_v_resid")]
        if patched_indices != cross_attn_indices:
            raise RuntimeError(
                "[GR00T-ValueResidual-FFS audit] patched cross-attn blocks "
                f"{patched_indices} != expected {cross_attn_indices}"
            )
        if self._n_value_residual_blocks != len(cross_attn_indices):
            raise RuntimeError(
                "[GR00T-ValueResidual-FFS audit] install returned "
                f"{self._n_value_residual_blocks} but expected {len(cross_attn_indices)} cross-attn blocks"
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
                "[GR00T-ValueResidual-FFS audit] configured FFS key prefixes "
                f"have no state_dict keys: {missing}"
            )

    def _prepare_ffs_for_vlm(self, batch_images: List) -> None:
        return None

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _encode_last_hidden_with_inputs(self, batch_images: List, instructions: List[str]):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        last_hidden = self._run_qwenvl_forward(qwen_inputs)
        return qwen_inputs, last_hidden

    def _compute_aligned_ffs(self, batch_images: List, qwen_inputs, last_hidden: torch.Tensor) -> torch.Tensor:
        net0 = self._compute_ffs_feature(batch_images)
        return build_primary_aligned_ffs(
            net0=net0,
            input_ids=qwen_inputs["input_ids"],
            image_grid_thw=qwen_inputs["image_grid_thw"],
            last_hidden=last_hidden,
            image_token_id=self._image_token_id,
            num_cameras=self.num_cameras,
            spatial_merge_size=self.spatial_merge_size,
            primary_cam_id=self.primary_cam_id,
            label="GR00T-ValueResidual-FFS",
        )

    def _repeated_diffusion_steps(self) -> int:
        return int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            if self.config and hasattr(self.config, "framework")
            else 4
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        clear_aligned_ffs()
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs, last_hidden = self._encode_last_hidden_with_inputs(batch_images, instructions)
        aligned_ffs = self._compute_aligned_ffs(batch_images, qwen_inputs, last_hidden)

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

            set_aligned_ffs(aligned_ffs)
            try:
                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                )
            finally:
                clear_aligned_ffs()

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: str):
        clear_aligned_ffs()
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs, last_hidden = self._encode_last_hidden_with_inputs(batch_images, instructions)
        aligned_ffs = self._compute_aligned_ffs(batch_images, qwen_inputs, last_hidden)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            set_aligned_ffs(aligned_ffs)
            try:
                pred_actions = self.action_model.predict_action(last_hidden, state)
            finally:
                clear_aligned_ffs()
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _ffs_key_prefixes(self) -> Tuple[str, ...]:
        prefixes = ["ffs."]
        for idx, block in enumerate(self.action_model.model.transformer_blocks):
            if getattr(block, "cross_attention_dim", None) is None:
                continue
            prefixes.append(
                f"action_model.model.transformer_blocks.{idx}.attn1.to_v_resid."
            )
        return tuple(prefixes)

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        remapped, n_legacy = _remap_legacy_attn1_keys(
            state_dict,
            own_keys=set(self.state_dict().keys()),
        )
        if n_legacy:
            logger.info(
                "[GR00T-ValueResidual-FFS audit] remapped %d legacy attn1.* trunk keys to attn1.base.*",
                n_legacy,
            )
        if init_from_baseline:
            own_keys = set(self.state_dict().keys())
            provided_keys = set(remapped.keys())
            own_value_keys = {key for key in own_keys if ".attn1.to_v_resid." in key}
            provided_value_keys = own_value_keys & provided_keys
            missing_value_keys = own_value_keys - provided_value_keys
            if provided_value_keys and missing_value_keys:
                raise RuntimeError(
                    "[GR00T-ValueResidual-FFS audit] checkpoint contains a partial "
                    "ValueResidual adapter family: "
                    f"{len(provided_value_keys)} present, "
                    f"{len(missing_value_keys)} missing; first missing="
                    f"{sorted(missing_value_keys)[:10]}"
                )
        return super().load_state_dict(
            remapped,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )
