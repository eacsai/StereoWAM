# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.modules.stereo import install_cam_branch, install_stereo_cam_rope_hooks


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
            # === Stereo Camera-Frame RoPE (ported from QwenPI) ===
            "stereo_cam_rope_enabled": False,
            "stereo_cam_rope_d_c": 16,
            "stereo_cam_rope_num_cameras": 2,
            "stereo_cam_rope_baseline_m": 0.06,
            "stereo_cam_rope_fovy_degrees": 45.0,
            "stereo_cam_rope_image_width": 256,
            "stereo_cam_rope_image_height": 256,
            "stereo_cam_rope_spatial_merge": 2,
            "stereo_cam_rope_init_mode": "zero",
            "stereo_cam_rope_right_first": True,
            "stereo_extra_image_cam_id": None,
            # === Phase 4 epipolar attention mask ===
            "stereo_epipolar_mask_enabled": False,
            # === Parallel PRoPE camera branch (default off = exact legacy behavior) ===
            "stereo_cam_branch_enabled": False,
            "stereo_cam_branch_heads": 4,
            "stereo_cam_branch_head_dim": 128,
            "stereo_cam_branch_layers": None,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        cam_rope_enabled = bool(self.config.framework.qwenvl.get("stereo_cam_rope_enabled", False))
        cam_branch_enabled = bool(self.config.framework.qwenvl.get("stereo_cam_branch_enabled", False))
        if cam_rope_enabled and cam_branch_enabled:
            raise RuntimeError(
                "stereo_cam_rope_enabled and stereo_cam_branch_enabled are mutually exclusive. "
                "cam_branch is the parallel PRoPE replacement; disable cam_rope first."
            )

        # === Stereo Camera-Frame RoPE install (ported from QwenPI) ===
        self._stereo_cam_rope_state = None
        self.stereo_cam_rope_layers = None
        if cam_rope_enabled:
            import torch.nn as _nn_for_stereo
            self._stereo_cam_rope_state, scl_modules = install_stereo_cam_rope_hooks(
                self.qwen_vl_interface.model,
                d_c=int(self.config.framework.qwenvl.get("stereo_cam_rope_d_c", 16)),
                num_cameras=int(self.config.framework.qwenvl.get("stereo_cam_rope_num_cameras", 2)),
                baseline_m=float(self.config.framework.qwenvl.get("stereo_cam_rope_baseline_m", 0.06)),
                fovy_degrees=float(self.config.framework.qwenvl.get("stereo_cam_rope_fovy_degrees", 45.0)),
                image_width=int(self.config.framework.qwenvl.get("stereo_cam_rope_image_width", 256)),
                image_height=int(self.config.framework.qwenvl.get("stereo_cam_rope_image_height", 256)),
                spatial_merge_size=int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2)),
                init_mode=str(self.config.framework.qwenvl.get("stereo_cam_rope_init_mode", "zero")),
                epipolar_mask_enabled=bool(self.config.framework.qwenvl.get("stereo_epipolar_mask_enabled", False)),
                right_first=bool(self.config.framework.qwenvl.get("stereo_cam_rope_right_first", True)),
                extra_image_cam_id=self.config.framework.qwenvl.get("stereo_extra_image_cam_id", None),
            )
            if not scl_modules:
                raise RuntimeError(
                    "stereo_cam_rope_enabled=True but install_stereo_cam_rope_hooks returned 0 layers. "
                    "Did the model architecture change so no attention layers found?"
                )
            self.stereo_cam_rope_layers = _nn_for_stereo.ModuleList(scl_modules)

        # === Parallel PRoPE camera branch install ===
        self._stereo_cam_branch_state = None
        self.stereo_cam_branch_layers_modules = None
        if cam_branch_enabled:
            import torch.nn as _nn_for_stereo

            trainer_cfg = getattr(self.config, "trainer", {})
            logging_frequency = (
                int(trainer_cfg.get("logging_frequency", 20))
                if hasattr(trainer_cfg, "get")
                else 20
            )
            self._stereo_cam_branch_state, branch_modules = install_cam_branch(
                self.qwen_vl_interface.model,
                branch_heads=int(self.config.framework.qwenvl.get("stereo_cam_branch_heads", 4)),
                branch_head_dim=int(self.config.framework.qwenvl.get("stereo_cam_branch_head_dim", 128)),
                layers=self.config.framework.qwenvl.get("stereo_cam_branch_layers", None),
                num_cameras=int(self.config.framework.qwenvl.get("stereo_cam_rope_num_cameras", 2)),
                baseline_m=float(self.config.framework.qwenvl.get("stereo_cam_rope_baseline_m", 0.06)),
                fovy_degrees=float(self.config.framework.qwenvl.get("stereo_cam_rope_fovy_degrees", 45.0)),
                image_width=int(self.config.framework.qwenvl.get("stereo_cam_rope_image_width", 256)),
                image_height=int(self.config.framework.qwenvl.get("stereo_cam_rope_image_height", 256)),
                spatial_merge_size=int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2)),
                right_first=bool(self.config.framework.qwenvl.get("stereo_cam_rope_right_first", True)),
                extra_image_cam_id=self.config.framework.qwenvl.get("stereo_extra_image_cam_id", None),
                logging_frequency=logging_frequency,
            )
            if not branch_modules:
                raise RuntimeError(
                    "stereo_cam_branch_enabled=True but install_cam_branch returned 0 layers. "
                    "Did the model architecture change so no attention layers found?"
                )
            self.stereo_cam_branch_layers_modules = _nn_for_stereo.ModuleList(branch_modules)

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        """Fail-closed load with warm-start exceptions for inert legacy stereo keys."""
        own_keys = set(self.state_dict().keys())
        cam_branch_enabled = bool(self.config.framework.qwenvl.get("stereo_cam_branch_enabled", False))
        legacy_audit_label = "cam_branch audit" if cam_branch_enabled else "legacy cam_rope audit"

        def _is_legacy_cam_rope_key(key: str) -> bool:
            # The inert legacy cam_rope registers the SAME modules under two families
            # (framework ModuleList + attention child); a cam_rope-enabled baseline
            # checkpoint (e.g. B) carries both. Any CAM_ROPE=0 warm-start target
            # has no such family, including cam_branch and FFS frameworks.
            return key.startswith("stereo_cam_rope_layers.") or ".stereo_cam_layer." in key

        if init_from_baseline and self.stereo_cam_rope_layers is None:
            legacy_keys = [k for k in state_dict.keys() if _is_legacy_cam_rope_key(k)]
            if legacy_keys:
                nonzero = [k for k in legacy_keys if state_dict[k].abs().max().item() != 0.0]
                if nonzero:
                    raise RuntimeError(
                        f"[{legacy_audit_label}] baseline checkpoint has NON-ZERO legacy cam_rope "
                        f"weights ({nonzero[:5]}...). cam_rope was provably inert (all-zero); "
                        "non-zero values mean this is not the expected baseline. Refusing."
                    )
                state_dict = {k: v for k, v in state_dict.items() if not _is_legacy_cam_rope_key(k)}
                logger.info(
                    "[%s] dropped %d inert legacy cam_rope keys (all verified zero) "
                    "from the warm-start checkpoint",
                    legacy_audit_label,
                    len(legacy_keys),
                )

        if not cam_branch_enabled:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        provided_keys = set(state_dict.keys())

        def _is_cam_branch_key(key: str) -> bool:
            return key.startswith("stereo_cam_branch_layers_modules.")

        # Missing branch keys are tolerated ONLY for a true baseline checkpoint that
        # has ZERO cam_branch keys. A checkpoint with SOME branch keys is a partial/
        # truncated cam_branch checkpoint — loading it with fresh-initialized gaps
        # would silently corrupt the run.
        provided_has_branch = any(_is_cam_branch_key(key) for key in provided_keys)
        allowed_missing = (
            {key for key in own_keys - provided_keys if _is_cam_branch_key(key)}
            if (init_from_baseline and not provided_has_branch)
            else set()
        )
        suspicious_missing = (own_keys - provided_keys) - allowed_missing
        unexpected = provided_keys - own_keys

        if suspicious_missing:
            sample = sorted(suspicious_missing)[:20]
            raise RuntimeError(
                "[cam_branch audit] warm-start would leave non-cam_branch params uninitialized; "
                f"missing {len(suspicious_missing)} keys, first={sample}"
            )
        if unexpected:
            sample = sorted(unexpected)[:20]
            raise RuntimeError(
                f"[cam_branch audit] checkpoint has {len(unexpected)} unexpected keys; "
                f"first={sample}. Refusing silent drop."
            )
        if allowed_missing:
            logger.info(
                "[cam_branch audit] allowing %d missing cam_branch keys during warm-start",
                len(allowed_missing),
            )

        forwarded_strict = strict and not allowed_missing
        return super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
