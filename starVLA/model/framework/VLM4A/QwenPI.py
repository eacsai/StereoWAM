# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.modules.stereo import install_stereo_cam_rope_hooks

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPIDefaultConfig:
    """QwenPI (QwenFM) framework default parameters.

    Layer-wise cross-DiT flow-matching action prediction conditioned on
    multi-layer VLM hidden states.  All fields can be overridden by the
    corresponding key in the YAML ``framework:`` section.
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenPI"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (auto-set at runtime from model config)
            "vl_hidden_dim": 2048,
            # Number of VL transformer layers (auto-set at runtime)
            "num_vl_layers": 36,
            # === Phase 3 B: stereo Camera-Frame RoPE (dim expansion, opt-in) ===
            # When True, patch every Qwen3_5Attention layer to add a d_c branch
            # carrying camera-conditioned positional rotation (StereoWorld Eq 6-8).
            # Image tokens get cam_id-dependent P_t rotation; text tokens get 0.
            # Zero Init so step-0 output equals the mono baseline byte-identical.
            "stereo_cam_rope_enabled": False,
            "stereo_cam_rope_d_c": 16,             # new dim per head
            "stereo_cam_rope_num_cameras": 2,
            "stereo_cam_rope_baseline_m": 0.06,    # 6cm right of agentview
            "stereo_cam_rope_fovy_degrees": 45.0,  # LIBERO agentview default
            "stereo_cam_rope_image_width": 256,
            "stereo_cam_rope_image_height": 256,
            "stereo_cam_rope_spatial_merge": 2,    # Qwen3.5-VL default
            # init_mode default "zero": byte-identical step-0 to baseline.
            # (copy_temporal_mrope variant for Qwen3-VL-4B was removed —
            #  4B+GR00T NaN'd upstream, Issue 171.)
            "stereo_cam_rope_init_mode": "zero",
            # === Phase 4 epipolar attention mask (StereoWorld paper Sec 3.3) ===
            "stereo_epipolar_mask_enabled": False,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "LayerwiseFM",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            # Repeat factor for flow-matching loss
            "repeated_diffusion_steps": 2,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # DiT architecture settings — shape fields (num_layers,
            # input_embedding_dim, cross_attention_dim, num_attention_heads)
            # are auto-populated by populate_layerwise_dit_cfg at runtime.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFM")
@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model (PI variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Layer-wise cross-DiT diffusion head fed by multi-layer VLM hidden states

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    #
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
        self.config = merge_framework_config(QwenPIDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors;
        # we keep the last num_hidden_layers of them, so DiT depth must match.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # QwenPI: DiT runs at the LLM hidden size (no compression).  Tell the
        # action head exactly that — the head itself does not look at qwenvl.*.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # === Phase 3 B: Stereo Camera-Frame RoPE (dim expansion, opt-in) ===
        # Patches each standard attention layer to add q_cam/k_cam projections;
        # image tokens get a camera-conditioned rotation in the new d_c dim.
        # Zero Init → step-0 byte-identical to baseline.
        self._stereo_cam_rope_state = None
        self._stereo_cam_rope_layers = None
        self._stereo_cam_rope_top_hook_handle = None
        self._stereo_cam_rope_top_hook_fn = None
        self._stereo_cam_rope_reinstall_top_hook = None
        if bool(self.config.framework.qwenvl.get("stereo_cam_rope_enabled", False)):
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
            )
            self._stereo_cam_rope_top_hook_handle = getattr(
                self._stereo_cam_rope_state, "top_pre_hook_handle", None
            )
            self._stereo_cam_rope_top_hook_fn = getattr(self._stereo_cam_rope_state, "top_pre_hook_fn", None)
            self._stereo_cam_rope_reinstall_top_hook = getattr(
                self._stereo_cam_rope_state, "reinstall_top_pre_hook", None
            )
            # B-1a (codex round-1 fix): fail-closed startup check. install_*_hooks
            # already raises if no Qwen3_5Attention layers exist, but we double-check
            # here because future model surgery / freeze_modules cfg may silently
            # leave the list empty without raising.
            if not scl_modules:
                raise RuntimeError(
                    "stereo_cam_rope_enabled=True but install_stereo_cam_rope_hooks returned 0 layers. "
                    "Did the model architecture change so no Qwen3_5Attention layers are present?"
                )
            # Register the per-layer projection modules as a ModuleList child so
            # they participate in state_dict / DDP / DeepSpeed sharding.
            self.stereo_cam_rope_layers = nn.ModuleList(scl_modules)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Permissive load that survives Phase 3 stereo module toggles.

        Phase 3 introduces opt-in stereo modules (cam_embed, cam_rope). When the
        ckpt and the model disagree on which are enabled, the rest of the
        network is fine — we don'''t want a mismatch on the optional branch to
        block resume / eval.
        """
        def _is_stereo_key(k):
            if k.startswith("stereo_cam_embed.") or k.startswith("stereo_cam_rope_layers."):
                return True
            if ".language_model.layers." in k and ".self_attn.stereo_cam_layer." in k:
                return True
            return False

        own_keys = set(self.state_dict().keys())
        provided_keys = set(state_dict.keys())

        missing_stereo = {k for k in own_keys - provided_keys if _is_stereo_key(k)}
        extra_stereo = {k for k in provided_keys - own_keys if _is_stereo_key(k)}

        if missing_stereo or extra_stereo:
            import logging
            if missing_stereo:
                logging.warning(
                    f"[stereo] ckpt missing {len(missing_stereo)} stereo module key(s) — "
                    f"loading older ckpt into Phase 3 model. New modules stay at zero-init (safe)."
                )
            if extra_stereo:
                logging.warning(
                    f"[stereo] ckpt has {len(extra_stereo)} extra stereo module key(s) — "
                    f"loading Phase 3 ckpt into model with module disabled. Dropping extras."
                )
                state_dict = {k: v for k, v in state_dict.items() if k not in extra_stereo}
            strict = False

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> List[torch.Tensor]:
        """Run QwenVL and return the last-N layer-wise hidden states for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: encode through QwenVL
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat features for each layer
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(  # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to directly regress future actions (no diffusion sampling).

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

        # Step 1: encode through QwenVL
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state
            )  # (B, chunk_len, action_dim)

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

    model = Qwen_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
