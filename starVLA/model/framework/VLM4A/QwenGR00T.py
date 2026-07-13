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
from starVLA.model.modules.stereo import install_cam_branch


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
            # Auxiliary dense scene-flow head. Default off means no decoder module,
            # no hidden-state return from DiT, and no extra loss path.
            "scene_flow": {
                "enabled": False,
                "flow_lambda": 0.05,
                "online_grad_norm_enabled": False,
                "target_grad_ratio": 0.05,
                "lambda_min": 1.0e-3,
                "lambda_max": 1.0e4,
                "lambda_ema_decay": 0.97,
                "online_grad_norm_every_n_steps": 10,
                "probe_samples": 4,
                "lambda_jump_cap": 2.0,
                "lambda_warmup_steps": 100,
                "min_supervised_pixels": 128.0,
                "grad_norm_epsilon": 1.0e-12,
                "grid_size": 16,
                "hidden_layer": -1,
                "decoder_layers": 2,
                "decoder_heads": 8,
                "decoder_mlp_ratio": 4.0,
                "mask_mode": "dynamic",
                "dynamic_fallback_to_valid": True,
                "smooth_l1_beta": 0.01,
                "step0_flow_warmup_steps": 1,
                "step0_action_loss_audit": True,
                "step0_action_loss_audit_atol": 1.0e-4,
                "grad_ratio_steps": 20,
                "grad_ratio_min_warn": 0.01,
                "grad_ratio_failure_warn_every": 10,
                "grad_ratio_raise_on_failure": False,
            },
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
                # Dual-DiT cascade: width of the scene-flow DiT hidden that the
                # action DiT's zero-init motion coupler reads. None -> no coupler
                # (baseline unchanged). Set at runtime from scene_predictor.
                "motion_coupler_dim": None,
            },
        }
    )

    # === Scene-flow dual-DiT cascade (predicted-motion / static-geometry predictor) ===
    # Default disabled -> no scene predictor, no coupler, base behavior unchanged.
    scene_predictor: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "action_model_type": "DiT-B",
            "grid_size": 16,
            "field_dim": 3,
            "hidden_size": 1024,
            "target_key": "flow_gt",  # "pointmap_gt" for the static-geometry twin
            "valid_key": "flow_valid",
            "dynamic_key": "flow_dynamic",
            "dynamic_fallback_to_valid": False,  # stage-1 standalone: don't learn to predict zero
            "tap_hidden_index": 10,              # into all_hidden_states=[input]+per-block; avoid final
            "motion_extract_t_bucket": 0,
            "motion_extract_noise_mode": "zeros",
            "detach_conditioning": True,         # v1: detach; False = end-to-end ablation
            "flow_lambda": 0.1,                  # stage-2 joint weight
            "smooth_l1_beta": 0.01,
            "z_weight": 2.0,
            "dynamic_loss_weight": 1.0,
            "static_zero_loss_weight": 0.0,
            "dynamic_direction_loss_weight": 0.0,
            "dynamic_magnitude_loss_weight": 0.0,
            "direction_loss_eps": 1e-6,
            "conditioning_past_motion_gate": False,
            "conditioning_static_scale": 1.0,
            "conditioning_motion_threshold": 1e-5,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,     # aligned to VLM hidden at runtime
                "num_layers": 16,
                "output_dim": 768,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
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

        cam_branch_enabled = bool(self.config.framework.qwenvl.get("stereo_cam_branch_enabled", False))
        # cam_rope (StereoCamRoPELayer d_c branch) was removed — it was a
        # provably-inert double-zero no-op. Fail closed if a stale config still
        # asks for it instead of silently ignoring the flag.
        if bool(self.config.framework.qwenvl.get("stereo_cam_rope_enabled", False)):
            raise NotImplementedError(
                "stereo_cam_rope_enabled=True but the cam_rope module was removed "
                "(inert double-zero). Use stereo_cam_branch_enabled instead."
            )
        # Inert legacy attributes kept so load_state_dict's legacy cam_rope-key
        # drop (below) still works on old cam_rope-trained baseline checkpoints.
        self._stereo_cam_rope_state = None
        self.stereo_cam_rope_layers = None

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
        _vlm_hidden = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = _vlm_hidden

        # === Scene-flow dual-DiT cascade ===
        # Set the action DiT's motion-coupler width BEFORE building the action head
        # so the zero-init coupler is constructed. Gated by scene_predictor.enabled
        # (default False -> unchanged baseline).
        self._scene_predictor_cfg = getattr(self.config.framework, "scene_predictor", None)

        def _truthy(v):  # string-aware: quoted "false"/"0"/"no" must NOT enable (review M2)
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        self.scene_predictor_enabled = (
            _truthy(self._scene_predictor_cfg.get("enabled", False))
            if (self._scene_predictor_cfg is not None and hasattr(self._scene_predictor_cfg, "get"))
            else False
        )
        _coupler_dim = None
        if self.scene_predictor_enabled:
            from starVLA.model.modules.action_model.GR00T_ActionHeader import DiTConfig as _DiTCfg

            _sp_type = self._scene_predictor_cfg.get("action_model_type", "DiT-B")
            _coupler_dim = int(_DiTCfg[_sp_type]["input_embedding_dim"])
            self.config.framework.action_model.diffusion_model_cfg.motion_coupler_dim = _coupler_dim
            _legacy_sf = self.config.framework.action_model.get("scene_flow", {}) or {}
            if hasattr(_legacy_sf, "get") and _truthy(_legacy_sf.get("enabled", False)):
                raise ValueError(
                    "scene_predictor.enabled=True conflicts with legacy "
                    "action_model.scene_flow.enabled=True; enable exactly one."
                )
        elif hasattr(self.config.framework.action_model, "diffusion_model_cfg"):
            # Never build an unused coupler when the cascade is off (else strict baseline
            # loads fail on missing action_model.model.motion_coupler.* keys — review L1).
            self.config.framework.action_model.diffusion_model_cfg.motion_coupler_dim = None

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # Build the scene-flow / static-geometry predictor DiT (shares the VLM as its
        # upstream encoder via cross-attn). Its tapped hidden conditions the action
        # DiT through the zero-init motion coupler.
        self.scene_predictor = None
        if self.scene_predictor_enabled:
            from starVLA.model.modules.action_model.flow_matching_head.scene_flow_head import (
                SceneFieldMatchingHead,
            )

            sp_cfg = self.config.framework.scene_predictor
            if hasattr(sp_cfg, "diffusion_model_cfg"):
                sp_cfg.diffusion_model_cfg.cross_attention_dim = _vlm_hidden
            self.scene_predictor = SceneFieldMatchingHead(self.config, sp_cfg)
            # rollout FIFO of last-K self-predicted flow fields (past-flow ControlNet at
            # inference); persistent across calls, reset per episode via reset_past_flow().
            self._pf_fifo = []
            self._pf_K = int(getattr(self.scene_predictor, "n_past_steps", 2))
            # single-source width: the coupler K/V dim MUST equal the scene DiT hidden width
            # (review M4). Reject width-changing scene_predictor.diffusion_model_cfg overrides.
            _scene_inner = int(self.scene_predictor.model.inner_dim)
            if _scene_inner != _coupler_dim:
                raise ValueError(
                    f"scene DiT inner_dim ({_scene_inner}) != action coupler dim ({_coupler_dim}); "
                    "do not override num_attention_heads/attention_head_dim in "
                    "scene_predictor.diffusion_model_cfg (it breaks the coupler K/V width)."
                )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _collect_scene_targets(self, examples, device):
        """Stack per-sample scene-predictor GT into (B,...) tensors for the scene head.
        Returns None if NO sample has the target key (conditioning-only / step-0 smoke).
        Fixes (review 2026-07-06): per-sample all-or-none GT (no keying off examples[0]);
        normalize target HWC->BCHW; targets fp32 + masks bool INDEPENDENT of VLM dtype
        (else the fp32 loss island uses bf16-rounded targets); honor flow_has_gt."""
        if self.scene_predictor is None or not examples:
            return None
        tk = self.scene_predictor.target_key
        present = [tk in e for e in examples]
        if not any(present):
            return None
        if not all(present):
            raise RuntimeError(
                f"scene_predictor: mixed batch — {sum(present)}/{len(present)} samples have "
                f"'{tk}'. All-or-none required (mixed cotrain not wired). Fix the sampler/data_mix."
            )

        def _to_bchw(arr):
            t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)  # (H,W,3) or (3,H,W)
            if t.dim() == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
                t = t.permute(2, 0, 1)  # HWC -> CHW
            return t.contiguous()

        def _to_kchw(arr):
            t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)  # (K,H,W,3) or (K,3,H,W)
            if t.dim() == 4 and t.shape[-1] == 3 and t.shape[1] != 3:
                t = t.permute(0, 3, 1, 2)  # (K,H,W,3) -> (K,3,H,W)
            return t.contiguous()

        target = torch.stack([_to_bchw(e[tk]) for e in examples]).to(device)  # (B,3,H,W) fp32
        batch = {tk: target}
        for key in (self.scene_predictor.valid_key, self.scene_predictor.dynamic_key):
            if all(key in e for e in examples):
                batch[key] = torch.as_tensor(
                    np.stack([np.asarray(e[key]) for e in examples]), dtype=torch.bool, device=device
                )
        # per-sample GT flag (dummy no-GT samples still carry zero flow_gt in the loader)
        if all("flow_has_gt" in e for e in examples):
            batch["flow_has_gt"] = torch.as_tensor(
                [bool(e["flow_has_gt"]) for e in examples], dtype=torch.bool, device=device
            )
        # past-flow ControlNet inputs (present only when the dataloader emits them)
        pf_key = getattr(self.scene_predictor, "past_flow_key", "past_flow_gt")
        if all(pf_key in e for e in examples):
            batch[pf_key] = torch.stack([_to_kchw(e[pf_key]) for e in examples]).to(device)  # (B,K,3,H,W)
            pv_key = getattr(self.scene_predictor, "past_valid_key", "past_flow_valid")
            if all(pv_key in e for e in examples):
                batch[pv_key] = torch.as_tensor(
                    np.stack([np.asarray(e[pv_key]) for e in examples]), dtype=torch.bool, device=device
                )  # (B,K,H,W)
            hp_key = getattr(self.scene_predictor, "has_past_key", "has_past_flow")
            if all(hp_key in e for e in examples):
                batch[hp_key] = torch.as_tensor(
                    [bool(e[hp_key]) for e in examples], dtype=torch.bool, device=device
                )  # (B,)
        return batch

    def _scene_past_only_batch(self, scene_batch):
        """Keep only historical past-flow keys for action-conditioning extraction.

        The full scene batch also contains current/future supervision (`flow_gt`,
        `flow_dynamic`). Passing a past-only dict makes future-mask leakage impossible
        by construction when optional token gating is enabled.
        """
        if self.scene_predictor is None or scene_batch is None:
            return None
        keys = (
            getattr(self.scene_predictor, "past_flow_key", "past_flow_gt"),
            getattr(self.scene_predictor, "past_valid_key", "past_flow_valid"),
            getattr(self.scene_predictor, "has_past_key", "has_past_flow"),
        )
        out = {k: scene_batch[k] for k in keys if k in scene_batch}
        return out or None

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

        if not cam_branch_enabled and not getattr(self, "scene_predictor_enabled", False):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        provided_keys = set(state_dict.keys())

        def _is_cam_branch_key(key: str) -> bool:
            return key.startswith("stereo_cam_branch_layers_modules.")

        def _is_scene_cascade_key(key: str) -> bool:
            # Scene-flow dual-DiT cascade families that fresh-init on a plain warm-start
            # (do NOT strip them from a stage-1 checkpoint — freeze via freeze_modules).
            return (
                key.startswith("scene_predictor.")
                or key.startswith("action_model.motion_coupler.")
                or ".motion_coupler." in key
            )

        # Missing branch keys are tolerated ONLY for a true baseline checkpoint that
        # has ZERO cam_branch keys. A checkpoint with SOME branch keys is a partial/
        # truncated cam_branch checkpoint — loading it with fresh-initialized gaps
        # would silently corrupt the run. Same all-or-none rule for the scene cascade.
        provided_has_branch = any(_is_cam_branch_key(key) for key in provided_keys)
        provided_has_scene = any(_is_scene_cascade_key(key) for key in provided_keys)
        allowed_missing = set()
        if init_from_baseline and not provided_has_branch:
            allowed_missing |= {key for key in own_keys - provided_keys if _is_cam_branch_key(key)}
        if init_from_baseline and not provided_has_scene:
            allowed_missing |= {key for key in own_keys - provided_keys if _is_scene_cascade_key(key)}
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
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
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
            backbone_attention_mask_repeated = None
            if backbone_attention_mask is not None:
                backbone_attention_mask_repeated = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # Scene-flow cascade: extract the (detached, GT-free, deterministic)
            # motion conditioning hidden and repeat it to match the diffusion-step
            # expansion. fork_rng keeps the action head's RNG stream untouched.
            motion_hidden_repeated = None
            _enc_mask = (
                backbone_attention_mask.to(torch.bool) if backbone_attention_mask is not None else None
            )
            # Collect the scene batch ONCE (incl. past-flow inputs); reuse for both the
            # action-conditioning extract and the supervised flow forward below.
            _scene_batch = (
                self._collect_scene_targets(examples, last_hidden.device)
                if self.scene_predictor_enabled else None
            )
            _scene_past_batch = self._scene_past_only_batch(_scene_batch) if self.scene_predictor_enabled else None
            if self.scene_predictor_enabled:
                with torch.random.fork_rng(
                    devices=[last_hidden.device.index] if last_hidden.is_cuda else []
                ):
                    motion_hidden = self.scene_predictor.extract_conditioning_hidden(
                        last_hidden, encoder_attention_mask=_enc_mask, past_flow_batch=_scene_past_batch
                    )
                motion_hidden_repeated = motion_hidden.repeat(repeated_diffusion_steps, 1, 1)

            action_output = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=backbone_attention_mask_repeated,
                motion_hidden=motion_hidden_repeated,
            )  # (B, chunk_len, action_dim)
            if isinstance(action_output, dict):
                if "flow_pred" in action_output:
                    raise RuntimeError(
                        "action_model.scene_flow (legacy aux) is not supported together with "
                        "the scene_predictor cascade; enable exactly one."
                    )
                action_loss = action_output["action_loss"]
            else:
                action_loss = action_output

            losses = {"action_loss": action_loss}
            if self.scene_predictor_enabled:
                flow_batch = _scene_batch
                if flow_batch is not None:
                    scene_out = self.scene_predictor(
                        last_hidden, flow_batch, encoder_attention_mask=_enc_mask
                    )
                    losses["flow_loss"] = scene_out["flow_loss"]
                    for key in (
                        "flow_supervised_cells",
                        "flow_dynamic_cells",
                        "flow_static_zero_cells",
                        "flow_weighted_cells",
                        "flow_effective_cells",
                        "flow_fm_loss",
                        "flow_dynamic_loss",
                        "flow_static_zero_loss",
                        "flow_direction_loss",
                        "flow_magnitude_loss",
                    ):
                        if key in scene_out:
                            losses[key] = scene_out[key]

        return losses

    def reset_past_flow(self):
        """Clear the rollout past-flow FIFO. Call at each episode boundary (the model server holds
        ONE persistent framework object across all episodes, else past flow leaks between episodes)."""
        self._pf_fifo = []

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
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
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
            motion_hidden = None
            if self.scene_predictor_enabled:
                _pf_on = getattr(self.scene_predictor, "past_flow_enabled", False)
                # Build past-flow from the rollout FIFO (last-K self-predicted fields). Empty/
                # partial FIFO -> None -> gate off (cold start; matches training no-past contract).
                pf_batch = None
                if _pf_on and len(getattr(self, "_pf_fifo", [])) >= self._pf_K:
                    _past = torch.stack(self._pf_fifo[-self._pf_K:], dim=1)  # (B,K,3,g,g) fp32
                    pf_batch = {
                        self.scene_predictor.past_flow_key: _past,
                        self.scene_predictor.has_past_key: torch.ones(
                            _past.shape[0], dtype=torch.bool, device=_past.device
                        ),
                    }
                with torch.random.fork_rng(
                    devices=[last_hidden.device.index] if last_hidden.is_cuda else []
                ):
                    motion_hidden = self.scene_predictor.extract_conditioning_hidden(
                        last_hidden, encoder_attention_mask=backbone_attention_mask, past_flow_batch=pf_batch
                    )
                    if _pf_on:
                        # sample the CURRENT field (same past-flow) and push to the FIFO for next chunk
                        _cur = self.scene_predictor.sample_field(
                            last_hidden, encoder_attention_mask=backbone_attention_mask, past_flow_batch=pf_batch
                        )  # (B,3,g,g) fp32
                if _pf_on:
                    if not hasattr(self, "_pf_fifo"):
                        self._pf_fifo = []
                    self._pf_fifo.append(_cur.detach().float())
                    if len(self._pf_fifo) > self._pf_K:
                        self._pf_fifo.pop(0)
            pred_actions = self.action_model.predict_action(
                last_hidden, state, encoder_attention_mask=backbone_attention_mask,
                motion_hidden=motion_hidden,
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
