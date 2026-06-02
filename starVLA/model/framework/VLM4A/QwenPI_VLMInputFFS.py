# QwenPI + Fast-FoundationStereo, VLM-INPUT injection (ControlNet-style, zero-gated).
#
# Where this sits among the FFS variants:
#   QwenPI_ControlNet_FFS  : residual onto VLM *output* hidden states (action cond).
#   QwenPI_ControlVLA_FFS  : parallel K/V branch inside the action DiT cross-attn.
#   QwenPI_VLMInputFFS     : THIS — residual onto the merged inputs_embeds at the
#                            primary-view image-token positions, so the FFS stereo
#                            signal flows through the WHOLE VLM and fuses with the
#                            language instruction (cf. StereoPolicy / StereoVLA,
#                            which both add stereo at the VLM input). Differentiator
#                            = ControlNet-style zero-gated injection (step-0 byte
#                            identical, warm-start safe).
#
# Injection mechanism + the two pre-hooks live in
# starVLA/model/modules/stereo/ffs_vlm_inject.py. This framework only:
#   * loads the frozen FFS and computes net[0] (gru_hidden, the real post-cost
#     -volume disparity-aware feature) — same proven path as QwenPI_ControlVLA_FFS,
#   * owns the FFSVLMInjector params,
#   * stashes the FFS feature on the module-level state each forward.
#
# Default feature_source = gru_hidden (the real stereo feature). "backbone"
# (monocular concat[L,R]) kept only for an ablation toggle.

from dataclasses import dataclass, field
from typing import List, Optional
import os, sys, logging

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.stereo.ffs_vlm_inject import (
    FFSVLMInjector,
    install_ffs_vlm_input_hooks,
    set_ffs_feature,
    clear_ffs_state,
)

logger = logging.getLogger(__name__)


@dataclass
class QwenPIVLMInputFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPIVLMInputFFS"
    ffs_vlm_input: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_scale": 0,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
            "ffs_expected_sha256": None,
            # which FFS tensor to inject (see QwenPI_ControlVLA_FFS for the full
            # rationale). gru_hidden = post-cost-volume disparity-aware net[0].
            "ffs_feature_source": "gru_hidden",
            # net[0] channel dim for the distilled 20-30-48 FFS (empirically 16).
            "gru_hidden_dim": 16,
            # backbone-only: concat[L,R] feature pyramid level channel count is
            # derived from the model; gru_hidden ignores this.
            "inject_hidden_dim": 256,    # FFSVLMInjector bottleneck width
            "num_cameras": 2,            # primary + right
        }
    )


# Bootstrap Fast-FoundationStereo import path (same as the other FFS frameworks).
_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPIVLMInputFFS")
class QwenPIVLMInputFFS(QwenPI):
    """QwenPI + FFS stereo injected at the VLM input (zero-gated ControlNet style)."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIVLMInputFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_vlm_input", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_scale = int(ffs_cfg.get("ffs_scale", 0))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "gru_hidden"))
        self.num_cameras = int(ffs_cfg.get("num_cameras", 2))
        if not (0 <= self.primary_idx < self.num_cameras):
            raise ValueError(
                f"[VLMInput-FFS] primary_idx={self.primary_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if not (0 <= self.right_view_idx < self.num_cameras):
            raise ValueError(
                f"[VLMInput-FFS] right_view_idx={self.right_view_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if self.primary_idx == self.right_view_idx:
            raise ValueError(
                f"[VLMInput-FFS] primary_idx and right_view_idx must differ; "
                f"got {self.primary_idx}"
            )
        if self.ffs_feature_source not in ("backbone", "gru_hidden"):
            raise ValueError(
                f"[VLMInput-FFS] ffs_feature_source must be 'backbone' or 'gru_hidden', "
                f"got {self.ffs_feature_source!r}"
            )

        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")

        # SHA256 guard (pickle safety — same as the sibling frameworks).
        expected_sha256 = ffs_cfg.get("ffs_expected_sha256")
        if expected_sha256:
            import hashlib
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"[VLMInput-FFS] SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info(f"[VLMInput-FFS] FFS SHA256 verified ({expected_sha256[:12]}…)")
        else:
            logger.warning(
                f"[VLMInput-FFS] no ffs_expected_sha256 set — pickle load via weights_only=False"
            )

        _ffs_register_fixup()
        logger.info(f"[VLMInput-FFS] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for p in self.ffs.parameters():
            p.requires_grad = False

        if self.ffs_feature_source == "gru_hidden":
            # net[0] = FFS GRU hidden after cost volume + refinement, a single fused
            # disparity-aware map (B, C_net0, H/4, W/4). C_net0 is 16 for the
            # distilled 20-30-48 model (a forward-time assert validates it).
            self.ffs_feat_dim = int(ffs_cfg.get("gru_hidden_dim", 16))
            # FFS.forward wraps in fp16 autocast when args.mixed_precision; under bf16
            # training that collides with bf16 frozen params. Disable it and run FFS
            # fully fp32 (frozen + no_grad, cheap). Mirrors QwenPI_ControlVLA_FFS.
            try:
                self.ffs.args.mixed_precision = False
            except Exception:
                self.ffs.args["mixed_precision"] = False
            self._ffs_captured_net0 = None

            def _capture_net0_hook(_module, _inputs, output):
                # update_block returns (net_list, mask, delta_disp); net_list[0] is
                # the refined hidden. Fires per GRU iter -> last write = final state.
                self._ffs_captured_net0 = output[0][0]

            self.ffs.update_block.register_forward_hook(_capture_net0_hook)
            logger.info(
                f"[VLMInput-FFS] feature_source=gru_hidden net0_dim={self.ffs_feat_dim}, "
                f"valid_iters={int(self.ffs.args.valid_iters)}"
            )
        else:
            ffs_feat_dim_per_view = int(self.ffs.feature.d_out[self.ffs_scale])
            self.ffs_feat_dim = ffs_feat_dim_per_view * 2  # concat L+R
            logger.info(
                f"[VLMInput-FFS] feature_source=backbone ffs_scale={self.ffs_scale} "
                f"per_view={ffs_feat_dim_per_view} concat={self.ffs_feat_dim}"
            )

        # Build the injector (owns trainable params) + install the two pre-hooks.
        llm_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        self.ffs_vlm_injector = FFSVLMInjector(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            hidden_dim=int(ffs_cfg.get("inject_hidden_dim", 256)),
        )
        get_input_embeddings = getattr(self.qwen_vl_interface.model, "get_input_embeddings", None)
        input_embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
        if input_embeddings is not None and getattr(input_embeddings, "weight", None) is not None:
            base_dtype = input_embeddings.weight.dtype
        else:
            base_dtype = next(self.qwen_vl_interface.model.parameters()).dtype
        # Match cam_rope_hook's base_dtype cast using the input-embedding dtype;
        # bf16 ZeRO dislikes mixed-dtype freshly-created branch params next to the bf16 Qwen backbone.
        self.ffs_vlm_injector = self.ffs_vlm_injector.to(base_dtype)
        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._ffs_vlm_state = install_ffs_vlm_input_hooks(
            self.qwen_vl_interface.model,
            injector=self.ffs_vlm_injector,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.primary_idx,
        )
        logger.info(
            f"[VLMInput-FFS] injector built (in_ch={self.ffs_feat_dim} -> llm_dim={llm_dim}); "
            f"VLM-input zero-gated injection active."
        )

    # --- helpers ---------------------------------------------------------- #

    def _imgs_to_ffs_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        device = next(self.parameters()).device
        size = self.ffs_image_size
        resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)
        out = []
        for example_imgs in batch_images:
            img = example_imgs[view_idx]
            if not torch.is_tensor(img):
                img = to_tensor(img)
            img = resize(img.unsqueeze(0)).squeeze(0)
            out.append(img * 255.0)
        return torch.stack(out, dim=0).to(device).float()

    def _compute_ffs_feature(self, batch_images: List) -> torch.Tensor:
        """Frozen FFS -> (B, C_ffs, h, w) stereo feature map. No grad."""
        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)
        right = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)
        B = primary.shape[0]
        ffs_dtype = next(self.ffs.parameters()).dtype
        stacked = torch.cat([primary, right], dim=0).to(dtype=ffs_dtype)  # (2B, 3, H, W)
        with torch.no_grad():
            mean = stacked.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = stacked.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            stacked_normed = (stacked / 255.0 - mean) / std
            if self.ffs_feature_source == "gru_hidden":
                # FFS.forward normalises 0-255 input INTERNALLY -> pass RAW stacked,
                # not stacked_normed (double-normalize bug). Run fully fp32.
                if next(self.ffs.parameters()).dtype != torch.float32:
                    self.ffs.float()
                image1 = stacked[:B].float()
                image2 = stacked[B:].float()
                self._ffs_captured_net0 = None
                with torch.amp.autocast("cuda", enabled=False):
                    self.ffs(image1, image2, iters=int(self.ffs.args.valid_iters), test_mode=True)
                if self._ffs_captured_net0 is None:
                    raise RuntimeError(
                        "[VLMInput-FFS] gru_hidden: update_block hook did not fire — FFS forward changed?"
                    )
                ffs_feat = self._ffs_captured_net0  # (B, C_net0, H/4, W/4)
                if ffs_feat.shape[1] != self.ffs_feat_dim:
                    raise RuntimeError(
                        f"[VLMInput-FFS] net[0] channels {ffs_feat.shape[1]} != configured "
                        f"ffs_feat_dim {self.ffs_feat_dim}; set ffs_vlm_input.gru_hidden_dim "
                        f"={ffs_feat.shape[1]}."
                    )
            else:
                ffs_pyramid = self.ffs.feature(stacked_normed)
                feats_left = [o[:B] for o in ffs_pyramid]
                feats_right = [o[B:] for o in ffs_pyramid]
                ffs_feat = torch.cat(
                    [feats_left[self.ffs_scale], feats_right[self.ffs_scale]], dim=1
                )  # (B, 2C, h, w)
        return ffs_feat.detach()

    # --- override parent encode ------------------------------------------- #

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> List[torch.Tensor]:
        clear_ffs_state()  # fresh per forward (paired with set below)
        # 1) Compute FFS feature and stash it; the inner LM pre-hook reads it
        #    DURING the VLM forward below (the outer hook fills cam-id + grid).
        ffs_feat = self._compute_ffs_feature(batch_images)
        set_ffs_feature(ffs_feat)
        # 2) Run the normal VLM forward; the two pre-hooks inject the zero-gated
        #    residual onto the primary-view image tokens of the merged inputs_embeds.
        vl_embs_list = super()._encode_vl_hidden_states(batch_images, instructions)
        clear_ffs_state()
        # 3) vl_embs returned UNCHANGED here — the injection already happened
        #    inside the VLM (it flowed through every VLM layer).
        return vl_embs_list

    # --- init ckpt audit (matches the sibling frameworks' pattern) -------- #

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        own_keys = set(self.state_dict().keys())
        provided = set(state_dict.keys())

        def _is_branch_key(k: str) -> bool:
            return k.startswith("ffs.") or k.startswith("ffs_vlm_injector.")

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    "qwen_vl_interface.model.model.language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        if init_from_baseline:
            allowed_missing = {
                k for k in own_keys - provided
                if _is_branch_key(k) or _is_stereo_cam_rope_key(k)
            }
        else:
            allowed_missing = set()

        suspicious_missing = (own_keys - provided) - allowed_missing
        unexpected = provided - own_keys

        if allowed_missing:
            logger.info(
                f"[VLMInput-FFS audit] ckpt missing {len(allowed_missing)} optional-branch keys "
                f"(ffs / ffs_vlm_injector / stereo_cam_rope)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if strict:
                raise RuntimeError(
                    f"[VLMInput-FFS audit] {len(suspicious_missing)} unexpected missing keys "
                    f"under strict=True; first: {sample}"
                )
            logger.warning(
                f"[VLMInput-FFS audit] {len(suspicious_missing)} missing keys "
                f"(strict=False, accepting); first: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[VLMInput-FFS audit] {len(unexpected)} UNEXPECTED keys in ckpt not present "
                f"in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " — refusing silent drop.")
            logger.warning(msg + " — dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        forwarded_strict = strict and not allowed_missing
        return super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)
