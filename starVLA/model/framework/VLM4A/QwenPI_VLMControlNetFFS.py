# QwenPI + true VLM ControlNet FFS stereo injection.
#
# This framework freezes the Qwen VLM and injects the frozen FFS gru_hidden map
# through a trainable six-layer ControlNet branch copied from the cam_rope
# softmax decoder layers. Per-depth zero-init linears make step-0 behavior match
# the warm-started frozen VLM exactly.

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import os
import sys

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
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

logger = logging.getLogger(__name__)


@dataclass
class QwenPIVLMControlNetFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPIVLMControlNetFFS"
    ffs_vlm_controlnet: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_scale": 0,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
            "num_cameras": 2,
            "hint_hidden_dim": 256,
        }
    )


_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPIVLMControlNetFFS")
class QwenPI_VLMControlNetFFS(QwenPI):
    """QwenPI with a true VLM ControlNet branch for frozen FFS gru_hidden."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIVLMControlNetFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_vlm_controlnet", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_scale = int(ffs_cfg.get("ffs_scale", 0))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))
        self.num_cameras = int(ffs_cfg.get("num_cameras", 2))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "gru_hidden"))
        if self.ffs_feature_source != "gru_hidden":
            raise ValueError(
                f"[VLMControlNet-FFS] only ffs_feature_source='gru_hidden' is supported; "
                f"got {self.ffs_feature_source!r}"
            )
        if not (0 <= self.primary_idx < self.num_cameras):
            raise ValueError(
                f"[VLMControlNet-FFS] primary_idx={self.primary_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if not (0 <= self.right_view_idx < self.num_cameras):
            raise ValueError(
                f"[VLMControlNet-FFS] right_view_idx={self.right_view_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if self.primary_idx == self.right_view_idx:
            raise ValueError("[VLMControlNet-FFS] primary_idx and right_view_idx must differ")
        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")

        expected_sha256 = ffs_cfg.get("ffs_expected_sha256")
        if expected_sha256:
            import hashlib
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"[VLMControlNet-FFS] SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info(f"[VLMControlNet-FFS] FFS SHA256 verified ({expected_sha256[:12]})")
        else:
            logger.warning(
                "[VLMControlNet-FFS] no ffs_expected_sha256 set; pickle load uses weights_only=False"
            )

        _ffs_register_fixup()
        logger.info(f"[VLMControlNet-FFS] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for param in self.ffs.parameters():
            param.requires_grad = False

        self.ffs_feat_dim = int(ffs_cfg.get("gru_hidden_dim", 16))
        try:
            self.ffs.args.mixed_precision = False
        except Exception:
            self.ffs.args["mixed_precision"] = False
        self._ffs_captured_net0 = None

        def _capture_net0_hook(_module, _inputs, output):
            self._ffs_captured_net0 = output[0][0]

        self.ffs.update_block.register_forward_hook(_capture_net0_hook)
        logger.info(
            f"[VLMControlNet-FFS] feature_source=gru_hidden net0_dim={self.ffs_feat_dim}, "
            f"valid_iters={int(self.ffs.args.valid_iters)}"
        )

        self._vlm_controlnet_inject_depths = self._discover_cam_rope_depths()
        if len(self._vlm_controlnet_inject_depths) != 6:
            raise RuntimeError(
                f"[VLMControlNet-FFS] expected 6 cam_rope softmax layers for Qwen3.5-0.8B, "
                f"got {len(self._vlm_controlnet_inject_depths)} at {self._vlm_controlnet_inject_depths}"
            )
        trunk_layers = self._vlm_controlnet_trunk_layers()
        trunk_scls = self._vlm_controlnet_trunk_scls()
        self._assert_softmax_layers(trunk_layers)

        llm_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        input_embeddings = self.qwen_vl_interface.model.get_input_embeddings()
        base_dtype = input_embeddings.weight.dtype
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
            primary_cam_id=self.primary_idx,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
        )
        branch_params = sum(p.numel() for p in self.ffs_controlnet_branch.parameters() if p.requires_grad)
        hint_params = sum(p.numel() for p in self.ffs_controlnet_hint.parameters() if p.requires_grad)
        logger.info(
            f"[VLMControlNet-FFS] installed true VLM ControlNet; depths={self._vlm_controlnet_inject_depths}, "
            f"trainable_branch_params={branch_params}, trainable_hint_params={hint_params}, "
            "zero_convs all-zero confirmed"
        )

    def _language_model_layers(self):
        inner = getattr(self.qwen_vl_interface.model, "model", None) or self.qwen_vl_interface.model
        lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
        if lm is None or not hasattr(lm, "layers"):
            raise RuntimeError("[VLMControlNet-FFS] could not locate language_model.layers")
        return lm.layers

    def _discover_cam_rope_depths(self) -> List[int]:
        depths: List[int] = []
        for idx, layer in enumerate(self._language_model_layers()):
            # Qwen3.5-0.8B interleaves Qwen3_5Attention (softmax, cam_rope-patched)
            # with Qwen3_5GatedDeltaNet layers whose mixer is NOT under
            # self_attn/attention/attn — skip those gracefully (mirrors cam_rope's
            # own dispatch which only patches the softmax layers).
            attn = self._get_layer_attn(layer, required=False)
            if attn is not None and hasattr(attn, "stereo_cam_layer"):
                depths.append(idx)
        return depths

    @staticmethod
    def _get_layer_attn(layer: nn.Module, required: bool = True):
        for child_name in ("self_attn", "attention", "attn"):
            if hasattr(layer, child_name):
                return getattr(layer, child_name)
        if required:
            raise RuntimeError(f"[VLMControlNet-FFS] no attention child on {type(layer).__name__}")
        return None

    def _vlm_controlnet_trunk_layers(self) -> List[nn.Module]:
        layers = self._language_model_layers()
        return [layers[idx] for idx in self._vlm_controlnet_inject_depths]

    def _vlm_controlnet_trunk_scls(self) -> List[nn.Module]:
        out = []
        for layer in self._vlm_controlnet_trunk_layers():
            attn = self._get_layer_attn(layer)
            scl = getattr(attn, "stereo_cam_layer", None)
            if scl is None:
                raise RuntimeError("[VLMControlNet-FFS] trunk cam_rope layer missing stereo_cam_layer")
            out.append(scl)
        return out

    def _assert_softmax_layers(self, layers: List[nn.Module]) -> None:
        bad = []
        for depth, layer in zip(self._vlm_controlnet_inject_depths, layers):
            attn_name = type(self._get_layer_attn(layer)).__name__
            if attn_name != "Qwen3_5Attention":
                bad.append((depth, attn_name))
        if bad:
            raise RuntimeError(f"[VLMControlNet-FFS] non-softmax cam_rope layers found: {bad}")

    def _imgs_to_ffs_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        device = next(self.parameters()).device
        resize = transforms.Resize(
            (self.ffs_image_size, self.ffs_image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        out = []
        for example_imgs in batch_images:
            img = example_imgs[view_idx]
            if not torch.is_tensor(img):
                img = to_tensor(img)
            img = resize(img.unsqueeze(0)).squeeze(0)
            out.append(img * 255.0)
        return torch.stack(out, dim=0).to(device).float()

    def _compute_ffs_feature(self, batch_images: List) -> torch.Tensor:
        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)
        right = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)
        B = primary.shape[0]
        stacked = torch.cat([primary, right], dim=0)
        with torch.no_grad():
            if next(self.ffs.parameters()).dtype != torch.float32:
                self.ffs.float()
            image1 = stacked[:B].float()
            image2 = stacked[B:].float()
            self._ffs_captured_net0 = None
            with torch.amp.autocast("cuda", enabled=False):
                self.ffs(image1, image2, iters=int(self.ffs.args.valid_iters), test_mode=True)
            if self._ffs_captured_net0 is None:
                raise RuntimeError(
                    "[VLMControlNet-FFS] gru_hidden: update_block hook did not fire; FFS forward changed?"
                )
            ffs_feat = self._ffs_captured_net0
            if ffs_feat.shape[1] != self.ffs_feat_dim:
                raise RuntimeError(
                    f"[VLMControlNet-FFS] net[0] channels {ffs_feat.shape[1]} != configured "
                    f"gru_hidden_dim {self.ffs_feat_dim}"
                )
        return ffs_feat.detach()

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        # Clear stale state from the previous microbatch before computing this one.
        # Do not clear immediately after the VLM forward: Qwen gradient checkpointing
        # can rerun decoder layer hooks during backward, and those recomputes still
        # need the FFS feature/control hidden/injection state.
        clear_state()
        ffs_feat = self._compute_ffs_feature(batch_images)
        set_ffs_feature(ffs_feat)
        vl_embs_list = super()._encode_vl_hidden_states(batch_images, instructions)
        return vl_embs_list

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        own_keys = set(self.state_dict().keys())
        provided = set(state_dict.keys())

        def _is_branch_key(k: str) -> bool:
            return (
                k.startswith("ffs.")
                or k.startswith("ffs_controlnet_hint.")
                or k.startswith("ffs_controlnet_branch.")
            )

        def _is_controlnet_param_key(k: str) -> bool:
            return k.startswith("ffs_controlnet_hint.") or k.startswith("ffs_controlnet_branch.")

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    "qwen_vl_interface.model.model.language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        expected_controlnet_keys = {k for k in own_keys if _is_controlnet_param_key(k)}
        provided_controlnet_keys = {k for k in provided if _is_controlnet_param_key(k)}
        branch_keys_absent = not provided_controlnet_keys
        if provided_controlnet_keys and provided_controlnet_keys != expected_controlnet_keys:
            missing = sorted(expected_controlnet_keys - provided_controlnet_keys)[:20]
            extra = sorted(provided_controlnet_keys - expected_controlnet_keys)[:20]
            raise RuntimeError(
                "[VLMControlNet-FFS audit] partial ControlNet checkpoint detected; "
                f"missing_first={missing}, unexpected_controlnet_first={extra}. "
                "Provide all ffs_controlnet_hint/branch keys or none so warm-start copy-init is unambiguous."
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
                f"[VLMControlNet-FFS audit] ckpt missing {len(allowed_missing)} optional keys "
                "(ffs / vlm_controlnet / stereo_cam_rope)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if init_from_baseline:
                raise RuntimeError(
                    f"[VLMControlNet-FFS audit] warm-start ckpt architecture mismatch: "
                    f"these non-ControlNet params would stay randomly initialized; "
                    f"align the launcher config (e.g. interleave_self_attention) with the checkpoint. "
                    f"missing {len(suspicious_missing)} keys; first: {sample}"
                )
            if strict:
                raise RuntimeError(
                    f"[VLMControlNet-FFS audit] {len(suspicious_missing)} unexpected missing keys "
                    f"under strict=True; first: {sample}"
                )
            logger.warning(
                f"[VLMControlNet-FFS audit] {len(suspicious_missing)} missing keys "
                f"(strict=False, accepting); first: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[VLMControlNet-FFS audit] {len(unexpected)} UNEXPECTED keys in ckpt not present "
                f"in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " -- refusing silent drop.")
            logger.warning(msg + " -- dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        forwarded_strict = strict and not allowed_missing
        result = super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)

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
            logger.info(
                "[VLMControlNet-FFS audit] post-load branch copy-init passed: "
                "trunk equality, distinct storage, cam_rope copied/frozen, zero-convs zero"
            )
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
            raise RuntimeError("[VLMControlNet-FFS audit] fresh branch round-trip key mismatch")
        for key, value in ref.items():
            if not torch.equal(value.detach().cpu(), got[key].detach().cpu()):
                raise RuntimeError(f"[VLMControlNet-FFS audit] fresh branch mismatch at {key}")
