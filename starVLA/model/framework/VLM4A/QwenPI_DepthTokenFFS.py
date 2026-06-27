# QwenPI + Fast-FoundationStereo depth-token sequence insertion.
#
# This variant projects the frozen FFS net[0] stereo feature into 16 learned
# depth tokens and inserts them into the Qwen VLM input sequence immediately
# after the primary-view image-token run. The VLM internally sees S+16 tokens;
# before returning hidden states to the action head, the inserted depth-token
# output states are dropped so downstream action conditioning keeps the baseline
# image/text sequence length.

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import os
import sys

import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.stereo.depth_token_inject import (
    DepthTokenProjector,
    clear_state as clear_depth_state,
    get_state as get_depth_state,
    install_depth_token_hooks,
    set_depth_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class QwenPIDepthTokenFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPIDepthTokenFFS"
    ffs_depth_token: dict = field(
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
            "num_depth_tokens": 16,
            "pool_hw": 4,
        }
    )


# Bootstrap Fast-FoundationStereo import path (same as the other FFS frameworks).
_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPIDepthTokenFFS")
class QwenPI_DepthTokenFFS(QwenPI):
    """QwenPI + FFS net[0] inserted as depth tokens inside the VLM sequence."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIDepthTokenFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_depth_token", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_scale = int(ffs_cfg.get("ffs_scale", 0))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "gru_hidden"))
        self.num_cameras = int(ffs_cfg.get("num_cameras", 2))
        self.num_depth_tokens = int(ffs_cfg.get("num_depth_tokens", 16))
        self.depth_pool_hw = int(ffs_cfg.get("pool_hw", 4))
        if not (0 <= self.primary_idx < self.num_cameras):
            raise ValueError(
                f"[DepthToken-FFS] primary_idx={self.primary_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if not (0 <= self.right_view_idx < self.num_cameras):
            raise ValueError(
                f"[DepthToken-FFS] right_view_idx={self.right_view_idx} out of range "
                f"for num_cameras={self.num_cameras}"
            )
        if self.primary_idx == self.right_view_idx:
            raise ValueError(
                f"[DepthToken-FFS] primary_idx and right_view_idx must differ; "
                f"got {self.primary_idx}"
            )
        if self.ffs_feature_source != "gru_hidden":
            raise ValueError(
                f"[DepthToken-FFS] only ffs_feature_source='gru_hidden' is supported; "
                f"got {self.ffs_feature_source!r}"
            )

        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")

        # SHA256 guard (pickle safety - same as the sibling frameworks).
        expected_sha256 = ffs_cfg.get("ffs_expected_sha256")
        if expected_sha256:
            import hashlib
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"[DepthToken-FFS] SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info(f"[DepthToken-FFS] FFS SHA256 verified ({expected_sha256[:12]})")
        else:
            logger.warning(
                "[DepthToken-FFS] no ffs_expected_sha256 set; pickle load uses weights_only=False"
            )

        _ffs_register_fixup()
        logger.info(f"[DepthToken-FFS] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for p in self.ffs.parameters():
            p.requires_grad = False

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
            f"[DepthToken-FFS] feature_source=gru_hidden net0_dim={self.ffs_feat_dim}, "
            f"valid_iters={int(self.ffs.args.valid_iters)}"
        )

        llm_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        self.depth_token_projector = DepthTokenProjector(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            num_tokens=self.num_depth_tokens,
            pool_hw=self.depth_pool_hw,
        )

        hf_model = self.qwen_vl_interface.model
        inner = getattr(hf_model, "model", None) or hf_model
        lm = getattr(inner, "language_model", None)
        if lm is None or not hasattr(lm, "layers"):
            raise RuntimeError("[DepthToken-FFS] could not locate language_model.layers")
        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._depth_token_hook_handles = install_depth_token_hooks(
            hf_model=hf_model,
            lm=lm,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.primary_idx,
            cam_rope_state=getattr(self, "_stereo_cam_rope_state", None),
            image_token_id=int(hf_model.config.image_token_id),
        )
        logger.info(
            f"[DepthToken-FFS] depth_token_projector built (in_ch={self.ffs_feat_dim} "
            f"-> llm_dim={llm_dim}, num_depth_tokens={self.num_depth_tokens}); "
            "depth-token sequence-insertion active."
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
                        "[DepthToken-FFS] gru_hidden: update_block hook did not fire - FFS forward changed?"
                    )
                ffs_feat = self._ffs_captured_net0  # (B, C_net0, H/4, W/4)
                if ffs_feat.shape[1] != self.ffs_feat_dim:
                    raise RuntimeError(
                        f"[DepthToken-FFS] net[0] channels {ffs_feat.shape[1]} != configured "
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
        clear_depth_state()
        net0 = self._compute_ffs_feature(batch_images)
        depth_tokens = self.depth_token_projector(net0)
        set_depth_tokens(depth_tokens)

        vl_embs_list = super()._encode_vl_hidden_states(batch_images, instructions)

        state = get_depth_state()
        keep_mask = state.keep_mask
        if keep_mask is None:
            raise RuntimeError(
                "[DepthToken-FFS] depth-token hook did not produce keep_mask; "
                "language_model pre-hook may not have run"
            )
        false_counts = (~keep_mask).sum(dim=1)
        expected = torch.full_like(false_counts, self.num_depth_tokens)
        if not torch.equal(false_counts, expected):
            raise RuntimeError(
                f"[DepthToken-FFS] expected {self.num_depth_tokens} inserted tokens per sample, "
                f"got {false_counts.detach().cpu().tolist()}"
            )
        batch_size = int(keep_mask.shape[0])
        kept_len = int(keep_mask.shape[1] - self.num_depth_tokens)
        out: List[torch.Tensor] = []
        for h in vl_embs_list:
            if h.shape[0] != batch_size or h.shape[1] != keep_mask.shape[1]:
                raise RuntimeError(
                    f"[DepthToken-FFS] hidden state shape {tuple(h.shape)} does not match "
                    f"expanded keep_mask {tuple(keep_mask.shape)}"
                )
            mask = keep_mask.to(device=h.device)
            out.append(h[mask].view(batch_size, kept_len, h.shape[-1]))
        return out

    # --- init ckpt audit --------------------------------------------------- #

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        own_keys = set(self.state_dict().keys())
        provided = set(state_dict.keys())

        def _is_branch_key(k: str) -> bool:
            return k.startswith("ffs.") or k.startswith("depth_token_projector.")

        def _is_depth_projector_key(k: str) -> bool:
            return k.startswith("depth_token_projector.")

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    "qwen_vl_interface.model.model.language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        expected_depth_keys = {k for k in own_keys if _is_depth_projector_key(k)}
        provided_depth_keys = {k for k in provided if _is_depth_projector_key(k)}
        if provided_depth_keys and provided_depth_keys != expected_depth_keys:
            missing = sorted(expected_depth_keys - provided_depth_keys)[:20]
            extra = sorted(provided_depth_keys - expected_depth_keys)[:20]
            raise RuntimeError(
                "[DepthToken-FFS audit] partial depth-token projector checkpoint detected; "
                f"missing_first={missing}, unexpected_depth_first={extra}. "
                "Provide all depth_token_projector keys or none for baseline warm-start."
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
                f"[DepthToken-FFS audit] ckpt missing {len(allowed_missing)} optional-branch keys "
                "(ffs / depth_token_projector / stereo_cam_rope)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if strict:
                raise RuntimeError(
                    f"[DepthToken-FFS audit] {len(suspicious_missing)} unexpected missing keys "
                    f"under strict=True; first: {sample}"
                )
            logger.warning(
                f"[DepthToken-FFS audit] {len(suspicious_missing)} missing keys "
                f"(strict=False, accepting); first: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[DepthToken-FFS audit] {len(unexpected)} UNEXPECTED keys in ckpt not present "
                f"in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " -- refusing silent drop.")
            logger.warning(msg + " -- dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        forwarded_strict = strict and not allowed_missing
        return super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)
