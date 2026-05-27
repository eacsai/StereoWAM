# QwenPI + Fast-FoundationStereo ControlNet (v2: real stereo)
# Stacks on top of cam_rope + epipolar mask. Zero-init residual injection.
# v2 fixes (from codex review 2026-05-27):
#   - FIX #1 (HIGH): concat[L,R] features so projector sees BOTH views (was monocular L-only)
#   - FIX #3 (MEDIUM): explicit init-ckpt key audit in load_state_dict

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os, sys, logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class QwenPIControlNetFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPIControlNetFFS"
    ffs_controlnet: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_scale": 0,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
        }
    )


class FFSControlNetProj(nn.Module):
    """Per-DiT-layer projector: FFS stereo spatial feature -> Qwen-shaped residual.
    Zero-init final linear so step-0 output = 0 -> baseline preserved.
    """
    def __init__(self, in_ch: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )
        self.zero_proj = nn.Linear(out_dim, out_dim, bias=True)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, ffs_feat: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        # ffs_feat: [B, C_concat, H, W]  ->  [B, T, D]
        x = self.spatial_proj(ffs_feat)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        if x.shape[1] != target_seq_len:
            x_lin = x.transpose(1, 2).unsqueeze(-1)  # [B, D, T_src, 1]
            x_lin = F.interpolate(x_lin, size=(target_seq_len, 1), mode="bilinear", align_corners=False)
            x = x_lin.squeeze(-1).transpose(1, 2)
        x = self.zero_proj(x)  # ===> 0 at step 0
        return x


# === Bootstrap: add Fast-FoundationStereo to sys.path so we can import its modules ===
_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPIControlNetFFS")
class QwenPIControlNetFFS(QwenPI):
    """QwenPI + Fast-FoundationStereo (real stereo) ControlNet branch.

    Injects a zero-init residual derived from FFS concat[left, right] features into
    each VL hidden_state that the LayerwiseFM DiT consumes.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIControlNetFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_controlnet", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_scale = int(ffs_cfg.get("ffs_scale", 0))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))

        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")
        # SECURITY (codex round-2 HIGH#2): torch.load(weights_only=False) executes
        # pickle. We only intend to load NVIDIA's official FFS release. Set
        # ffs_controlnet.ffs_expected_sha256 in config to refuse any swapped artifact.
        expected_sha256 = ffs_cfg.get("ffs_expected_sha256", None)
        if expected_sha256:
            import hashlib
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"[FFS-CN security] ffs_model_path SHA256 mismatch:\n"
                    f"  expected: {expected_sha256}\n"
                    f"  actual:   {actual}\n"
                    f"  refusing pickle load of unverified artifact at {ffs_model_path}"
                )
            logger.info(f"[FFS-CN security] SHA256 verified ({expected_sha256[:12]}…)")
        else:
            logger.warning(
                f"[FFS-CN security] no ffs_controlnet.ffs_expected_sha256 in config; "
                f"loading {ffs_model_path} via weights_only=False (executes pickle). "
                f"Set ffs_expected_sha256 to enforce artifact integrity."
            )
        _ffs_register_fixup()
        logger.info(f"[FFS-ControlNet] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for p in self.ffs.parameters():
            p.requires_grad = False

        # FFS feature channels at chosen scale (per view) * 2 because we concat L+R
        ffs_feat_dim_per_view = int(self.ffs.feature.d_out[self.ffs_scale])
        ffs_feat_dim = ffs_feat_dim_per_view * 2
        llm_hidden_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        n_dit_layers = len(self.action_model.model.transformer_blocks)
        logger.info(
            f"[FFS-ControlNet v2 stereo] ffs_scale={self.ffs_scale} per_view={ffs_feat_dim_per_view} "
            f"concat={ffs_feat_dim} -> llm_hidden_dim={llm_hidden_dim}, n_dit_layers={n_dit_layers}"
        )

        self.ffs_controlnet_projs = nn.ModuleList([
            FFSControlNetProj(in_ch=ffs_feat_dim, out_dim=llm_hidden_dim)
            for _ in range(n_dit_layers)
        ])

    # --- helpers ---

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

    # --- override parent encode ---

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> List[torch.Tensor]:
        # 1) Base Qwen forward (cam_rope + epipolar baked in by parent hooks).
        vl_embs_list = super()._encode_vl_hidden_states(batch_images, instructions)

        # 2) Stereo pair -> FFS feature pyramid for BOTH views.
        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)
        right   = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)
        B = primary.shape[0]
        # Dtype-aware: training keeps FFS+projector fp32; eval server --use_bf16 casts to bf16.
        # Match input dtype to whatever the submodule currently holds.
        ffs_dtype = next(self.ffs.parameters()).dtype
        stacked = torch.cat([primary, right], dim=0).to(dtype=ffs_dtype)            # [2B, 3, H, W]
        with torch.no_grad():
            mean = stacked.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = stacked.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            stacked_normed = (stacked / 255.0 - mean) / std
            ffs_pyramid = self.ffs.feature(stacked_normed)
            ffs_feat = ffs_pyramid[self.ffs_scale]              # [2B, C, h, w]
            ffs_feat_left  = ffs_feat[:B]
            ffs_feat_right = ffs_feat[B:]
            # === v2 FIX: concat both views so projector receives STEREO signal ===
            ffs_feat_stereo = torch.cat([ffs_feat_left, ffs_feat_right], dim=1).detach()  # [B, 2C, h, w]

        # 3) Inject zero-init residual into each layer the DiT will read.
        for i, h in enumerate(vl_embs_list):
            T = int(h.shape[1])
            proj_dtype = next(self.ffs_controlnet_projs[i].parameters()).dtype
            residual = self.ffs_controlnet_projs[i](
                ffs_feat_stereo.to(dtype=proj_dtype), T
            )
            vl_embs_list[i] = h + residual.to(h.dtype)

        return vl_embs_list

    # --- init ckpt audit (codex finding #3) ---

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        """Audited permissive load (codex round-2 HIGH#1 fix).

        Permissiveness is gated on init_from_baseline=True. When True we allow
        missing ffs.* / ffs_controlnet_projs.* (loaded separately / zero-init) and
        missing stereo_cam_rope_* (inherited from parent's permissive logic).
        When False we honor strict=True for every key.

        We also audit UNEXPECTED keys (ckpt has keys current model doesn't): drop
        them loudly under strict=False, raise under strict=True.
        """
        own_keys = set(self.state_dict().keys())
        provided_keys = set(state_dict.keys())

        def _is_ffs_key(k: str) -> bool:
            return k.startswith("ffs.") or k.startswith("ffs_controlnet_projs.")

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
                k for k in own_keys - provided_keys
                if _is_ffs_key(k) or _is_stereo_cam_rope_key(k)
            }
        else:
            allowed_missing = set()

        suspicious_missing = (own_keys - provided_keys) - allowed_missing
        unexpected = provided_keys - own_keys

        if allowed_missing:
            logger.info(
                f"[FFS-CN audit] ckpt missing {len(allowed_missing)} expected optional-branch keys "
                f"(ffs.* / ffs_controlnet_projs.* / stereo_cam_rope.*)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if strict:
                raise RuntimeError(
                    f"[FFS-CN audit] {len(suspicious_missing)} unexpected missing keys with strict=True; "
                    f"refusing silent load. First few: {sample}"
                )
            logger.warning(
                f"[FFS-CN audit] {len(suspicious_missing)} missing keys (strict=False, accepting); "
                f"first few: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[FFS-CN audit] {len(unexpected)} UNEXPECTED keys in ckpt not present in current model; "
                f"first few: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " — refusing silent drop (strict=True).")
            logger.warning(msg + " — dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        # Honor caller's strict request. Drop to strict=False only when we have
        # known allowed_missing branches (else parent would re-raise on those).
        forwarded_strict = strict and not allowed_missing
        return super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)
