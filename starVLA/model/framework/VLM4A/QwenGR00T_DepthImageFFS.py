from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TFFSBase
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


_DEFAULT_DEPTH_PROMPT = "Below is the stereo disparity (depth) map of the left view:"

def _build_turbo_lut() -> np.ndarray:
    """256-entry turbo RGB LUT via the Anton Mikhailov polynomial approximation of
    Google's turbo colormap. Pure-numpy (no matplotlib / no optional top-level import)
    so the framework stays safe to auto-import via FRAMEWORK_REGISTRY, and the table is
    generated (never miscopied) so it always has exactly 256 entries."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    lut = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    return (lut * 255.0).round().astype(np.uint8)


_TURBO_LUT = _build_turbo_lut()  # (256, 3) uint8


def _pil_bilinear_resample():
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _coerce_image_size(value, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, int):
        return (int(value), int(value))
    try:
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    except TypeError:
        pass
    raise ValueError(f"obs_image_size must be an int or length-2 sequence, got {value!r}")


def _read_obs_image_size(config, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        vla_data = config.datasets.vla_data
        value = vla_data.get("obs_image_size", None)
    except Exception:
        value = None
    return _coerce_image_size(value, fallback)


def _set_depthimage_cam_rope_flag(config):
    """Set the extra-image cam_rope flag before QwenGR00T installs hooks."""

    cfg = OmegaConf.create({}) if config is None else config
    target = getattr(cfg, "_cfg", cfg)
    inject_cam_id = 1
    if isinstance(target, str):
        target = OmegaConf.load(target)
        cfg = target

    if OmegaConf.is_config(target):
        ffs_cfg = OmegaConf.select(target, "framework.ffs_depth_image")
        if ffs_cfg is not None:
            inject_cam_id = int(ffs_cfg.get("inject_cam_id", inject_cam_id))
        OmegaConf.update(
            target,
            "framework.qwenvl.stereo_extra_image_cam_id",
            inject_cam_id,
            force_add=True,
        )
    elif isinstance(target, dict):
        framework = target.setdefault("framework", {})
        ffs_cfg = framework.get("ffs_depth_image", {})
        inject_cam_id = int(ffs_cfg.get("inject_cam_id", inject_cam_id))
        framework.setdefault("qwenvl", {})["stereo_extra_image_cam_id"] = inject_cam_id
    else:
        raise TypeError(f"Unsupported config type for QwenGR00T_DepthImageFFS: {type(config).__name__}")
    return cfg


def render_disp_tensor_as_turbo_pils(
    disp_up: torch.Tensor,
    image_size: Tuple[int, int],
) -> List[Image.Image]:
    """Render FFS pixel disparity as per-sample normalized turbo RGB PIL images."""

    if not torch.is_tensor(disp_up):
        raise TypeError(f"disp_up must be a torch.Tensor, got {type(disp_up).__name__}")
    if disp_up.ndim == 3:
        disp_up = disp_up.unsqueeze(1)
    if disp_up.ndim != 4 or int(disp_up.shape[1]) != 1:
        raise ValueError(f"disp_up must have shape (B, 1, H, W), got {tuple(disp_up.shape)}")

    disp = disp_up.detach().float()
    # Sanitize (codex Stage-4 MEDIUM): FFS should never emit NaN/Inf, but a bad input must
    # not poison the per-sample min/max or the LUT index. Fail closed if NO finite pixels.
    if not torch.isfinite(disp).all():
        if not torch.isfinite(disp).any():
            raise RuntimeError("[GR00T-DepthImage-FFS] FFS disparity has no finite pixels")
        disp = torch.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    batch = int(disp.shape[0])
    flat = disp.reshape(batch, -1)
    d_min = flat.min(dim=1).values.view(batch, 1, 1, 1)
    d_max = flat.max(dim=1).values.view(batch, 1, 1, 1)
    norm = ((disp - d_min) / (d_max - d_min + 1e-6)).clamp_(0.0, 1.0)
    lut_idx = (norm[:, 0] * 255.0).round().to(torch.long).cpu().numpy()
    rgb = _TURBO_LUT[lut_idx]

    resample = _pil_bilinear_resample()
    return [
        Image.fromarray(rgb[b], mode="RGB").resize(tuple(image_size), resample=resample)
        for b in range(batch)
    ]


@dataclass
class QwenGR00TDepthImageFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_DepthImageFFS"
    ffs_depth_image: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "num_cameras": 2,
            "left_ref_idx": 1,
            "primary_view_idx": 0,
            "inject_cam_id": 1,
            "depth_prompt": _DEFAULT_DEPTH_PROMPT,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_DepthImageFFS")
class QwenGR00T_DepthImageFFS(QwenGR00TFFSBase):
    """GR00T + FFS #4: feed rendered FFS disparity as a third VLM image."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = _set_depthimage_cam_rope_flag(config)
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TDepthImageFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_depth_image", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-DepthImage-FFS]")
        self.depth_prompt = str(ffs_cfg.get("depth_prompt", _DEFAULT_DEPTH_PROMPT))
        self.depth_image_size = _read_obs_image_size(
            self.config,
            fallback=(self.ffs_image_size, self.ffs_image_size),
        )

        extra_cam_id = self.config.framework.qwenvl.get("stereo_extra_image_cam_id", None)
        if bool(self.config.framework.qwenvl.get("stereo_cam_rope_enabled", False)):
            if extra_cam_id is None or int(extra_cam_id) != int(self.inject_cam_id):
                raise RuntimeError(
                    "[GR00T-DepthImage-FFS] framework.qwenvl.stereo_extra_image_cam_id "
                    f"must be set to inject_cam_id={self.inject_cam_id} before QwenGR00T.__init__ "
                    f"installs cam_rope hooks; got {extra_cam_id}"
                )

    def _compute_ffs_disp_image(self, batch_images) -> List[Image.Image]:
        disp_up = self._compute_ffs_disparity(batch_images)
        return render_disp_tensor_as_turbo_pils(disp_up, self.depth_image_size)

    def _build_depthimage_qwenvl_inputs(self, batch_images, instructions, depth_pils: List[Image.Image]):
        assert len(batch_images) == len(instructions), "Images and instructions must have the same length"
        assert len(depth_pils) == len(batch_images), "Depth image count must match batch size"
        messages = []
        for imgs, instruction, depth_pil in zip(batch_images, instructions, depth_pils):
            if len(imgs) <= max(self.primary_view_idx, self.left_ref_idx):
                raise ValueError(
                    "[GR00T-DepthImage-FFS] expected primary,left_view stereo images with "
                    f"at least {max(self.primary_view_idx, self.left_ref_idx) + 1} views, got {len(imgs)}"
                )

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                task_prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                task_prompt = instruction

            content = [
                {"type": "image", "image": imgs[self.primary_view_idx]},
                {"type": "image", "image": imgs[self.left_ref_idx]},
                {"type": "text", "text": self.depth_prompt},
                {"type": "image", "image": depth_pil},
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

    def _encode_last_hidden_with_ffs(self, batch_images, instructions) -> torch.Tensor:
        depth_pils = self._compute_ffs_disp_image(batch_images)
        qwen_inputs = self._build_depthimage_qwenvl_inputs(
            batch_images=batch_images,
            instructions=instructions,
            depth_pils=depth_pils,
        )
        return self._run_qwenvl_forward(qwen_inputs)

    def _ffs_key_prefixes(self):
        return ("ffs.",)
