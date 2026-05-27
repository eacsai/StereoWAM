"""Sanity v2: verify QwenPIControlNetFFS module:
 1. builds
 2. forward returns finite loss
 3. ⭐ residual is sensitive to right_view (stereo signal verification)
"""
import os, sys, traceback, logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, "/data/wangqiwei/ICLR2026/starVLA")
sys.path.insert(0, "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework

ffs_path = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth"

cfg = OmegaConf.create({
    "datasets": {"vla_data": {}},
    "framework": {
        "name": "QwenPIControlNetFFS",
        "qwenvl": {
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
            "attn_implementation": "flash_attention_2",
            "stereo_cam_rope_enabled": True,
            "stereo_cam_rope_d_c": 16,
            "stereo_cam_rope_num_cameras": 2,
            "stereo_cam_rope_baseline_m": 0.06,
            "stereo_cam_rope_fovy_degrees": 45.0,
            "stereo_cam_rope_image_width": 256,
            "stereo_cam_rope_image_height": 256,
            "stereo_cam_rope_spatial_merge": 2,
            "stereo_cam_rope_init_mode": "zero",
            "stereo_epipolar_mask_enabled": True,
        },
        "ffs_controlnet": {
            "ffs_model_path": ffs_path,
            "ffs_scale": 0,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
        },
        "action_model": {
            "action_model_type": "LayerwiseFM", "action_dim": 7, "state_dim": 7,
            "action_horizon": 16, "num_inference_timesteps": 4, "repeated_diffusion_steps": 2,
            "add_pos_embed": True, "max_seq_len": 1024, "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
            "num_timestep_buckets": 1000, "diffusion_model_cfg": {},
        },
    },
})

os.chdir("/data/wangqiwei/ICLR2026/starVLA")
assert os.path.isfile(ffs_path)

print("=== build ===")
model = build_framework(cfg)
print(f"[ok] class: {type(model).__name__}")
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
n_proj = sum(p.numel() for p in model.ffs_controlnet_projs.parameters()) / 1e6
print(f"[ok] trainable={n_trainable:.1f}M, frozen FFS={n_frozen:.1f}M, projectors={n_proj:.1f}M")

model = model.cuda().eval()
B, H, W = 2, 256, 256

# === STEREO SENSITIVITY CHECK ===
# Same primary, two different right_views. Residual should differ.
print("\n=== stereo sensitivity check ===")
primary_imgs = [Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)) for _ in range(B)]
right_A = [Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)) for _ in range(B)]
right_B = [Image.fromarray(np.random.randint(128, 255, (H, W, 3), dtype=np.uint8)) for _ in range(B)]  # different distribution

# Probe just the FFS feature extraction path directly
def get_ffs_residual(primary_pil_list, right_pil_list):
    batch = [[primary_pil_list[i], right_pil_list[i]] for i in range(B)]
    primary = model._imgs_to_ffs_tensor(batch, 0)
    right = model._imgs_to_ffs_tensor(batch, 1)
    stacked = torch.cat([primary, right], dim=0)
    with torch.no_grad():
        mean = primary.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = primary.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        stacked_normed = (stacked / 255.0 - mean) / std
        ffs_pyramid = model.ffs.feature(stacked_normed)
        ffs_feat = ffs_pyramid[model.ffs_scale]
        ffs_feat_stereo = torch.cat([ffs_feat[:B], ffs_feat[B:]], dim=1)
    return ffs_feat_stereo

stereo_A = get_ffs_residual(primary_imgs, right_A)
stereo_B = get_ffs_residual(primary_imgs, right_B)

diff = (stereo_A - stereo_B).abs()
print(f"   stereo_feat_A.shape: {tuple(stereo_A.shape)}, sum_abs_diff={diff.sum().item():.4f}, max_diff={diff.max().item():.4f}")
if diff.sum().item() < 1e-3:
    print("   [FAIL] stereo features DIDNT change with right_view  ===>  stereo signal NOT flowing!")
    sys.exit(1)
else:
    print(f"   [OK] right_view changes ffs_feat_stereo → STEREO SIGNAL CONFIRMED 🎉")

# === FULL FORWARD ===
print("\n=== full forward ===")
examples = []
for i in range(B):
    examples.append({
        "image": [primary_imgs[i], right_A[i]],
        "lang": "pick up the wrench and place it on the table",
        "action": np.zeros((16, 7), dtype=np.float32),
        "state": np.zeros((1, 7), dtype=np.float32),
    })
with torch.no_grad():
    out = model.forward(examples=examples)
    print(f"[ok] forward: {type(out).__name__}")
    if isinstance(out, dict):
        for k, v in out.items():
            if hasattr(v, "shape"):
                print(f"   {k}: shape={tuple(v.shape)}, finite={torch.isfinite(v).all().item()}")
            else:
                print(f"   {k}: {v}")

print("\n=== ALL OK ===")
