#!/usr/bin/env python3
"""Verify the FFS-order bug fix (un-rotate + flip-back) in QwenGR00T_FFSCommon.py.

The fix (2026-06-25) replaced the OLD buggy call
    disp = ffs(primary, right, test_mode=True)
with the EXACT logic
    disp = flip( ffs(flip(primary), flip(right), test_mode=True) )
so that on the rot180-stored LIBERO stereo frames `right_view` becomes the
geometric LEFT eye of FoundationStereo (which searches NON-NEGATIVE disparity
only), and the disparity/net0 output is rotated back into the stored primary
token grid the residual injects onto.

This script reuses the KNOWN-WORKING FFS load + preprocessing from
scripts/tools/ffs_order_net0_check.py (torch.load weights_only=False, eval,
freeze, mixed_precision=False, valid_iters, test_mode=True; ToTensor+Resize256+*255)
and the camera intrinsics K from scripts/tools/stereo_pointcloud_viz.py
(fovy=45, 256x256, baseline 0.06) to numerically confirm correct geometry.
"""
import os
import sys
import numpy as np
import torch

FFS_REPO = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo"
FFS_WEIGHTS = os.path.join(FFS_REPO, "weights/20-30-48/model_best_bp2_serialize.pth")
PRIMARY_PNG = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz/primary.png"
RIGHTVIEW_PNG = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz/right_view.png"
OUT_DIR = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz"

# Training/eval constants (from QwenGR00T_FFSCommon + cfg.yaml).
FFS_IMAGE_SIZE = 256
VALID_ITERS = 8

# Geometry (from stereo_pointcloud_viz.py).
H = W = 256
FOVY_DEG = 45.0
BASELINE_M = 0.06
FX = FY = (H / 2.0) / np.tan(np.deg2rad(FOVY_DEG) / 2.0)  # 309.0193...
CX = W / 2.0
CY = H / 2.0

if FFS_REPO not in sys.path:
    sys.path.insert(0, FFS_REPO)
import core.foundation_stereo as _fs  # noqa: F401  (register classes for unpickle)

from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def imgs_to_ffs_tensor(pil_img, device):
    """EXACT replica of _imgs_to_ffs_tensor / ffs_order_net0_check.imgs_to_ffs_tensor."""
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        (FFS_IMAGE_SIZE, FFS_IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    img = to_tensor(pil_img)              # [3,H,W] in [0,1]
    img = resize(img.unsqueeze(0)).squeeze(0)
    img = img * 255.0
    return img.unsqueeze(0).to(device).float()  # [1,3,H,W]


def load_ffs(device):
    """Load FFS exactly as the training/eval code does."""
    ffs = torch.load(FFS_WEIGHTS, map_location="cpu", weights_only=False)
    ffs.eval()
    for p in ffs.parameters():
        p.requires_grad = False
    try:
        ffs.args.mixed_precision = False
    except Exception:
        ffs.args["mixed_precision"] = False
    ffs = ffs.to(device).float()
    ffs.eval()
    return ffs


@torch.no_grad()
def run_ffs_disp(ffs, img1, img2):
    """Single FFS forward, test_mode=True -> disparity [1,1,H,W]. tf32 off, autocast off."""
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.amp.autocast("cuda", enabled=False):
            disp = ffs(img1.float(), img2.float(), iters=VALID_ITERS, test_mode=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn
    return disp.detach().float()


def disp_median_masked(disp):
    """median over valid (>0) disparity pixels, in pixels."""
    d = disp.flatten().double()
    m = d > 0
    if m.sum() == 0:
        return float("nan"), 0.0
    valid = d[m]
    return valid.median().item(), 100.0 * m.sum().item() / d.numel()


def depth_from_disp(disp, fx, baseline, lo=0.05, hi=3.0):
    """Z = fx*baseline/disp; mask invalid (<=0) and clamp range. disp [1,1,H,W]."""
    d = disp[0, 0].cpu().numpy()
    Z = np.zeros_like(d, dtype=np.float32)
    valid = d > 1e-3
    Z[valid] = fx * baseline / d[valid]
    inrange = valid & (Z >= lo) & (Z <= hi)
    return Z, inrange


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  torch={torch.__version__}")
    if device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}  CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    primary = Image.open(PRIMARY_PNG).convert("RGB")
    rightview = Image.open(RIGHTVIEW_PNG).convert("RGB")
    print(f"[img] primary={primary.size}  right_view={rightview.size}")
    print(f"[K] fx=fy={FX:.4f}  cx=cy={CX}  baseline={BASELINE_M} m")

    ffs = load_ffs(device)
    print(f"[ffs] loaded. valid_iters={VALID_ITERS} max_disp={int(ffs.args.max_disp)}")

    p = imgs_to_ffs_tensor(primary, device)     # stored primary token-grid orientation
    r = imgs_to_ffs_tensor(rightview, device)

    # OLD buggy order: ffs(primary, right)
    disp_old = run_ffs_disp(ffs, p, r)

    # FIXED: flip both inputs, run, flip disparity back  (EXACT code-under-test logic)
    pf = torch.flip(p, dims=(-2, -1)).contiguous()
    rf = torch.flip(r, dims=(-2, -1)).contiguous()
    disp_fixed = torch.flip(run_ffs_disp(ffs, pf, rf), dims=(-2, -1)).contiguous()

    med_old, valid_old = disp_median_masked(disp_old)
    med_fixed, valid_fixed = disp_median_masked(disp_fixed)
    print("\n" + "=" * 70)
    print("STEP 2 — DISPARITY MEDIANS (masked >0px)")
    print("=" * 70)
    print(f"  disp_old   (buggy ffs(primary,right))      median = {med_old:8.3f} px  (valid {valid_old:.1f}%)")
    print(f"  disp_fixed (flip->ffs->flip-back)          median = {med_fixed:8.3f} px  (valid {valid_fixed:.1f}%)")

    # STEP 3 — depth + scale (from disp_fixed)
    Z_fixed, inrange_fixed = depth_from_disp(disp_fixed, FX, BASELINE_M)
    Z_old, inrange_old = depth_from_disp(disp_old, FX, BASELINE_M)
    zf = Z_fixed[inrange_fixed]
    print("\n" + "=" * 70)
    print("STEP 3 — DEPTH (Z = fx*baseline/disp_fixed), meters")
    print("=" * 70)
    if zf.size > 0:
        p5, p50, p95 = np.percentile(zf, [5, 50, 95])
        print(f"  disp_fixed depth: p5={p5:.3f}  median={p50:.3f}  p95={p95:.3f}  "
              f"min={zf.min():.3f}  max={zf.max():.3f}  (valid px in [0.05,3]m = {inrange_fixed.sum()})")
    else:
        p5 = p50 = p95 = float("nan")
        print("  disp_fixed depth: NO valid pixels in range")

    # STEP 4 — ALIGNMENT sanity
    print("\n" + "=" * 70)
    print("STEP 4 — ALIGNMENT to primary.png")
    print("=" * 70)
    # (a) tilted-plane sign: corr between image-row index v and depth Z over valid pixels.
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    vv = vs[inrange_fixed].astype(np.float64)
    zz = Z_fixed[inrange_fixed].astype(np.float64)
    if vv.size > 10 and zz.std() > 0:
        corr_row_depth = float(np.corrcoef(vv, zz)[0, 1])
    else:
        corr_row_depth = float("nan")
    print(f"  corr(image-row v, depth Z) over valid = {corr_row_depth:+.4f}  "
          f"(earlier known-good tilted-plane sign ~ -0.944)")

    # (b) disp_fixed must NOT equal a 180-deg rotation of itself = genuinely re-aligned,
    #     not symmetric. Compare disp_fixed vs flip(disp_fixed,180).
    df = disp_fixed[0, 0]
    df_rot = torch.flip(df, dims=(-2, -1))
    rel_self_vs_rot = (df - df_rot).abs().mean().item() / (df.abs().mean().item() + 1e-9)
    print(f"  mean|disp_fixed - rot180(disp_fixed)| / mean|disp_fixed| = {rel_self_vs_rot:.4f}  "
          f"(>>0 => genuinely re-aligned, not a 180-symmetric map)")

    # (c) disp_fixed vs disp_old: confirm they are different maps (fix changed geometry).
    rel_fixed_vs_old = (disp_fixed - disp_old).abs().mean().item() / (disp_old.abs().mean().item() + 1e-9)
    print(f"  mean|disp_fixed - disp_old| / mean|disp_old| = {rel_fixed_vs_old:.4f}")

    # (d) side-by-side PNG: primary.png vs disp_fixed depth colormap (same orientation),
    #     plus disp_old depth for contrast.
    prim_np = np.asarray(primary.resize((W, H)))
    Zf_disp = np.where(inrange_fixed, Z_fixed, np.nan)
    Zo_disp = np.where(inrange_old, Z_old, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(prim_np)
    axes[0].set_title("primary.png (stored)")
    axes[0].axis("off")
    if np.isfinite(Zf_disp).any():
        vmin, vmax = np.nanpercentile(Zf_disp, [2, 98])
    else:
        vmin, vmax = 0, 1
    im1 = axes[1].imshow(Zf_disp, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"disp_fixed depth (m)\nmedian={p50:.3f}m")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    if np.isfinite(Zo_disp).any():
        vmin2, vmax2 = np.nanpercentile(Zo_disp, [2, 98])
    else:
        vmin2, vmax2 = 0, 1
    im2 = axes[2].imshow(Zo_disp, cmap="turbo", vmin=vmin2, vmax=vmax2)
    axes[2].set_title("disp_old depth (m) [buggy, contrast]")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "verify_fixed.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  [saved] {out_png}")

    # ---- VERDICT ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    cond_b = (10.0 <= med_fixed <= 30.0) and (med_old > 100.0)
    cond_c = (not np.isnan(p50)) and (0.5 <= p50 <= 1.5)
    # alignment: tilted-plane sign present (|corr| reasonably large) AND map genuinely re-aligned
    cond_d = (abs(corr_row_depth) > 0.3) and (rel_self_vs_rot > 0.2)
    print(f"  (b) disp_fixed median {med_fixed:.2f} in [10,30] AND disp_old {med_old:.2f} > 100 : {cond_b}")
    print(f"  (c) depth median {p50:.3f} in [0.5,1.5] m                                : {cond_c}")
    print(f"  (d) tilted-plane corr |{corr_row_depth:.3f}|>0.3 AND re-aligned {rel_self_vs_rot:.3f}>0.2 : {cond_d}")
    overall = cond_b and cond_c and cond_d
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
