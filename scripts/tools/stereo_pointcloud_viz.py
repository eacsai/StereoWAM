#!/usr/bin/env python3
"""
LIBERO stereo point-cloud visualization to verify FoundationStereo depth accuracy.

Pipeline:
  primary.png (LEFT) + right_view.png (RIGHT)
    -> FoundationStereo disparity (left-frame, pixels, disp = x_left - x_right > 0)
    -> metric depth  Z = fx * baseline / disp  (left/primary camera frame)
    -> unproject every valid pixel to (X,Y,Z), colored by primary RGB
    -> colored PLY + matplotlib diagnostics (depth heatmap + 3 cloud views)

Geometry constants (LIBERO stereo rig, 256x256, fovy=45, baseline 0.06, parallel +X):
  fx = fy = 309.0193 px, cx = cy = 128.0  (cx,cy = W/2,H/2 convention, NOT (W-1)/2)
  This matches starVLA's production utonia_pointcloud.py backprojection.

Runs headless (matplotlib Agg). Writes its own PLY (no open3d dependency).
"""
import os, sys, argparse
import numpy as np

FFS_ROOT = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo"
sys.path.insert(0, FFS_ROOT)

import torch
import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE  # torch.float16

# ---- Geometry (cam scout NUMERIC_K + baseline) ----
H = W = 256
FOVY_DEG = 45.0
BASELINE_M = 0.06
FX = FY = (H / 2.0) / np.tan(np.deg2rad(FOVY_DEG) / 2.0)  # 309.0193...
CX = W / 2.0  # 128.0
CY = H / 2.0  # 128.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)

# Expected LIBERO tabletop depth range for PASS/FAIL sanity
EXPECT_LO, EXPECT_HI = 0.30, 1.50  # meters (objects/table ~0.6-1.5; bulk ~0.8-1.3)
VALID_LO, VALID_HI = 0.05, 3.0     # hard clamp per utonia pipeline

MODEL_PATH = os.path.join(FFS_ROOT, "weights/20-30-48/model_best_bp2_serialize.pth")
OUT_DIR = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz"


def load_rgb(path):
    a = iio.imread(path)
    if a.ndim == 2:
        a = np.tile(a[..., None], (1, 1, 3))
    return a[..., :3].astype(np.uint8)


def run_ffs_disparity(left_rgb, right_rgb, valid_iters=8, max_disp=192):
    """Return left-frame disparity (H,W) float32, pixels, clipped >=0."""
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)

    img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(img0, img1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(left_rgb.shape[0],
                                                            left_rgb.shape[1])
    return disp.clip(0, None).astype(np.float32)


def disparity_to_depth(disp, fx, baseline, disp_eps=0.001):
    """Z = fx*baseline/disp; invalid where disp<=eps -> 0 (will be masked)."""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > disp_eps
    depth[valid] = fx * baseline / disp[valid]
    return depth, valid


def unproject(depth, rgb, K, valid_mask, lo, hi):
    """Unproject valid pixels to (X,Y,Z) in left/primary camera frame."""
    H_, W_ = depth.shape
    vs, us = np.meshgrid(np.arange(H_), np.arange(W_), indexing="ij")
    Z = depth
    X = (us - K[0, 2]) * Z / K[0, 0]
    Y = (vs - K[1, 2]) * Z / K[1, 1]
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    keep = valid_mask.reshape(-1) & (Z.reshape(-1) >= lo) & (Z.reshape(-1) <= hi)
    return pts[keep], cols[keep]


def write_ply(path, pts, cols):
    """Binary little-endian PLY with per-vertex RGB. No external deps."""
    n = pts.shape[0]
    cols = cols.astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    buf = np.empty(n, dtype=dt)
    buf["x"], buf["y"], buf["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    buf["r"], buf["g"], buf["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def save_depth_heatmap(depth, valid, path):
    d = np.where(valid, depth, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d, cmap="turbo", vmin=np.nanpercentile(d, 2),
                   vmax=np.nanpercentile(d, 98))
    ax.set_title("FFS metric depth (primary/left frame)")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    cb = fig.colorbar(im, ax=ax); cb.set_label("depth Z (m)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_cloud_views(pts, cols, out_dir, n_sub=40000):
    if pts.shape[0] > n_sub:
        idx = np.random.default_rng(0).choice(pts.shape[0], n_sub, replace=False)
        p, c = pts[idx], cols[idx]
    else:
        p, c = pts, cols
    c01 = c.astype(np.float32) / 255.0
    # Equal aspect bounds
    mins, maxs = p.min(0), p.max(0)
    ctr = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0
    angles = [(-60, 20), (-30, 60), (-120, 10)]  # (azim, elev)
    for i, (az, el) in enumerate(angles, 1):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c01, s=1, marker=".",
                   linewidths=0, depthshade=False)
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z depth (m)")
        ax.view_init(elev=el, azim=az)
        ax.set_title(f"stereo cloud view {i} (azim={az}, elev={el})")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cloud_view{i}.png"), dpi=120)
        plt.close(fig)


def stats_and_verdict(depth, valid, total_px, tag):
    d = depth[valid]
    pct_valid = 100.0 * valid.sum() / total_px
    p5, p50, p95 = np.percentile(d, [5, 50, 95])
    print(f"\n=== DEPTH STATS [{tag}] ===")
    print(f"  valid pixels: {valid.sum()} / {total_px} ({pct_valid:.1f}%)")
    print(f"  min={d.min():.3f}  p5={p5:.3f}  median={p50:.3f}  "
          f"p95={p95:.3f}  max={d.max():.3f}  (meters)")
    ok = (EXPECT_LO <= p50 <= EXPECT_HI) and (d.min() > 0)
    print(f"  expected tabletop range: [{EXPECT_LO}, {EXPECT_HI}] m")
    print(f"  SANITY: {'PASS' if ok else 'FAIL'} "
          f"(median {p50:.3f} m {'in' if EXPECT_LO<=p50<=EXPECT_HI else 'OUT of'} range)")
    return ok, (p5, p50, p95, d.min(), d.max(), pct_valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default=os.path.join(OUT_DIR, "primary.png"))
    ap.add_argument("--right", default=os.path.join(OUT_DIR, "right_view.png"))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--valid_iters", type=int, default=8)
    ap.add_argument("--max_disp", type=int, default=192)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"K =\n{K}\nbaseline={BASELINE_M} m  fx=fy={FX:.4f}  cx=cy={CX}")
    left = load_rgb(args.left)   # primary = LEFT = reference
    right = load_rgb(args.right) # right_view = RIGHT
    print(f"left(primary) {left.shape}  right(right_view) {right.shape}")

    total_px = H * W
    sign_flipped = False

    # --- Attempt 1: primary=LEFT, right_view=RIGHT (geometrically correct) ---
    disp = run_ffs_disparity(left, right, args.valid_iters, args.max_disp)
    print(f"\ndisparity raw: min={disp.min():.3f} median={np.median(disp):.3f} "
          f"max={disp.max():.3f}  (>0 px = {100.0*(disp>0.001).sum()/total_px:.1f}%)")
    depth, valid = disparity_to_depth(disp, FX, BASELINE_M)
    ok, _ = stats_and_verdict(depth, valid, total_px, "primary=LEFT, right_view=RIGHT")

    if not ok:
        # Diagnose: try swapping left/right (handles flipped baseline convention).
        print("\n[!] First attempt FAILED sanity. Trying SWAPPED left/right "
              "(right_view as LEFT, primary as RIGHT)...")
        disp2 = run_ffs_disparity(right, left, args.valid_iters, args.max_disp)
        depth2, valid2 = disparity_to_depth(disp2, FX, BASELINE_M)
        ok2, _ = stats_and_verdict(depth2, valid2, total_px, "SWAPPED")
        if ok2:
            print("[FIX] Swapping left/right produced sane depth. Using swapped. "
                  "NOTE: colored by the LEFT-of-swap image (right_view).")
            disp, depth, valid = disp2, depth2, valid2
            left = right  # color cloud by the reference (left) view actually used
            sign_flipped = True
            ok = True
        else:
            print("[!] Swap also failed. Reporting original (primary=LEFT) result; "
                  "geometry/scale likely off — see caveats.")

    # --- Build outputs from the chosen (disp, depth, valid) ---
    save_depth_heatmap(depth, valid, os.path.join(args.out, "depth_colormap.png"))
    pts, cols = unproject(depth, left, K, valid, VALID_LO, VALID_HI)
    print(f"\npoint cloud: {pts.shape[0]} valid points after [{VALID_LO},{VALID_HI}]m clamp")
    write_ply(os.path.join(args.out, "stereo_cloud.ply"), pts, cols)
    save_cloud_views(pts, cols, args.out)

    print(f"\nsign/left-right fix needed: {sign_flipped}")
    print(f"FINAL VERDICT: {'PASS' if ok else 'FAIL'}")
    print("\nOutputs in", args.out)
    for fn in ["depth_colormap.png", "cloud_view1.png", "cloud_view2.png",
               "cloud_view3.png", "stereo_cloud.ply"]:
        p = os.path.join(args.out, fn)
        print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {fn} "
              f"({os.path.getsize(p) if os.path.exists(p) else 0} bytes)")


if __name__ == "__main__":
    main()
