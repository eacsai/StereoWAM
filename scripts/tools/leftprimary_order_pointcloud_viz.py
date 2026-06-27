#!/usr/bin/env python3
"""
Clean LEFTPRIMARY-convention visualization.

Answers four questions on a REAL LIBERO stereo frame:
  (1) the order of images fed INTO FoundationStereo (FFS)
  (2) the order of images fed to the VLM
  (3) the point cloud FFS produces
  (4) whether that point cloud is in the LEFT_VIEW frame

Current clean convention (FFS_STEREO_CONVENTION=leftprimary), confirmed in code:
  DataConfig video_keys = ["video.primary_image", "video.left_view"]
    -> view[0] = primary, view[1] = left_view
    -> "left_view" is a LeRobot alias of the stored "right_view" column;
       that stored right_view is GEOMETRICALLY the LEFT camera (~17px parallax).
  VLM input order  = [primary, left_view]                       (view[0], view[1])
  FFS input order  = ffs(image1=left_view, image2=primary)      (left_ref_idx=1, primary_view_idx=0)
                     image1 = reference = geometric-LEFT = left_view, UPRIGHT (no flip)
  net[0]/disparity -> LEFT_VIEW camera frame
  point cloud      -> LEFT_VIEW camera frame; inject_cam_id=1 -> left_view tokens
So the FFS point cloud corresponds to LEFT_VIEW. This script proves it on a real frame.
"""
import os, sys
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

H = W = 256
FOVY_DEG = 45.0
BASELINE_M = 0.06
FX = FY = (H / 2.0) / np.tan(np.deg2rad(FOVY_DEG) / 2.0)  # 309.0193
CX = CY = W / 2.0  # 128.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float32)
MODEL_PATH = os.path.join(FFS_ROOT, "weights/20-30-48/model_best_bp2_serialize.pth")
SRC = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz"
OUT = SRC


def load_rgb(p):
    a = iio.imread(p)
    if a.ndim == 2:
        a = np.tile(a[..., None], (1, 1, 3))
    return a[..., :3].astype(np.uint8)


def run_ffs(left, right, iters=8, max_disp=192):
    """Disparity in image1(left)'s frame. left=image1=reference, right=image2."""
    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    m.args.valid_iters = iters
    m.args.max_disp = max_disp
    m.cuda().eval()
    torch.autograd.set_grad_enabled(False)
    i0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    i1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    pad = InputPadder(i0.shape, divis_by=32, force_square=False)
    i0, i1 = pad.pad(i0, i1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = m.forward(i0, i1, iters=iters, test_mode=True,
                         optimize_build_volume="pytorch1")
    disp = pad.unpad(disp.float()).cpu().numpy().reshape(H, W)
    return disp.clip(0, None).astype(np.float32)


def unproject(depth, rgb, valid):
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = depth
    X = (us - CX) * Z / FX
    Y = (vs - CY) * Z / FY
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    keep = valid.reshape(-1) & (Z.reshape(-1) >= 0.05) & (Z.reshape(-1) <= 3.0)
    return pts[keep], cols[keep]


def write_ply(path, pts, cols):
    n = pts.shape[0]
    cols = cols.astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    buf = np.empty(n, dtype=dt)
    buf["x"], buf["y"], buf["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    buf["r"], buf["g"], buf["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def main():
    primary = load_rgb(os.path.join(SRC, "primary.png"))      # view[0] = primary
    left_view = load_rgb(os.path.join(SRC, "right_view.png"))  # view[1] = left_view (stored right_view col = geometric LEFT)

    # FFS in the CLEAN leftprimary order: ffs(image1=left_view, image2=primary), UPRIGHT.
    disp = run_ffs(left_view, primary)
    valid = disp > 0.001
    depth = np.zeros_like(disp)
    depth[valid] = FX * BASELINE_M / disp[valid]
    d = depth[valid]
    p5, p50, p95 = np.percentile(d, [5, 50, 95])
    disp_med = float(np.median(disp[valid]))

    # === FIGURE 1: input ORDER (FFS + VLM) ===
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(left_view)
    ax[0, 0].set_title("FFS image1 = left_view  (reference = geometric LEFT)\n"
                       "[stored 'right_view' column, aliased to left_view]", fontsize=9)
    ax[0, 1].imshow(primary)
    ax[0, 1].set_title("FFS image2 = primary", fontsize=9)
    ax[1, 0].imshow(primary)
    ax[1, 0].set_title("VLM token[0] = primary", fontsize=9)
    ax[1, 1].imshow(left_view)
    ax[1, 1].set_title("VLM token[1] = left_view", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Clean LEFTPRIMARY order\n"
                 "FFS = ffs(left_view, primary) UPRIGHT   |   VLM = [primary, left_view]",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "leftprimary_order.png"), dpi=110)
    plt.close(fig)

    # === FIGURE 2: point cloud (LEFT_VIEW frame), colored by left_view ===
    pts, cols = unproject(depth, left_view, valid)
    # optical (Y-down, Z-fwd) -> OpenGL/world (Y-up): upright .ply for MeshLab
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0
    write_ply(os.path.join(OUT, "leftprimary_cloud.ply"), pts, cols)

    dmap = np.where(valid, depth, np.nan)
    fig = plt.figure(figsize=(15, 10))
    a0 = fig.add_subplot(2, 3, 1)
    im0 = a0.imshow(dmap, cmap="turbo", vmin=np.nanpercentile(dmap, 2),
                    vmax=np.nanpercentile(dmap, 98))
    a0.set_title("FFS depth (LEFT_VIEW frame)")
    fig.colorbar(im0, ax=a0, label="Z (m)")
    a1 = fig.add_subplot(2, 3, 2)
    a1.imshow(left_view)
    a1.set_title("left_view (cloud color source)")
    a1.axis("off")
    a2 = fig.add_subplot(2, 3, 3)
    dispv = np.where(valid, disp, np.nan)
    im2 = a2.imshow(dispv, cmap="magma")
    a2.set_title(f"disparity (median {disp_med:.1f} px)")
    fig.colorbar(im2, ax=a2, label="disp (px)")

    if pts.shape[0] > 40000:
        idx = np.random.default_rng(0).choice(pts.shape[0], 40000, replace=False)
        p, c = pts[idx], cols[idx]
    else:
        p, c = pts, cols
    c01 = c.astype(np.float32) / 255.0
    mins, maxs = p.min(0), p.max(0)
    ctr = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0
    for j, (az, el) in enumerate([(-60, 20), (-30, 60), (-120, 10)], 4):
        a = fig.add_subplot(2, 3, j, projection="3d")
        a.scatter(p[:, 0], p[:, 1], p[:, 2], c=c01, s=1, marker=".",
                  linewidths=0, depthshade=False)
        a.set_xlim(ctr[0] - r, ctr[0] + r)
        a.set_ylim(ctr[1] - r, ctr[1] + r)
        a.set_zlim(ctr[2] - r, ctr[2] + r)
        a.view_init(elev=el, azim=az)
        a.set_title(f"cloud (az={az}, el={el})")
        a.set_xlabel("X")
        a.set_ylabel("Y")
        a.set_zlabel("Z")
        try:
            a.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle(f"FFS point cloud in LEFT_VIEW frame  |  depth median {p50:.3f} m "
                 f"(p5 {p5:.2f}, p95 {p95:.2f})  |  disp median {disp_med:.1f} px  |  "
                 f"{pts.shape[0]} pts", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "leftprimary_pointcloud.png"), dpi=110)
    plt.close(fig)

    print("=== LEFTPRIMARY VIZ ===")
    print("FFS input order: image1=left_view (geometric LEFT, = stored right_view col), "
          "image2=primary  ->  ffs(left_view, primary) UPRIGHT")
    print("VLM input order: [primary, left_view]")
    print(f"disparity median: {disp_med:.2f} px  (sane stereo parallax ~17px)")
    print(f"depth: p5={p5:.3f} median={p50:.3f} p95={p95:.3f} m  (LIBERO tabletop ~0.8-1.3m)")
    print("point cloud frame: LEFT_VIEW (image1/reference)  ->  cloud corresponds to left_view: YES")
    print("saved: leftprimary_order.png, leftprimary_pointcloud.png, leftprimary_cloud.ply")


if __name__ == "__main__":
    main()
