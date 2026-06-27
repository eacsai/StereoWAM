#!/usr/bin/env python3
"""
Flip the leftprimary stereo point cloud from camera-optical frame (Y-down, Z-forward)
to an upright world/OpenGL frame (Y-up). Renders candidate transforms for visual check,
then saves the chosen one. Binary little-endian PLY (xyz float32 + rgb uint8).

Candidate transforms (all proper rotations, no mirror):
  A = (x, -y, -z)  optical->OpenGL (negate Y and Z), 180 deg about X   [standard]
  B = (-x, -y,  z)  180 deg about Z (in-plane spin of the front view)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.environ.get("PLY_SRC",
    "/Users/agiuser/Documents/ICLR2026/stereo_pointcloud_viz/leftprimary_cloud.ply")
OUT_DIR = os.environ.get("PLY_OUT_DIR",
    "/Users/agiuser/Documents/ICLR2026/stereo_pointcloud_viz")
DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")])


def load_ply(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        n = None
        for ln in header.split(b"\n"):
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
        buf = np.frombuffer(f.read(), dtype=DT, count=n).copy()
    return header, buf


def save_ply(path, header, buf):
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def transform(buf, sx, sy, sz):
    out = buf.copy()
    out["x"] = sx * buf["x"]
    out["y"] = sy * buf["y"]
    out["z"] = sz * buf["z"]
    return out


def render(buf, title, png, up_axis="y"):
    """Render with the chosen world-up axis as the vertical (mpl Z) axis so we can
    visually judge uprightness: table should sit at the BOTTOM, objects rise UP."""
    p = np.stack([buf["x"], buf["y"], buf["z"]], -1).astype(np.float32)
    c = np.stack([buf["r"], buf["g"], buf["b"]], -1).astype(np.float32) / 255.0
    if len(p) > 40000:
        idx = np.random.default_rng(0).choice(len(p), 40000, replace=False)
        p, c = p[idx], c[idx]
    # remap so up_axis becomes the vertical (3rd) coordinate in the plot
    order = {"y": (0, 2, 1), "z": (0, 1, 2), "x": (1, 2, 0)}[up_axis]
    pp = p[:, order]
    vert = pp[:, 2]
    fig = plt.figure(figsize=(15, 7))
    for j, (az, el) in enumerate([(-75, 12), (105, 12)], 1):
        a = fig.add_subplot(1, 2, j, projection="3d")
        # color RGB (left) and by height (right) so the table plane vs raised objects is obvious
        col = c if j == 1 else plt.cm.viridis((vert - vert.min()) / (vert.ptp() + 1e-9))[:, :3]
        a.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c=col, s=2, marker=".",
                  linewidths=0, depthshade=False)
        a.view_init(elev=el, azim=az)
        a.set_title(f"{title}  {'RGB' if j == 1 else 'by height'}  az={az} el={el}")
        a.set_xlabel("X"); a.set_ylabel("depth"); a.set_zlabel(f"UP ({up_axis})")
        try:
            a.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle(f"{title}: table should be at BOTTOM, objects rise UP", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=115)
    plt.close(fig)
    print("saved", png)


def main():
    header, buf = load_ply(SRC)
    print("loaded", SRC, "n=", len(buf))
    A = transform(buf, 1, -1, -1)   # optical -> OpenGL
    B = transform(buf, -1, -1, 1)   # 180 about Z
    render(A, "A=(x,-y,-z)", os.path.join(OUT_DIR, "cand_A_preview.png"), up_axis="y")
    render(B, "B=(-x,-y,z)", os.path.join(OUT_DIR, "cand_B_preview.png"), up_axis="y")
    # also stash the candidate plys so we can promote the chosen one without recompute
    save_ply(os.path.join(OUT_DIR, "cand_A.ply"), header, A)
    save_ply(os.path.join(OUT_DIR, "cand_B.ply"), header, B)
    print("done")


if __name__ == "__main__":
    main()
