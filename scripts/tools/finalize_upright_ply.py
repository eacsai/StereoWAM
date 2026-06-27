#!/usr/bin/env python3
"""
Promote the leftprimary stereo cloud to the upright (Y-up) frame and overwrite in place.
Transform = optical (Y-down, Z-forward) -> OpenGL/world (Y-up): negate Y and Z, i.e.
180 deg about X (proper rotation, X kept so left-right is NOT mirrored). numpy-only.
Backs up the original optical-frame file as <name>_optical_orig.ply (once).
"""
import os
import shutil
import numpy as np

PLY = os.environ.get("PLY_PATH",
    "/Users/agiuser/Documents/ICLR2026/stereo_pointcloud_viz/leftprimary_cloud.ply")
DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")])

with open(PLY, "rb") as f:
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

backup = PLY[:-4] + "_optical_orig.ply"
if not os.path.exists(backup):
    shutil.copyfile(PLY, backup)
    print("backed up original ->", backup)

buf["y"] = -buf["y"]
buf["z"] = -buf["z"]
with open(PLY, "wb") as f:
    f.write(header)
    f.write(buf.tobytes())
print(f"wrote upright (Y-up) cloud -> {PLY}  n={n}")
