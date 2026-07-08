#!/usr/bin/env python3
"""
Probe v2(试水验证): stereo 左图点云 → 世界坐标三视图 + 轴测 + PLY。

vs v1 的改动(用户反馈):
  - camera→world 变换: 从 LIBERO MuJoCo scene 读 agentview 真实相机矩阵,
    再按 starVLA 渲染脚本的真实 rightview baseline 得到 right_view pose
  - table leveling: gated auto fallback；只在残余大且平面可信时启用，避免遮挡帧被转飞
  - 右下格: legend → 轴测图(matplotlib 3D scatter, 立体感)
  - 保存 PLY 点云(世界坐标, 外部 3D viewer 看)
  - 多帧模式(从 leftprimary dataset 取 N 帧)

Conventions (leftprimary, 跟生产 QwenGR00T_FFSCommon 一致):
  - s['image'][0] = primary    = 几何右眼
  - s['image'][1] = right_view = 几何左眼 (别名 left_view)
  - FFS: ffs(image1=几何左=right_view, image2=几何右=primary) → 视差是几何左 frame 的
  - unproject 到 stored right_view 图像坐标 (X 右, Y 下, Z 前=深度)
  - OURRENDER_PW 存盘图像做了 180deg rotate: stored OpenCV X右/Y下/Z前
    → MuJoCo 相机 -X/-Y/-Z → 世界

世界三视图(观察者在场景外围):
  top   = 从世界 +Z 俯视, 看 XY 平面(桌面布局)  u=+X(右), v=-Y(上=北), near=max Z(高处近)
  front = 从世界 +Y 正视, 看 XZ 平面             u=+X(右), v=-Z(上),    near=max Y(前近)
  side  = 从世界 +X 侧视, 看 YZ 平面             u=-Y(右=后),v=-Z(上),  near=max X(近)
"""
import os, sys, argparse, io
import numpy as np
import torch
import imageio.v2 as iio

FFS_ROOT = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", "/data/wangqiwei/ICLR2026/starVLA")
sys.path.insert(0, FFS_ROOT)
from core.utils.utils import InputPadder  # noqa
from Utils import AMP_DTYPE  # noqa

H = W = 256
FOVY_DEG = 45.0
BASELINE_M = 0.06
AUTO_LEVEL_MIN_DEG = 5.0
AUTO_LEVEL_MAX_DEG = 20.0
AUTO_LEVEL_MIN_INLIERS = 5000
AUTO_LEVEL_MIN_INLIER_FRAC = 0.55
FX = FY = (H / 2.0) / np.tan(np.deg2rad(FOVY_DEG) / 2.0)
CX = W / 2.0
CY = H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
FFS_MODEL = os.path.join(FFS_ROOT, "weights/20-30-48/model_best_bp2_serialize.pth")

WORKSPACE_ORIGIN_CAM = np.array([0.0, 0.0, 0.8], dtype=np.float32)  # 几何左相机坐标(前 0.8m)
WORKSPACE_SIZE = 0.8
VALID_LO, VALID_HI = 0.30, 1.50
IMG_SIZE = 224
SPLAT_RADIUS_M = 0.012
METERS_PER_PX = WORKSPACE_SIZE / IMG_SIZE  # ≈3.57 mm/px


def to_np(img):
    if hasattr(img, "numpy"):
        a = img.numpy()
    else:
        a = np.array(img)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.max() <= 1.5:
        a = a * 255.0
    return a.astype(np.uint8)[..., :3]


def run_ffs_disparity(img_left, img_right, valid_iters=8, max_disp=192):
    """FFS: image1=left=几何左(reference), image2=right=几何右. 返回 left-frame 视差."""
    model = torch.load(FFS_MODEL, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)
    a = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    b = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(a, b, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(img_left.shape[0], img_left.shape[1])
    return disp.clip(0, None).astype(np.float32)


def disparity_to_depth(disp, fx, baseline, disp_eps=1e-3):
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > disp_eps
    depth[valid] = fx * baseline / disp[valid]
    return depth, valid


def unproject(depth, rgb, valid_mask, lo, hi):
    """反投影到几何左相机坐标 (X右, Y下, Z前)."""
    H_, W_ = depth.shape
    vs, us = np.meshgrid(np.arange(H_), np.arange(W_), indexing="ij")
    Z = depth
    X = (us - K[0, 2]) * Z / K[0, 0]
    Y = (vs - K[1, 2]) * Z / K[1, 1]
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    zv = Z.reshape(-1)
    keep = valid_mask.reshape(-1) & (zv >= lo) & (zv <= hi)
    return pts[keep].astype(np.float32), cols[keep].astype(np.uint8)


def resolve_starvla_path(path):
    return path if os.path.isabs(path) else os.path.join(STARVLA_ROOT, path)


def quat_wxyz_to_rotmat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


OURRENDER_SUITE_ORDER = ("libero_object", "libero_goal", "libero_spatial", "libero_10")
OURRENDER_RIGHTVIEW_POSES = {
    # Measured from the actual OURRENDER_PW generation path:
    # openvla_debug/render_rightview_state_replay.py ->
    # starVLA/SSF/render_scripts/_stereo_render_utils.make_stereo_env().
    # rightview = agentview translated by +0.06m along agentview local +X,
    # same quaternion, fovy=45, with saved RGB frames rotated 180deg.
    "libero_object": (
        np.array([0.89657688, 0.06000057, 0.65000004], dtype=np.float32),
        np.array([0.61821675, 0.34323086, 0.34323086, 0.61821775], dtype=np.float32),
    ),
    "libero_goal": (
        np.array([0.65861294, 0.06000006, 1.61035000], dtype=np.float32),
        np.array([0.63801769, 0.30484985, 0.30484985, 0.63801769], dtype=np.float32),
    ),
    "libero_spatial": (
        np.array([0.65861294, 0.06000006, 1.61035000], dtype=np.float32),
        np.array([0.63801769, 0.30484985, 0.30484985, 0.63801769], dtype=np.float32),
    ),
    "libero_10": (
        np.array([0.60657688, 0.06000005, 0.96000004], dtype=np.float32),
        np.array([0.61821675, 0.34323086, 0.34323086, 0.61821775], dtype=np.float32),
    ),
}


def load_right_view_pose(suite_name):
    """Return OURRENDER_PW stored right_view MuJoCo camera pose as local → world."""
    try:
        pos, quat = OURRENDER_RIGHTVIEW_POSES[suite_name]
    except KeyError as e:
        raise RuntimeError(f"unknown LIBERO suite for right_view pose: {suite_name}") from e
    return pos.copy(), quat_wxyz_to_rotmat(quat)


def suite_name_for_index(global_index, dataset_lengths):
    if dataset_lengths is None:
        return OURRENDER_SUITE_ORDER[0]
    offset = 0
    for suite_name, n in zip(OURRENDER_SUITE_ORDER, dataset_lengths):
        offset += int(n)
        if global_index < offset:
            return suite_name
    return OURRENDER_SUITE_ORDER[-1]


# OURRENDER_PW render chain stores all LIBERO camera frames as raw[::-1, ::-1].
# Relative to MuJoCo's natural camera image, that extra horizontal flip means
# stored OpenCV +X maps to MuJoCo camera -X.
OPENCV_TO_MUJOCO_CAM = np.diag([-1.0, -1.0, -1.0]).astype(np.float32)


def camera_to_world(pts, right_view_pos_w, right_view_R_mjc_to_w):
    """OpenCV 相机坐标(X右,Y下,Z前) → 世界坐标(mujoco LIBERO, Z上).
    点云来自 OURRENDER_PW stored right_view frame；180deg 存盘旋转要 flip X/Y/Z."""
    Rfull = right_view_R_mjc_to_w @ OPENCV_TO_MUJOCO_CAM
    return (pts @ Rfull.T + right_view_pos_w).astype(np.float32)


def rotation_align(src, dst):
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    src /= max(float(np.linalg.norm(src)), 1e-8)
    dst /= max(float(np.linalg.norm(dst)), 1e-8)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float32)
    axis = np.cross(src, dst)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    axis /= axis_norm
    Kx = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float32)
    angle = np.arccos(dot)
    return (np.eye(3, dtype=np.float32) +
            np.sin(angle) * Kx + (1.0 - np.cos(angle)) * (Kx @ Kx)).astype(np.float32)


def estimate_table_plane(pts_w, cols):
    """Estimate the visible tabletop plane for leveling the stereo point cloud."""
    if len(pts_w) < 1000:
        return None
    brightness = cols.astype(np.float32).mean(axis=1)
    z_lo, z_hi = np.percentile(pts_w[:, 2], [2.0, 70.0])
    keep = ((brightness > 55.0) & (brightness < 250.0) &
            (pts_w[:, 2] >= z_lo) & (pts_w[:, 2] <= z_hi))
    if int(keep.sum()) < 1000:
        keep = (brightness > 55.0) & (brightness < 250.0)
    pts = pts_w[keep]
    if len(pts) < 1000:
        return None

    rng = np.random.default_rng(0)
    best_count, best_normal, best_offset = 0, None, 0.0
    iters = min(2500, max(300, len(pts) // 8))
    thresh = 0.008
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        angle_deg = float(np.rad2deg(np.arccos(np.clip(normal[2], -1.0, 1.0))))
        if angle_deg > AUTO_LEVEL_MAX_DEG:
            continue
        offset = -float(np.dot(normal, a))
        count = int((np.abs(pts @ normal + offset) < thresh).sum())
        if count > best_count:
            best_count, best_normal, best_offset = count, normal.astype(np.float32), offset

    if best_normal is None or best_count < 1000:
        return None
    inliers = np.abs(pts @ best_normal + best_offset) < thresh
    table_pts = pts[inliers]
    center = table_pts.mean(axis=0).astype(np.float32)
    _, _, vh = np.linalg.svd(table_pts - center, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    if normal[2] < 0:
        normal = -normal
    return center, normal, int(table_pts.shape[0]), int(pts.shape[0])


def maybe_level_table(pts_w, cols, mode):
    plane = estimate_table_plane(pts_w, cols)
    if plane is None:
        return pts_w, np.nan, 0, False, "no_plane"
    center, normal, inliers, candidates = plane
    angle_deg = float(np.rad2deg(np.arccos(np.clip(normal[2], -1.0, 1.0))))

    should_level = False
    reason = mode
    if mode == "always":
        should_level = True
    elif mode == "auto":
        enough_plane = inliers >= AUTO_LEVEL_MIN_INLIERS
        enough_fraction = (inliers / max(float(candidates), 1.0)) >= AUTO_LEVEL_MIN_INLIER_FRAC
        residual_large = angle_deg > AUTO_LEVEL_MIN_DEG
        normal_plausible = angle_deg <= AUTO_LEVEL_MAX_DEG
        should_level = enough_plane and enough_fraction and residual_large and normal_plausible
        if not enough_plane:
            reason = "auto_skip_few_inliers"
        elif not enough_fraction:
            reason = "auto_skip_low_support"
        elif not residual_large:
            reason = "auto_skip_small_residual"
        elif not normal_plausible:
            reason = "auto_skip_implausible_normal"
        else:
            reason = "auto_apply"
    elif mode == "off":
        reason = "off"
    else:
        raise ValueError(mode)

    if not should_level:
        return pts_w, angle_deg, inliers, False, reason

    R_level = rotation_align(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    pts_level = ((pts_w - center) @ R_level.T + center).astype(np.float32)
    return pts_level, angle_deg, inliers, True, reason


def workspace_crop(pts_w, cols, origin_w):
    half = WORKSPACE_SIZE / 2.0
    rel = pts_w - origin_w
    keep = np.all(np.abs(rel) <= half, axis=1)
    return rel[keep], cols[keep]


_SPLAT_CACHE = {}


def _splat_offsets(r):
    if r not in _SPLAT_CACHE:
        pairs = [(du, dv)
                 for du in range(-r, r + 1)
                 for dv in range(-r, r + 1)
                 if du * du + dv * dv <= r * r]
        a = np.asarray(pairs, dtype=np.int32)
        _SPLAT_CACHE[r] = (a[:, 0], a[:, 1])
    return _SPLAT_CACHE[r]


def _ordered_f32_key(x):
    bits = x.astype(np.float32, copy=False).view(np.uint32)
    return np.where((bits >> 31) == 0,
                    bits ^ np.uint32(0x80000000),
                    ~bits).astype(np.uint32)


def render_world_view_ref(pts_rel, cols, view):
    """正交投影 + splat + z-buffer(近覆盖远). pts_rel = 相对 workspace 原点(世界)."""
    if view == "top":
        u, v, w = pts_rel[:, 0], -pts_rel[:, 1], pts_rel[:, 2]   # X, -Y, Z
    elif view == "front":
        u, v, w = pts_rel[:, 0], -pts_rel[:, 2], pts_rel[:, 1]   # X, -Z, Y
    elif view == "side":
        u, v, w = -pts_rel[:, 1], -pts_rel[:, 2], pts_rel[:, 0]  # -Y, -Z, X
    else:
        raise ValueError(view)

    u_px = (u / METERS_PER_PX + IMG_SIZE / 2.0).astype(np.int32)
    v_px = (v / METERS_PER_PX + IMG_SIZE / 2.0).astype(np.int32)
    splat_r = max(1, int(round(SPLAT_RADIUS_M / METERS_PER_PX)))

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    depth_buf = np.full((IMG_SIZE, IMG_SIZE), -np.inf, dtype=np.float32)  # near=max w
    order = np.argsort(w)  # w 小(远)先画, w 大(近)后覆盖
    for i in order:
        pu, pv, wi = int(u_px[i]), int(v_px[i]), float(w[i])
        if pu < -splat_r or pv < -splat_r or pu >= IMG_SIZE + splat_r or pv >= IMG_SIZE + splat_r:
            continue
        for du in range(-splat_r, splat_r + 1):
            cu = pu + du
            if cu < 0 or cu >= IMG_SIZE:
                continue
            for dv in range(-splat_r, splat_r + 1):
                if du * du + dv * dv > splat_r * splat_r:
                    continue
                cv = pv + dv
                if cv < 0 or cv >= IMG_SIZE:
                    continue
                if wi > depth_buf[cv, cu]:   # near(max w) 覆盖 far
                    depth_buf[cv, cu] = wi
                    img[cv, cu] = cols[i]
    return img


def render_world_view(pts_rel, cols, view):
    """Vectorized orthographic splat + z-buffer; matches render_world_view_ref."""
    if view == "top":
        u, v, w = pts_rel[:, 0], -pts_rel[:, 1], pts_rel[:, 2]   # X, -Y, Z
    elif view == "front":
        u, v, w = pts_rel[:, 0], -pts_rel[:, 2], pts_rel[:, 1]   # X, -Z, Y
    elif view == "side":
        u, v, w = -pts_rel[:, 1], -pts_rel[:, 2], pts_rel[:, 0]  # -Y, -Z, X
    else:
        raise ValueError(view)

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    if len(pts_rel) == 0:
        return img

    u_px = (u / METERS_PER_PX + IMG_SIZE / 2.0).astype(np.int32)
    v_px = (v / METERS_PER_PX + IMG_SIZE / 2.0).astype(np.int32)
    splat_r = max(1, int(round(SPLAT_RADIUS_M / METERS_PER_PX)))
    du, dv = _splat_offsets(splat_r)

    order = np.argsort(w)
    rank = np.empty(len(w), dtype=np.uint32)
    rank[order] = np.arange(len(w), dtype=np.uint32)

    depth_key = _ordered_f32_key(w)
    tie_key = (np.uint32(len(w) - 1) - rank).astype(np.uint32)
    point_key = (depth_key.astype(np.uint64) << 32) | tie_key.astype(np.uint64)

    keep = (
        (u_px >= -splat_r) & (v_px >= -splat_r) &
        (u_px < IMG_SIZE + splat_r) & (v_px < IMG_SIZE + splat_r) &
        (~np.isnan(w)) & (w > -np.inf)
    )
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return img

    cu = u_px[idx, None] + du[None, :]
    cv = v_px[idx, None] + dv[None, :]
    valid = (cu >= 0) & (cu < IMG_SIZE) & (cv >= 0) & (cv < IMG_SIZE)
    if not np.any(valid):
        return img

    pix = cv[valid].astype(np.intp) * IMG_SIZE + cu[valid].astype(np.intp)
    src = np.repeat(idx, len(du))[valid.ravel()]
    key = point_key[src]

    best = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.uint64)
    np.maximum.at(best, pix, key)

    winner = key == best[pix]
    img.reshape(-1, 3)[pix[winner]] = cols[src[winner]]
    return img


def render_axonometric(pts_rel, cols, size=IMG_SIZE):
    """轴测图(matplotlib 3D scatter, 等轴测视角)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from PIL import Image

    n = len(pts_rel)
    if n > 8000:
        idx = np.random.default_rng(0).choice(n, 8000, replace=False)
        p, c = pts_rel[idx], cols[idx]
    else:
        p, c = pts_rel, cols
    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c.astype(np.float32) / 255.0,
               s=14, linewidths=0, depthshade=False)
    lo, hi = p.min(0), p.max(0)
    ctr = (lo + hi) / 2.0
    r = max((hi - lo).max() / 2.0, 1e-3)
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.view_init(elev=25, azim=-55)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_facecolor("black"); fig.patch.set_facecolor("black"); ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white"); ax.set_zlabel("Z", color="white"); ax.tick_params(colors="white")
    ax.set_title("axonometric", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.fromarray(iio.imread(buf)).resize((size, size)))
    return img[..., :3].astype(np.uint8)


def save_ply(path, pts, cols):
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


def make_legend(size=IMG_SIZE):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    bar_px = max(20, int(0.1 / METERS_PER_PX))
    bx, by = 18, size // 2 - 40
    d.line([(bx, by), (bx + bar_px, by)], fill=(255, 255, 255), width=2)
    d.line([(bx, by - 4), (bx, by + 4)], fill=(255, 255, 255), width=2)
    d.line([(bx + bar_px, by - 4), (bx + bar_px, by + 4)], fill=(255, 255, 255), width=2)
    d.text((bx, by + 7), "100 mm", fill=(255, 255, 255))
    cx, cy = size // 2, size // 2 + 35
    d.line([(cx, cy), (cx + 32, cy)], fill=(255, 90, 90), width=2); d.text((cx + 35, cy - 6), "X", fill=(255, 90, 90))
    d.line([(cx, cy), (cx, cy - 32)], fill=(90, 255, 90), width=2); d.text((cx + 4, cy - 44), "Y", fill=(90, 255, 90))
    d.line([(cx, cy), (cx - 22, cy + 22)], fill=(90, 180, 255), width=2); d.text((cx - 34, cy + 20), "Z", fill=(90, 180, 255))
    d.text((4, size - 18), "Top/Front/Side @ %.1f mm/px" % (METERS_PER_PX * 1000), fill=(200, 200, 200))
    return np.array(img)


def compose_grid(top, front, side, legend):
    from PIL import Image, ImageDraw
    gap = 6
    sz = IMG_SIZE
    G = np.zeros((sz * 2 + gap, sz * 2 + gap, 3), dtype=np.uint8)
    G[:sz, :sz] = top
    G[:sz, sz + gap:] = front
    G[sz + gap:, :sz] = side
    G[sz + gap:, sz + gap:] = legend
    img = Image.fromarray(G)
    d = ImageDraw.Draw(img)
    d.text((5, 3), "TOP", fill=(255, 255, 255))
    d.text((sz + gap + 5, 3), "FRONT", fill=(255, 255, 255))
    d.text((5, sz + gap + 3), "SIDE", fill=(255, 255, 255))
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--camera-pitch", type=float, default=45.0,
                    help="deprecated; ignored (camera pose comes from OURRENDER_PW rightview extrinsics)")
    ap.add_argument("--mode", choices=["grid", "single_axo"], default="grid",
                    help="grid=三视图+轴测田字格; single_axo=只一张大轴测图")
    ap.add_argument("--axo-size", type=int, default=512)
    ap.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW")
    ap.add_argument("--data-mix", default="libero_all_sfstereo_leftprimary")
    ap.add_argument("--frame-step", type=int, default=300)
    ap.add_argument("--level-table", choices=["off", "auto", "always"], default="auto",
                    help="off=trust real right_view pose; auto=only gated fallback; always=debug old behavior")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(resolve_starvla_path("examples/LIBERO/train_files/starvla_cotrain_libero.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.create(
        {"datasets": {"vla_data": {"data_root_dir": resolve_starvla_path(args.data_root),
                                   "data_mix": args.data_mix}}}))

    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=42)
    dataset_lengths = getattr(ds, "dataset_lengths", getattr(ds, "_dataset_lengths", None))

    for i in range(args.n_frames):
        sample_index = i * args.frame_step
        suite_name = suite_name_for_index(sample_index, dataset_lengths)
        right_view_pos_w, right_view_R_mjc_to_w = load_right_view_pose(suite_name)
        s = ds[sample_index]
        primary = to_np(s["image"][0])     # 几何右
        right_view = to_np(s["image"][1])  # 几何左
        disp = run_ffs_disparity(right_view, primary)
        depth, valid = disparity_to_depth(disp, FX, BASELINE_M)
        pts, cols = unproject(depth, right_view, valid, VALID_LO, VALID_HI)
        pts_w = camera_to_world(pts, right_view_pos_w, right_view_R_mjc_to_w)
        pts_w, table_deg, table_inliers, leveled, level_reason = maybe_level_table(
            pts_w, cols, args.level_table)
        origin_w = pts_w.mean(0)  # 点云质心当 workspace 中心 → 自动居中桌面
        pts_rel, cols_rel = workspace_crop(pts_w, cols, origin_w)

        if args.mode == "single_axo":
            axo = render_axonometric(pts_rel, cols_rel, size=args.axo_size)
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_axo.png" % i), axo)
            save_ply(os.path.join(args.out_dir, "frame%02d_cloud.ply" % i), pts_rel, cols_rel)
            print("frame %02d [%s]: %d pts -> single axo (%dpx) + ply "
                  "(real right_view extrinsic, table %.2fdeg/%d inliers, "
                  "level=%s:%s)" %
                  (i, suite_name, len(pts_rel), args.axo_size, table_deg, table_inliers,
                   "on" if leveled else "off", level_reason))
        else:
            top = render_world_view(pts_rel, cols_rel, "top")
            front = render_world_view(pts_rel, cols_rel, "front")
            side = render_world_view(pts_rel, cols_rel, "side")
            axo_grid = render_axonometric(pts_rel, cols_rel)                    # 224, 拼田字格
            grid = compose_grid(top, front, side, axo_grid)
            axo_big = render_axonometric(pts_rel, cols_rel, size=args.axo_size)  # 大轴测, 单独存供检查
            # 存全套供人工逐项检查: 输入 stereo 对 + 三视图分张 + 轴测 + grid 总览 + 点云
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_primary.png" % i), primary)        # 输入 几何右眼
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_right_view.png" % i), right_view)  # 输入 几何左眼
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_top.png" % i), top)
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_front.png" % i), front)
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_side.png" % i), side)
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_axo.png" % i), axo_big)
            iio.imwrite(os.path.join(args.out_dir, "frame%02d_grid.png" % i), grid)
            save_ply(os.path.join(args.out_dir, "frame%02d_cloud.ply" % i), pts_rel, cols_rel)
            print("frame %02d [%s]: %d pts -> primary+right_view+top+front+side+axo+grid+ply "
                  "(real right_view extrinsic, table %.2fdeg/%d inliers, "
                  "level=%s:%s, %.1fmm/px)" %
                  (i, suite_name, len(pts_rel), table_deg, table_inliers,
                   "on" if leveled else "off", level_reason, METERS_PER_PX * 1000))
    print("PROBE_V2_OK", args.out_dir)


if __name__ == "__main__":
    main()
