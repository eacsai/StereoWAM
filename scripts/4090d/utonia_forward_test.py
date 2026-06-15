"""Utonia pre-flight step 2: load the frozen encoder + run a forward on a synthetic
back-projected point cloud, confirm per-point multi-scale features (expect dim 1386) come out.
Run with enable_flash=False so no flash-attn build is needed."""
import os, sys, torch, numpy as np
import utonia

CKPT = "/data/wangqiwei/ICLR2026/starVLA/playground/Pretrained_models/Utonia/utonia.pth"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "torch:", torch.__version__)

# load (enc_mode True per ckpt config); disable flash for a dependency-light test
model = utonia.load(CKPT, custom_config=dict(enable_flash=False, enc_patch_size=[1024]*5))
model = model.to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
print("loaded utonia, params:", sum(v.numel() for v in model.parameters()))

# synthetic "back-projected from a left image" point cloud: a 64x64 patch grid -> 4096 points
H = W = 64
ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
z = 0.5 + 0.3 * np.random.rand(H, W)               # fake metric depth (m)
coord = np.stack([(xs - W / 2) * 0.002 * z, (ys - H / 2) * 0.002 * z, z], -1).reshape(-1, 3).astype(np.float32)
color = (np.random.rand(H * W, 3) * 255).astype(np.float32)
normal = np.zeros_like(coord)                       # missing-modality -> zeros
print("input points:", coord.shape)

t = utonia.transform.default(scale=4.0, apply_z_positive=True, normalize_coord=False)
point = t({"coord": coord, "color": color, "normal": normal})
for k in list(point.keys()):
    if torch.is_tensor(point[k]):
        point[k] = point[k].to(dev)
print("grid-sampled points (after transform):", int(point["feat"].shape[0]), "feat dim:", int(point["feat"].shape[1]))

with torch.inference_mode():
    out = model(point)
print("raw out.feat:", tuple(out.feat.shape))

# walk the pooling chain to concat all encoder scales -> per-point multi-scale feature
o = out
levels = 0
while "pooling_parent" in o.keys():
    parent = o.pop("pooling_parent"); inv = o.pop("pooling_inverse")
    parent.feat = torch.cat([parent.feat, o.feat[inv]], dim=-1)
    o = parent; levels += 1
print("multi-scale concat: levels walked =", levels, " feat dim =", int(o.feat.shape[1]))

# map back to original N input points via the transform's inverse
per_point = o.feat[point["inverse"]] if "inverse" in point.keys() else o.feat
print("PER-POINT FEATURE (aligned to original points):", tuple(per_point.shape))
print("UTONIA_FORWARD_OK feat_dim=", int(per_point.shape[-1]))
