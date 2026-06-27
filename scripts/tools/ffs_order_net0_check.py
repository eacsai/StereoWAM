#!/usr/bin/env python3
"""Decisive numeric check: is the TRAINING-order FFS net[0] feature degenerate (~mono)?

Loads Fast-FoundationStereo EXACTLY as the training code does
(starVLA/model/framework/VLM4A/QwenGR00T_FFSCommon.py: _imgs_to_ffs_tensor /
_compute_ffs_feature / _compute_ffs_disparity), captures net[0] = update_block
output[0][0] (first GRU hidden state, the disparity-bearing feature that training
injects), and compares input orders:

  A = ffs(right_view, primary)    geometrically CORRECT  (image1=left=right_view, image2=right=primary)
  B = ffs(primary, right_view)    TRAINING order (suspected inverted)
  C = ffs(primary, primary)       degenerate zero-disparity baseline
  D = ffs(right_view, right_view) second degenerate baseline

NOTE on naming: FoundationStereo.forward(image1, image2) treats image1 as the
LEFT camera and image2 as the RIGHT camera and searches NON-NEGATIVE disparity
(left pixel is to the RIGHT of the matching right pixel). On the 180deg-rotated
stored frames the geometric LEFT camera is `right_view.png`. So the geometrically
correct call is ffs(image1=right_view, image2=primary) == A. Training calls
ffs(image1=primary, image2=right_view) == B (inverted) -> negative true disparity
-> clamped to ~0 by the non-negative search -> suspected degenerate.
"""
import os
import sys
import contextlib
import numpy as np
import torch
import torch.nn.functional as F

FFS_REPO = "/data/wangqiwei/ICLR2026/Fast-FoundationStereo"
FFS_WEIGHTS = os.path.join(FFS_REPO, "weights/20-30-48/model_best_bp2_serialize.pth")
PRIMARY_PNG = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz/primary.png"
RIGHTVIEW_PNG = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz/right_view.png"
OUT_DIR = "/data/wangqiwei/ICLR2026/stereo_pointcloud_viz"

# Training constants (from QwenGR00T_FFSCommon._init_frozen_ffs_net0 / cfg.yaml).
FFS_IMAGE_SIZE = 256
VALID_ITERS = 8          # cfg.yaml valid_iters
GRU_HIDDEN_DIM = 16      # ffs_feat_dim = gru_hidden_dim default in training (_init_frozen_ffs_net0)

if FFS_REPO not in sys.path:
    sys.path.insert(0, FFS_REPO)
import core.foundation_stereo as _fs  # noqa: F401  (register classes for unpickle)

from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def imgs_to_ffs_tensor(pil_img, device):
    """Replicates _imgs_to_ffs_tensor for a single image -> [1,3,H,W] float in [0,255]."""
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
    """Load FFS exactly as training does: torch.load full object, eval, freeze,
    mixed_precision=False, float32, register update_block hook capturing net[0]."""
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

    captured = {"net0": None}

    def _hook(_m, _i, output):
        # update_block returns (net, mask, delta_disp); net[0] = first GRU hidden.
        captured["net0"] = output[0][0]

    ffs.update_block.register_forward_hook(_hook)
    return ffs, captured


@torch.no_grad()
def run_ffs(ffs, captured, img1, img2):
    """Run a single FFS forward; return (disp_up [1,1,H,W], net0 [1,C,h,w]).
    Replicates _compute_ffs_feature / _compute_ffs_disparity: tf32 off,
    autocast disabled, test_mode=True, iters=valid_iters."""
    captured["net0"] = None
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.amp.autocast("cuda", enabled=False):
            disp_up = ffs(img1.float(), img2.float(), iters=VALID_ITERS, test_mode=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn
    net0 = captured["net0"]
    assert net0 is not None, "update_block hook did not fire"
    assert net0.shape[1] == GRU_HIDDEN_DIM, f"net0 channels {net0.shape[1]} != {GRU_HIDDEN_DIM}"
    return disp_up.detach().float(), net0.detach().float()


def tensor_stats(name, t):
    f = t.flatten().double()
    return {
        "name": name,
        "shape": tuple(t.shape),
        "mean": f.mean().item(),
        "std": f.std(unbiased=False).item(),
        "var": f.var(unbiased=False).item(),
        "l2": f.norm(p=2).item(),
        "min": f.min().item(),
        "max": f.max().item(),
    }


def cosine(a, b):
    av = a.flatten().double()
    bv = b.flatten().double()
    return F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item()


def rel_l2(a, b):
    av = a.flatten().double()
    bv = b.flatten().double()
    denom = av.norm(p=2).item()
    if denom == 0:
        denom = 1e-12
    return (av - bv).norm(p=2).item() / denom


def disp_summary(name, disp, max_disp_full):
    """disp is [1,1,H,W] in pixels. max_disp_full = full-res search ceiling."""
    d = disp.flatten().double()
    eps_zero = 0.5    # pixels: 'pinned at d~0' if disparity < 0.5 px
    eps_max = 0.5     # pixels from the ceiling
    n = d.numel()
    pct_near_zero = 100.0 * (d.abs() < eps_zero).sum().item() / n
    pct_near_max = 100.0 * (d > (max_disp_full - eps_max)).sum().item() / n
    return {
        "name": name,
        "median": d.median().item(),
        "mean": d.mean().item(),
        "p10": torch.quantile(d, 0.10).item(),
        "p90": torch.quantile(d, 0.90).item(),
        "min": d.min().item(),
        "max": d.max().item(),
        "pct_near_zero": pct_near_zero,
        "pct_near_maxdisp": pct_near_max,
    }


def save_heatmap(net0, path, title):
    # net0 [1,C,h,w] -> per-pixel L2 magnitude over channels.
    mag = net0[0].pow(2).sum(0).sqrt().cpu().numpy()  # [h,w]
    plt.figure(figsize=(5, 5))
    plt.imshow(mag, cmap="viridis")
    plt.colorbar(label="net[0] channel-L2 magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  torch={torch.__version__}")
    if device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}  id={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    primary = Image.open(PRIMARY_PNG).convert("RGB")
    rightview = Image.open(RIGHTVIEW_PNG).convert("RGB")
    print(f"[img] primary={primary.size}  right_view={rightview.size}")

    ffs, captured = load_ffs(device)
    max_disp_full = int(ffs.args.max_disp)
    print(f"[ffs] loaded. valid_iters={VALID_ITERS} max_disp(full-res)={max_disp_full} "
          f"net0_channels={GRU_HIDDEN_DIM}")

    p = imgs_to_ffs_tensor(primary, device)
    r = imgs_to_ffs_tensor(rightview, device)

    # A=correct(right_view,primary)  B=training(primary,right_view)
    # C=(primary,primary)  D=(right_view,right_view)
    disp_A, net0_A = run_ffs(ffs, captured, r, p)
    disp_B, net0_B = run_ffs(ffs, captured, p, r)
    disp_C, net0_C = run_ffs(ffs, captured, p, p)
    disp_D, net0_D = run_ffs(ffs, captured, r, r)

    print("\n" + "=" * 78)
    print("net[0] PER-TENSOR STATS (the SAME feature training injects)")
    print("=" * 78)
    hdr = f"{'tensor':<6}{'shape':<18}{'mean':>10}{'std':>10}{'L2norm':>12}{'min':>9}{'max':>9}"
    print(hdr)
    for nm, t in [("A", net0_A), ("B", net0_B), ("C", net0_C), ("D", net0_D)]:
        s = tensor_stats(nm, t)
        print(f"{nm:<6}{str(s['shape']):<18}{s['mean']:>10.4f}{s['std']:>10.4f}"
              f"{s['l2']:>12.2f}{s['min']:>9.3f}{s['max']:>9.3f}")
    print("  A=ffs(right_view,primary)=CORRECT  B=ffs(primary,right_view)=TRAINING")
    print("  C=ffs(primary,primary)=degenerate  D=ffs(right_view,right_view)=degenerate")

    print("\n" + "=" * 78)
    print("net[0] PAIRWISE SIMILARITY")
    print("=" * 78)
    pairs = [
        ("B,C", net0_B, net0_C, "training vs degenerate(prim,prim)"),
        ("B,D", net0_B, net0_D, "training vs degenerate(rv,rv)"),
        ("B,A", net0_B, net0_A, "training vs CORRECT stereo"),
        ("A,C", net0_A, net0_C, "correct vs degenerate"),
        ("A,D", net0_A, net0_D, "correct vs degenerate(rv,rv)"),
        ("C,D", net0_C, net0_D, "degenerate vs degenerate"),
    ]
    print(f"{'pair':<8}{'cosine':>10}{'rel_L2':>10}   meaning")
    sims = {}
    for tag, x, y, meaning in pairs:
        cs = cosine(x, y)
        rl = rel_l2(x, y)
        sims[tag] = (cs, rl)
        print(f"{tag:<8}{cs:>10.5f}{rl:>10.5f}   {meaning}")

    print("\n" + "=" * 78)
    print("DISPARITY MAP COMPARISON (test_mode disparity, pixels @ full res)")
    print("=" * 78)
    print(f"{'tensor':<6}{'median':>9}{'mean':>9}{'p10':>8}{'p90':>8}{'min':>9}{'max':>9}"
          f"{'%~0px':>9}{'%~maxd':>9}")
    for nm, d in [("A", disp_A), ("B", disp_B), ("C", disp_C), ("D", disp_D)]:
        s = disp_summary(nm, d, max_disp_full)
        print(f"{nm:<6}{s['median']:>9.3f}{s['mean']:>9.3f}{s['p10']:>8.3f}{s['p90']:>8.3f}"
              f"{s['min']:>9.3f}{s['max']:>9.3f}{s['pct_near_zero']:>9.2f}{s['pct_near_maxdisp']:>9.2f}")

    # heatmaps
    save_heatmap(net0_A, os.path.join(OUT_DIR, "net0_A.png"), "net[0] |mag| A=ffs(right_view,primary) CORRECT")
    save_heatmap(net0_B, os.path.join(OUT_DIR, "net0_B.png"), "net[0] |mag| B=ffs(primary,right_view) TRAINING")
    save_heatmap(net0_C, os.path.join(OUT_DIR, "net0_C.png"), "net[0] |mag| C=ffs(primary,primary) DEGENERATE")
    print(f"\n[saved] {OUT_DIR}/net0_A.png net0_B.png net0_C.png")

    # ---- VERDICT logic ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    bc_cos, bc_rl = sims["B,C"]
    bd_cos, bd_rl = sims["B,D"]
    ba_cos, ba_rl = sims["B,A"]
    ac_cos, ac_rl = sims["A,C"]
    # B is "degenerate" if it's much closer to C/D than to A.
    b_deg_cos_margin = max(bc_cos, bd_cos) - ba_cos
    print(f"  sim(B,C) cos={bc_cos:.4f}  sim(B,D) cos={bd_cos:.4f}  sim(B,A) cos={ba_cos:.4f}")
    print(f"  sim(A,C) cos={ac_cos:.4f}  (how far CORRECT stereo sits from degenerate)")
    print(f"  cosine margin  max(B-vs-degenerate) - (B-vs-correct) = {b_deg_cos_margin:+.4f}")
    dB = disp_summary("B", disp_B, max_disp_full)
    dA = disp_summary("A", disp_A, max_disp_full)
    dC = disp_summary("C", disp_C, max_disp_full)
    print(f"  disp median: A(correct)={dA['median']:.2f}px  B(training)={dB['median']:.2f}px  C(degen)={dC['median']:.2f}px")
    print(f"  disp %~0px : A={dA['pct_near_zero']:.1f}%  B={dB['pct_near_zero']:.1f}%  C={dC['pct_near_zero']:.1f}%")

    verdict = []
    if max(bc_cos, bd_cos) > ba_cos and b_deg_cos_margin > 0.05:
        verdict.append("B (training order) net[0] is CLOSER to the degenerate zero-disparity baselines (C/D) than to the correct stereo A.")
    else:
        verdict.append("B (training order) net[0] is NOT closer to degenerate than to correct stereo A.")
    if dB["pct_near_zero"] > 60 and dB["median"] < 1.0:
        verdict.append("B disparity is pinned at ~0 px over the image -> non-negative search clamped the inverted pair -> feature carries ~no parallax.")
    elif dB["median"] < dA["median"] * 0.5:
        verdict.append("B disparity median is far below A -> training pair under-resolves real parallax.")
    else:
        verdict.append("B disparity still carries substantial magnitude (not fully pinned at 0).")
    for v in verdict:
        print("  - " + v)
    print("=" * 78)


if __name__ == "__main__":
    main()
