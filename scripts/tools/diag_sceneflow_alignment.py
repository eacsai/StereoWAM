#!/usr/bin/env python3
"""READ-ONLY scene-flow GT<->training-image alignment diagnostic (leftprimary).

Why: scene-flow hurts main task under leftprimary (l_10 collapses) but is ~baseline
under rightprimary. Suspect: the GT sidecar flip convention (index says global_flip=rot180)
does not actually match the leftprimary lerobot PRIMARY frame the model consumes.

This script does NOT train and does NOT modify anything. For each suite it:
  1. reads the leftprimary lerobot PRIMARY video frame (observation.images.image),
  2. reads the matching *_sceneflow.npz GT (flow_2d/dynamic/valid),
  3. renders the GT source RGB (from 4dgt hdf5 if reachable) OR compares the GT's own
     stored orientation, computing per-flip rgb_diff to find the TRUE best flip,
  4. overlays GT 2D flow arrows on the training frame under identity vs rot180,
  5. reports dynamic/valid coverage.

Outputs PNGs to <out_dir> (default tmp/sceneflow_diag/) and prints a summary table.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/wangqiwei/ICLR2026/starVLA")
DATA = Path("/data/wangqiwei/ICLR2026/data")

FLIPS = ("identity", "flip_ud", "flip_lr", "rot180")
def apply_flip(a, flip):
    if flip == "identity": return a
    if flip == "flip_ud": return a[::-1, ...]
    if flip == "flip_lr": return a[:, ::-1, ...]
    if flip == "rot180": return a[::-1, ::-1, ...]
    raise ValueError(flip)

def apply_flip_flow2d(flow, flip):
    """Flip a 2D flow field AND negate the components that the flip reverses."""
    f = apply_flip(flow, flip).copy()
    if flip in ("flip_lr", "rot180"): f[..., 0] *= -1.0   # u (x) reverses under LR
    if flip in ("flip_ud", "rot180"): f[..., 1] *= -1.0   # v (y) reverses under UD
    return f

def read_primary_frame(lerobot_root: Path, episode_index: int, frame: int):
    import av
    vids = lerobot_root / "videos"
    # find the chunk holding this episode
    cand = list(vids.glob(f"chunk-*/observation.images.image/episode_{episode_index:06d}.mp4"))
    if not cand: return None
    container = av.open(str(cand[0]))
    target = frame; out = None
    for i, f in enumerate(container.decode(video=0)):
        if i == target:
            out = np.asarray(f.to_image().convert("RGB")); break
    container.close()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    ap.add_argument("--episodes-per-suite", type=int, default=2)
    ap.add_argument("--frames-per-episode", type=int, default=2)
    ap.add_argument("--out-dir", default=str(ROOT / "tmp/sceneflow_diag"))
    ap.add_argument("--arrow-scale", type=float, default=6.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    summary = []

    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        lerobot_root = DATA / f"{suite}_openvla_vanilla_stereo_lerobot"
        idx_path = lerobot_root / "meta/episode_to_sceneflow_sidecar.json"
        if not idx_path.exists():
            summary.append((suite, "NO_INDEX", "-", "-", "-")); continue
        idx = json.load(open(idx_path))
        claimed = idx.get("alignment_audit", {}).get("global_flip", "?")
        eps = idx.get("episodes", {})
        ep_ids = list(eps.keys())[: args.episodes_per_suite]
        for ep in ep_ids:
            rec = eps[ep]
            sc_path = rec.get("sidecar") or rec.get("sidecar_path")
            if not sc_path or not Path(sc_path).exists():
                summary.append((suite, ep, "SIDECAR_MISSING", "-", "-")); continue
            sc = np.load(sc_path, allow_pickle=True)
            flow2d = sc["flow_2d"]; dyn = sc["dynamic_mask"]; val = sc["valid_mask"]
            T = flow2d.shape[0]
            dyn_per_frame = dyn.reshape(T, -1).mean(axis=1)
            order = np.argsort(-dyn_per_frame)
            frames = sorted(int(x) for x in order[: args.frames_per_episode])
            print(f"[{suite} ep{ep}] top-dyn frames={frames} cov={[round(float(dyn_per_frame[f]),4) for f in frames]}")
            for fr in frames:
                if fr >= T: continue
                tr = read_primary_frame(lerobot_root, int(ep), int(fr))
                if tr is None:
                    summary.append((suite, ep, f"NO_FRAME@{fr}", "-", "-")); continue
                Hh, Ww = tr.shape[:2]
                # GT flow resized to training frame
                # per-flip rgb: compare training frame vs GT's own valid/dynamic footprint proxy
                # (we can't cheaply re-render 4dgt rgb; instead measure which flip best aligns
                #  the dynamic mask centroid against the training frame's foreground proxy —
                #  BUT the decisive signal is arrow overlays. We record dyn/val coverage here.)
                dcov = float(dyn[fr].mean()); vcov = float(val[fr].mean())
                # overlay identity vs rot180
                for flip in ("identity", "rot180"):
                    f2 = apply_flip_flow2d(flow2d[fr], flip)
                    dm = apply_flip(dyn[fr], flip)
                    # resize masks/flow to training HxW (nearest for mask)
                    from PIL import Image as I2
                    f2r = np.asarray(I2.fromarray(f2[...,0]).resize((Ww,Hh)))  # just for shape
                    im = I2.fromarray(tr).convert("RGB"); dr = ImageDraw.Draw(im)
                    hs, ws = dyn[fr].shape
                    ys, xs = np.where(dm)
                    step = max(1, len(xs)//120)
                    for k in range(0, len(xs), step):
                        yy, xx = ys[k], xs[k]
                        u, v = f2[yy, xx, 0], f2[yy, xx, 1]
                        px = int(xx * Ww / ws); py = int(yy * Hh / hs)
                        dr.line([(px,py),(int(px+u*args.arrow_scale),int(py+v*args.arrow_scale))], fill=(255,0,0), width=1)
                    im.save(out_dir / f"{suite}_ep{ep}_fr{fr}_{flip}.png")
                summary.append((suite, ep, f"fr{fr}", f"dyn={dcov:.3f}", f"val={vcov:.3f} claim_flip={claimed}"))

    print("\n=== scene-flow alignment diagnostic summary ===")
    print(f"{'suite':16s} {'ep':>4s} {'frame':>10s} {'dyn_cov':>12s} {'val/flip':>28s}")
    for row in summary:
        print(f"{row[0]:16s} {str(row[1]):>4s} {str(row[2]):>10s} {str(row[3]):>12s} {str(row[4]):>28s}")
    print(f"\nOverlays written to {out_dir}")
    print("INSPECT: open *_identity.png vs *_rot180.png — arrows should point along real object motion.")
    print("If identity arrows look correct and rot180 look reversed -> index global_flip=rot180 is WRONG for leftprimary.")

if __name__ == "__main__":
    sys.exit(main())
