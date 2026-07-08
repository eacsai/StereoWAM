#!/usr/bin/env python3
"""READ-ONLY correct visualization of scene-flow GT on leftprimary training frames.

Uses the AUTHORITATIVE rules from gen_gt_sceneflow_rerender.py:
  - flow_2d is image-plane pixel displacement (analytic_projection_displacement).
  - geometry_to_training_flow2d(flip): spatial reorder + sign flip (lr->u, ud->v).
    rot180 => flip BOTH u and v signs.
Draws GT flow_2d arrows on the actual leftprimary PRIMARY training frame at the
most-dynamic frames. Also draws a t->t+1 endpoint check: arrow tail at (x,y) on
frame t, head at (x+u, y+v); overlaid on frame t+1 the head should land on the
object's NEW position if the GT is correct. No writes except PNGs to out_dir.
"""
import argparse, json
from pathlib import Path
import numpy as np

DATA = Path("/data/wangqiwei/ICLR2026/data")
FLIP_BITS = {"identity": (False, False), "flip_ud": (True, False),
             "flip_lr": (False, True), "rot180": (True, True)}

def apply_flip(a, flip):
    if flip == "identity": return a
    if flip == "flip_ud": return a[::-1, ...]
    if flip == "flip_lr": return a[:, ::-1, ...]
    if flip == "rot180": return a[::-1, ::-1, ...]
    raise ValueError(flip)

def geometry_to_training_flow2d(flow, flip):
    """EXACT copy of generator rule."""
    out = apply_flip(flow, flip).copy()
    ud, lr = FLIP_BITS[flip]
    if lr: out[..., 0] *= -1.0
    if ud: out[..., 1] *= -1.0
    return out

def geometry_to_training_mask(mask, flip):
    return apply_flip(mask, flip)

def read_frames(root, ep, frs):
    import av
    cand = list((root / "videos").glob("chunk-*/observation.images.image/episode_%06d.mp4" % ep))
    if not cand: return None
    c = av.open(str(cand[0])); want = set(frs); got = {}
    for i, fr in enumerate(c.decode(video=0)):
        if i in want: got[i] = np.asarray(fr.to_image().convert("RGB"))
        if len(got) == len(want): break
    c.close(); return got

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    ap.add_argument("--out-dir", default="/data/wangqiwei/ICLR2026/starVLA/tmp/sceneflow_viz_correct")
    ap.add_argument("--arrow-scale", type=float, default=4.0)
    ap.add_argument("--max-arrows", type=int, default=150)
    args = ap.parse_args()
    from PIL import Image, ImageDraw
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        root = DATA / ("%s_openvla_vanilla_stereo_lerobot" % suite)
        idx = json.load(open(root / "meta/episode_to_sceneflow_sidecar.json"))
        eps = idx["episodes"]; ek = list(eps.keys())[0]
        rec = eps[ek]; flip = rec.get("sidecar_to_training_flip", idx.get("alignment_audit", {}).get("global_flip", "rot180"))
        sc = np.load(rec.get("sidecar") or rec.get("sidecar_path"), allow_pickle=True)
        if "flow_2d" not in sc.files:
            print("%s: sidecar has no flow_2d, skip" % suite); continue
        f2 = sc["flow_2d"]; dyn = sc["dynamic_mask"]; T = dyn.shape[0]
        t = int(np.argmax(dyn.reshape(T, -1).mean(1)))
        if t + 1 >= T: t = max(0, T - 2)
        fr = read_frames(root, int(ek), [t, t + 1])
        if not fr: print("%s: no frames" % suite); continue

        f2t = geometry_to_training_flow2d(f2[t], flip)     # -> training orientation
        dmt = geometry_to_training_mask(dyn[t], flip)
        H, W = fr[t].shape[:2]; hs, ws = dmt.shape
        ys, xs = np.where(dmt)
        step = max(1, len(xs) // args.max_arrows)

        # panel A: arrows on frame t (should point along motion direction)
        imA = Image.fromarray(fr[t]).convert("RGB"); dA = ImageDraw.Draw(imA)
        # panel B: same arrows drawn on frame t+1 (heads should land on object's new pos)
        imB = Image.fromarray(fr[t + 1]).convert("RGB"); dB = ImageDraw.Draw(imB)
        for k in range(0, len(xs), step):
            yy, xx = ys[k], xs[k]
            u = float(f2t[yy, xx, 0]); v = float(f2t[yy, xx, 1])
            px = int(xx * W / ws); py = int(yy * H / hs)
            hx = int(px + u * args.arrow_scale); hy = int(py + v * args.arrow_scale)
            dA.line([(px, py), (hx, hy)], fill=(255, 0, 0), width=1)
            dA.ellipse([px-1, py-1, px+1, py+1], fill=(0, 255, 0))          # tail=green
            dB.ellipse([hx-2, hy-2, hx+2, hy+2], outline=(255, 0, 0), width=1)  # head on t+1
        imA.save(out / ("%s_t%d_A_arrows_on_t.png" % (suite, t)))
        imB.save(out / ("%s_t%d_B_heads_on_t+1.png" % (suite, t)))
        print("%-14s t=%d flip=%s dyn_px=%d  wrote A(arrows on t) + B(heads on t+1)" % (suite, t, flip, len(xs)))
    print("\nOut:", out)
    print("READ: A = arrows should point along the moving object's motion.")
    print("      B = red circles (arrow HEADS) should land on where the object MOVED TO in t+1.")

if __name__ == "__main__":
    main()
