#!/usr/bin/env python3
"""READ-ONLY: empirically determine the CORRECT dx/dy sign-flip rule for rot180 GT flow.

Principle: scene flow at frame t describes t->t+1 motion. If we take frame t's grayscale,
warp each dynamic pixel FORWARD by its (dx,dy), the warped image should look more like
frame t+1 than frame t does — but ONLY if the flow direction is correct. We test the 4
sign combos on the rot180-aligned flow and score each by how much the forward-warp reduces
|warped_t - (t+1)| vs the static |t-(t+1)| baseline over dynamic pixels. Best combo wins.

No writes. Prints a table. This pins the axis/sign rule before we touch dataloader code.
"""
import numpy as np, json, av
from pathlib import Path

DATA = Path("/data/wangqiwei/ICLR2026/data")

def rot180(a):  # spatial rot180 (pixel reorder), matches _apply_spatial_alignment
    return a[::-1, ::-1, ...]

def read_gray(root, ep, frs):
    cand = list((root / "videos").glob("chunk-*/observation.images.image/episode_%06d.mp4" % ep))
    if not cand: return None
    c = av.open(str(cand[0])); want = set(frs); got = {}
    for i, fr in enumerate(c.decode(video=0)):
        if i in want: got[i] = np.asarray(fr.to_image().convert("L"), dtype=np.float32)
        if len(got) == len(want): break
    c.close(); return got

def score_combo(gray_t, gray_t1, flow_xy, dyn, sx, sy):
    """Forward-warp gray_t by (sx*dx, sy*dy) on dynamic pixels; return residual reduction.
    flow_xy: HxWx2 already rot180-spatial-aligned to training frame. dyn: HxW bool same grid.
    Positive return => this sign combo makes warped_t closer to t+1 (i.e. correct direction)."""
    H, W = gray_t.shape
    hs, ws = dyn.shape
    # resize flow & mask to gray HxW (nearest)
    yi = (np.arange(H) * hs / H).astype(int); xi = (np.arange(W) * ws / W).astype(int)
    fx = flow_xy[np.ix_(yi, xi, [0])][..., 0] * sx
    fy = flow_xy[np.ix_(yi, xi, [1])][..., 0] * sy
    dm = dyn[np.ix_(yi, xi)]
    # flow units: flow_3d is in meters (camera frame), tiny; scale to pixels heuristically.
    # We only care about DIRECTION, so normalize each vector to a fixed small pixel step.
    mag = np.sqrt(fx**2 + fy**2) + 1e-9
    STEP = 6.0
    ux = (fx / mag) * STEP; uy = (fy / mag) * STEP
    ys, xs = np.where(dm)
    if len(xs) == 0: return None
    base = 0.0; warped = 0.0; n = 0
    for k in range(0, len(xs), max(1, len(xs)//400)):
        y, x = ys[k], xs[k]
        nx = int(round(x + ux[y, x])); ny = int(round(y + uy[y, x]))
        if not (0 <= nx < W and 0 <= ny < H): continue
        # static residual at source vs t+1; warped residual: t's intensity moved to (nx,ny) vs t+1 there
        base += abs(gray_t[y, x] - gray_t1[y, x])
        warped += abs(gray_t[y, x] - gray_t1[ny, nx])
        n += 1
    if n == 0: return None
    return (base - warped) / n   # >0 means forward-warp along this sign matches t+1 motion

def main():
    combos = {"none(+dx,+dy)": (1, 1), "dx_only(-dx,+dy)": (-1, 1),
              "dy_only(+dx,-dy)": (1, -1), "both/rot180(-dx,-dy)": (-1, -1)}
    print("suite            fr   residual-reduction per sign-combo  [HIGH=correct direction]")
    agg = {k: [] for k in combos}
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        root = DATA / ("%s_openvla_vanilla_stereo_lerobot" % suite)
        idx = json.load(open(root / "meta/episode_to_sceneflow_sidecar.json"))
        eps = idx["episodes"]
        for ek in list(eps.keys())[:2]:
            rec = eps[ek]; sc = np.load(rec.get("sidecar") or rec.get("sidecar_path"), allow_pickle=True)
            flow3d = sc["flow_3d"]; dyn = sc["dynamic_mask"]; T = dyn.shape[0]
            cov = dyn.reshape(T, -1).mean(1); order = np.argsort(-cov)
            for t in [int(order[0])]:
                if t + 1 >= T: continue
                gr = read_gray(root, int(ek), [t, t + 1])
                if not gr or t not in gr or t + 1 not in gr: continue
                # rot180 spatial-align flow xy (dx,dy) and mask to training-frame orientation
                fxy = rot180(flow3d[t][..., :2])
                dm = rot180(dyn[t])
                line = "%-14s %4d " % (suite, t)
                for name, (sx, sy) in combos.items():
                    s = score_combo(gr[t], gr[t + 1], fxy, dm, sx, sy)
                    if s is not None: agg[name].append(s)
                    line += " %s=%s" % (name.split("(")[0], ("%.2f" % s) if s is not None else "NA")
                print(line)
    print("\n=== mean residual-reduction across all frames (HIGHEST = correct sign rule) ===")
    for k in sorted(agg, key=lambda x: -(np.mean(agg[x]) if agg[x] else -1e9)):
        v = np.mean(agg[k]) if agg[k] else float("nan")
        print("  %-22s mean=%.3f  (n=%d)" % (k, v, len(agg[k])))
    print("\nNOTE: flow_3d is meters/camera-frame; we normalized to direction-only, so magnitudes")
    print("are heuristic. The RANKING of sign combos is what matters, not absolute values.")

if __name__ == "__main__":
    main()
