#!/usr/bin/env python3
"""READ-ONLY objective flip test: correct flip => GT dynamic_mask overlaps the REAL
inter-frame image change region. No training, no writes (prints only)."""
import numpy as np, json, av
from pathlib import Path

DATA = Path("/data/wangqiwei/ICLR2026/data")
FLIPS = ("identity", "flip_ud", "flip_lr", "rot180")

def fm(a, f):
    if f == "identity": return a
    if f == "flip_ud": return a[::-1]
    if f == "flip_lr": return a[:, ::-1]
    if f == "rot180": return a[::-1, ::-1]
    raise ValueError(f)

def read_frames(root, ep, frs):
    cand = list((root / "videos").glob("chunk-*/observation.images.image/episode_%06d.mp4" % ep))
    if not cand: return None
    c = av.open(str(cand[0])); want = set(frs); got = {}
    for i, fr in enumerate(c.decode(video=0)):
        if i in want: got[i] = np.asarray(fr.to_image().convert("L"), dtype=np.float32)
        if len(got) == len(want): break
    c.close(); return got

def main():
    print("suite            fr   overlap(dynamic_mask, real_change) per flip   [HIGH=correct]")
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        root = DATA / ("%s_openvla_vanilla_stereo_lerobot" % suite)
        idx = json.load(open(root / "meta/episode_to_sceneflow_sidecar.json"))
        eps = idx["episodes"]; ek = list(eps.keys())[0]
        rec = eps[ek]; sc_path = rec.get("sidecar") or rec.get("sidecar_path")
        sc = np.load(sc_path, allow_pickle=True)
        dyn = sc["dynamic_mask"]; T = dyn.shape[0]
        cov = dyn.reshape(T, -1).mean(1); order = np.argsort(-cov)
        frs = [int(order[0]), int(order[1])]
        need = sorted(set(frs) | {f + 1 for f in frs})
        fr = read_frames(root, int(ek), need)
        if not fr:
            print("%-14s no-frames" % suite); continue
        for t in frs:
            if t not in fr or (t + 1) not in fr: continue
            chg = np.abs(fr[t + 1] - fr[t]); H, W = chg.shape
            change = chg >= max(np.percentile(chg, 90), 8.0)
            hs, ws = dyn[t].shape
            yi = (np.arange(H) * hs / H).astype(int); xi = (np.arange(W) * ws / W).astype(int)
            row = []
            for fl in FLIPS:
                dm = fm(dyn[t], fl)[np.ix_(yi, xi)]
                frac = float((dm & change).sum()) / float(dm.sum() + 1e-6)
                row.append((fl, frac))
            best = max(row, key=lambda x: x[1])[0]
            print("%-14s %4d  %s  -> BEST=%s" % (
                suite, t, "  ".join("%s=%.3f" % (fl, v) for fl, v in row), best))

if __name__ == "__main__":
    main()
