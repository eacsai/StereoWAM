#!/usr/bin/env python3
"""Spot-check: REGENERATED FFS net0 cache uses the FIXED stereo order.

Compares pooled net[0] of rows that are 'done' in BOTH the freshly-regenerated
cache and the .confounded old cache. Row order is pinned identically by
all_steps_sha256, so row N == same sample in both. The order fix changes the
disparity-bearing GRU hidden state, so the two should DIFFER substantially
(cosine well below 1.0). Cosine ~1.0 would mean the regen did NOT change order.

Usage: python ffs_regen_spotcheck.py [SUITE]
"""
import os, sys, pickle
import numpy as np

NEW = "/data/wangqiwei/ICLR2026/ffs_net0_cache_4090d"
OLD = "/data/wangqiwei/ICLR2026/ffs_net0_cache_4090d.confounded"
SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_object_no_noops_1.0.0_lerobot"


def load_cache(root):
    d = os.path.join(root, SUITE)
    net0 = np.load(os.path.join(d, "net0_pooled.f32.npy"), mmap_mode="r")
    done = np.load(os.path.join(d, "done.npy"), mmap_mode="r")
    return net0, done


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float("nan") if (na == 0 or nb == 0) else float(a.dot(b) / (na * nb))


def rel_l2(ref, other):
    ref = ref.astype(np.float64).ravel(); other = other.astype(np.float64).ravel()
    d = np.linalg.norm(ref)
    return float(np.linalg.norm(ref - other) / (d if d else 1e-12))


print(f"suite={SUITE}")
new_net0, new_done = load_cache(NEW)
old_net0, old_done = load_cache(OLD)
print(f"new net0 shape={new_net0.shape}  confounded shape={old_net0.shape}")
assert new_net0.shape == old_net0.shape, "shape mismatch new vs confounded"
both = np.where(np.asarray(new_done) & np.asarray(old_done))[0]
print(f"rows done in BOTH = {both.size}")
rows = both[:8].tolist()
if not rows:
    print("(no overlapping done rows yet; rerun shortly)"); sys.exit(0)
cosines = []
for r in rows:
    a = np.asarray(new_net0[r]); b = np.asarray(old_net0[r])
    c = cosine(a, b); cosines.append(c)
    v = "DIFFERS (good)" if c < 0.999 else "IDENTICAL (BAD!)"
    print(f"  row {r:6d}: cosine(new,confounded)={c:+.5f}  relL2={rel_l2(b,a):.4f}  -> {v}")
print(f"mean cosine over {len(rows)} rows = {np.mean(cosines):+.5f}")
print("EXPECT cosine << 1.0  (order fix changed the disparity-bearing net[0]).")
