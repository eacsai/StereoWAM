#!/usr/bin/env python3
"""Make eval_one_ckpt.sh's server-ready wait env-overridable (SERVER_READY_ITERS, default 90 = unchanged).
Lets slow model init under GPU neighbor contention finish instead of FATAL-exiting at the fixed 180s.
Default-preserving (90 iters x 2s = 180s when unset). Atomic + .bak."""
import os, shutil, tempfile
F = "/data/wangqiwei/ICLR2026/starVLA/scripts/4090d/eval_one_ckpt.sh"
src = open(F).read()
if "SERVER_READY_ITERS" in src:
    print("ALREADY_PATCHED -> no-op"); raise SystemExit(0)
old = "for i in $(seq 1 90); do"
assert src.count(old) == 1, f"expected 1x ready-loop, found {src.count(old)}"
new = ('SERVER_READY_ITERS="${SERVER_READY_ITERS:-90}"  # 90*2s=180s default; raise under GPU contention\n'
       'for i in $(seq 1 "$SERVER_READY_ITERS"); do')
src = src.replace(old, new)
shutil.copy(F, F + ".bak_readytimeout")
d = os.path.dirname(F); fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
with os.fdopen(fd, "w") as fh: fh.write(src)
os.replace(tmp, F)
print("READYTIMEOUT_OK")
