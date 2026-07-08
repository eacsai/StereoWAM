#!/usr/bin/env bash
# Diagnostic: dump the LIVE eval-time orthogonal-grid (田字格) images the orthogrid
# ckpt actually consumed during LIBERO rollouts, for visual inspection and comparison
# against the training sqlite cache. Runs a TINY eval (1 task x 1 trial per suite) with
# ORTHO_DUMP_DIR set so the policy server saves each rendered grid + the t=0 stereo pair.
# READ-ONLY w.r.t. training. Grids land under the run dir (project, not /tmp).
set -uo pipefail
STARVLA=/data/wangqiwei/ICLR2026/starVLA
RID=qwen0p8_groot_stereo_orthogrid_leftprimary_fromscratch_fullft_eff128_maskfix_30k
CKPT=$STARVLA/playground/Checkpoints/$RID/checkpoints/steps_30000_pytorch_model.pt
DUMP=$STARVLA/playground/Checkpoints/$RID/ortho_grid_dumps
CACHE=$STARVLA/playground/Caches/ortho_views_leftprimary_probe_v1
[ -f "$CKPT" ] || { echo "[FATAL] ckpt missing: $CKPT"; exit 1; }
rm -rf "$DUMP"; mkdir -p "$DUMP"
export ORTHO_DUMP_DIR="$DUMP" ORTHO_DUMP_CAP="${ORTHO_DUMP_CAP:-60}"
export EVAL_MAX_TASKS=1 EVAL_NUM_TRIALS=1

run_suite () {  # suite gpu port
  local S=$1 G=$2 P=$3 ED=$DUMP/_eval_$1
  rm -rf "$ED"; mkdir -p "$ED"
  echo "[dump] $S on GPU$G port$P -> grids in $DUMP/$S"
  bash "$STARVLA/scripts/4090d/eval_one_ckpt.sh" "$CKPT" "$G" "$P" "$ED" "$S" primary,left_view 1 0 > "$ED/lane.log" 2>&1
  echo "[dump] $S lane rc=$? grids=$(ls "$DUMP/$S"/grid_*.png 2>/dev/null | wc -l)"
}

# wave 1: 3 suites in parallel on the 3 free GPUs (0,2,6)
run_suite libero_spatial 0 6810 &
run_suite libero_object  2 6811 &
run_suite libero_goal    6 6812 &
wait
# wave 2: libero_10 on GPU0 (freed after spatial)
run_suite libero_10 0 6813

# extract training cache grids (row 0 / mid / last) per suite for side-by-side compare
"$STARVLA/.venv/bin/python" - "$CACHE" "$DUMP" <<'PY'
import sys, io, sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
cache, dump = Path(sys.argv[1]), Path(sys.argv[2])
for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    key = next((p.name for p in sorted(cache.iterdir()) if p.name.startswith(s)), None)
    if not key:
        print(f"[cache] no dir for {s}"); continue
    db = sqlite3.connect(f"file:{cache/key/'ortho_views.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in db.execute("SELECT row FROM ortho_views ORDER BY row")]
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, 3).astype(int)]
    outd = dump / s; outd.mkdir(parents=True, exist_ok=True)
    for r in picks:
        blob = db.execute("SELECT grid_png FROM ortho_views WHERE row=?", (int(r),)).fetchone()[0]
        Image.open(io.BytesIO(blob)).convert("RGB").save(outd / f"cache_row{int(r)}.png")
    db.close()
    print(f"[cache] {s}: wrote {len(picks)} cache grids from {key}")
PY
echo "ALL_DONE dump=$DUMP"
ls -R "$DUMP" | head -80
