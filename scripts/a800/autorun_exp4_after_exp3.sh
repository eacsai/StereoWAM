#!/usr/bin/env bash
# Autonomous serial chain on a800b (single A800): wait for ③ utonia eff128 to finish,
# then GPU-smoke + run ④ sceneflow eff128 (DepthTokenFFS KEEP, from-scratch full-ft, Fix#1).
# Survives operator disconnect (run in tmux). Idempotent: skips ④ if its 30k ckpt exists;
# aborts if ③ died without a ckpt. Author: 4090d (authority); synced to a800b; run on a800b.
set -uo pipefail
cd /home/wangqiwei/ICLR2026/starVLA

CKPTS=playground/Checkpoints
EXP3=qwen3p5_0p8b_utonia_prompttoken_leftprimary_fromscratch_eff128_maskfix_30k
EXP4=qwen0p8_groot_depthtoken_keep_sceneflow_all4_lambda130_leftprimary_fromscratch_fullft_eff128_maskfix_30k
LOG=tmp/autorun_exp4.log
mkdir -p tmp

ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
say(){ echo "[autorun-exp4 $(ts)] $*" | tee -a "$LOG"; }

exp3_done(){ [ -f "$CKPTS/$EXP3/checkpoints/steps_30000_pytorch_model.pt" ]; }
exp4_done(){ [ -f "$CKPTS/$EXP4/checkpoints/steps_30000_pytorch_model.pt" ]; }

say "=== chain start ==="
if exp4_done; then say "④ already has steps_30000 -> nothing to do"; exit 0; fi

# 1) wait for ③ (poll 5 min); abort if ③ tmux gone AND no ckpt = ③ failed
say "waiting for ③ ($EXP3) steps_30000 ..."
while ! exp3_done; do
  if ! tmux has-session -t exp3_eff128_maskfix 2>/dev/null; then
    say "ABORT: ③ tmux gone but no steps_30000 ckpt — ③ likely failed. NOT starting ④."
    exit 1
  fi
  sleep 300
done
say "③ DONE (steps_30000 present). proceeding to ④."

# ④ env (from-scratch full-ft override of the warmstart+frozen launcher defaults)
EXP4_ENV=(
  FRAMEWORK=QwenGR00T_DepthTokenFFS GPUS=0
  DATA_MIX=libero_all_sfstereo_leftprimary
  BS=16 DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml
  SAVE_INTERVAL=10000 PRETRAINED_CKPT= FREEZE_MODULES= CAM_ROPE=0
  STRIP_DEPTH=0 NUM_DEPTH_TOKENS=64 POOL_HW=8
  SCENE_FLOW=1 FLOW_LAMBDA=130 FLOW_ONLINE_GRADNORM=0 FLOW_GRID=16 FLOW_MASK=dynamic
  FLOW_GT_ONLY=1 FLOW_EXPECTED_FLIP=rot180 FLOW_STEP0_AUDIT=1 FLOW_STEP0_WARMUP=1
)

# 2) ④ GPU smoke (real deepspeed, 2 steps, BS16) — gates the 30k
say "④ GPU smoke (MAX_STEPS=2, real bf16) ..."
if ! env "${EXP4_ENV[@]}" MAX_STEPS=2 RUN_ID=${EXP4}_smoke bash scripts/a800/run_qwen0p8_groot_ffs.sh > tmp/exp4_smoke.log 2>&1; then
  say "ABORT: ④ smoke FAILED (see tmp/exp4_smoke.log)"
  exit 1
fi
say "④ smoke OK -> 30k full run"

# 3) ④ full 30k
env "${EXP4_ENV[@]}" MAX_STEPS=30000 RUN_ID=$EXP4 bash scripts/a800/run_qwen0p8_groot_ffs.sh > tmp/exp4_full.log 2>&1
if exp4_done; then say "=== ④ 30k DONE ==="; else say "WARN: ④ full ended but no steps_30000 ckpt — check tmp/exp4_full.log"; fi
say "=== chain end ==="
