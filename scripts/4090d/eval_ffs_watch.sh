#!/usr/bin/env bash
# Order-independent multi-run eval watcher (runs ON 4090d). 谁先到目标 step 先评, 不互相阻塞。
# 每个 (run, step) 独立: ckpt 在 h100b ready + 4090d 有真正空闲卡(used<FREE_MB) → 评 → 记 SR。
# 区分 FFS run(需 ffs_model_path 覆盖 → eval_ffs_4suite.sh) vs 非 FFS 对照(plain → eval_qwen2p5vl_4suite.sh)。
# 已评过的(results.txt 有该 run+step 头)跳过 → 可重启幂等。fail-closed: 非零退出不记录。
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA

H100B="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
FREE_MB=${FREE_MB:-40000}
NEED_GPUS=${NEED_GPUS:-2}
LOG=playground/Checkpoints/eval_ffs_watch.log
RESULTS=playground/Checkpoints/eval_ffs_results.txt
log(){ echo "[ffsevalwatch $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

# run_id | steps | kind(ffs|plain)
# ⭐ 2026-06-08 eval-mode-fix + #2 softmax-only 重跑集 (P1-P6, 全 FFS, 评 20k+30k)。
# 旧 buggy 结果已归档到 eval_ffs_results.prebugfix.txt; 此处 run_id 同名 → 新 ckpt 重评。
TARGETS=(
  "qwen3p5_0p8b_ffs_vlminput_fromscratch_fullinject_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_vlminput_warmstartB_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_controlnet_warmstartB_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_vlmcontrolnet_warmstartB_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_controlnet_warmstartB_camfrozen_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_vlmcontrolnet_warmstartB_camfrozen_30k|20000 30000|ffs"
  # 2026-06-09 #4 injection-modality ablation (from-scratch, 64 depth tokens):
  "qwen3p5_0p8b_ffs_depthtoken_keep_fromscratch_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_depthtoken_strip_fromscratch_30k|20000 30000|ffs"
  "qwen3p5_0p8b_ffs_depthimage_fromscratch_30k|20000 30000|ffs"
)

pick_free_gpus(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v t="$FREE_MB" '$2<t{print $2" "$1}' | sort -n | head -"$NEED_GPUS" | awk '{print $2}' | tr '\n' ' '; }
ckpt_ready(){ local sz; sz=$($H100B "stat -c %s ${H100B_CKPT}/$1/checkpoints/steps_$2_pytorch_model.pt" 2>/dev/null | tr -dc 0-9); [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 1000000 ]; }
# Dedup requires the COMPLETE block (header anchored at line start + final suite line):
# a watcher killed mid-record must NOT leave a truncated block that reads as "done".
already_done(){ grep -A4 -E "^===== ${1} step=${2} " "$RESULTS" 2>/dev/null | grep -q "libero_10: SR="; }

record(){ local rid="$1" step="$2"
  local edir="playground/Checkpoints/${rid}/eval_step${step}_right_view_primary"
  # Build the whole block first, then append atomically (single write): a kill
  # between header and suite lines would otherwise permanently truncate the record.
  local block s sr
  block="===== ${rid} step=${step} ($(date -u)) ====="$'\n'
  for s in libero_spatial libero_object libero_goal libero_10; do
    sr=$(grep -iE "Total success rate" "${edir}/${s}/client.log" 2>/dev/null | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    block+="  ${s}: SR=${sr:-NA}"$'\n'
  done
  printf '%s' "$block" >> "$RESULTS"
  printf '%s' "$block"
}

# Per-target failure budget: a structurally-broken target (e.g. config mismatch)
# must not spin the watcher forever. After GIVEUP_AFTER non-zero evals the target
# is skipped for this watcher's lifetime; restart the watcher to retry it.
declare -A FAILCOUNT
GIVEUP_AFTER=${GIVEUP_AFTER:-3}

log "=== order-independent eval watcher 启动 (FREE_MB=$FREE_MB NEED_GPUS=$NEED_GPUS; $((${#TARGETS[@]})) targets) ==="
while true; do
  remaining=0
  gaveup=0
  for row in "${TARGETS[@]}"; do
    IFS='|' read -r rid steps kind <<< "$row"
    for step in $steps; do
      already_done "$rid" "$step" && continue
      key="${rid}|${step}"
      [ "${FAILCOUNT[$key]:-0}" -ge "$GIVEUP_AFTER" ] && { gaveup=$((gaveup + 1)); continue; }
      remaining=1
      ckpt_ready "$rid" "$step" || continue          # ckpt 没好, 下轮再看
      gpus=$(pick_free_gpus); [ "$(echo $gpus | wc -w)" -ge "$NEED_GPUS" ] || { log "$rid step$step ready 但无 $NEED_GPUS 空闲卡(现 '$gpus'), 等"; break 2; }
      log "$rid step$step ($kind) → GPUs [$gpus] eval"
      if [ "$kind" = "ffs" ]; then
        ok=0; bash scripts/4090d/eval_ffs_4suite.sh "$rid" "$step" "$gpus" >> "$LOG" 2>&1 && ok=1
      else
        ok=0; bash scripts/4090d/eval_qwen2p5vl_4suite.sh "$rid" "$step" right_view,primary "$gpus" >> "$LOG" 2>&1 && ok=1
      fi
      if [ "$ok" = 1 ]; then
        record "$rid" "$step"; log "EVAL_DONE $rid step$step"
      else
        FAILCOUNT[$key]=$(( ${FAILCOUNT[$key]:-0} + 1 ))
        if [ "${FAILCOUNT[$key]}" -ge "$GIVEUP_AFTER" ]; then
          log "FATAL eval 非零 $rid step$step 已失败 ${FAILCOUNT[$key]} 次 — 放弃该 target (重启 watcher 可重试)"
        else
          log "WARN eval 非零 $rid step$step (未记录, fail ${FAILCOUNT[$key]}/${GIVEUP_AFTER})"
        fi
      fi
    done
  done
  if [ "$remaining" = 0 ]; then
    if [ "$gaveup" -gt 0 ]; then
      log "=== watcher 退出: ${gaveup} 个 target 被放弃 (达到 GIVEUP_AFTER, 结果缺失!) — 重启 watcher 可重试 ==="
      exit 1
    fi
    log "=== 所有 target 已评, watcher 退出 ==="
    break
  fi
  sleep 120
done
