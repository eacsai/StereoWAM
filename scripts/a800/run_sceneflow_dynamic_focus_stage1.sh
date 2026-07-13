#!/bin/bash
# Stage1-only scene-flow prediction experiment: dynamic cells first, static-zero regularized.
# Built on 4090d authoritative repo. This wrapper intentionally does NOT launch or chain stage2.
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export REPO_DIR

export STAGE=${STAGE:-learn_scene_flow}
export CAM_BRANCH=${CAM_BRANCH:-1}
export PAST_FLOW=${PAST_FLOW:-1}
export PAST_FLOW_FULL_INJECT=${PAST_FLOW_FULL_INJECT:-0}
export PAST_FLOW_K=${PAST_FLOW_K:-2}
export PAST_FLOW_DELTA=${PAST_FLOW_DELTA:-1}
export PAST_FLOW_DROPOUT=${PAST_FLOW_DROPOUT:-0.3}
export PAST_FLOW_NOISE=${PAST_FLOW_NOISE:-0.0}

export DYNAMIC_LOSS_WEIGHT=${DYNAMIC_LOSS_WEIGHT:-1.0}
export STATIC_ZERO_LOSS_WEIGHT=${STATIC_ZERO_LOSS_WEIGHT:-0.05}
export DYNAMIC_DIRECTION_LOSS_WEIGHT=${DYNAMIC_DIRECTION_LOSS_WEIGHT:-0.2}
export DYNAMIC_MAGNITUDE_LOSS_WEIGHT=${DYNAMIC_MAGNITUDE_LOSS_WEIGHT:-0.2}
export DIRECTION_LOSS_EPS=${DIRECTION_LOSS_EPS:-1e-6}
# Stage1 trains the predictor only; action-conditioning gates are for later action ablations.
export CONDITIONING_PAST_MOTION_GATE=${CONDITIONING_PAST_MOTION_GATE:-0}

export MAX_STEPS=${MAX_STEPS:-30000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export WARMUP_STEPS=${WARMUP_STEPS:-5000}
export BS=${BS:-16}
export RUN_ID=${RUN_ID:-qwen0p8_groot_cascade_motion_cambranch_pastflow_dynamic_focus_stage1_fromscratch_leftprimary}

if [ "${STAGE}" != "learn_scene_flow" ]; then
  echo "[guard] dynamic-focus stage1 wrapper requires STAGE=learn_scene_flow, got ${STAGE}" >&2
  exit 3
fi
case "${RUN_ID}" in
  *dynamic_focus_stage1*) ;;
  *) echo "[guard] RUN_ID must contain dynamic_focus_stage1 to avoid confusing it with joint/stage2 runs: ${RUN_ID}" >&2; exit 3 ;;
esac

exec "${SCRIPT_DIR}/run_sceneflow_cascade.sh" "$@"
