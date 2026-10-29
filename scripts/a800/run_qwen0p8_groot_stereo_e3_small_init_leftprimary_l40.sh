#!/bin/bash
set -e

# E3: Camera Branch + stereo, small nonzero out_proj initialization.
# This wrapper preserves the E2 launcher and changes only the run id/init knob.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RUN_ID=${RUN_ID:-qwen0p8_groot_stereo_cambranch_e3_small_init_std1e-3_bidirectional_leftprimary_l40_4gpu_bs8ga4_eff128_seed42_fromscratch_0721}
export CAM_BRANCH_OUT_PROJ_INIT_STD=${CAM_BRANCH_OUT_PROJ_INIT_STD:-1e-3}
exec ${SCRIPT_DIR}/run_qwen0p8_groot_stereo_e2_leftprimary_l40.sh $@
