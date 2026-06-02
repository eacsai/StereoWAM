#!/usr/bin/env bash
# Parallel 25k/30k sweep across the 3 fromscratch stereo runs on 4090d free GPUs (3,4,6,7).
# ALL lanes use video_keys=primary,right_view (STEREO, NO wrist) — these models train on right_view.
# Lane layout (4 GPUs, 6 evals; GPU3 & GPU6 lanes do 2 serial each):
#   GPU3 : ControlVLA 30k -> ControlVLA 25k
#   GPU6 : no-epi V1 25k  -> no-epi V1 30k   (30k re-eval: prior 74% was wrist, invalid)
#   GPU4 : epi V1 30k
#   GPU7 : epi V1 25k
set -uo pipefail

EVAL=/data/wangqiwei/ICLR2026/starVLA/scripts/4090d/eval_one_ckpt.sh
CKPTS=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints
CTRLVLA=$CKPTS/pi_qwen0p8_camrope_controlvla_branch_fromscratch_h100b_0527
NOEPI=$CKPTS/pi_qwen0p8_camrope_NOEpipolar_ffs_controlnet_fromscratch_4090d_0527
EPI=$CKPTS/pi_qwen0p8_camrope_epipolar_ffs_controlnet_fromscratch_h100b_0527
VK=primary,right_view

# single-eval lanes
nohup bash $EVAL $EPI/checkpoints/steps_30000_pytorch_model.pt 4 6711 $EPI/eval_logs_rightview/step30000 libero_goal $VK > /tmp/lane_epi_30k.log 2>&1 &
nohup bash $EVAL $EPI/checkpoints/steps_25000_pytorch_model.pt 7 6713 $EPI/eval_logs_rightview/step25000 libero_goal $VK > /tmp/lane_epi_25k.log 2>&1 &

# GPU3: ControlVLA 30k then 25k (serial)
( bash $EVAL $CTRLVLA/checkpoints/steps_30000_pytorch_model.pt 3 6710 $CTRLVLA/eval_logs_rightview/step30000 libero_goal $VK
  bash $EVAL $CTRLVLA/checkpoints/steps_25000_pytorch_model.pt 3 6710 $CTRLVLA/eval_logs_rightview/step25000 libero_goal $VK
) > /tmp/lane_controlvla.log 2>&1 &

# GPU6: no-epi 25k then 30k (serial; 30k is re-eval with right_view)
( bash $EVAL $NOEPI/checkpoints/steps_25000_pytorch_model.pt 6 6712 $NOEPI/eval_logs_rightview/step25000 libero_goal $VK
  bash $EVAL $NOEPI/checkpoints/steps_30000_pytorch_model.pt 6 6712 $NOEPI/eval_logs_rightview/step30000 libero_goal $VK
) > /tmp/lane_noepi.log 2>&1 &

echo "[driver] 4 lanes launched (6 evals, all right_view): ctrlvla(30k->25k) noepi(25k->30k) epi30k epi25k"
wait
echo "=== ALL SWEEP LANES DONE $(date) ==="
