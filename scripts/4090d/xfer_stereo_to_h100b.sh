#!/bin/bash
# Transfer the remaining stereo 4-suite LeRobot datasets 4090d -> h100b (direct, tar over ssh).
# spatial already transferred. Excludes the stray LLaVA-OneVision-COCO subdir.
set -u
cd /data/wangqiwei/ICLR2026/data
DEST=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Datasets/LEROBOT_LIBERO_STEREO_4SUITE
H100='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93'
for s in object goal 10; do
  echo "[xfer $(date +%H:%M:%S)] libero_${s} ..."
  tar cf - --exclude=LLaVA-OneVision-COCO "libero_${s}_openvla_vanilla_stereo_lerobot" \
    | $H100 "mkdir -p $DEST && cd $DEST && tar xf - && rm -rf libero_${s} && mv libero_${s}_openvla_vanilla_stereo_lerobot libero_${s} && echo GOT_libero_${s}_\$(find libero_${s}/data -name '*.parquet' | wc -l)pq"
done
echo "ALL_STEREO_TRANSFERRED"
