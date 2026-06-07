#!/usr/bin/env bash
# Transfer the 2 missing official no_noops suites 4090d -> h100b via tar over ssh (h100b has no rsync).
set -e
cd /data/wangqiwei/ICLR2026/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA
H=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA
echo "[xfer] tar spatial+lib10 -> h100b ..."
tar cf - libero_spatial_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot \
  | ssh -o StrictHostKeyChecking=no root@10.112.2.93 "cd $H && tar xf -"
echo "XFER_DONE_TAR"
ssh -o StrictHostKeyChecking=no root@10.112.2.93 "ls $H/"
