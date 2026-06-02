# pi_qwen0p8_camrope_controlnet_actionattn_initdisp_from_baseline96_h100b_0529

**Role:** ControlVLA + init-disparity, **warm-started from the strongest cam_rope baseline** (phase3b @30k).
**Status:** TRAINING IN PROGRESS on h100b (2× H100). 10k checkpoint evaluated (right_view). 30k ETA ~07:09 UTC 05/29.

## What it is
The most principled attempt to show a stereo benefit: take the best plain cam_rope baseline
(`goal_phase3b_camrope_0523` @30k = 0.94/0.96, saved as `goal_phase3b_camrope_0523_baseline_for_initdisp_init`)
as initialization, then continue training with the **ControlVLA parallel K/V branch** (`QwenPIControlVLAFFS`)
plus **`use_init_disp=true`** (explicit FFS disparity channel). Branch is zero-init, so step-0 = baseline behavior.

## Architecture / config
- framework: QwenPIControlVLAFFS, `use_init_disp=true`, `ffs_pool_size=8`, ffs pyramid level 0.
  FFS token dim = 224/view × 2 + 1 (init_disp) = 449.
- `interleave_self_attention: false` (all 24 DiT layers cross-attn → ControlVLA branch in every layer).
- cam_rope d_c16, epipolar off, init_mode zero.
- `trainer.pretrained_checkpoint` = phase3b baseline 30k (warm start).
- 2× H100 80GB, deepspeed zero2 ga2, per_device 24 → eff-batch 96, 30k steps, save/5k.

## Eval — libero_goal, 100ep, primary+right_view
| step | SR | note |
|---|---|---|
| 10k | **0.87** | evaled 05/29 04:06 UTC |
| 25k | 0.92 | |
| 30k | 0.92 | training done; converges ~0.92 (saturated band) |

## Observations
- 10k = **0.87, which is BELOW the 0.94/0.96 baseline it was initialized from** (gap > eval noise).
  i.e. after 10k steps of continued training with the ControlVLA+initdisp branch, performance has dipped
  below the strong init rather than improved. Whether it recovers/exceeds the baseline by 30k is the key
  question — if it stays below, it's further evidence the FFS branch adds no benefit (and may perturb) on
  the saturated libero_goal suite.
- Comparison: fromscratch ControlVLA+initdisp (`..._controlvla_branch_initdisp_fromscratch_...`) had
  10k=0.80, 25k=0.90, 30k=0.87. Warm-start is ahead at 10k (0.87 vs 0.80) as expected, but both sit ~0.87-0.90.

## Cross-machine eval note
This ckpt trained on h100b; to eval on 4090d, config.yaml had to be replaced with config.full.yaml
(sparse config.yaml was missing interleave_self_attention / action_horizon / future_action_window_size),
ffs_model_path rewritten to the 4090d path, and dataset_statistics.json transferred alongside the .pt.
