# pi_qwen0p8_camrope_epipolar_ffs_controlnet_fromscratch_h100b_0527

**Role:** the "+ epipolar mask" arm of the core comparison.
**Status:** trained 30k, evaluated right_view. From scratch on h100b (8× H100).

## Hypothesis
On top of the FFS-ControlNet stereo residual, restricting attention with an **epipolar mask** (each
left-view token attends only along its epipolar line in the right view) should sharpen stereo
correspondence and help. (Outcome: it did NOT help; see decision to drop epipolar.)

## Architecture (deltas vs no-epi sibling)
- framework: QwenPIControlNetFFS (identical to no-epi) **except** `stereo_epipolar_mask_enabled: true`.
- everything else identical: cam_rope d_c16, interleave_self_attn=true, FFS pyramid level 0, zero-init residual.

## Data / training
- same data/schedule as no-epi. per-device vla batch = **12** (h100b 8-GPU), eff-batch ≈ 96.

## Eval — libero_goal, 100ep, primary+right_view
| step | SR |
|---|---|
| 5k | 0.31 |
| 10k | 0.74 |
| 20k | 0.91 |
| 25k | 0.88 (05/28) / 0.89 (05/29) |
| 30k | 0.91 (05/28) / 0.90 (05/29) |

## Conclusion
Converges to ~0.90 — **tied** with the no-epi baseline, but **slower early** (5k=31% vs no-epi 45%;
10k=74%). Combined with codex's 3 high-severity findings on the epipolar mask and a ~4pp early drop,
this is why **epipolar is NOT reported in the paper** (see memory: drop-epipolar decision).
