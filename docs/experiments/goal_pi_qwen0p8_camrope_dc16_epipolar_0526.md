# goal_pi_qwen0p8_camrope_dc16_epipolar_0526

**Role:** the "epipolar mask WITHOUT ControlNet" cell of the 2×2 — the only such run.
**Status:** trained 30k (6 ckpts), evaluated right_view (full curve).

## What it is
Plain `QwenPI` (NO FFS-ControlNet / ControlVLA branch). Stereo signal enters ONLY through camera-frame
RoPE (d_c=16), and the **epipolar attention mask is ON**. This isolates the effect of the epipolar mask
on a cam_rope-only stereo model, with no FFS feature injection at all.

## Architecture
- framework: **QwenPI** (base PI action expert; no ffs_controlnet / ffs_controlvla block).
- cam_rope: enabled, d_c=16, init_mode=zero, baseline 0.06m, 2 cams.
- `stereo_epipolar_mask_enabled: true`.
- data `libero_goal_stereo`, 30k steps.

## Eval — libero_goal, 100ep, primary+right_view
| step | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|
| SR | 0.74 | 0.79 | 0.86 | 0.91 | 0.92 | **0.90** |

## Conclusion
Converges ~0.90 — same ballpark as everything else. The epipolar mask on a cam_rope-only model neither
clearly helps nor hurts at convergence vs the no-epipolar cam_rope runs (phase3b/d/e). Combined with the
FFS-ControlNet epipolar arm (also no gain, slower early), this reinforces dropping epipolar from the paper.
