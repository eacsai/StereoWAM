# goal_phase3b_camrope_0523

**Role:** the plain cam_rope-only stereo baseline (no epipolar, no FFS branch). **Strongest stereo config on libero_goal.**
**Status:** trained 30k (6 ckpts), evaluated right_view.

## What it is
`QwenPI` + camera-frame RoPE (d_c=16) only. No epipolar mask, no FFS-ControlNet/ControlVLA. The minimal
"stereo via cam_rope" model — the left edge of the design-space 2×2.

## Architecture
- framework: **QwenPI**; cam_rope enabled, d_c=16; epipolar off; no FFS block.
- data `libero_goal_stereo`, 30k steps.

## Eval — libero_goal, 100ep
| step | 10k | 15k | 20k | 30k (right_view) | 30k (wrist, wrong) |
|---|---|---|---|---|---|
| SR | 0.86 | 0.88 | 0.82 | **0.94 / 0.96** | 0.84 |

(30k evaled twice with right_view: 0.94 in run dir, 0.96 in /tmp — both within noise. The 0.84 is a
wrist eval and should be ignored.)

## Conclusion
**0.94–0.96 at 30k — the highest of ALL stereo variants**, and not beaten by any FFS-ControlNet,
ControlVLA, epipolar, or larger-d_c (d32/d64) addition. Direct evidence that on the saturated
`libero_goal` suite the extra stereo-injection machinery buys nothing over plain cam_rope.

## Sibling
`goal_phase3b_camrope_30k_0522` is an earlier same-config run with 6 ckpts but **no eval yet**.
