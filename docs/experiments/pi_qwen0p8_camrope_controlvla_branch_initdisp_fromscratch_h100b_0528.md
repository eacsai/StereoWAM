# pi_qwen0p8_camrope_controlvla_branch_initdisp_fromscratch_h100b_0528

**Role:** ControlVLA + FFS init-disparity variant.
**Status:** trained (5k–30k ckpts), evaluated right_view. From scratch on h100b.

## Hypothesis
On top of the ControlVLA branch, also feed FFS's **initial disparity** estimate (`use_init_disp: true`)
into the branch, giving an explicit depth/disparity prior rather than only deep features.

## Architecture (deltas vs ControlVLA sibling)
- framework: QwenPIControlVLAFFS (same as ControlVLA) **plus** `ffs_controlvla.use_init_disp: true`,
  `ffs_expected_sha256` pinned. interleave_self_attn=false; cam_rope d_c16; FFS pyramid level 0.

## Data / training
- same data/schedule. per-device vla batch = **48** (h100b), eff-batch ≈ 96.

## Eval — libero_goal, 100ep, primary+right_view
| step | SR |
|---|---|
| 5k | (incomplete log) |
| 10k | 0.80 |
| 15k | 0.74 |
| 25k | 0.90 |
| 30k | 0.87 |

## Conclusion
Converges to ~0.87–0.90, **no better** than plain ControlVLA or the ControlNet baseline on libero_goal.
init-disparity provides no measurable lift on this saturated suite. Same caveat: needs a harder eval to tell.
