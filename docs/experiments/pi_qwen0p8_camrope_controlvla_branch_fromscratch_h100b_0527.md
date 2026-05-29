# pi_qwen0p8_camrope_controlvla_branch_fromscratch_h100b_0527

**Role:** the ControlVLA-style redesign of stereo injection (the paper's candidate "new feature").
**Status:** trained 30k, evaluated right_view. From scratch on h100b.

## Hypothesis
Instead of adding FFS features as a residual into the VL stream (ControlNet style), inject them as a
**parallel K/V branch inside the action cross-attention** (ControlVLA style): the action queries attend
to FFS stereo features through a separate, strict step-0-preserving K/V path. Expected to use the
stereo signal more directly than a residual and beat the ControlNet baseline.

## Architecture (deltas vs FFS-ControlNet runs)
- framework: **QwenPIControlVLAFFS** — `starVLA/model/modules/stereo/controlvla_branch.py` adds a
  per-DiT-layer parallel K/V branch, replacing attn1; FFS features pooled (`ffs_pool_size: 8`).
- `interleave_self_attention: **false**` (forced; ControlVLA branch owns that path).
- `stereo_epipolar_mask_enabled: false`; cam_rope d_c16; FFS pyramid level 0.
- 4 codex-reviewed + sanity-verified fixes: zero-init parity (diff 0), per-sample `.repeat(2)`
  alignment, branch fires only when K_z/V_z nonzero, coverage assert.

## Data / training
- same data/schedule. per-device vla batch = **24** (h100b), eff-batch ≈ 96.

## Eval — libero_goal, 100ep, primary+right_view
| step | SR |
|---|---|
| 25k | 0.90 (05/29) |
| 30k | 0.90 (05/28) / 0.91 (05/29) |

## Conclusion
Converges to ~0.90–0.91 — **does NOT clearly beat the no-epi ControlNet baseline** (also ~0.90) on
`libero_goal`. The architectural change is within eval noise here. To demonstrate a real benefit, this
run needs a non-saturated / perturbation eval. This is the central open question for the paper story.
