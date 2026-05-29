# pi_qwen0p8_camrope_NOEpipolar_ffs_controlnet_fromscratch_4090d_0527

**Role:** baseline of the core Phase-3 comparison — "stereo via FFS-ControlNet, no epipolar mask".
**Status:** trained 30k (6 ckpts), evaluated right_view. **Trained from scratch** on 4090d (6× RTX4090D-48GB).

## Hypothesis
Injecting frozen Fast-FoundationStereo (FFS) spatial features as a zero-init ControlNet residual into
the Qwen-VL layers gives the action model stereo/disparity signal, on top of cam_rope. The epipolar
attention mask is *not* needed and may even hurt — this run is the no-mask control.

## Architecture (deltas vs the shared core base)
- framework: **QwenPIControlNetFFS** — per-DiT-layer projector turns FFS concat[left,right] features
  into a Qwen-shaped residual, added to `vl_embs` of each layer (`h + residual`, residual zero-init+learned).
- `stereo_epipolar_mask_enabled: **false**`
- `interleave_self_attention: true`
- cam_rope: enabled, d_c=16, 2 cams, baseline 0.06m, init_mode=zero
- FFS: weights `20-30-48/model_best_bp2_serialize.pth` (frozen), `ffs_scale: 0` (= pyramid level 0, finest).

## Data / training
- data: `LEROBOT_LIBERO_STEREO_DATA`, mix `libero_goal_stereo`, action `delta_qpos`.
- 30k steps, warmup 5k, save/5k, cosine LR (base 2.5e-5 / vl 1e-5 / action 1e-4), grad-accum 4.
- per-device vla batch = **16** (4090d); eff-batch ≈ 96.

## Eval — libero_goal, 100ep, primary+right_view
| step | SR | note |
|---|---|---|
| 5k | (45%) | from prior snapshot; right_view log not on disk |
| 25k | 0.90 (05/28) / 0.91 (05/29) | |
| 30k | 0.92 (05/28) / 0.90 (05/29) | the 05/29 re-eval was redundant (05/28 already done) |
| — | ~~74% (wrist)~~ | INVALID: a 05/28 manual 30k eval used wrist by mistake |

## Conclusion
Converges to ~0.90, tied with epi / ControlVLA at 25k+. Strong early sample efficiency (5k≈45%,
faster than epi's 31%). On `libero_goal` (saturated) it is indistinguishable from the mono baseline.
