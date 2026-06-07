# Qwen2.5-VL-3B × 4-suite input-view ablation (stereo vs mono vs primary+wrist)

> Status: **IN-PROGRESS** (2026-06-04). Registers the first Qwen2.5-VL-3B Stereo-VLA
> baseline line. Purpose: validate the stereo pipeline on a **strong backbone + hard
> 4-suite joint data** before adding cam_rope / FFS-hybrid. Updated as ckpts/evals land.

## Setup (identical across all 3 runs — only the visual input differs)
- **Backbone / head**: Qwen2.5-VL-3B-Instruct + GR00T action head (DiT). framework `QwenGR00T`.
- **No cam_rope, no FFS** — these are the plain input-view baselines.
- **Data**: ONE dataset `LEROBOT_LIBERO_STEREO_4SUITE` (4 suites: spatial/object/goal/libero_10,
  each weight 1.0, joint). Channels available: image(primary) / right_view / wrist_image.
  The 3 runs only change *which channels the DataConfig reads*:
  | run_id | DATA_MIX | reads channels | note |
  |---|---|---|---|
  | `qwen2p5vl3b_4suite_stereo_primaryright_0604` | libero_all_stereo | primary + right_view | stereo |
  | `qwen2p5vl3b_4suite_monoprimary_0604` | libero_all_mono_replay | primary only | **pixel-identical to stereo minus right_view** → clean Z baseline |
  | `qwen2p5vl3b_4suite_primarywrist_0604` | libero_all_mono_selfrender | primary + wrist | upstream recipe (libero_franka) |
- **Recipe**: BS16 × 2×H100 × GA4 = **eff_batch 128**, 30k steps, **fromscratch** (no warm-start),
  deepspeed_zero2_ga4, freeze="" (full finetune). h100b GPU0,1, serial (chain orchestrator).
- **Eval**: 4 suites in parallel on 4090d, eval-time live right_view render for stereo
  (`eval_qwen2p5vl_4suite.sh`, per-run video_keys). gripper openvla {0,1}.

## Why stereo-vs-mono here is the clean apples-to-apples
mono (`libero_all_mono_replay`) reads the SAME replayed frames as stereo but drops right_view
(never had wrist). So stereo−mono isolates exactly "model has right_view or not". primary+wrist
is the original recipe (has wrist, no right_view) — useful as the standard reference but NOT a
clean stereo control (different DataConfig).

## Results (SR per suite; — = pending)
| run | step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|---|
| **stereo** (primary+right) | 10k | 0.45 | 0.39 | 0.53 | 0.07 | ≈0.36 |
| stereo | 20k | — | — | — | — | — |
| stereo | 30k | — | — | — | — | — |
| **mono** (primary only) | 20k | — | — | — | — | — |
| mono | 30k | — | — | — | — | — |
| **primary+wrist** | 20k | — | — | — | — | — |
| primary+wrist | 30k | — | — | — | — | — |

### Notes on the 10k stereo point (short-validation gate — PASSED)
- 10k is fromscratch + 4-suite-joint + hardest LIBERO mix → these are normal *early* numbers,
  not a ceiling. Prior fromscratch single-suite refs: gru_hidden 5k=0.42→30k=0.92, depth-token
  Run C 10k=0.57→30k=0.94. 4-suite joint is harder (capacity split 4 ways) so 10k mean ≈0.36 is
  in-range. libero_10 (long-horizon) expected to lag early and climb late.
- **State path confirmed unused / not a confound**: model `state_encoder=Linear(7,1024)` but data
  state is 8-dim → feeding it would crash training (it didn't) ⇒ training ran with state=None
  (state_encoder weights at init, std 0.216 ≈ Kaiming init 0.218). Eval (reverted state-fix) also
  uses state=None → **train/eval matched, no proprioception mismatch**. Low early SR is purely
  early-training, not a state bug. (Proprioception is a currently-dead path; aligning state_dim
  7↔8 + feeding it is a separate optional enhancement, may help libero_10.)

## Next (after this line completes)
1. Add cam_rope (reverse-order) on the 4-suite stereo data → new baseline.
2. Freeze it + add FFS-hybrid (LLaMA-adapter / the 8-method injection comparison set).
3. Real stereo-vs-mono signal likely needs a non-saturated depth-sensitive suite (RLBench/RoboCasa);
   libero_goal is saturated (stereo≈mono≈0.92 across all prior FFS variants).

参见 [[project_experiment_registry]] [[ffs_into_vlm_injection_comparison]] [[project_ffs_features_are_monocular]]
