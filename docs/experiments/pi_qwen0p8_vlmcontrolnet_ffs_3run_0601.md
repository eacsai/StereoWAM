# VLM-ControlNet-FFS 3-run ablation (2026-06-01/02)

**Question**: does a *true* VLM-side ControlNet (frozen Qwen VLM + trainable trunk-copy branch + zero-conv
residual into the VLM hidden states), fed the real post-cost-volume stereo feature (FFS gru_hidden net[0]),
add value over the plain cam_rope baseline on libero_goal?

## Setup (shared)
- Framework: `QwenPIVLMControlNetFFS` (Run2/Run3) / `QwenPI` head-only (Run1).
- Frozen trunk = cam_rope 0.94 warm-start `goal_phase3b_camrope_0523/steps_30000` (interleave_self_attention=true).
- Control branch = copy of the 6 cam_rope softmax layers {3,7,11,15,19,23}; per-depth zero-conv Linear(1024->1024)
  adds a residual into the frozen VLM hidden states → step-0 == baseline byte-exact. Copied cam_rope frozen.
- FFS feature = FoundationStereo **gru_hidden net[0]** (B,16,h,w), **post-cost-volume (real disparity)**, FFS frozen,
  fp32 forward, captured via `update_block` forward hook (output[0][0]). (Confirmed by static trace 2026-06-02.)
- Suite libero_goal, video=primary,right_view (NO wrist), 100ep (10 trials/task). eff_batch 96
  (BS24 x 2GPU x GA2, deepspeed_zero2_ga2), 30k steps, warm-start. h100b train -> 4090d eval.

## The three runs (right_view SR)
| run | run_id | what trains | 10k | 20k | 30k |
|---|---|---|---|---|---|
| Run1 head-only (control) | pi_qwen0p8_camrope_frozenvlm_headonly_warmstart_0601 | action head only, NO FFS | 0.88 | 0.91 | 0.92 |
| Run2 head+ControlNet | pi_qwen0p8_vlmcontrolnet_ffs_trainhead_warmstart_0601 | head + ControlNet branch | 0.93 | 0.96 | 0.93 |
| Run3 pure ControlNet | pi_qwen0p8_vlmcontrolnet_ffs_frozenhead_warmstart_0601 | ControlNet branch only (head frozen) | 0.95 | 0.93 | 0.95 |

Reference (no FFS): plain cam_rope `goal_phase3b_camrope_0523` = **0.94/0.96**.

## Decomposition (30k SR; baseline = 0.94)
- Run3 - baseline = 0.95 - 0.94 = **+0.01** -> pure FFS-ControlNet on a frozen 0.94 policy: net-neutral.
- Run2 - Run1    = 0.93 - 0.92 = **+0.01** -> FFS effect when the head can also adapt: net-neutral.
- Run1 - baseline = 0.92 - 0.94 = **-0.02** -> freezing VLM + retraining head recovers ~baseline.
- All within libero_goal noise (+/-3-5% @100ep). **No statistically meaningful signal.**

## Conclusion
1. **No positive signal on libero_goal** — consistent with the saturation finding: even a real post-cost-volume
   stereo ControlNet cannot beat ~0.94 here.
2. **BUT this is the first FFS-injection that does NOT degrade the baseline.** Prior FFS variants (ControlNet->V,
   ControlVLA branch, gru_hidden on those) drifted down to 0.87-0.92 (net-negative). The true VLM-ControlNet
   (trunk-copy + zero-conv) holds baseline (Run3 30k 0.95 ~= 0.94, peak 0.95-0.96). Clean/harmless injection,
   no drift — matches the 0530 diagnosis that the *drift*, not the feature, was the problem.
3. **Decisive test still pending**: a depth/occlusion-sensitive, non-saturated suite where mono+wrist cannot solve.
   -> stereo4d proposal section 5 de-risk. libero_goal is exhausted for this question.

Raw orchestrator output: playground/Checkpoints/vlmcontrolnet_3way_comparison.txt (not git-tracked).
See also project_ffs_features_are_monocular, project_experiment_registry.
