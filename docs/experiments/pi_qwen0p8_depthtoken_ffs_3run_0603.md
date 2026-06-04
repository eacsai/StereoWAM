# Depth-token FFS 3-run ablation (2026-06-03)

**Question**: instead of injecting the FFS stereo feature as a *residual* (ControlNet→V, ControlVLA K/V,
VLM-ControlNet), insert it as **feature TOKENS in the Qwen image+text token sequence** (sequence grows
S → S+16, the VLM's own self-attention fuses the depth tokens). Does giving the VLM the real post-cost-volume
stereo signal as first-class tokens beat the plain cam_rope baseline on libero_goal — and does it matter how
the VLM adapts (frozen / LoRA / full finetune)?

## Mechanism (depth-token injection)
- FFS `gru_hidden` **net[0]** (B,16,64,64), post-cost-volume (real disparity), FFS frozen, fp32 forward.
- `adaptive_pool` 4×4 → 16 spatial cells → `Linear(16→1024, zero-init)` → **16 depth tokens** (step-0 == baseline byte-exact).
- Inserted via an inner `language_model` forward-pre-hook that rewrites `inputs_embeds` (S→S+16) + `position_ids`
  + `attention_mask` + cam_rope `per_token_cam_id` (depth tokens = cam_id −1 → plain M-RoPE). No HF forward rewrite needed.
- Insert point = **before ALL image tokens** (HIGH-2 fix) so the primary image can causally attend back to the depth tokens.
- Depth-token positions = the 4×4 cell centers of the primary image token grid (MED-4 fix).
- Output hidden of the 16 depth tokens is **sliced out** (keep_mask) before the action expert → action-side token count = baseline, downstream unchanged.
- Frameworks: `QwenPIDepthTokenFFS` (Run A & C) / `QwenPIDepthTokenLoRAFFS` (Run B).

## Setup (shared)
- Suite libero_goal, video = **primary,right_view** (NO wrist), 100ep (10 trials/task). Reference (no FFS):
  plain cam_rope `goal_phase3b_camrope_0523` = **0.94/0.96** @30k.
- eff_batch 96 (BS24 × 2 H100 × GA2, deepspeed_zero2_ga2), 30k steps. FFS frozen, fp32. h100b train → 4090d eval.

## The three runs (right_view SR; verified from each run's auto_eval_results.txt)
| run | run_id | what trains / init | LR (projector) | 10k | 20k | 30k |
|---|---|---|---|---|---|---|
| Run A frozen-VLM | pi_qwen0p8_depthtoken_ffs_frozenvlm_warmstart_0603 | projector + head; VLM frozen; warm-start 0.94 | base ~1e-5 (low) | 0.89 | 0.92 | 0.90 |
| Run B LoRA | pi_qwen0p8_depthtoken_lora_warmstart_0603 | LoRA r16/α32 q/k/v/o + projector + head; warm-start 0.94 | **1e-4 (high)**, base 2.5e-5 | 0.83 | **0.94** | 0.93 |
| Run C full / fromscratch (STRESS) | pi_qwen0p8_depthtoken_ffs_fullfinetune_fromscratch_0603 | everything (freeze=""), base Qwen, NO warm-start | base LR | 0.57 | 0.88 | 0.94 |

## Decomposition (30k SR; baseline = 0.94)
- Run A − baseline = 0.90 − 0.94 = **−0.04** → frozen VLM + depth-token + trained projector/head: ≈ baseline.
- Run B − baseline = 0.93 − 0.94 = **−0.01** → LoRA + high projector LR: ≈ baseline (peak 0.94 @20k).
- Run C − baseline = 0.94 − 0.94 = **0.00** → full fromscratch climbs the usual fromscratch curve (10k 0.57 → 30k 0.94).
- All within libero_goal noise (±3-5% @100ep). **No statistically meaningful signal; no variant clears baseline.**

## Conclusion
1. **No positive signal on libero_goal** — all three adaptation paradigms (frozen+low-LR, LoRA+high-LR,
   full-finetune-fromscratch) converge into the **same 0.90-0.94 saturation band** as every other FFS-injection
   variant (gru_hidden 0.92, VLM-ControlNet 0.92-0.95). The Nth confirmation that the bottleneck is the
   **suite, not the mechanism / feature / capacity / adaptation strategy**.
2. **A vs B is a clean null**: LoRA + high projector LR (B) does NOT beat frozen + low LR (A); the ±3-5% gaps
   are noise — cannot claim LoRA is better. High LR on the depth path did not break saturation either.
3. **Run C is a STRESS run** (fromscratch + full finetune + depth-token all confounded) and is NOT a clean
   attribution point; its 0.94 @30k just shows depth-token training is stable and reaches the same band.
4. **Decisive test still pending**: a depth/occlusion-sensitive, non-saturated suite (stereo4d proposal §5;
   echoes concurrent StereoVLA 2512.21970 needing real-robot + 5M synthetic to show a gap). libero_goal is exhausted.

## Engineering notes (shared bugs, for future FFS work)
- **Eval dtype bug** (shared with gru_hidden lesson): `DepthTokenProjector.forward` Linear input net0 is fp32
  (FFS all-fp32) but eval `--use_bf16` casts the projector weight to bf16 → `float != BFloat16` crash. Training
  didn't crash (no global cast), smoke didn't crash (ran fp32) → only the eval path surfaced it (Run A 10k eval
  FAILED first). Fix: `return self.proj(x.to(self.proj.weight.dtype))` (safe in train + eval).
- **LoRA-B injection bug**: peft `BaseTuner.forward` bypasses the top-level module hook → depth tokens were
  not injected. Fix: after wrapping, `reinstall_depth_token_outer_hook` / `_reinstall_stereo_cam_rope_outer_hook`
  re-attach the outer hook to the `PeftModel`. Warm-start of a non-LoRA 0.94 ckpt into the LoRA-wrapped model
  needs key remap (`base_model.model.` prefix); 486 keys remapped, verified at GPU smoke.
- **Eval GPU contention**: 4090d is shared; daemons hit CUDA OOM when their pinned GPU was taken (Run B 10k
  FAILED on GPU7 @46/49GB) → moved to a free GPU and re-evaluated cleanly (0.83). Not a code bug.

Raw eval logs: each run's `playground/Checkpoints/<run_id>/auto_eval_results.txt` (gitignored; numbers recorded here).
See also project_ffs_features_are_monocular, project_experiment_registry, pi_qwen0p8_vlmcontrolnet_ffs_3run_0601.
