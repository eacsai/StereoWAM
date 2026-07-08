# Handoff — codex spec-review context: Past-Flow ControlNet for the Scene-Flow DiT

**Date**: 2026-07-07. **Purpose**: holistic session context for a codex (read-only) spec-review of the design-bearing spec at `~/.claude/plans/sceneflow-past-flow-controlnet-spec.md`. This is a **Stage-1.5 pre-code review** — evaluate whether the DESIGN is sound before any code is written. Do NOT rewrite the spec; surface issues.

Code authoritative source: `4090d:/data/wangqiwei/ICLR2026/starVLA` (never push; GPU-smoke gate mandatory before long runs).

## What is being proposed (read the spec, don't duplicate)
Spec: `~/.claude/plans/sceneflow-past-flow-controlnet-spec.md`. In one line: add a **zero-init ControlNet side-branch on the scene-flow DiT** that feeds it the **last K=2–3 previous scene-flow fields** (dense 16×16×3, spatially aligned), to fix reversed-direction flow predictions caused by single-frame motion ambiguity. Train with GT past-flow (teacher-forcing), infer with the model's own chained predictions; scheduled-sampling + noise-aug built into config as the exposure-bias escape hatch.

## Why (the motivating evidence)
- **The base system** = scene-flow dual-DiT cascade: shared VLM (Qwen3.5-0.8B, raw leftprimary stereo) → scene-flow DiT (SceneFieldMatchingHead, DiT-B, flow-matching, predicts 16×16×3 camera-frame flow, conditions on VLM hidden via cross-attn) + GR00T action DiT, coupled by a zero-init gated cross-attn (MotionCoupler) reading the scene DiT's tapped hidden (tap=10, detached). Base spec: `~/.claude/plans/sceneflow-dualdit-cascade-spec.md`.
- **The problem** (offline flow-prediction check, 2026-07-07, script `scripts/4090d/eval_scene_flow_prediction.py`, results `docs/experiments/stage1_flow_prediction_check/`): the plain stage-1 flow predictor is NOT degenerate (dynamic-cell cosine 0.70, loaded 0 missing/0 unexpected) but WEAK — RelEPE 0.80 vs zero-flow baseline (only 20% better than predicting zero), under-shoots magnitude, and **reverses direction on some cells**. Hypothesis: single-frame conditioning cannot disambiguate motion direction.
- **Literature backing** (note `notes/2605.12090_WorldActionModels.md` + workflow survey of DreamVLA/GaussianDream/mu0/GAM): essentially all world/motion predictors condition on the PAST (RSSM recurrent, autoregressive past-token, video spatiotemporal). DreamVLA=7-frame causal, GaussianDream=3 sparse frames ("motion cues", 0.5s spacing), mu0=8-step past keypoint trajectory as suffix tokens, GAM=~7-step causal frames+proprio+prev-action. Only DepthVLA is single-frame — and it predicts STATIC depth (no motion, no direction ambiguity). Our single-frame flow predictor is the outlier. mu0/GAM inject past as sparse TOKENS into the predictor stream (not VLM, not ControlNet); we deliberately chose a dense spatially-aligned signal (past flow FIELD) → ControlNet is the fitting tool.

## Current experiment landscape (for judging where this fits)
- **Joint-from-scratch cascade** (a800b, done+evaled): 4-suite MEAN **0.85** (spatial .93/object .98/goal .84/long .65). Key: this **ties the stereo+cam_branch baseline (0.85)** — the correct apples-to-apples comparison, since the cascade also uses cam_branch. plain-stereo (NO cam_branch) = 0.8275; mono = 0.8325. So adding the scene-flow branch on top of stereo+cam_branch gave ZERO task gain.
- **stage-2** (a800a GPU0, running ~2.5k/30k): joint_cascade warmstart from the plain stage-1 flow ckpt — tests whether "learn flow first, then joint" beats 0.85. GPU-smoke passed (1078 trainable/0 frozen, warmstart clean).
- **Utonia stage-1** (a800a GPU1, ~28.5k/30k, near done): #5 prompt+point-token injection variant, flow-only.
- Best overall so far = ⑤ cambranch+Utonia 0.8825.
- **Tension worth noting**: the WAM survey motivates world-modeling as improving generalization, yet our joint cascade shows predicting flow ≠ task gain. The flow-check shows flow prediction itself is weak. The past-flow spec targets the flow-quality half of that chain.

## Files the review should read (the spec will touch these)
- `starVLA/model/modules/action_model/flow_matching_head/scene_flow_head.py` (SceneFieldMatchingHead — the DiT to add the ControlNet branch to; forward() @175, _encode @141, _pool_target @148, _supervision_mask @162, extract_conditioning_hidden @233).
- `starVLA/dataloader/gr00t_lerobot/datasets.py` (needs to expose `past_flow_gt` from t−k·Δ sidecars; MUST reuse the exact rot180/leftprimary transform path used for `flow_gt` — historic silent-bug class, see [[project_sceneflow_flip_vector_bug]]).
- `starVLA/model/framework/VLM4A/QwenGR00T.py` (framework forward that calls scene_predictor; where past_flow_batch gets threaded in).
- Rollout/eval harness (for the autoregressive inference-time past-flow buffer — scope what state needs adding).
- GT pipeline: `gt_sceneflow_impl/gen_gt_sceneflow_rerender.py` (GT_METHOD=sim_rerender_analytic, per-body analytic camera-frame flow; camera fixed in LIBERO agentview → past flow directly comparable).

## The 5 review questions the spec explicitly wants judged (spec §9)
1. ControlNet depth: full trunk-copy vs shallow (2–4 block) copy for a 16-layer DiT-B? (spec leans shallow.)
2. Is a ControlNet warranted vs simply concatenating past-flow tokens to the DiT's cross-attention conditioning (lighter)? — dense spatial alignment favors ControlNet, but list the token-concat as a cheap ablation.
3. Autoregressive inference: does the rollout harness have a place to stash the previous-step prediction? Scope the rollout-state addition.
4. Multi-step past (K>1): per-field time-position + cross-time attention (spec's choice) vs channel-stack vs K separate zero-conv branches.
5. Polarity/flip: guarantee past_flow_gt goes through the identical rot180/leftprimary transform as flow_gt (reuse one code path).
Plus general: is the exposure-bias mitigation (teacher-forcing v1 + scheduled-sampling/noise-aug) adequate given K-step chaining of a weak (RelEPE 0.80) predictor? Any silent-bug/assumption/redundancy concerns? Simpler equivalent designs?

## Hard constraints / conventions (must respect)
- Code only on 4090d (authoritative); never push; commit needs explicit user OK.
- Injection methods MUST be step-0==baseline (zero-init) and pass `/gpu-smoke-train --injection`.
- leftprimary convention, wrist-free, eff-batch 128, eval vk=primary,left_view; ~6pp eval noise floor (don't over-read small SR deltas).
- Descriptive naming (no stage1/v2/abcd); files in project not /tmp; results to git-tracked docs/experiments.

## Suggested skills for the continuing agent
- `/codex-implement` (this review is its Stage-1.5); codex driver = `~/.claude/skills/codex-implement/codex_tmux.sh run <name> <repo> <promptfile> review 4090d` (locality: code on 4090d → codex on 4090d).
- `/gpu-smoke-train` (before any long run of the implemented code).
- `/sync-code-from-4090d` if syncing to a training box.

/Users/agiuser/.claude/projects/-Users-agiuser-Documents-ICLR2026/handoffs/handoff-2026-07-07T08-08-25Z.md
