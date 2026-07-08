# Scene-flow dual-DiT cascade — motion arm, from-scratch (launched 2026-07-06)

Primary bet: **predicting future motion (scene flow) coupled into the action head > static point features**.
Directly comparable to the leftprimary from-scratch table (same eff128 / 30k / leftprimary / vk=primary,left_view).

## Run (LIVE on A800)
- **run_id**: `qwen0p8_groot_cascade_motion_cambranch_joint_cascade_fromscratch_leftprimary`
- **host**: A800 `10.13.32.3:11089`, single A800-SXM4-80GB (GPU0), repo `/home/wangqiwei/ICLR2026/starVLA`
- **launched**: 2026-07-06 ~18:02 UTC, tmux session = run_id, log `playground/Checkpoints/<run_id>.train.log`
- **launcher**: `scripts/a800/run_sceneflow_cascade.sh` (STAGE=joint_cascade)

## Config (single-variable vs plain / static twin)
- Architecture: DiT4DiT-style **dual-DiT cascade** — shared VLM (raw leftprimary stereo, cam_branch=ON) → scene-flow DiT + GR00T action DiT, coupled by a **zero-init gated cross-attn** on the scene DiT's mid-late tapped hidden (tap_hidden_index=10, detach_conditioning=true).
- STAGE=joint_cascade (from-scratch, no warm-start — the end-to-end variant, table-comparable), loss_mode=joint = action + **FLOW_LAMBDA=0.1**·flow.
- TARGET_KEY=flow_gt (predicted future scene flow; dataloader `scene_flow.enabled` + `gt_only_sampler` + `expected_sidecar_to_training_flip=rot180`).
- cam_branch=ON (parallel PRoPE geometry branch); BS16 × 1GPU × GA8 = **eff128**; MAX_STEPS=30000, save=2000, warmup=5000.

## Validation before launch (all PASS)
- Unit smoke (coupler byte-zero) + full-VLM step-0 smoke (`STEP0_EQ_BASELINE=0`, real Qwen VLM, incl. cam_branch) + real bf16 MAX_STEPS=2 training (4090d BS8 + A800 BS16, no OOM/hang, joint loss = action+λ·flow verified).

## Code state (UNCOMMITTED on 4090d, HEAD d665b8c — rsynced to A800)
- scene-flow cascade: `cross_attention_dit.py` (MotionCoupler), `scene_flow_head.py` (new SceneFieldMatchingHead), `GR00T_ActionHeader.py`, `QwenGR00T.py`, `train_starvla.py`.
- cam_branch: the exp#5 defer refactor (`cam_branch_attention.py` + `depth_token_inject.py` + `QwenGR00T_FFSCommon.py`) + a NameError typo fix (:601) + a fail-closed deferred-refresh guard (codex HIGH). Reviewed by 2 codex + 3 Claude agents (0 disagreements). Note: reported result ⑤ (cambranch+Utonia 0.8825) depends on this same uncommitted refactor (committed code hard-raises for FFS+cam_branch, so ⑤ necessarily used it).

## Results (final eval checked 2026-07-08 09:24 UTC)

Protocol for the final rows below: `step=30000`, `video_keys=primary,left_view`, 4-suite LIBERO eval. Treat deltas below roughly 6pp as eval noise unless they repeat across seeds/reruns.

### Plain scene-flow joint cascade, from-scratch

Run id: `qwen0p8_groot_cascade_motion_cambranch_joint_cascade_fromscratch_leftprimary`

| suite | SR |
|---|---:|
| libero_spatial | 0.93 |
| libero_object | 0.98 |
| libero_goal | 0.84 |
| libero_10 | 0.65 |
| mean | 0.8500 |

Interpretation: ties the stereo+cam_branch task baseline around 0.84-0.85; no clear task gain from the scene-flow cascade on saturated standard LIBERO.

### Coupled-Utonia scene-flow stage-2

Run id: `qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary`

| suite | SR |
|---|---:|
| libero_spatial | 0.89 |
| libero_object | 0.93 |
| libero_goal | 0.92 |
| libero_10 | 0.53 |
| mean | 0.8175 |

Interpretation: did not win. It is below plain joint from-scratch by 3.25pp and below the previous Utonia top baseline 0.8825 by 6.5pp. The long-horizon `libero_10` drop is the main loss.

### Past-flow ControlNet joint cascade, from-scratch

Run id: `qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary`

| suite | SR |
|---|---:|
| libero_spatial | 0.91 |
| libero_object | 0.94 |
| libero_goal | 0.91 |
| libero_10 | 0.67 |
| mean | 0.8575 |

Interpretation: numerically +0.75pp over plain joint from-scratch 0.8500, which is inside the 6pp eval-noise band. It is also 2.5pp below the previous Utonia top baseline 0.8825. Register as neutral/negative for the current standard-LIBERO claim, not as a clear improvement.

### Bottom line

- Scene-flow auxiliary objectives have not produced a robust standard-LIBERO gain so far.
- Past-flow ControlNet is the best of these scene-flow variants at mean 0.8575, but the margin over 0.8500 is too small to claim.
- Further work should treat these as ablations or move to a pre-registered OOD/generalization split rather than continuing blind standard-LIBERO tuning.
