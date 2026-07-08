# Spec: Past-Flow ControlNet into the Scene-Flow DiT (temporal disambiguation of scene-flow direction)

Status: DRAFT for user review → (on approval) /handoff + codex spec-review BEFORE any code.
Author context: StereoVLA scene-flow dual-DiT cascade. All code on 4090d authoritative (`/data/wangqiwei/ICLR2026/starVLA`). Never push. GPU-smoke gate mandatory before any long run.
Related: `~/.claude/plans/sceneflow-dualdit-cascade-spec.md` (base cascade), memory [[project_sceneflow_repo_lessons]], [[project_future_t3_video_timeaxis_4d]], note `2605.12090_WorldActionModels.md`.

## 1. Problem & hypothesis
- Empirical (2026-07-07 stage-1 flow-prediction check): the scene-flow predictor is NOT degenerate (dyn-cell cosine 0.70, RelEPE 0.80 vs zero-baseline) but **under-shoots magnitude and reverses direction on some cells**.
- Root-cause hypothesis: the predictor conditions on a **single current observation** (VLM hidden of the current stereo frame). A single frame cannot disambiguate motion DIRECTION (gripper opening vs closing; object moving left vs right). Every world model in the WAM survey + our 4 studied predictors (DreamVLA/GaussianDream/mu0/GAM) condition on the PAST; only DepthVLA is single-frame and it predicts static depth (no motion). Our predictor is the outlier.
- Fix: give the scene-flow DiT the **previous flow field** (what just moved) as a spatially-aligned conditioning signal, so it has the velocity/momentum cue to resolve direction.

## 2. Design choice (approved by user)
- **Signal = the last K previous scene-flow fields** (each the same 16×16×3 camera-frame grid the DiT predicts), from t−K·Δ .. t−Δ. Spatially aligned with the DiT's output grid. **K = n_past_steps, default 2–3, ablate {1,2,3}** (user 2026-07-07: keep a few past steps). Rationale: 1 step → velocity (enough for direction sign), 2 → acceleration (helps magnitude), >3 diminishing + worse inference error-compounding. Do NOT go large (unlike mu0's 8 — that is cheap sparse (u,v,z) with GT-at-train; our dense self-predicted flow-field chain compounds error far more).
- **Injection = ControlNet-style side branch on the scene-flow DiT** (SceneFieldMatchingHead), zero-init (zero-conv) so step-0 == baseline exactly. Chosen over token/adaLN because the signal is DENSE + spatially aligned (ControlNet's home turf), unlike mu0/GAM's sparse-token approach.
- **Multi-step encoding**: do NOT just channel-stack the K fields (loses time order). Tokenize each past field like the head's `_encode`, add a **per-field time-position embedding**, and let the ControlNet encoder attend ACROSS time (small temporal self-attn) so the model can compute velocity/acceleration and knows which field is t−1 vs t−2.
- **Do NOT touch the VLM or the action head** in this experiment → clean single-variable ablation ("does past-flow conditioning fix flow direction?"). Action head benefits indirectly via the existing zero-init coupler reading the (now-better) scene-DiT hidden.

## 3. Architecture detail
- The scene DiT is `SceneFieldMatchingHead.model` (DiT-B, 16 layers, inner 768) at `starVLA/model/modules/action_model/flow_matching_head/scene_flow_head.py`. It cross-attends to VLM hidden (`encoder_hidden_states=vl_embs`), input = noised flow tokens (B,256,3) + timestep.
- **ControlNet branch**:
  - Encode the past-flow field (B,3,16,16 → B,256,3, same tokenization as the head's `_encode`) through a **trainable copy of the first N DiT blocks** (ControlNet convention) OR a lightweight parallel encoder (decide in review; a full trunk-copy may be overkill for a 16-layer DiT-B — a shallow 2–4 block copy likely suffices).
  - Its per-block outputs are added to the main DiT's per-block hidden via **zero-initialized 1×1 projections (zero-conv)**. At init all zero → main path unchanged → **step-0 == baseline** (verify in GPU-smoke, injection mode).
  - The past-flow tokens carry their own positional/time encoding; they are the CONDITION, not denoised.
- Config gate: `framework.scene_predictor.past_flow_controlnet.enabled` (default false). New keys: `enabled`, `n_control_blocks` (shallow copy, ~2–4), `spacing_delta` (Δ frames), `n_past_steps` (default 2; ablate {1,2,3}), `zero_init` (true), `scheduled_sampling_p` (default 0 = pure teacher-forcing v1), `past_flow_noise_aug` (default off). Multi-step past encoded with per-field time-position + cross-time attention (§2), NOT channel-stack.

## 4. What "previous flow" means + data plumbing
- Our GT flow (`flow_gt`) is per-pair camera-frame 3D displacement (`gen_gt_sceneflow_rerender.py`, sim_rerender_analytic). For sample at time t, the "past flow" = `flow_gt` of the sample at t−Δ (the previous transition).
- **Dataloader change** (`starVLA/dataloader/gr00t_lerobot/datasets.py`): additionally expose `past_flow_gt` (and `past_flow_valid`) = the flow sidecar of the frame Δ steps earlier in the same episode. Guarded (only when the ControlNet is enabled + the earlier frame exists; episode-start frames → zero past-flow + a `has_past_flow=0` flag so the ControlNet contributes nothing there).
- **Camera frame**: LIBERO agentview camera is fixed → past flow (camera-frame) is directly comparable to current flow, no re-alignment needed. (Note in review: confirm leftprimary convention + rot180 flip applied identically to past_flow_gt as to flow_gt — reuse the SAME transform path to avoid a silent polarity bug like the historic one.)

## 5. Train / inference distribution gap (the key risk) — DECISION (user 2026-07-07)
- **Decided**: TRAIN feeds GT past-flow fields (`past_flow_gt` from the t−k·Δ sidecars); INFERENCE feeds the model's OWN previous-step predicted flow fields (autoregressive chaining, à la GaussianDream chaining / mu0 self-solved trace). This is teacher-forcing.
- **⚠️ Exposure-bias trap (flagged, expected)**: training only on clean GT → the model never sees its own noisy predictions → at rollout the self-predicted past is dirtier than anything seen in training → distribution shift → error compounding. WORST exactly in our case: multi-step past (K=2–3) × a still-weak predictor (RelEPE 0.80, cosine 0.70) × pure teacher-forcing. Expected symptom: offline (GT-fed) direction looks fixed, but rollout (self-fed) drifts/degrades. Do not be surprised by an offline-vs-rollout gap.
- **Mitigations — BUILD INTO CONFIG FROM DAY 1 (do not bolt on later)**; v1 runs with them off, flip on if offline improves but rollout drifts:
  1. `scheduled_sampling_p` (ramps 0→p over training): with prob p, replace a GT past-flow field with the model's own predicted one (or a detached cached prediction) so it learns to tolerate self-error.
  2. `past_flow_noise_aug`: perturb GT past-flow magnitude/direction to approximate the measured inference error (~0.80 RelEPE / 0.70 cosine), calibrated to the flow-check stats.
- **Rollout order**: episode start → no past → zero-fill + `has_past_flow=0` → ControlNet zero-contributes (baseline). As the episode proceeds, the K-slot past buffer fills with the model's own predictions (a short FIFO in rollout state, §9 Q3).
- **Reporting**: always report BOTH the teacher-forced offline flow-check (cosine/RelEPE/reversed-cell %) AND the rollout 4-suite SR, and explicitly track the gap between them.

## 6. Temporal spacing Δ (GaussianDream lesson)
- Consecutive frames have too-little motion (our motions are mm-scale). GaussianDream uses 0.5s spacing (5 frames @10fps) deliberately so motion is visible.
- Spec Δ as a config; **ablate Δ ∈ {1, 3, 5} frames**. Default start Δ=5 (matching GaussianDream's "visible motion" intuition), reconcile with our control frequency.

## 7. Step-0 identity + GPU-smoke (mandatory)
- With zero-init ControlNet, step-0 flow_loss and action path MUST equal the no-ControlNet baseline byte-for-byte. Add a `STEP0_EQ_BASELINE=PASS/FAIL` marker; run `/gpu-smoke-train ... --injection` — NO-GO if the marker is absent (this is an injection method).
- Also smoke: warmstart load clean (0 unexpected), joint loss finite, 2 steps advance, no bf16×fp32 crash (past-flow ControlNet params fp32 island like the scene DiT).

## 8. Experiment / ablation plan
Compare on the SAME protocol (leftprimary, from-scratch or warmstart-from-stage-1, eff128, 30k, wrist-free, eval vk=primary,left_view):
- **A. baseline** = current scene-flow cascade (no past-flow) — already have (joint 0.85 / stage-2 running).
- **B. + past-flow ControlNet (teacher-forced GT past)** — main arm.
- **C. B but self-predicted past (scheduled sampling)** — closes train/infer gap.
- **D. Δ sweep** {1,3,5}.
- **Primary offline metric**: the flow-prediction check (dyn-cell cosine ↑, RelEPE ↓, fewer reversed cells) — does past-flow fix DIRECTION? (This is measurable offline regardless of task SR.)
- **Downstream**: 4-suite SR vs the 0.85 line. NOTE per WAM-survey tension: better flow prediction ≠ guaranteed task gain — report both.

## 9. Open questions for codex spec-review
1. ControlNet depth: full trunk-copy vs shallow (2–4 block) copy for a 16-layer DiT-B? (lean shallow.)
2. Is a ControlNet warranted vs simply concatenating the past-flow tokens to the DiT's cross-attention conditioning (lighter)? Decide: dense spatial alignment favors ControlNet, but a token-concat ablation is cheap and worth listing.
3. Autoregressive inference: does the eval/rollout harness currently give the scene DiT a place to stash its previous-step prediction? (Likely needs a small rollout-state addition — scope it.)
4. Multi-step past (n_past_steps>1): stack K past flow fields as ControlNet input channels, or K separate zero-conv branches?
5. Polarity/flip: guarantee past_flow_gt goes through the identical rot180/leftprimary transform as flow_gt (reuse one code path; this is the historic silent-bug class).

## 10. Non-goals (this spec)
- Not touching the VLM (no motion tokens into VLM — that's the deferred "second step" if this works and we want the action head to directly benefit).
- Not changing the action head / coupler.
- Not the sparse-token (mu0/GAM) variant — listed only as a possible cheap alternative arm.
