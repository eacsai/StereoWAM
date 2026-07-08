# Spec — Scene-Flow Dual-DiT Cascade for StereoVLA

Status: DRAFT for codex design review (codex-implement Stage 1.5) → then Claude writes on 4090d.
Supersedes the earlier `sceneflow-staged-inject-spec.md` (that was an action-side light-head design; the finalized architecture below is a DiT4DiT-style dual-DiT cascade).
Repo (authoritative): `4090d:/data/wangqiwei/ICLR2026/starVLA` (branch `ffs_controlnet_dev`). NESTED package: repo-relative `starVLA/model/...` = abs `.../starVLA/starVLA/model/...`.

## 0. One-paragraph summary
A shared VLM (Qwen3.5-0.8B + cam_branch) encodes the raw stereo pair + language into `vl_embs`. Two downstream flow-matching **GR00T-DiTs** consume it: a **scene-flow DiT** (predicts future 3D scene flow) and the existing **action DiT**. Following DiT4DiT, the action DiT additionally reads the scene-flow DiT's tapped mid-late hidden through a NEW **zero-init gated cross-attention** (our deviation from DiT4DiT, which uses a bare add and leans on a pretrained Cosmos — we are from-scratch so we need the gate). Stage-1 pretrains the scene-flow DiT (+VLM) on flow only; Stage-2 joint-trains both DiTs (+VLM) with the coupling **detached**. The research bet: does injecting a **predicted-future-motion** hidden beat injecting a **static-geometry** hidden (a matched static-target-DiT twin) — measured on an OOD split, since standard LIBERO is saturated.

## 1. Goal & hypothesis
Research question: **does a predicted-future-dynamics (scene-flow) representation, coupled into the action DiT, beat a static-geometry representation coupled identically?**
The claim only lives on a **generalization / OOD split** (standard LIBERO is saturated; our prior scene-flow diagnosis found it neutral there). Non-goal: a higher standard-LIBERO number.

Grounding (verified this session):
- Scene-flow GT is GOOD and unchanged: `/data/wangqiwei/ICLR2026/gt_sceneflow_impl/gen_gt_sceneflow_rerender.py` (`GT_METHOD="sim_rerender_analytic"`, OUTSIDE the repo cwd — a codex `-C starVLA` run won't see it, that's expected) = per-pixel MuJoCo segmentation → per-body GT pose delta → exact rigid 3D motion (camera-frame meters).
- The prior Utonia+sceneflow −9pp was joint interference at the action DiT's **final** layer (`GR00T_ActionHeader.py`, `scene_flow.hidden_layer=-1`). The cascade **structurally avoids** this: the flow objective trains a SEPARATE DiT, not the action DiT's own layers.
- Precedents (deep-read this session): DiT4DiT (dual-DiT cascade, bare coupling, warm-starts Cosmos), DepthVLA (3-expert MoT with the VLM inside a shared-attention mixture — we are NOT copying that; ours is a two-DiT cross-attn cascade with the VLM as an upstream encoder). DepthVLA endorsements we DO take: "predict the modality > feed GT" and "the expert needs a good init / pretraining, not random init".

## 2. Final architecture (data flow)
```
raw stereo (primary+left_view) + language
        │
   Qwen VLM (+cam_branch)  → vl_embs        [shared upstream encoder; NOT a mixture expert]
        │
   ┌────┴───────────────────────┐
   ▼                             ▼
 scene-flow DiT (GR00T-arch)   action DiT (GR00T-arch, existing head)
   cross-attn → vl_embs          cross-attn → vl_embs            (unchanged path)
   flow-matching → scene flow    + NEW zero-init gated cross-attn → scene-flow-DiT tapped hidden (detached)
   (16×16×3 camera-frame)        flow-matching → action chunk
```
- Both DiTs are the **same GR00T DiT architecture** (DiT-B: dim 768, 16 layers, `interleave_self_attention=True` → 8 cross-attn (even) + 8 self-attn (odd) layers). Verify: `GR00T_ActionHeader.py:191`, `flow_matching_head/cross_attention_dit.py:230-233`.
- The scene-flow DiT is a NEW module family; the action DiT is the existing `FlowmatchingActionHead`.
- Coupling detail in §4.

## 3. Three configurations (single-variable comparison) — addresses codex H1/H5/H6
Descriptive names, minimal set:
1. **plain-stereo** (anchor): VLM (+cam_branch) + action DiT, NO third DiT, NO coupling. Establishes the backbone floor (does EITHER coupled feature help?).
2. **static-geometry twin** (strict baseline): identical to config 3 but the third DiT is trained (stage-1) to predict the **current-frame camera-frame XYZ pointmap** (3-channel, SAME 16×16 grid / loss / mask / bound policy as flow) instead of future flow; coupled identically. Only the target's *temporal content* (current vs future) differs → true single variable. NOT depth-only (changes channels/density), NOT zero-flow (that ≈ plain-stereo → keep zero-flow only as a negative-control diagnostic). Generate current-pointmap GT from the same sim-rerender pipeline (unproject current metric depth + intrinsics). (round-2 N2)
3. **scene-flow motion** (the bet): third DiT predicts **future scene flow**; coupled identically.
Single variable between (2) and (3) = the third DiT's **training target** (current geometry vs future motion); EVERYTHING else identical — same DiT architecture, same tap layer, same coupling, same gate, same token/hidden shapes, same freeze/joint schedule, same VLM+cam_branch, same data/steps/optimizer/eval order (`primary,left_view`), wrist-free.
- #5 (VLM-side Utonia 0.8825) and prior runs are **reference numbers only** (different injection paradigm) — never the A/B partner (fixes the §-contradiction codex H5 flagged).
- Standard LIBERO gates ONLY wiring / step-0 identity / no-crash / **parity within ~6pp noise** among {plain, static, motion}. It CANNOT show motion>static (saturated). The win/lose verdict is on the OOD split (§10).

## 4. Coupling (DiT4DiT-style single-tap + zero-init gate) — addresses H4
- **Tap**: ONE **mid-late** layer of the scene-flow DiT (AVOID the final layer — output-specialized, the −9pp lesson). Start ~block 10-12 of 16; ablate {8,10,12}. Semantics: `all_hidden_states = [input_emb] + per-block outputs` (len 17 for 16 layers; `cross_attention_dit.py:279`), so "tap block k" = `all_hidden_states[k]` = output of transformer_block[k-1]; state the index explicitly in config.
- **Conditioning extraction (⭐ round-2 N1, label-leakage fix, VERIFIED):** the action DiT MUST read the scene-flow hidden from a SEPARATE `extract_conditioning_hidden()` path that **never sees `flow_gt` / the static GT**. A flow-matching DiT mixes its target into the token stream (`noisy = (1-t)*noise + t*target`, verified `GR00T_ActionHeader.py:424`), so tapping the *supervised* (teacher-forced) forward would leak future GT into the action → inflated training, collapse at eval. This extraction = a SINGLE forward at a **deterministic** input (fixed `motion_extract_t_bucket` + `motion_extract_noise_mode`, per-sample deterministic), detached, with **identical construction in train / eval / `predict_action`** (round-2 N5). It is DISTINCT from the supervised flow-loss forward — state explicitly whether the flow loss reuses this pass or runs a second logged pass. Bounds inference to ~2× (single forward, not N×).
- **Detach** the tapped hidden before it enters the action DiT (Stage-2 gradients from the action loss do NOT flow into the scene-flow DiT).
- **Inject**: at the action DiT's existing 8 cross-attn (even) layers only (NOT all 16 — codex MEDIUM), a NEW gated cross-attention added as a residual:
  `hidden = hidden + zero_out(motion_xattn(norm(hidden), tapped_hidden))`, where `zero_out` is a **zero-initialized final Linear** (codebase idiom — there is NO adaLN-zero in `BasicTransformerBlock`). All 8 injectors read the same tapped hidden (single-tap broadcast). Per-layer aligned read (flow block L → action block L) is the FIRST ablation upgrade (DepthVLA-endorsed, ~same params) if single-tap plateaus.
- **Step-0 identity (H4 + round-2 N4, verified)**: the DiT uses `dropout: 0.2, final_dropout: True` (`QwenGR00T.py:163`); the motion branch MUST use `dropout=0`, and the zero-init Linear must be the FINAL op immediately before the residual add (no motion-conditioned norm/adaLN on the main path). ⭐ PLUS: the scene-flow extraction forward runs BEFORE the action head and consumes dropout/noise RNG → shifts the action head's RNG stream → breaks byte-identity even with dropout=0. Wrap ALL scene-flow extraction in `torch.random.fork_rng` (snapshot-restore the global generator) so the action stream is untouched. GPU-smoke asserts enabled-vs-disabled TENSOR equality with RNG restored in BOTH `eval()` and `train()` — `zero_proj_w_linf`, `zero_proj_b_linf`, per-injector `motion_residual_linf`, `dit_output_maxabsdiff`, `action_loss_absdiff`, plus a NONZERO `zero_proj_grad_linf` — same batch/noise/t, before optimizer step, print `STEP0_EQ_BASELINE=PASS`.

## 5. Training pipeline — addresses H2 + round-2 N3 (VLM purity + two-source init)
- **Stage-0 (plain-stereo pretrain = config 1 anchor):** train a plain-stereo policy (VLM + cam_branch + action DiT, no third DiT, no coupler). This checkpoint is the SHARED init for BOTH the static and motion arms → the VLM is identical across arms (a motion win cannot come from VLM reshaping).
- **Stage-1 (third-DiT pretrain, VLM FROZEN):** from the Stage-0 checkpoint, **freeze the VLM + cam_branch**, train ONLY the third DiT (`scene_flow_dit` for motion / `static_geometry_dit` for the twin) on its target. Needs a real **target-only trainer mode** — `train_only` merely freezes params; the loop always forms `total_loss = action_loss (+ flow_weight*flow_loss)` (`train_starvla.py:1050`, verified). Add `loss_mode=flow_only` that skips action loss/eval/audits and backprops only the third-DiT target loss (set `dynamic_fallback_to_valid=False` here — round-2, verified `QwenGR00T_FFSCommon.py:680`: fallback trains the DiT to predict zero on no-motion frames). Good geometric init (DepthVLA: random-init 51% vs pretrained 74.8%). **Kill criterion:** do NOT proceed to Stage-2 unless the third DiT beats the zero/no-motion baseline on held-out flow with a pre-registered margin (§7).
- **Stage-2 (joint):** BOTH arms init VLM+action IDENTICALLY from the Stage-0 plain checkpoint + load their Stage-1 third DiT + a FRESH zero-init coupler (a **two-source audited merge**, §6). Then joint-train: `total_loss = action_loss + λ_flow·flow_loss`, λ_flow weak (~0.1). Coupling detached (§4). Only the third DiT's target (current pointmap vs future flow) differs between arms.
  - Knobs (ablation, not v1): freeze-vs-joint the third DiT in Stage-2 (DepthVLA: frozen ≈ trained); whether the VLM is trainable in Stage-2 (if trained, both arms train it identically from the same init → still fair).

## 6. Checkpoint / module-family rules — addresses H3 (verified)
- New families use clean TOP-LEVEL prefixes: `scene_flow_dit.`, `action_model.motion_coupler.` (the 8 gated x-attns), and `static_geometry_dit.` for the twin. Avoid `train_only` substring collisions (e.g. don't name things so `motion` matches both extractor and coupler).
- Add these prefixes to the warm-start **fresh-init allow-list** (currently only FFS + `action_model.scene_flow_decoder`; `QwenGR00T_FFSCommon.py:876`) so a stage-2 arch with new coupler keys loads a stage-1 ckpt without raising.
- **Do NOT** add `scene_flow_dit.` to `_frozen_encoder_key_prefixes()` (`QwenGR00T_FFSCommon.py:824`, verified: it STRIPS `ffs.` from the saved state_dict). If treated like frozen FFS, the stage-1-trained scene-flow DiT would be stripped from the checkpoint and stage-2 would fresh-init the "frozen" predictor. Freeze via `freeze_modules` only; stage-1 ckpt MUST contain the scene-flow DiT weights.
- All-or-none load audit for each new family. `train_starvla` resume is weight-only (AdamW reset, `train_starvla.py:866`) — stage-2 uses `pretrained_checkpoint` + fresh optimizer, NOT `is_resume`.
- **Freeze safety (H-MEDIUM)**: `freeze_modules` dotted-path typos only warn+continue (`trainer_utils/trainer_tools.py:202`). Add post-freeze assertions: target params `requires_grad=False`, absent from optimizer groups, `.eval()` reasserted in the frozen forward.

## 7. GT & loss
- Keep analytic sim-rerender `flow_3d`. Consume via dataloader `example["flow_gt"]/flow_valid/flow_dynamic` (which applies `sidecar_to_training_flip` via `_apply_spatial_alignment`, `datasets.py:127`) — NOT raw sidecar reads; assert camera frame `agentview` + no extra sign flip.
- Loss: smooth_l1 beta 0.01, z-channel 2×, robust valid_mask (existing). For the standalone scene-flow DiT: **near-zero flow head** (xavier gain≈0.01, zero bias) — NOT exact-zero (exact-zero gives no first-step gradient into the DiT; reserve exact-zero for the action-coupler residual gate). fp32 loss.
- **Flow scale/bound policy**: print GT flow p50/p95/p99 per channel; if bounding, `flow_bound_m ≥ 1.2·p99`; log clipped ratio. Stage-1 LR ~1e-4–3e-4 + grad clip 1.0.
- Diagnostics: **zero/no-motion baseline** (rename the "copy-current" idea — for dense flow, copy-current ≈ predict-zero) + held-out flow quality (dynamic-pixel EPE, directional cosine, predictor-vs-zero margin per suite) as a **stage-1 acceptance gate** before stage-2 (proves the flow target is informative + the DiT learns it). Visualize predicted vs GT flow.

## 8. Inference — addresses threading + cost
- ~2× baseline (both DiTs run). The scene-flow DiT runs a single high-noise forward per obs to yield the tapped hidden; the action DiT runs its normal denoise loop, reusing that hidden across denoise steps.
- Thread the motion hidden through BOTH `FlowmatchingActionHead.forward` AND `predict_action` (separate DiT call path, `GR00T_ActionHeader.py:484/530`), and down into `DiT.forward` → `BasicTransformerBlock.forward`. Compute the scene-flow hidden ONCE per observation.
- **Repeated-diffusion batch expansion (H-MEDIUM, verified)**: training repeats VLM/action tensors by `repeated_diffusion_steps` before the action head (`QwenGR00T_FFSCommon.py:736`). Compute the scene-flow hidden once on B, then REPEAT it (and any mask) by the same factor — do NOT recompute the scene-flow DiT after expansion (8× waste), and do NOT pass an unrepeated `(B,·)` into a `(B*R,·)` action head (shape mismatch).

## 9. Wiring gotchas
- **cam_branch gate is not just an allow-list**: `QwenGR00TFFSBase` allows only `QwenGR00T_UtoniaPromptTokenFFS` and sets `defer_position_cache_until_token_insert=True` (`QwenGR00T_FFSCommon.py:516`) — correct for VLM token-insertion, WRONG for our raw-stereo (no token insert) case. Either do NOT inherit `QwenGR00TFFSBase`, or split the gate into capability flags (token-inserting vs raw-stereo). The new framework class must still enable cam_branch.
- **Resolution/camera parity (H-MEDIUM)**: training resizes camera frames to 224 (`datasets.py:1767`); flow sidecars are 256; predict may resize via `obs_image_size`. Add a `scene_flow_dit_image_size` + assert train/eval image sizes and `[primary,left_view]` order before the scene-flow DiT forward.
- **Stale paths in the OLD spec, corrected here**: `cross_attention_dit.py` is under `starVLA/model/modules/action_model/flow_matching_head/`; `trainer_tools.py` is under `starVLA/training/trainer_utils/`.

## 10. Eval — LIBERO-first mechanism gate, OOD is Milestone 0 for the claim
- **Milestone 0 (BEFORE long runs, codex H-assumptions):** pick the OOD split, define + validate a SMALL subset, run {plain, static, #5-reference} baselines on it, lock metrics. `examples/LIBERO-plus` is an inert template + full set is ~10,030 tasks — do NOT discover feasibility after stage-2. Candidate: LIBERO-plus small subset (primary) + a stereo camera-viewpoint-shift eval (on-narrative bonus).
- Standard LIBERO = mechanism gate only (wiring/step-0/parity), per §3. Note the launcher default is 10 trials/task not the README 50 (`eval_one_ckpt.sh:148`).
- Power: single train+eval seed on saturated suites can drown a 2-6pp effect. Pre-register a minimum detectable effect; use paired eval seeds/initial states across arms; report CIs / paired bootstrap; multi-seed the final OOD comparison.

## 11. Adoptions from the 4-repo + DepthVLA deep-read
- **DiT4DiT**: the dual-DiT cascade skeleton + single-tap of a mid (not final) layer + detach + single-step hidden extraction. Our addition: the zero-init gate (they bare-add on a pretrained Cosmos).
- **DepthVLA**: "predict the modality > feed GT" (predict flow, don't feed it); "expert needs pretraining, not random init" (stage-1 flow pretrain + geometric init); "frozen expert ≈ trained" (freeze-in-stage-2 is a valid knob). We are NOT copying its 3-expert shared-attention MoT (our VLM is upstream, not a mixture member).
- **mu0**: inject the LATENT (tapped hidden), not decoded geometry — already the design. B-spline horizon compression = later option if we move to a trajectory target.
- **GaussianDream**: multi-horizon chaining (later ablation), near-zero head + staged warmup + weak stage-2 weight (adopted).
- **DreamVLA**: moving-region / arm-down-weight mask (optional ablation).

## 12. Phasing — LOCKED DECISIONS (user 2026-07-05)
Locked: static twin target = current camera-frame XYZ pointmap ✓; inference ~2× accepted ✓; **SKIP the standalone −9pp hygiene run (choice 乙)** — the cascade avoids −9pp by construction; keep the mid-tap-recovery only as a later paper ablation.
1. **Build (code; no long train yet):** current-pointmap GT gen (extend sim-rerender); `SceneFlowMatchingHead` (scene-flow DiT + flow-matching wrapper) + `flow_only`/target-only trainer mode; `action_model.motion_coupler` (zero-init gated x-attn) + `extract_conditioning_hidden` + threading (forward + `predict_action` + repeat + RNG fork); config schema `framework.scene_predictor.*`; checkpoint family rules. **GPU-smoke step-0 identity (both static & motion configs) BEFORE any long run.**
2. **Stage-0:** train plain-stereo anchor (shared init for both arms).
3. **Stage-1:** pretrain scene-flow DiT + static-geometry DiT (VLM frozen), flow-only; pass the acceptance gate (§7) + visualize; kill-criterion check.
4. **Milestone 0:** build + validate the small OOD split; run {plain, static-ref} baselines; lock metrics (must be before Stage-2).
5. **Stage-2:** two-source merge + joint train {static, motion}; standard-LIBERO parity gate; decisive OOD comparison.
6. **Ablations (deferred, memory `project_future_sceneflow_cascade_ablations`):** per-layer coupling; end-to-end (no staging/detach); tap-layer; freeze-vs-joint; arm-mask; multi-horizon; the −9pp mid-tap recovery.

## 13. Risks
- Predicted dynamics may TIE static on saturated LIBERO and only separate on OOD (honest prior). The zero/no-motion baseline + OOD split + acceptance gate are the referees.
- Joint stage-2 reintroduces mild shared-VLM competition (milder than −9pp); detach + weak λ + stage-1 pretrain + gate mitigate.
- 2× inference is the cascade cost (accepted); distill-to-baseline-cost is a later, separate line.
- Convention/flip discipline: pin flip/rotation at GT time, assert in loader (prior silent-bug history).
- from-scratch stability: gate + near-zero head + LayerScale/grad-clip; watch for bf16 dtype at the gate.

## 14. Open items for the codex review to pressure-test
Single-variable purity of the static-geometry twin (is "current geometry" the right static target, or should it be current-flow=zero / an identity target?); whether stage-1 pretrain of a from-scratch scene-flow DiT on LIBERO-scale flow is strong enough to carry motion signal; the tap-layer + detach choices; the OOD split feasibility; and any silent wiring bug in threading the tapped hidden through train (repeat) + `predict_action`.

## 15. Round-2 codex review resolutions (folded; supersede/augment the sections noted)
HIGH N1 (leakage) → §4 conditioning-extraction path. N2 (static target = current XYZ pointmap) → §3 config 2. N3 (VLM-purity + two-source init) → §5 Stage-0/1/2. N4 (RNG fork) + step-0 tensor smoke → §4. N5 (deterministic extraction, same train/eval) → §4.

**N6 — OOD Milestone 0 is a HARD gate (augments §10):** before ANY Stage-2 long run, LOCK an OOD manifest: task ids/categories, a **stereo camera adapter** (LIBERO-plus scripts feed `agentview+wrist`, NOT `primary,left_view` — must be adapted + asserted), a subset runner (full LIBERO-plus ≈ 10,030 tasks — use a validated subset), initial states/seeds, trial counts, paired eval protocol, baseline SR/variance, and a pre-registered MDE/CI method. Power: `eval_one_ckpt.sh:148` defaults to 10 trials/task (~400 rollouts over 40 tasks → ~7pp 80%-power MDE); use PAIRED init states/RNG across arms, paired bootstrap CIs, train-seed pairing, and multi-seed the final OOD comparison. Standard-LIBERO parity is a no-crash/no-regression smoke ONLY (report enabled-vs-disabled coupler + gate norms to prove the motion branch is actually used).

**MEDIUM (fold into implementation):**
- **`SceneFlowMatchingHead` contract (must define, not just "a GR00T DiT"):** the bare `DiT` is only a transformer over caller-supplied tokens; the action wrapper supplies noised tokens + a velocity target (`GR00T_ActionHeader.py:419-424`). Define for the flow/pointmap head: normalized target, noised-token encoder, timestep/noise schedule, **velocity-vs-reconstruction** target (state which), output decoder, loss mask, and the exact hidden slice used for coupling.
- **`flow_only` = real `_train_step` surgery** (`_train_step` unconditionally reads `output_dict["action_loss"]`, `train_starvla.py:1050`): add the mode, skip action loss/eval/scene-flow-grad-ratio audits, require dynamic-pixel supervision.
- **Coupler = clean side module:** `action_model.motion_coupler` as a separate `ModuleList` called from block forward — do NOT wrap `attn1` (ControlVLA-style wrapping renames trunk keys to `attn1.base.*` → breaks the clean-prefix ckpt rule). Assert it covers exactly the even cross-attn blocks `[0,2,…,14]`.
- **Config namespace:** put the new design under `framework.scene_predictor.*` / `framework.scene_conditioning.*`; the legacy aux decoder still owns `framework.action_model.scene_flow.*` (`QwenGR00T.py:129`) — assert mutual exclusion.
- **fp32 island:** wrap the flow/pointmap loss + reconstruction in `autocast(enabled=False)` + explicit `.float()`; cast the zero-gate residual back to `hidden.dtype` before the add (repo has non-robust autocast-fp32 spots, e.g. `QwenGR00T_FFSCommon.py:728`).
- **Repeat semantics:** `motion_hidden.detach().repeat(R,1,1)` (matches training's `repeated_diffusion_steps`, `QwenGR00T_FFSCommon.py:736`), NOT `repeat_interleave` (same shape, wrong sample order); add a B=2,R=3 sentinel expecting `[0,1,0,1,0,1]`. Compute the scene-flow hidden ONCE on B, then repeat.
- **Hidden-level acceptance gate (augments §7):** probe the EXACT tapped hidden (not just decoded flow) — linear-probe EPE/cosine vs zero on held-out dynamic pixels, a current-geometry-regressed residual-motion probe, static-vs-motion CKA, and a temporal-shuffle / time-reversal negative control. Add mean-flow / per-task-prior baselines beyond zero.
- **plain-stereo = anchor floor, not a causal "third-DiT helps" control** (it lacks the aux pretraining) — also run coupler-OFF controls for static/motion to isolate target-induced VLM-only gains.
- **v1 = exactly 3 configs** (plain / static-pointmap / scene-flow), one chosen tap, detach, zero gate. Defer tap-sweep, per-layer coupling, freeze-vs-joint, arm-mask, multi-horizon, B-spline to post-v1 ablations. The §12.1 mid-layer aux hygiene is optional/timeboxed, not a pre-Milestone-0 gate.

**LOW — corrected anchors (the OLD ones were imprecise):**
- Flow flip is applied via `_apply_spatial_alignment` (`datasets.py:127`, used in the scene-flow sample path ~`:881-899`); `datasets.py:926` is metadata validation, not flip application.
- `_frozen_encoder_key_prefixes` is at `QwenGR00T_FFSCommon.py:827` (not 824); warm-start allow-list edits touch both prefix construction (~`:881`) and allowed-missing logic (~`:897`).
- Tap indices: with `all_hidden_states=[input]+per-block`, `8/10/12` are POST-self-attn outputs; `9/11/13` are POST-cross-attn. State the choice deliberately; config key `tap_hidden_index`, validate `1 ≤ index < num_layers`, forbid `-1`/`16`.
- Add a synthetic rot180 test: spatial positions move, XYZ channel signs unchanged (camera-frame 3D flow convention).
