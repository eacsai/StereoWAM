# Experiment registry — StereoVLA / LIBERO

**Canonical, git-tracked home for experiment results.** Checkpoints under `playground/Checkpoints/<run_id>/`
are **gitignored** — eval numbers there are NOT durable. Record every result HERE.

> ## ⚠️ Why this file exists (2026-05-29 lesson)
> A full `right_view` eval sweep (20k/25k/30k for the 3 core runs) ran on 05/28 but its results landed in
> **`/tmp/*.log` under misleading filenames** (`sweep_cnetvlembs_epi_25k.log` actually evaluated the
> `epipolar_ffs_controlnet_fromscratch` run; `sweep_cnetactionattn_initdisp_25k.log` evaluated the
> `controlvla_branch_initdisp` run) and were not in the session snapshot. On resume 05/29, 5 of 6 evals
> were **re-run redundantly**. Root cause: results in `/tmp` + non-descriptive names + stale snapshot.
> **Rule: log results here keyed by `run_id`; identify any eval by the `pretrained_path` inside its log,
> never by filename/dirname.**

## Eval protocol

- Suite: LIBERO **`libero_goal`**, 10 tasks × 10 trials = **100 episodes** (≈ ±3% std error; <3pp gaps = noise).
- Stereo ckpts: client MUST pass `--args.video-keys primary,right_view` (rightview baseline **0.06 m**).
  Default is `primary,wrist` — **never** eval stereo ckpts with wrist (gives wrong/lower numbers, e.g.
  no-epi 30k = 74% wrist vs 90% right_view; phase3b 30k = 84% wrist vs 96% right_view).
- Mono ckpts: `--args.video-keys primary`.
- Tooling: `scripts/4090d/eval_one_ckpt.sh` (one lane) / `scripts/4090d/sweep_25k30k_parallel.sh` (parallel sweep).

## 🧭 The stereo design space (2×2) — answers "what combos have we run?"

Stereo signal can enter two independent ways, and the epipolar attention mask is an independent toggle:

|  | **No ControlNet** (stereo via cam_rope only, framework `QwenPI`) | **ControlNet→V** (FFS residual into vl_embs, `QwenPIControlNetFFS`) | **ControlVLA** (FFS K/V branch in action attn, `QwenPIControlVLAFFS`) |
|---|---|---|---|
| **no epipolar** | phase3b d16 = **0.94/0.96** · phase3d d32 = 0.89 · phase3e d64 = 0.86 · groot d16 = 0.91 | NOEpipolar_ffs_controlnet = **0.90** | controlvla = **0.90/0.91** · +initdisp = 0.87 |
| **+ epipolar** | dc16_epipolar_0526 = **0.90** | epipolar_ffs_controlnet = **0.90** | (not run) |

(values = 30k right_view SR.) **Headline: the simplest plain cam_rope d16 (phase3b, 0.94–0.96) is the
best of all variants; no FFS-ControlNet / ControlVLA / epipolar combination beats it on libero_goal.**

## A. Core Phase-3 FFS comparison (QwenPI 0.8B, cam_rope d_c16, FFS fed by right_view, 30k)

Shared: Qwen3.5-0.8B, LayerwiseFM action head (24 DiT layers), `libero_goal_stereo`, warmup 5k, cosine LR,
grad-accum 4, eff-batch ≈ 96. FFS frozen, pyramid level 0 (`ffs_scale`=pyramid index, NOT a scale), zero-init learnable residual.

| run_id | framework | epipolar | branch | 5k | 10k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|---|---|
| `..._NOEpipolar_ffs_controlnet_fromscratch_4090d_0527` | QwenPIControlNetFFS | OFF | residual→vl_embs | (45% wrist) | — | — | 0.90 | 0.92 |
| `..._epipolar_ffs_controlnet_fromscratch_h100b_0527` | QwenPIControlNetFFS | ON | residual→vl_embs | 0.31 | 0.74 | 0.91 | 0.88/0.89 | 0.91/0.90 |
| `..._controlvla_branch_fromscratch_h100b_0527` | QwenPIControlVLAFFS | OFF | K/V in action attn | — | — | — | 0.90 | 0.90/0.91 |
| `..._controlvla_branch_initdisp_fromscratch_h100b_0528` | QwenPIControlVLAFFS | OFF | K/V + use_init_disp | — | 0.80 | — | 0.90 | 0.87 |

`a/b` = 05/28 / 05/29 (consistent). Per-run detail docs: `docs/experiments/<run_id>.md`.

## B. Warm-start / init variants (right_view)

| run_id | framework | init from | 5k | 10k |
|---|---|---|---|---|
| `..._epipolar_ffs_controlnet_init_epi25k_h100b_0527` | QwenPIControlNetFFS | epi-fromscratch 25k | 0.82 | 0.83 |
| `..._NOEpipolar_ffs_controlnet_init_camrope30k_4090d_0527` | QwenPIControlNetFFS | camrope 30k | 0.85 | — |
| (launcher only, no ckpts found) `..._controlnet_actionattn_initdisp_from_baseline96` | QwenPIControlVLAFFS | phase3b baseline | — | — |

## C. cam_rope-only stereo (NO FFS branch) — framework QwenPI / QwenGR00T

The "no ControlNet" column of the 2×2. Stereo enters purely through camera-frame RoPE (or, for nocamrope, raw stereo data).

| run_id | backbone | cam_rope | epipolar | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|---|---|---|
| `goal_pi_qwen0p8_camrope_dc16_epipolar_0526` | QwenPI 0.8B | d16 | **ON** | 0.74 | 0.79 | 0.86 | 0.91 | 0.92 | 0.90 |
| `goal_phase3b_camrope_0523` | QwenPI 0.8B | d16 | off | — | 0.86 | 0.88 | 0.82 | — | **0.94/0.96** |
| `goal_phase3d_camrope_dc32_0524` | QwenPI 0.8B | d32 | off | — | — | — | 0.89 | — | 0.89 |
| `goal_phase3e_camrope_dc64_0524` | QwenPI 0.8B | d64 | off | — | — | — | — | — | 0.86 |
| `goal_groot_qwen0p8_camrope_dc16_0525` | QwenGR00T 0.8B | d16 | off | — | — | — | — | — | 0.91 |
| `goal_phase2_stereo_nocamrope_0524` | QwenPI 0.8B | **off** | off | — | — | — | — | — | 0.89 |
| `goal_groot_qwen4b_camrope_dc48_copyinit_0525` | QwenGR00T 4B | d48 copyinit | off | — | — | — | — | — | (incomplete) |
| `goal_phase3c_v3_groot_copyinit_0525` | QwenGR00T 4B | d48 copyinit | off | — | — | — | — | — | (incomplete) |
| `goal_phase3b_camrope_30k_0522` | QwenPI 0.8B | d16 | off | **NOT EVALUATED** (6 ckpts on disk) | | | | | |

## D. Mono baselines (video_keys=primary)

| run_id | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|
| `goal_only_30k_qwenpi_0519` | 0.91 | 0.90 | 0.94 | 0.96 | 0.94 |
| `goal_only_30k_mono_primary_official_0520` | 0.88 | 0.81 | 0.94 | 0.88 | 0.91 |
| `goal_mono_newrender_0523` | 0.89 | — | 0.85 | — | — |

> ⚠️ **CORRECTION (2026-05-29):** `goal_only_30k_qwenpi_0519` is `data_mix=libero_goal` = **primary+WRIST** (Libero4in1DataConfig default), NOT mono — its 0.94/0.96 reflect a *second view (wrist)*, not mono. **True mono-primary** = `goal_only_30k_mono_primary_official_0520` (0.88/0.91, mix libero_goal_mono_official) and `goal_mono_newrender_0523` (0.85-0.89, video_keys=[primary]). So mono-primary ≈ **0.85-0.91**, and adding a second view (wrist OR stereo right_view) lifts it to ~0.94-0.96.


## 🔑 Headline findings

1. **`libero_goal` is saturated.** Mono (0.91–0.96), plain cam_rope stereo (0.86–0.96), FFS-ControlNet (0.90),
   ControlVLA (0.90–0.91), epipolar on/off, d_c 16/32/64, cam_rope on/off — **all ~0.90 at 25k/30k, within ±3% noise.**
2. **Plain cam_rope d16 (phase3b, 0.94–0.96) is the strongest stereo config** and is NOT beaten by any
   FFS-ControlNet / ControlVLA / epipolar addition. The extra injection machinery shows no gain here.
3. **Stereo ≈ mono at convergence** → no measurable stereo benefit on this suite. Variants only separate in
   the *early* curve (no-epi 5k=45% vs epi 5k=31%; epipolar slows early training).
4. **Open**: to make any stereo / ControlVLA contribution measurable, need a harder / less-saturated suite
   (libero_10/long, LIBERO-plus) or a perturbation eval (camera shift, occlusion) where depth must matter.

## Notes / caveats (from exhaustive 2026-05-29 audit of all 72 eval logs)

- `ffs_scale` in configs = FFS feature-**pyramid level index** (`ffs_pyramid[ffs_scale]`, 0=finest), NOT a residual scale. Rename candidate: `ffs_pyramid_level`.
- `sanity_baseline_with_controlvla_branch_zero` is a zero-init parity sanity check, not a real experiment.
- ControlVLA framework files (`QwenPI_ControlVLA_FFS.py`, `controlvla_branch.py`) are **untracked** (not yet committed to `ffs_controlnet_dev`).
- Many runs have BOTH a `primary,wrist` (wrong, lower) and `primary,right_view` (correct) eval — always cite the right_view one for stereo ckpts.

## How to log a new experiment

1. Create `docs/experiments/<run_id>.md` (copy a core-run doc as template).
2. Add a row to the relevant table above.
3. Record run_id, framework + config deltas, dataset, train setup, per-step right_view SR, conclusion.
4. Do it right after the eval finishes — never leave results in `/tmp`.

## E. FFS-injection comparison — does introducing FFS beat plain cam_rope?

All right_view, libero_goal, 100ep. **Reference (NO FFS): plain cam_rope d16 `goal_phase3b_camrope_0523` = 0.94/0.96 @30k.**
Every row below injects FFS *backbone (monocular)* features — EXCEPT the gru_hidden row (post-cost-volume net[0], the real stereo signal).

| injection mechanism | run | init | 5k | 10k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|---|
| ControlNet→V (residual into vl_embs) | NOEpipolar_ffs_controlnet_fromscratch | fromscratch | (0.45) | — | — | 0.90 | 0.90–0.92 |
| | epipolar_ffs_controlnet_fromscratch | fromscratch | 0.31 | 0.74 | 0.91 | 0.89 | 0.90 |
| | epipolar_ffs_controlnet_init_epi25k | warm@epi25k | 0.82 | 0.83 | — | — | — |
| | NOEpipolar_ffs_controlnet_init_camrope30k | warm@camrope30k | 0.85 | — | — | — | — |
| ControlVLA branch (parallel K/V in action attn) | controlvla_branch_fromscratch | fromscratch | — | — | — | 0.90 | 0.90–0.91 |
| | controlvla_branch_initdisp_fromscratch | fromscratch | — | 0.80 | (15k 0.74) | 0.90 | 0.87 |
| | controlvla_..._initdisp_from_baseline96 | warm@phase3b | — | 0.87 | — | 0.92 | 0.92 |
| **ControlVLA + gru_hidden (REAL post-CV stereo net[0])** | controlvla_gruhidden_fromscratch_0529 | fromscratch | 0.42 | 0.74 | 0.84 | 0.89 | **0.92** |
| | controlvla_gruhidden_**warmstart**_0529 | warm@phase3b | 0.82 | 0.90 | (20k 0.94) | 0.94 | **0.92** |
| **REFERENCE (no FFS)** | **goal_phase3b_camrope_0523** | fromscratch | — | 0.86 | 0.82 | — | **0.94/0.96** |

### Findings
1. **No FFS-injection variant beats the no-FFS cam_rope baseline (0.94–0.96).** All converge to 0.87–0.92.
   Injecting FFS *backbone (monocular)* features is net-neutral-to-slightly-harmful on this (saturated) suite.
2. ControlNet→V (~0.90–0.92) ≈ ControlVLA branch (~0.87–0.92) at convergence — the injection mechanism does not matter.
3. warm-start consistently > fromscratch (ControlVLA+initdisp: 0.92 vs 0.87; ControlNet epi warm@5k=0.82 vs fromscratch 0.31),
   but neither clears the baseline.
4. init_disp adds nothing (ControlVLA+initdisp fromscratch 0.87 ≤ plain ControlVLA fromscratch 0.90–0.91).
5. **Open: gru_hidden** is the only run that injects the REAL post-cost-volume stereo feature (net[0]) instead of
   monocular backbone features — the test of whether a genuine stereo signal finally changes this picture.
   (All prior FFS rows were later found to inject monocular features — see project_ffs_features_are_monocular.)

## F. VLM-ControlNet 3-run ablation (true VLM ControlNet + zero-conv, real stereo net[0])

Detail: `pi_qwen0p8_vlmcontrolnet_ffs_3run_0601.md`. Framework `QwenPIVLMControlNetFFS`: frozen Qwen VLM +
trunk-copy of the 6 cam_rope softmax layers + per-depth zero-conv residual into VLM hidden states; FFS
gru_hidden net[0] (real post-cost-volume stereo). Reference (no FFS) = `goal_phase3b_camrope_0523` 0.94/0.96.

| run | what trains | 10k | 20k | 30k |
|---|---|---|---|---|
| Run1 head-only (control, no FFS) | action head only | 0.88 | 0.91 | 0.92 |
| Run2 head+ControlNet | head + branch | 0.93 | 0.96 | 0.93 |
| Run3 pure ControlNet (frozen head) | branch only | 0.95 | 0.93 | 0.95 |

Decomposition @30k (baseline 0.94): Run3-baseline=+0.01, Run2-Run1=+0.01, Run1-baseline=-0.02 — all within
+/-3-5% noise = **no signal** (libero_goal saturated). **Key contrast vs section E**: this true VLM-ControlNet
is the **first FFS-injection that stays AT baseline instead of drifting down to 0.87-0.92** (zero-conv +
trunk-copy avoids the drift the section-E variants suffered). Clean but value-neutral here; decisive test needs a
non-saturated depth/occlusion suite (stereo4d proposal section 5).

## G. Depth-token FFS 3-run ablation (FFS feature as SEQUENCE TOKENS, not a residual)

Detail: `pi_qwen0p8_depthtoken_ffs_3run_0603.md`. New injection family: the FFS post-cost-volume stereo feature
(gru_hidden net[0]) is `adaptive_pool`'d to 16 cells and inserted as **16 depth TOKENS** into the Qwen
image+text sequence (S→S+16, the VLM's self-attention fuses them; tokens sliced out before the action expert
so downstream is unchanged). Distinct from sections E/F which inject a *residual* without changing sequence
length. Frameworks `QwenPIDepthTokenFFS` / `QwenPIDepthTokenLoRAFFS`. Reference (no FFS) = `goal_phase3b_camrope_0523` 0.94/0.96.

| run | adaptation | LR (projector) | 10k | 20k | 30k |
|---|---|---|---|---|---|
| Run A frozen-VLM (warm-start 0.94) | projector + head; VLM frozen | base ~1e-5 (low) | 0.89 | 0.92 | 0.90 |
| Run B LoRA (warm-start 0.94) | LoRA r16/α32 + projector + head | **1e-4 (high)** | 0.83 | 0.94 | 0.93 |
| Run C full / fromscratch (STRESS) | everything, no warm-start | base | 0.57 | 0.88 | 0.94 |

Decomposition @30k (baseline 0.94): A−base=−0.04, B−base=−0.01, C−base=0.00 — all within ±3-5% noise = **no
signal**. **A-vs-B is a clean null**: LoRA + high projector LR does NOT beat frozen + low LR. All three
adaptation paradigms land in the **same 0.90-0.94 band** as gru_hidden (0.92) and VLM-ControlNet (0.92-0.95)
→ Nth confirmation the bottleneck is the **suite, not the mechanism / feature / capacity / adaptation strategy**.
Run C is confounded (fromscratch+full+depth) and not a clean attribution point. Decisive test still needs a
non-saturated depth/occlusion suite (stereo4d proposal section 5).

## H. Qwen2.5-VL-3B 4-suite input-view ablation (stereo vs mono vs primary+wrist) — IN-PROGRESS

Strong-backbone (Qwen2.5-VL-3B + GR00T, **no cam_rope, no FFS**) plain input-view baselines on the
NEW 4-suite stereo data (spatial/object/goal/libero_10 joint, eff128, 30k, fromscratch). All 3 runs
share ONE dataset (`LEROBOT_LIBERO_STEREO_4SUITE`); only the read channels differ → stereo−mono is a
clean apples-to-apples (only right_view differs). Detail + live results table:
`docs/experiments/qwen2p5vl_3baseline_inputview_0604.md`.

| run | step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|---|
| stereo (primary+right) | 10k | 0.45 | 0.39 | 0.53 | 0.07 | ≈0.36 |
| stereo / mono / primary+wrist | 20k,30k | — | — | — | — | running (h100b chain) |

10k stereo = short-validation gate (PASSED: pipeline OK, state=None train/eval matched, numbers
in-range for early fromscratch 4-suite-joint). Purpose = validate stereo on strong backbone + hard
data before adding cam_rope / FFS-hybrid. Auto: training launchers in `scripts/h100b/` (run_qwen2p5vl3b_groot_4suite_stereo.sh + chain_*.sh),
eval daemon `scripts/4090d/auto_eval_qwen2p5vl_3baseline.sh`.


## I. LLaMA-Adapter prefix FFS injection (#5) — frozen-VLM arm DONE

Method #5 of the 8-method FFS port: a few learnable prompt tokens absorb the FFS `net[0]` stereo
features and are **zero-init-tanh-gated, prepended** into the last 6 softmax layers of the
Qwen3.5-0.8B VLM (faithful to LLaMA-Adapter v1: tanh gate, prefix un-RoPE'd, adapter-only training).
The **frozen** arm warm-starts from B and freezes the VLM (single-scalar gate); the **per-head**
from-scratch arm was SIGKILLed in the 2026-06-13 host-OOM and is not yet rerun. Detail:
`docs/experiments/qwen0p8_ffs5_llama_adapter_prefix_0611.md`.

| run | step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|---|
| frozen (warm-start B, freeze VLM) | 20k | 0.92 | 0.95 | 0.85 | 0.71 | 0.858 |
| frozen (warm-start B, freeze VLM) | 30k | 0.91 | 0.96 | 0.95 | 0.68 | **0.875** |
| per-head (from-scratch) | — | — | — | — | — | OOM-killed, not rerun |

**Verdict: #5 frozen 0.875 < mono 0.913 < depthtoken_keep 0.910 — in the 0.85–0.91 noise band, no
gain.** One more confirmation that freezing the VLM and learning only a LLaMA-Adapter prefix on
stereo-depth features does not beat mono on saturated LIBERO. See headline finding above.
