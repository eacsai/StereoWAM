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
