# Round-2 eval — scene-flow (online GradNorm vs static λ=130) + LLaMA-Adapter per-head — 4-suite (2026-06-25)

Eval on 4090d, 30k ckpts, **stereo `--args.video-keys right_view,primary` (RIGHT FIRST — these are
`*_rightprimary` models; see the camera-order incident below)**, baseline 0.06, 10 tasks × 10 trials = 100
episodes/suite. Trained on A800 (all4 `libero_all_sfstereo_rightprimary`, BS16×GA8 eff128; the two scene-flow runs
are warm-start B; per-head is fromscratch full-finetune).

> ## ⚠️⚠️ CAMERA-ORDER INCIDENT (2026-06-25) — a first eval pass was WRONG, do not cite it
> A first pass eval'd these right-primary models with `primary,right_view` (the `eval_one_ckpt.sh` default) →
> reversed cameras → systematically wrong stereo geometry / cam_rope camera-IDs / FFS left-right disparity → small
> persistent action errors that compounded on the long-horizon suite. It produced bogus low numbers
> (libero_10 0.36–0.51, "FFS+scene-flow hurts libero_10") that are ENTIRELY ARTIFACTUAL. Re-running with the
> correct `right_view,primary` order (matching training: data_mix `*_rightprimary`, data_config
> `[right_view, primary]`, FFS enforces `right_view_idx=0`) recovered everything (libero_10 0.62–0.74). The table
> below is the CORRECT `right_view,primary` result. Lesson: stereo eval MUST pass `right_view,primary`, never the
> script default. (codex found this; fix queued: make eval_one_ckpt.sh derive order from the ckpt's data_mix.)

## Results — success rate, CORRECT `right_view,primary` order

| run_id | what it is | goal | object | spatial | 10 (long) | **4-suite avg** |
|---|---|---|---|---|---|---|
| `qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k` | **baseline** (stereo cam_rope, NO FFS/scene-flow) | 0.91 | 0.95 | 0.94 | 0.74 | **0.885** |
| `qwen0p8_groot_depthtoken_keep_sceneflow_all4_gradnorm_warmstartB_30k` | FFS depthtoken-keep + scene-flow, **online GradNorm** | 0.89 | 0.96 | 0.93 | 0.74 | **0.880** |
| `qwen0p8_groot_depthtoken_keep_sceneflow_all4_lambda130_warmstartB_30k` | FFS depthtoken-keep + scene-flow, **static λ=130** | 0.92 | 0.96 | 0.91 | 0.72 | **0.878** |
| `qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k` | LLaMA-Adapter prefix + per-head gate, **fromscratch full-finetune** | 0.90 | 0.94 | 0.89 | 0.62 | **0.838** |

## Takeaways (with the corrected numbers)
- **Online GradNorm ≈ static λ=130** (0.880 vs 0.878, identical within noise on every suite). The online
  grad-balancing controller does NOT beat a fixed scene-flow loss weight → not worth its added complexity.
- **FFS depthtoken + scene-flow ≈ the clean stereo baseline** (0.878–0.880 vs 0.885): on (near-)saturated LIBERO,
  adding FFS geometry + scene-flow aux neither helps nor hurts. Consistent with the standing project finding —
  geometry/depth injection ≈ baseline on saturated 4-suite; the decisive test is OOD / generalization splits.
- **#5 per-head (fromscratch full-finetune) ~4–5pp lower** (0.838), mostly from libero_10 (0.62 vs 0.72–0.74) —
  fromscratch under-performs the warm-start arms on long-horizon. (Per user 2026-06-24, intentionally full-finetune;
  name it "full-finetune + adapter-prefix", not "faithful LLaMA-Adapter".)
- libero_10 (long-horizon, ~270-frame episodes) is the hardest/most-discriminating suite (0.62–0.74); object/spatial
  near-saturated (0.89–0.96); goal ~0.90.

Raw logs: `4090d:/data/wangqiwei/ICLR2026/eval_round2_RIGHTPRIMARY_20260625/<run>_<suite>/{server,client}.log`.
(Eval gotchas — config.yaml needs attn_implementation, FFS_REPO_DIR + 4090d ffs path, and the camera order above —
recorded in memory `[[project_autolaunch_watcher]]`.)
