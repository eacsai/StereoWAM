# Dynamic-focus stage-1 scene-flow visualization

Date: 2026-07-12 CST

This is a controlled one-sample comparison of two step-30000 stage-1
scene-flow checkpoints. Both evaluations use seed 0, 10 Euler steps, and the
first dataset sample with both past-flow history and raw dynamic-valid cells.
Matching GT statistics confirm that both runs evaluated the same sample.

## Provenance

- Model/training implementation: `368b072`
- Evaluation tool: `2bfbe33`
- Queue launchers: `0c11a32`
- Analytic GT generator: `gt-sceneflow-impl@f447e1c`
- Data mix: `libero_all_sfstereo_leftprimary`
- Suites and indexed episodes:
  `object=451`, `goal=427`, `spatial=434`, `libero_10=383`
- Sidecar convention: `sidecar_to_training_flip=rot180`
- Camera order: `right_view,primary`

The A800 checkpoint tensors and their original `config.full.yaml` files were
not uploaded. The launchers above are the reproducible configuration source.

## Runs

- Regular past-flow injection:
  `qwen0p8_groot_cascade_motion_cambranch_pastflow_dynamic_focus_stage1_fromscratch_leftprimary`
- Full past-flow injection:
  `qwen0p8_groot_cascade_motion_cambranch_pastflow_fullinject_dynamic_focus_stage1_fromscratch_leftprimary`

Both used `steps_30000_pytorch_model.pt`.

## Evaluation commands

Regular run on A800B:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/4090d/eval_scene_flow_prediction.py \
  --run-id qwen0p8_groot_cascade_motion_cambranch_pastflow_dynamic_focus_stage1_fromscratch_leftprimary \
  --ckpt-step 30000 --n-samples 1 --batch-size 1 \
  --n-euler-steps 10 --viz-top-k 1 \
  --require-past-flow --require-raw-dynamic --seed 0 \
  --out-dir docs/experiments/dynamic_focus_stage1_visualization/regular
```

Full-injection run on A800C used the same arguments except:

```bash
--run-id qwen0p8_groot_cascade_motion_cambranch_pastflow_fullinject_dynamic_focus_stage1_fromscratch_leftprimary
--out-dir docs/experiments/dynamic_focus_stage1_visualization/fullinject
```

## One-sample metrics

| Injection | Dynamic EPE | Zero-flow EPE | Relative EPE | Cosine | Dynamic cells |
|---|---:|---:|---:|---:|---:|
| Regular | 0.0066 m | 0.0119 m | 0.5542 | 0.8628 | 9 |
| Full | 0.0066 m | 0.0119 m | 0.5510 | 0.8493 | 9 |

Shared sample diagnostics:

- valid cells: 216
- dynamic-valid cells: 9
- mean GT dynamic displacement: 0.0119 m
- skipped one earlier sample with no raw dynamic cells

Both checkpoints recover the main dynamic-flow direction and beat the
zero-flow baseline on this sample. Their difference is negligible here.
The regular model has slightly higher direction cosine, while full injection
has slightly lower relative EPE. This single sample is qualitative evidence,
not a statistically meaningful ranking.

Each run directory contains `flow_temporal_context_vs_pred.png`,
`flow_pred_vs_gt.png`, and `flow_pred_metrics.json`.
