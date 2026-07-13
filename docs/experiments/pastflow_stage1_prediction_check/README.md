# Past-flow stage-1 prediction check

Date: 2026-07-09 CST.

## Provenance

- Checkpoint run:
  `qwen0p8_groot_cascade_motion_cambranch_pastflow_learn_scene_flow_fromscratch_leftprimary`
- Checkpoint: `steps_30000_pytorch_model.pt`
- Evaluation tool later formalized in commit `2bfbe33`
- Analytic GT generator: `gt-sceneflow-impl@f447e1c`
- Data mix: `libero_all_sfstereo_leftprimary`
- Camera order: `right_view,primary`
- Sidecar convention: `sidecar_to_training_flip=rot180`

Historical command:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/4090d/eval_scene_flow_prediction.py \
  --n-samples 1 --batch-size 8 --n-euler-steps 10 --viz-top-k 1 \
  --out-dir docs/experiments/pastflow_stage1_prediction_check
```

The pre-`2bfbe33` evaluator checked `--n-samples` only between batches.
Consequently this command accepted four dynamic samples from its batch even
though `--n-samples 1` was requested. The JSON correctly records
`n_samples=4`. Commit `2bfbe33` adds the missing within-batch stop.

## Metrics over the four accepted samples

- dynamic-valid model EPE: 0.00730 m
- zero-flow EPE: 0.01081 m
- relative EPE: 0.6753
- direction cosine: 0.7573
- dynamic-valid cells: 81

This is partial predictive ability: it beats zero flow and tracks the main
motion direction, but it is not a precise full-field predictor. The temporal
figure shows the two past-flow inputs, post-flow GT, and sampled post-flow
prediction.
