> Historical result record. The 30k step label is the trainer's logged counter,
> not a verified 30k optimizer-update budget. See
> [20261029_l40_results.md](../docs/experiments/20261029_l40_results.md)
> for DeepSpeed/Accelerate GA semantics and recovery limits.

# ICLR27 StereoVLA Baseline Experiments

- 记录日期：2026-07-19
- 远端主机：`wangqw@10.15.89.238`（`fineserver`）
- 项目目录：`/home/wangqw/ICLR27/starVLA`
- Git 分支：`ffs_controlnet_dev`
- 本文档用途：记录刚完成的两个 30k-step baseline 训练与 LIBERO 评估结果。
- 两个实验均使用 l40 的 GPU 4–7；未切换到公司仓库 `eacsai/StereoVLA`。

## 共同训练设置

| 项目 | 设置 |
|---|---|
| Base VLM | `./playground/Pretrained_models/Qwen3.5-0.8B` |
| Action model | `DiT-B` |
| 每卡 batch size | 8 |
| GPU 数量 | 4（GPU 4、5、6、7） |
| Gradient accumulation | 4 |
| Effective batch size | 4 × 8 × 4 = 128 |
| 最大训练步数 | 30,000 |
| Seed | 42 |
| Gradient checkpointing | true |
| WandB entity | disabled |

## Experiment A：StarVLA primary+wrist baseline

### 训练配置

- Run ID：`qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717`
- Data mix：`libero_all`
- 评估输入：`primary,wrist`
- `stereo_cam_branch_enabled`：`false`
- `stereo_cam_rope_enabled`：`false`
- 训练配置：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717/config.yaml`
- 完整保存配置：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717/config.full.yaml`

### Checkpoint

- 30k checkpoint：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717/checkpoints/steps_30000_pytorch_model.pt`
- 文件大小：2,499,988,954 bytes
- Final model：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717/final_model`

### LIBERO 评估

- 评估设置：10 trials/task，全部 task（`EVAL_MAX_TASKS=-1`）
- 观测设置：`num_obs_frames=1`，`obs_indices=0`
- 评估输出：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_baseline_primary_wrist_l40_4gpu_bs8ga4_eff128_queued_after_stereo_fromscratch_0717/eval_step30000_primary_wrist_0719/`
- 每个 suite 生成 100 个视频。

| LIBERO suite | Success rate |
|---|---:|
| libero_goal | 100% |
| libero_object | 100% |
| libero_spatial | 99% |
| libero_10 | 85% |
| 平均（四个 suite 简单平均） | **96.00%** |

## Experiment B：Plain stereo baseline（primary+left_view）

### 训练配置

- Run ID：`qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717`
- Data mix：`libero_all_sfstereo_leftprimary`
- 评估输入：`primary,left_view`
- 左右视图约定：`primary` 是第一路输入，逻辑上的 `left_view` 是第二路输入；本实验没有使用 `right_view`。
- `stereo_cam_branch_enabled`：`false`
- `stereo_cam_rope_enabled`：`false`
- 结论：这是 plain stereo baseline，不是 Camera Branch positional encoding 版本。
- 训练配置：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/config.yaml`
- 完整保存配置：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/config.full.yaml`

### Checkpoint

- 30k checkpoint：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/checkpoints/steps_30000_pytorch_model.pt`
- 文件大小：2,499,988,954 bytes
- Final model：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/final_model`

### Smoke test

- 1 task / 1 trial 的 `libero_goal` smoke test 通过，success rate 为 1.0。
- Smoke 输出：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/eval_smoke_goal_0719/`

### LIBERO 评估

- 评估设置：10 trials/task，全部 task（`EVAL_MAX_TASKS=-1`）
- 观测设置：`num_obs_frames=1`，`obs_indices=0`
- 评估输出：
  `/home/wangqw/ICLR27/starVLA/playground/Checkpoints/qwen0p8_groot_stereo_plain_leftprimary_l40_4gpu_bs8ga4_eff128_recovery_retry_fromscratch_0717/eval_step30000_primary_left_view_0719/`
- 每个 suite 生成 100 个视频。

| LIBERO suite | Success rate |
|---|---:|
| libero_spatial | 96% |
| libero_object | 98% |
| libero_goal | 89% |
| libero_10 | 78% |
| 平均（四个 suite 简单平均） | **90.25%** |

## 对比结论

| Experiment | 输入 | Camera Branch positional encoding | 平均 success rate |
|---|---|---|---:|
| primary+wrist baseline | `primary,wrist` | No | 96.00% |
| plain stereo baseline | `primary,left_view` | No | 90.25% |

这两个结果不能被称为 Camera Branch 对比实验；如果要验证 Camera Branch，需要另行训练并评估 `stereo_cam_branch_enabled=true` 的 checkpoint。

## 复现注意事项

1. 两个 30k checkpoint、`final_model` 和四套 LIBERO 评估结果均已保存。
2. mono 实验的 `train.log` 后段包含一次分布式 launcher 的 `SIGABRT`（记录在 20k checkpoint 之后）；但 30k checkpoint、final model 和四个评估 lane 均存在并已产出结果。若重新复现，应优先检查该日志异常。
3. 评估日志中可见结束阶段的 EGL cleanup traceback；每个 suite 同时存在 `[RESULT]` 和 `lane done`，因此本记录采用这些最终结果。
4. 本记录只保存实验元数据和结果路径，不包含数据集、模型权重、checkpoint 二进制文件或凭据。
