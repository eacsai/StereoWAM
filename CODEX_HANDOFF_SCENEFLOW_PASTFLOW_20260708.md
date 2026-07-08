# Codex 接手说明 — scene-flow / past-flow ControlNet（2026-07-08）

你是 Codex，后续这个项目完全交给你继续做；不要假设 Claude 会回来，也不要用 `/relay-takeover` 之类 Claude skill。按这份 Markdown 和仓库当前状态继续。

## 0. 最高优先级规则

- **代码唯一权威源**：`4090d:/data/wangqiwei/ICLR2026/starVLA`。所有代码修改只在这里做。
- **不要在 a800 / h100 / 本机快照上改代码**；其它机器只是运行/消费端，需要时从 4090d 单向同步。
- **不要擅自 kill 正常训练/eval**；kill 任何 `train_starvla` / `accelerate` / watcher 前必须问用户。
- **不要自动 push GitHub**。commit 可以先给用户看计划；push 必须用户明确批准。
- 面向用户用中文；讲解不用 LaTeX 公式。
- 结果和实验登记写进 `docs/experiments/`，不要留在 `/tmp`。

## 1. 当前机器 / 路径

- 4090d 权威仓库：`/data/wangqiwei/ICLR2026/starVLA`
- a800a：`ssh -p 30947 wangqiwei@10.13.32.4`，repo `/home/wangqiwei/ICLR2026/starVLA`
- a800b：`ssh -p 11089 wangqiwei@10.13.32.3`，repo `/home/wangqiwei/ICLR2026/starVLA`
- 端点会变；用前先探活。

## 2. 当前实验结论（最新只读核对：2026-07-08 09:02 UTC）

### coupled-Utonia scene-flow stage-2

Run id: `qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary`

状态：`EVAL_DONE`

| suite | SR |
|---|---:|
| libero_spatial | 0.89 |
| libero_object | 0.93 |
| libero_goal | 0.92 |
| libero_10 | 0.53 |
| 平均 | 0.8175 |

解读：没有赢，低于之前 Utonia 榜首 0.8825，也低于 plain joint from-scratch 0.85。

### past-flow ControlNet from-scratch

Run id: `qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary`

状态：`EVAL_DONE`

| suite | SR |
|---|---:|
| libero_spatial | 0.91 |
| libero_object | 0.94 |
| libero_goal | 0.91 |
| libero_10 | 0.67 |
| 平均 | 0.8575 |

解读：比 plain joint from-scratch 0.85 只高 0.75pp，在 6pp 噪声底内；没有明显翻盘。比 Utonia 榜首 0.8825 低 2.5pp，也在噪声底附近，但不是明确胜出。

## 3. 重要对照数字

- plain scene-flow joint from-scratch：平均 0.85（spatial 0.93 / object 0.98 / goal 0.84 / libero_10 0.65）
- plain scene-flow warmstart：平均 0.8275（0.89 / 0.93 / 0.88 / 0.61）
- coupled-Utonia stage-2：平均 0.8175（0.89 / 0.93 / 0.92 / 0.53）
- past-flow ControlNet：平均 0.8575（0.91 / 0.94 / 0.91 / 0.67）
- 旧 Utonia 榜首基线：0.8825
- 评测噪声口径：小于约 6pp 不要过度下结论。

## 4. 刚完成的代码工作

### past-flow eval / FIFO 接线已确认

关键点：

- `examples/LIBERO/eval_files/model2libero_interface.py`：每个 episode 带 `episode_id`。
- `deployment/model_server/policy_wrapper.py`：`episode_id` 变化时调用 `reset_past_flow()`。
- `starVLA/model/framework/VLM4A/QwenGR00T.py`：维护 `_pf_fifo`，有 `reset_past_flow()`，推理时 FIFO 满 K=2 后构造 `past_flow_batch`，并调用 `scene_predictor.sample_field(...)` 自预测当前 flow 入队。
- `starVLA/model/modules/action_model/flow_matching_head/scene_flow_head.py`：有 `PastFlowEncoder`、`_encode_past_flow()`、`sample_field()`。

之前做过 GPU smoke：FIFO 满 K=2、reset 清空、bf16 无 dtype 崩。

### 同步到 a800 已完成

必要 past-flow/FIFO 文件已从 4090d 同步到 a800a/a800b，并逐文件 md5：

- a800a：`MD5_OK`
- a800b：`MD5_OK`

同步过的文件包括 eval server/interface、`QwenGR00T`、`QwenGR00T_FFSCommon`、scene-flow head、DiT gate、dataloader、a800 launcher。

### H100 退役收尾已完成

已清理：

- 无 `scripts.h100b` import 残留。
- 无 4090d active scripts 中的 H100 endpoint 残留。
- 旧 H100 checkpoint auto-eval daemon 改成 fail-fast retired stub。
- `eval_qwen2p5vl_4suite.sh` / `eval_ffs_4suite.sh` 改成只用 4090d 本地 ckpt/config/stats，不再等待 h100b。
- 验证通过：bash syntax OK，Python compile OK。

## 5. 当前 git 状态

仓库有大量未提交改动，最新只读统计为约 183 条 `git status --short`。这不是干净工作树。

特别要注意：里面混了多类工作：

1. scene-flow / past-flow 训练与推理代码；
2. eval 接线和 watcher；
3. H100 退役清理；
4. 历史实验登记和脚本迁移；
5. 一些 untracked probe / patch 草稿。

**不要一把梭 commit。** 如要提交，请先分组给用户看计划，按 conventional commits 拆：

- `feat(sceneflow): past-flow ControlNet training and rollout FIFO`
- `fix(eval): reset past-flow FIFO per episode`
- `chore(scripts): retire H100 launch/eval endpoints`
- `docs(experiments): record scene-flow and past-flow results`

实际分组要以 `git diff --stat` / `git diff <files>` 为准。

## 6. Codex 下一步建议

### 6.1 先登记 past-flow 结果

把上述 Utonia / past-flow 结果写入权威实验登记，建议文件：

- `docs/experiments/sceneflow_cascade_motion_fromscratch_0706.md`
- 或新建/更新一个更明确的 `docs/experiments/pastflow_controlnet_review/README.md`

必须写清楚：

- run_id
- step=30000
- video_keys=`primary,left_view`
- 每 suite SR 和均值
- vs 0.85 / 0.8275 / 0.8825
- 6pp 噪声口径：past-flow 0.8575 对 plain 0.85 不是明确提升

### 6.2 检查 Tab 2 code-review findings

Claude 刚派了一个 reader-mode Tab 2 去 review scene-flow 相关代码。它只读不改。若用户把它的 findings 给你，优先处理 HIGH correctness finding。不要盲改，先定位代码再给 patch 计划。

### 6.3 可能的技术结论

目前主线看起来是：

- scene-flow 辅助没有稳定改善 LIBERO 成功率；
- past-flow ControlNet 也没有明显翻盘；
- 如果继续做，只能把它当 ablation / negative result，或者转向更强的后续 ablation：逐层读耦合、端到端 joint、3D泛化 split，而不是继续在当前 LIBERO 设定硬调。

### 6.4 若要继续实验

不要立即开新长训。先问用户要不要继续。候选：

- 逐层读耦合：flow 层 L 的 hidden 喂 action 层 L，而不是单 tap 广播。
- 一二阶段端到端联合训练：不 staging / 不 detach，但会弄脏 claim，只当 ablation。
- 3D 泛化 split：LIBERO 已饱和，可能需要换 benchmark 才看出 motion signal。

## 7. 常用只读命令

```bash
cd /data/wangqiwei/ICLR2026/starVLA

# watcher 状态
cat playground/Checkpoints/qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary.watcher.status
cat playground/Checkpoints/qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary.watcher.status

# past-flow eval 结果
root=playground/Checkpoints/qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary/eval_step30000_primary_left_view
for s in libero_spatial libero_object libero_goal libero_10; do
  grep -i 'Total success rate' "$root/$s/client.log" | tail -1
done

# Utonia eval 结果
root=playground/Checkpoints/qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary/eval_step30000_primary_left_view
for s in libero_spatial libero_object libero_goal libero_10; do
  grep -i 'Total success rate' "$root/$s/client.log" | tail -1
done

# 工作树概览
git status --short
git diff --stat
```

## 8. 不要做的事

- 不要 kill 任何还在跑的训练/eval，除非用户明确让你 kill。
- 不要 push GitHub。
- 不要在 a800/h100 上改代码。
- 不要把结果写到 `/tmp`。
- 不要把所有未提交改动混成一个大 commit。

## 9. 给用户的一句话版本

past-flow ControlNet 跑完了，结果平均 0.8575，只比 plain joint 0.85 高 0.75pp，在噪声内；coupled-Utonia 0.8175 没赢。建议先登记结果和整理未提交 diff，而不是继续盲目开新长训。
