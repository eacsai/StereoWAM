> Historical incident report. The "current running" and GA-plugin repair sections
> below describe an intermediate July 23 state, not the final handoff.
> The user later requested reverting the Accelerate GA plugin in train_starvla.py.
> The final successful legacyaccum_retry runs omitted the Accelerate CLI GA flag,
> retained DeepSpeed GA=4, and were then stopped by the user at saved step 14k.
> See [the verified continuation](20261029_l40_results.md) for actual step-count
> semantics. This report does not authorize restarting either job.

# 2026-07-23 训练启动事故复盘

## 范围

实验：Qwen3.5-0.8B、left-primary、Camera Branch visual-attention ablation；
分支：`ffs_controlnet_dev`；远端仓库：`/home/wangqw/ICLR27/starVLA`。

本记录只描述启动流程事故和防护措施，不改变模型结论，也不删除失败实验的日志或 provenance。

## 结果

最初的完整启动没有形成可运行的训练，原因是多个独立门禁连续失效：

1. causal 组在 GPU 占用预检阶段被 `fuser -v` 的正常 stderr 表格误判为异常。
2. image-bidir 组进入 step 0 后触发 Accelerate accumulation 与 DeepSpeed ZeRO-2 的 `no_sync` 冲突。
3. 启动脚本只在较晚位置设置 GPU 可见性，且没有验证进程实际看到的 GPU 数量，不能证明 0–3/4–7 隔离有效。
4. 原有 smoke 没有对每个实验臂分别完成真实 bf16/DeepSpeed 训练和启动后的健康检查，却进入了正式长跑。
5. tmux/session 存在被误当成训练健康，失败 run、进程树、日志和端口没有用一个唯一 manifest 统一核对。

这些问题叠加后，修复过程被迫逐层排查；远端 patch 上下文和双层 SSH 引号又增加了返工时间。核心教训是：过去训练成功只是历史证据，不是当前 launcher、环境和 GPU 分配组合的验收结果。

## 为什么过去实验没有自动避免这次问题

过去的成功实验并不等价于本次组合。当前运行实际使用了 Accelerate 1.5.2、DeepSpeed 0.16.9、ZeRO-2、GA=4，以及新的 visual-attention launcher；旧实验可能使用了不同的 launcher、版本、配置或没有走到同一个 `accumulate()` 路径。项目记忆中虽然有经验，但没有被编码成启动前的 fail-closed 检查，所以 agent 仍可能把“以前跑过”错误地当成“这次可以直接发车”。

## 已执行的修复

### 代码和 launcher

- `starVLA/training/train_starvla.py`：在 GA=4 下使用 `GradientAccumulationPlugin(sync_each_batch=True)`，避免 ZeRO-2 的 `no_sync` 冲突；解析后的 GA 不是 4 时直接失败。
- `scripts/a800/preflight_visual_attention_ablation.py`：正确处理 `fuser -v` 的返回码 0/1 和 stdout/stderr，只有异常返回码或不一致占用才阻断。
- `scripts/a800/run_qwen0p8_groot_stereo_visual_attention_l40.sh`：在启动 Python 前导出 `CUDA_VISIBLE_DEVICES`，并检查 `torch.cuda.device_count()` 等于 world size；修复 provenance 版本探针的引号错误；每次运行使用新的标识。

### 验证

- causal 和 image-bidir 两个配置分别完成真实 1-step GPU smoke。
- 两组都报告 GA=4、effective batch=128，没有 `no_sync`、Traceback、OOM 或 wiring 错误。
- 正式任务使用新 run id 和独立端口启动；启动后的只读检查确认 tmux、目标进程和日志都在推进。

## 以后每次训练的硬性 GO/NO-GO

在正式长训前，必须按以下顺序执行：

```text
探活/分支/环境 → 预留 GPU → 复核 GPU 占用
→ 导出 CUDA_VISIBLE_DEVICES → 验证可见卡数 == world size
→ 逐实验臂真 bf16/DeepSpeed smoke（至少推进 step 1，最好 step 2）
→ 检查 GA、effective batch、GPU 归属和错误扫描
→ 唯一 run id/端口/日志/输出保护
→ 正式启动
→ 30–90 秒检查目标 PID、日志 step/loss、GPU 归属
→ 登记运行信息
```

以下任何一项缺失都判 NO-GO：

- 没有每个实验臂的 smoke 结果；
- 没有实际 GPU 可见性计数；
- 只看到 tmux 而没有目标子进程和日志推进；
- 发现 `no_sync`、Traceback、OOM、KeyError、size mismatch 或参数接线错误；
- 复用了旧 run 的日志、端口或输出目录；
- 需要用宽泛 `pkill`/`killall` 才能清理。

## 当前长期规则

- `gpu-smoke-train` 负责发车前的 GPU/配置/真训练门禁；
- `launch-remote-watcher` 负责 tmux、目标进程、日志和精确进程树生命周期；
- `train-status-table` 只读监控，绝不承担修复或 kill；
- 失败证据必须保留，修复后用新 run id 重试；
- 不启动自动 reviewer gate；代码改动仍按先方案、获确认、再集中实现的两阶段流程执行。

## 当前运行

本次复盘写入时，以下两个正式任务仍在远端运行，未被本复盘操作停止：

- causal：`visual_causal_l40_retry_gpuvis`，日志 `logs/visual_attention/causal_full_retry_gpuvis_20260723.log`；
- image-bidir：`visual_image_bidir_l40_retry_gpuvis`，日志 `logs/visual_attention/image_bidir_full_retry_gpuvis_20260723.log`。

复盘后续只读查询不得把 session 存在当作唯一健康证据，必须同时检查 PID、日志和 GPU。
