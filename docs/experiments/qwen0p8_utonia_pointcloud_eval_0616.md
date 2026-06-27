# Utonia 点云注入 — 两变体 eval (2026-06-16)

## 背景 (#10 Utonia 点云注入)
把 左图 + 左深度 + 相机内参 反投影成点云,过一个**冻结的 Utonia 点云编码器 (137M)** 抽特征,
注入策略模型。本轮测两个"怎么注入 / 怎么算"的变体。
LIBERO 4-suite 已饱和,锚点:mono primary **0.913**,depthtoken_keep **0.910**,立体基线 B **0.908**。

## 变体 (arms)
| 变体(描述) | run_id | 注入方式 | 测的 checkpoint |
|---|---|---|---|
| 点云重采样版 | `qwen3p5_0p8b_utonia_resampler_fromscratch_30k` | resampler 模块 | **step 20000 中途快照**(仍在训到 30k) |
| 逐patch缓存版 | `qwen3p5_0p8b_utonia_perpatch_cached_30k` | per-patch 加性残差 + 预计算缓存 | **step 30000 最终** |

## 结果 (4-suite, our-render 数据, right_view+primary eval)
| 变体 / step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|
| 点云重采样版 @20k(中途) | 0.88 | 0.96 | 0.83 | 0.63 | **0.825** |
| 逐patch缓存版 @30k(最终) | 0.97 | 0.98 | 0.91 | 0.74 | **0.90** |

## 结论 / 注意
- **逐patch缓存版 30k = 0.90**,落在饱和 LIBERO 的 0.85–0.91 注入方法带内(≈ mono 0.913 / depthtoken_keep 0.910 / 立体基线 B 0.908),**未超 mono 上限** —— 与项目一贯结论一致:饱和 LIBERO 上第二视角 / 深度 / 几何 / 点云信号都测不出差异(要靠非饱和的 OOD / 泛化 split 才分得出)。
- **点云重采样版只是 step 20000 中途快照**(还在训到 30k),0.825 偏低但**未训完**,跟缓存版**不是公平对比**(步数 + 变体都不同)。等它训满 30k 再补测。
- 长任务 libero_10 两者都最低(0.63 / 0.74),符合预期(长程最难)。

## 踩坑
- ⚠️ 逐patch缓存版 eval 一开始 4 套件全崩 = 它的 `config.full.yaml` 里 FFS 权重路径写死成 **h100b 的 `/mnt/data/...`**(4090d 上不存在)→ server 启动即 `FileNotFoundError`。改成 4090d 路径(`/data/wangqiwei/ICLR2026/Fast-FoundationStereo/...`,SHA 一致)后重测通过。教训:跨机训练的 ckpt config 里的绝对路径在另一台机 eval 时要重定向。
- ⚠️ eval 基础设施:`eval_qwen2p5vl_4suite.sh` 的 BASE_PORT=6730 写死、默认 GPU `1 3 4 6` → **两个 4-suite eval 不能并发**(撞端口),要串行或改 BASE_PORT。

参见 [[project_experiment_registry]] [[reference_3dvla_utonia_pointcloud]]
