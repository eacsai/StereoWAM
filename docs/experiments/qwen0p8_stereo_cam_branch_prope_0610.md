# cam_branch 并行相机几何注意力支路（cam_rope 复活）— 2026-06-10

## Question
旧 cam_rope 被证实是摆设（q_cam/k_cam 双零初始化 → 梯度死锁，从未学习，memory
`project_cam_rope_inert_bug`），"立体相机几何编码有没有用"从未被真正测过。本实验用
StereoWorld 官方代码的正确结构（并行 PRoPE 支路 + 单边零出口）真正接通这个通道。

## Mechanism（详见 spec notes/cam_branch_parallel_prope_spec.md v2 + notes/stereoworld_code_walkthrough_20260610.md）
- Qwen3.5-0.8B 的 6 个 softmax 层 [3,7,11,15,19,23] 各挂一条窄并行注意力支路
  （4 头 × 128 维 = 512，新参数 ~25M ≈ 主干 3%），**只在图像 token 之间**做因果注意力。
- 支路内零参数 PRoPE 变换：Q×P^T、K/V×P^inv、输出×P（每 4 维一组；P 从已验证的
  K_norm/T 分别构造，禁合成逆）。同相机注意力分数不变；跨相机注入相对投影几何。
- 出口 out_proj 零初始化（唯一零点）→ step-0 逐位 == baseline，第一步起梯度非零。
- 主注意力完全不动 → FlashAttention 不掉（cam_rope 4 倍慢的教训不会重演）。

## Implementation provenance（全链路硬验证）
spec v1 → codex 设计 review（4H+4M+2L 合并）→ spec v2 → codex 写码 → 3 镜头 agent review
（1 HIGH: warm-start B 被 24 个摆设 cam_rope 旧键卡死 → 加载端过滤+全零断言修复; 4 LOW）
→ codex 终审（1H: cam-id 部分覆盖静默风险→fail-closed 计数断言; 3M: 部分支路 ckpt 拒载/
速度门日志同步污染/launcher 空 ckpt 静默 from-scratch guard; 2L: predict_action 覆盖+维度 guard）
→ **GPU smoke 7/7 PASS**（h100b, 2026-06-10）:
- step0_equivalence: 加载 B + 支路, last_hidden/loss 逐位 == 无支路 baseline; RNG 零消耗
- no_deadlock_grads: out_proj 梯度范数 ~7e4 非零; 第二步 q/k/v 梯度 4.4(链路通)
- matrix_triple: P@P_inv 误差=0; 同相机分数不变性 9.5e-07
- causal_no_leak(探出口前原始输出) / leftpad_no_nan / eval_path_predict(predict_action 触发 hook)
- speed_gate: 支路稳态开销 **0.041s/步**（BS2; 主干基准 ~0.9s）

## Arms（待用户批准 diff 后起跑; GPU 等 #4 三臂腾出）
| arm | run_id | 配置 | 验收 |
|---|---|---|---|
| 机制门 | `qwen3p5_0p8b_stereo_cam_branch_prope_warmstartB_mechgate_2k` | warm-start B 30k, 冻 VLM(支路+动作头可训), 2k steps, ~40min | step-0==B 重放; out_proj 离零; 各层支路范数增长; loss 无异常 |
| 论文臂 | `qwen3p5_0p8b_stereo_cam_branch_prope_fromscratch_30k` | from-scratch, 与 B 严格同配置(4-suite eff128 30k 全微调) | 4-suite 不掉点(vs B=0.908) + 支路在学 |

注: 机制门冻 VLM 是有意设计（更干净地观察支路学习; 支路注册在框架顶层不在 qwen_vl_interface
冻结子树内）。论文臂若不超 B, 结论限定为"6 层支路在饱和 4-suite 无增益", 非"相机几何无用";
真考场 = 非饱和深度敏感 suite。

## Results
| arm | step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|---|
| mechgate | — | — | — | — | — | 待批准 |
| fromscratch 30k | — | — | — | — | — | 待批准 |

参见 [[project_cam_rope_inert_bug]] [[reference_stereoworld_paper]] docs/experiments/qwen0p8_ffs4_injection_modality_camropeoff_0610.md
