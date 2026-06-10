# #4 FFS injection-modality ablation (keep / strip / depthimage), cam_rope-OFF relaunch — 2026-06-10

## Question
How should FFS stereo depth information be fed to the policy: as VLM-sequence tokens the
action head can read (**keep**), as VLM-context-only tokens stripped before the head
(**strip**), or as a rendered disparity image, i.e. a 3rd camera view (**depthimage**)?
Anchors: B stereo baseline 0.908, fullinject 0.870 (both eff128/30k, saturated LIBERO 4-suite).

## Arms (all from-scratch, eff128 = BS32 x GA4, 30k steps, save 10k, full finetune)
| arm | framework | run_id | knobs |
|---|---|---|---|
| keep | QwenGR00T_DepthTokenFFS | `qwen3p5_0p8b_ffs_depthtoken_keep_fromscratch_30k` | 64 tokens, pool 8x8, STRIP=0 |
| strip | QwenGR00T_DepthTokenFFS | `qwen3p5_0p8b_ffs_depthtoken_strip_fromscratch_30k` | 64 tokens, pool 8x8, STRIP=1 |
| depthimage | QwenGR00T_DepthImageFFS | `qwen3p5_0p8b_ffs_depthimage_fromscratch_30k` | turbo disparity as 3rd image, cam_id=1 |

Scheduler: `scripts/h100b/ffs4_ablation_scheduler.sh` (shared /mnt/data claim queue, h100b GPUs 0/1 +
h100a GPU 0; strip/depthimage auto-chain when the cam-frozen #2cf/#3cf runs free their GPUs).
Eval: 4090d `eval_ffs_watch.sh`, 20k + 30k, 4-suite.

## ⭐ All three arms run with stereo_cam_rope_enabled=false (CAM_ROPE=0)
The first keep attempt (2026-06-09, archived as `*.camropeON_killed_20260610`) trained with the
then-default cam_rope ON and ran ~4x slow (median 2-7s/step, spikes 20-34s, ~7.8s/step avg at
step 5076 after 11h). Root cause chain:
1. cam_rope is **provably inert**: `q_cam_proj` AND `k_cam_proj` are both zero-initialized
   (`cam_rope.py`), so the bilinear cam term AND its gradients are identically zero forever —
   the geometry encoding never trained in ANY stereo run (see memory `project_cam_rope_inert_bug`).
2. Its d_c=16 branch widens attention head_dim 256→272, past FlashAttention2's hard limit →
   SDPA math fallback → O(S²) materialized attention + allocator churn on varying seq lengths.
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was already set during the slow run → not
   the fix.

**Equivalence proof (GPU, h100b, 2026-06-10)** — `scripts/h100b/smoke_camrope_disable_equivalence.py`:
- ON→OFF weight transplant, both forced onto the same SDPA-MATH kernel: last_hidden
  **max|diff| = 0 (bitwise)**, loss diff = 0. Disabling the inert cam_rope changes nothing.
- Backward at the ON config: all 12 q_cam/k_cam grads are non-None and **exactly zero**
  (inertness re-proven on a real backward); shared-param grad cosine ON-vs-OFF ≥ 0.9999.
- OFF under production flash_attention_2 vs SDPA-math: loss rel diff ≤ 1e-3, hidden rel RMS
  ≈ 0.1 (known bf16 backend noise on random-init weights; scale-invariant gate 0.25).

**Comparability**: anchors B / fullinject trained with cam_rope ON but inert (output ≡ 0), so
cam_rope-OFF arms remain mathematically comparable; only the attention backend differs
(flash vs math bf16 noise — one-line footnote in the paper, no rerun of anchors needed).

## Relaunch log
- 2026-06-09 14:43Z keep (cam_rope ON) started → killed 2026-06-10 02:1xZ at step ~5076/30000.
- 2026-06-10: CAM_ROPE env added to `run_qwen0p8_groot_ffs.sh` (default 1; fail-closed guard:
  `*fromscratch*` depthtoken/depthimage run_ids REQUIRE CAM_ROPE=0); scheduler passes CAM_ROPE=0;
  gate smokes now build the cam_rope-OFF config (`--cam-rope-enabled`, default 0) and the gate
  list includes the equivalence smoke.
- 2026-06-10 02:34-02:38Z manual 3-smoke gate: ALL PASS.
- 2026-06-10 02:40:53Z keep relaunched on h100b GPU0 with CAM_ROPE=0 (banner confirms
  `stereo_cam_rope_enabled=false`, zero cam_rope install lines in log). Early speed: median
  ~1.8s/step with decaying compile-warmup spikes — tail re-check pending.

## Results
| arm | step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|---|
| keep | 20k/30k | — | — | — | — | running |
| strip | — | — | — | — | — | queued (after #2cf) |
| depthimage | — | — | — | — | — | queued (after #3cf) |

参见 [[project_cam_rope_inert_bug]] [[project_ffs_8methods_port_status]] [[project_experiment_registry]]

## ⚠️ Caveat (2026-06-10 Fable-5 全量 review 发现): FFS 训练/评测输入分辨率分歧
训练时 dataloader 输出 224×224 图(gr00t_lerobot/datasets.py 硬编码), FFS 前向把它升采样到
256(模糊立体对); 评测时 obs_image_size 未配置 → predict_action 的 resize 门跳过, FFS 吃到
原生 256×256 清晰帧 = 偏离训练分布的输入。**全部已完成 FFS run 都带这个分歧**(B 基线只带
VLM 侧的 224/256 偏移, FFS 臂额外带冻结立体网的特征分布偏移), 可能系统性压低 FFS 评测分,
是"FFS 无收益"结论的一个潜在混杂因子。处置: 引用 FFS-neutral 结论时注明本 caveat; 若要
排除, 需要 eval 侧 pin obs_image_size=[224,224] 重评一个代表性 ckpt 对比(未做)。

## ⚠️ Caveat 2 (codex 终审发现): cam-frozen 对照的真实差异要重新表述
cam_rope 模块在模块树挂了两处(attention 子树 + 框架顶层), FREEZE_MODULES=qwen_vl_interface
会经 attention 子树把同一份参数冻掉 → 凡是冻 VLM 的臂, cam_rope 反正已被冻结, 额外写
",stereo_cam_rope_layers" 是冗余 no-op。叠加 cam_rope 本身 inert(双零死锁, 梯度恒零, 冻不
冻无差), #2cf/#3cf vs 对应非冻臂的真实差别只剩"VLM 冻不冻", 与"相机层冻不冻"无关。引用
cf 对照(两者均 0.87)时按"冻 VLM 对照"口径表述。
