# CLAUDE.md — starVLA on 4090d (StereoVLA / Phase 1)

> 本文件是 **agentic 协作时的导航**，不是 user-facing docs。任何动手前必读。
> Last update: 2026-05-19

---

## 0. 这是什么仓库 / 用户在做什么

- **上游**: [starVLA](https://github.com/starVLA/starVLA) — 一个 Lego-like VLA 训练 codebase（Qwen-VL + action head，支持多种 head：π / OFT / FAST / GR00T）。
- **当前分支**: `starVLA_dev`（cfed863 上游 HEAD + 我们 1 处 `dist.get_rank` bug fix）。
- **用户身份**: ICLR 2026 "Stereo VLA" 项目 owner。
- **当前阶段**: **Paper baseline 复现**（StarVLA-π Qwen3-VL 95.7 avg），用上游 IPEC LeRobot 数据 + 官方 `run_libero_train.sh`。复现 OK 后再回到 Stereo VLA 实验（mv archive 回来）。
- **远程开发机**: `4090d` (192.168.22.219, user `wangqiwei`, SSH key 免密)。所有训练 / 渲染 / eval 都跑在这里。
- **本地 review 快照**: `/Users/agiuser/Documents/ICLR2026/codex-review-starvla/starVLA/`（rsync，**只读**；`sync-starvla-review.sh` 刷新）。
- **Stereo work 归档**: `/data/wangqiwei/ICLR2026/stereo_work_archive/20260519/`（`examples/LIBERO_STEREO/` + `SSF/render_scripts/` 整体 mv 出去；Phase 2 回归时 mv 回来即可）。

---

## 1. 全局规则（违反会被用户当场叫停）

1. **⛔ 项目内开发，文件必须放项目内**（远程同理）：
   - 集群相关 launcher / prep / 工具：`scripts/4090d/<…>.sh`
   - **不准**甩 `/tmp/`、用户 home 根目录、其他项目目录之外的任何地方
   - 例外：用户显式批准 / 一次性 log（产生它的 .sh 不能甩 /tmp）
2. **任何 LIBERO 操作前先 `cat examples/LIBERO/README.md`** — 上游写明了 2-process eval (server + sim) 工作流以及哪些路径要改。
3. **看 yaml `data_mix` 实际指向再选 eval script**：
   - 当前 `libero_all` / `libero_goal` 用 `libero_franka` config (primary+wrist) → eval 用 `examples/LIBERO/eval_files/eval_libero.py`
   - 错配 = success 直接 0%
4. **改代码走远程 ssh**，本地是只读快照（`sync-starvla-review.sh` 会覆盖本地工作树）。
5. **代码大修后自动后台跑 codex adversarial review**（全局 CLAUDE.md 规则）。
6. **ssh 命令尽量合并到一条**或并行多 Bash tool call，避免短时间多次连接被拒。
7. **不要 mock / 凭直觉填路径** — 这个 repo 大量 hardcoded 路径（如 jye624 集群的 `/home/jye624/...`、`bond0`、`mlx5_2`），都需要按本机改。
8. **长任务 launch 后立刻 update memory** — 用户 Mac 不稳定易重启，session 一断只有 memory 跨 session 可见。

---

## 2. 目录结构（only what matters）

```
starVLA/
├── starVLA/                              # 主 python 包
│   ├── training/
│   │   ├── train_starvla.py              # ⭐ 主训练入口
│   │   ├── train_starvla_cotrain.py      # VLA + VLM co-train
│   │   ├── train_starvlm.py              # 纯 VLM 训练
│   │   └── trainer_utils/
│   │       └── trainer_tools.py          # ⚠ 1 处本地 bug fix: dist.get_rank → dist.get_rank()
│   ├── model/
│   │   ├── framework/                    # QwenPI / OFT / FAST / GR00T / WM4A / VLM4A
│   │   │   └── base_framework.py         # ⭐ build_framework()
│   │   └── modules/                      # vlm / action_model (DiT) / world_model / projector / dino_model
│   ├── dataloader/
│   │   ├── __init__.py                   # ⭐ build_dataloader() 路由
│   │   ├── lerobot_datasets.py           # ⭐ get_vla_dataset() + collate_fn
│   │   └── gr00t_lerobot/                # ★ LeRobot 数据加载核心
│   │       ├── datasets.py               # LeRobotSingleDataset / MixtureDataset / __getitem__
│   │       ├── registry.py               # 自动从 examples/*/train_files/data_registry 合并 mixtures
│   │       └── transform/                # min-max normalize, state/action tensor
│   └── config/
│       ├── training/   starvla_cotrain_{oxe,libero}.yaml / starvla_train_adapter.yaml
│       ├── accelerate/ multi_gpu_bf16.yaml
│       └── deepseeds/  deepspeed_zero{2,3}.yaml   # ⭐ 官方训练用 zero2
├── deployment/
│   └── model_server/
│       ├── server_policy.py              # ⭐ eval websocket policy server (load ckpt → port → 处理 obs)
│       ├── policy_wrapper.py
│       └── policy_norm_processor.py      # action unnorm
├── examples/                             # **保持纯上游**，4090d 适配版放 scripts/4090d/
│   ├── LIBERO/                           # ★ paper baseline path
│   │   ├── README.md                     # ★★ 必读：完整 eval + train recipe
│   │   ├── data_preparation.sh           # 上游数据 prep（pip pin 老 hf 有 bug，我们在 scripts/4090d/ 有可用替代）
│   │   ├── train_files/
│   │   │   ├── run_libero_train.sh                  # 上游 launcher（jye624 路径，不能直接跑）
│   │   │   ├── starvla_cotrain_libero.yaml          # ⭐ 训练配置 (mono primary+wrist 4-suite joint)
│   │   │   ├── data_registry/data_config.py         # Libero4in1DataConfig + libero_all/libero_goal/libero_franka
│   │   │   └── modality.json                        # ⭐ 拷到每个 suite 的 meta/ 让 dataloader 看
│   │   └── eval_files/
│   │       ├── run_policy_server.sh                 # 上游 server 启动（jye624 路径）
│   │       ├── eval_libero.sh                       # 上游 eval 客户端（jye624 路径）
│   │       ├── eval_libero.py                       # ⭐ primary+wrist, gripper polarity inverted
│   │       └── model2libero_interface.py            # ws client wrapper
│   └── (Behavior, calvin, CoTrainVLM, Franka, Gemma4, LIBERO-plus, Robocasa_*, Robotwin,
│        SimplerEnv, VLA-Arena 等 — Phase 1 不涉及)
├── SSF/                                  # 只有 LIBERO 安装目录留下
│   └── libero_sim_env/
│       ├── LIBERO/                       # libero 包源码（fork，eval 必需）
│       └── .venv/                        # ⭐ LIBERO eval venv (libero + h5py + websockets + mujoco + tyro)
├── scripts/                              # ⭐ ⭐ ⭐ 我们写的、与上游解耦的 cluster-specific 脚本
│   └── 4090d/                            # 当前唯一 cluster
│       ├── data_preparation.sh           # pure aria2c data prep, 走 hf-mirror.com
│       ├── run_libero_train.sh           # 完整 4-suite × 80k 训练 (paper baseline 复现)
│       ├── run_libero_train_fast.sh      # 单 suite × 5k 训练 (smoke validation)
│       └── model_param_breakdown.py      # 工具：打印 framework 各 module 参数
├── playground/
│   ├── Pretrained_models/                # Qwen3.5-0.8B (0.8B VLM), Qwen3-VL-4B-Instruct (4B)
│   ├── Datasets/                         # 符号链接到外部数据集
│   └── Checkpoints/                      # 训练 ckpt + config.yaml + summary.jsonl + wandb/
├── docs/, assets/, .github/, README.md, pyproject.toml
├── .venv/                                # ⭐ starVLA 主训练 venv (torch 2.6.0+cu124)
└── pyproject.toml                        # package "starVLA" v1.0.1 (py>=3.10)
```

---

## 3. 关键 venv（两个，不可混用）

| venv | 路径 | 用途 | 关键依赖 |
|---|---|---|---|
| **starVLA 主训练** | `/data/wangqiwei/ICLR2026/starVLA/.venv/bin/python` | accelerate / 训练 / policy server | torch 2.6.0+cu124, transformers, accelerate, websockets, huggingface_hub 1.14 |
| **LIBERO 模拟** | `/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python` | eval client / MuJoCo 模拟 | libero 0.1.0, h5py 3.16, torch 2.12, mujoco 3.2, tyro, websockets |

eval / 渲染时 env vars:
```bash
export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:/data/wangqiwei/ICLR2026/starVLA:${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

## 4. 训练流程

### 入口（不要写新 launcher 到 /tmp）

| 用途 | 用哪个 sh |
|---|---|
| **fast smoke**（单 suite × 5k，~1.5h） | `scripts/4090d/run_libero_train_fast.sh` |
| **full paper baseline**（4 suite × 80k，~44h） | `scripts/4090d/run_libero_train.sh` |

启动方式：
```bash
cd /data/wangqiwei/ICLR2026/starVLA
nohup bash scripts/4090d/run_libero_train_fast.sh > playground/Checkpoints/<run_id>_launch.log 2>&1 &
```

里面调的核心命令：
```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 6 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  ...
```

### 必须 export 的 4090d 特定 env（已在 scripts/4090d/run_libero_*.sh 里设好）

- `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=ens20f0`（4090d 无 IB；上游 `bond0` + `mlx5_2,mlx5_3` 是 jye624 集群的，会 hang）
- `WANDB_MODE=disabled`（`--wandb_entity disabled` 只是改 entity 名不关 wandb；wandb 没登录会崩）

### config yaml 关键 schema

```yaml
framework: { name: QwenPI, qwenvl: {base_vlm, attn_implementation: flash_attention_2, vl_hidden_dim: 2048},
             action_model: {action_dim: 7, state_dim: 7, action_horizon: 8} }
datasets:
  vlm_data: { dataset_use: sharegpt4v_coco, model_type: qwen2.5vl, per_device_batch_size: 4, ... }
  vla_data: { data_root_dir, data_mix, action_type: delta_qpos,
              per_device_batch_size: 16, load_all_data_for_training: true, video_backend: torchvision_av }
trainer:   { max_train_steps, num_warmup_steps, save_interval, learning_rate {base, qwen_vl_interface, action_model},
             lr_scheduler_type: cosine_with_min_lr, loss_scale {vla, vlm}, gradient_checkpointing: true, optimizer {...} }
version_id: "0.21"
```

### data_mix 表（在 `examples/LIBERO/train_files/data_registry/data_config.py`）

| mixture | robot_type | 用途 |
|---|---|---|
| `libero_all` | libero_franka | **paper baseline** joint 4-suite (primary+wrist) |
| `libero_goal` | libero_franka | 单 suite goal (primary+wrist) — smoke validation |

**注**：Stereo-related mixtures (`libero_*_stereo`, `libero_*_mono_replay`) 当前在 stereo_work_archive 里，Phase 2 回归才用。

---

## 5. Eval 流程（2 进程，严格按 README）

### Server（GPU，starVLA venv）
```bash
cd /data/wangqiwei/ICLR2026/starVLA
CUDA_VISIBLE_DEVICES=<GPU> .venv/bin/python deployment/model_server/server_policy.py \
  --ckpt_path <ckpt.pt> --port 6694 --use_bf16
```
- 监听 `0.0.0.0:6694`，加载完毕日志: `server listening on 0.0.0.0:6694`
- ~15GB GPU mem（Qwen3.5-0.8B + DiT action head）

### Eval client（CPU+EGL，LIBERO venv）
```bash
cd /data/wangqiwei/ICLR2026/starVLA
export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:$(pwd):${PYTHONPATH}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python \
  examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path <ckpt.pt> \
  --args.host 127.0.0.1 --args.port 6694 \
  --args.task-suite-name libero_goal \                # libero_{spatial,object,goal,10}
  --args.num-trials-per-task 50 \
  --args.max-tasks -1 \                                # smoke: 2 trial × 1 task
  --args.video-out-path <out_dir>
```

### Eval 脚本（**只有上游 eval_libero.py**，我们删掉了 eval_libero_mono.py）

| 脚本 | image | gripper polarity | 用于 |
|---|---|---|---|
| `examples/LIBERO/eval_files/eval_libero.py` | primary + wrist | `1.0 - 2.0*(v>0.5)` (inverted) | **paper baseline ckpt** / `libero_franka` 训练 |

Eval 输出: `<model_root>/results/<task_suite>/<run>/<ckpt>/rollout_<task>_episodeN_{success,failure}.mp4`

### 上游官方 .sh 注意
`examples/LIBERO/eval_files/{run_policy_server.sh, eval_libero.sh}` 里写死 jye624 路径，**模板用**；需按本机改 STARVLA_DIR / LIBERO_HOME / venv path / CKPT，或直接 python 调（见上）。未来稳定 eval 流程时把适配版本写到 `scripts/4090d/eval_*.sh`。

---

## 6. 数据集状态（2026-05-19）

### ⭐ 官方 IPEC LeRobot（当前训练数据）
`/data/wangqiwei/ICLR2026/data/libero_official/libero/<suite>_no_noops_1.0.0_lerobot/`

| suite | parquet | videos | size | source |
|---|---|---|---|---|
| libero_spatial | 432 | 432+432 | 356M | HF IPEC-COMMUNITY (mirror) |
| libero_object | 454 | 454+454 | 520M | 同 |
| libero_goal | 428 | 428+428 | 325M | 同 |
| libero_10 | 379 | 379+379 | 609M | 同 |

每个 suite 4 个 meta + 1 个我们 cp 的 modality.json。

### `playground/Datasets/` 符号链接
```
LEROBOT_LIBERO_DATA -> /data/wangqiwei/ICLR2026/data/libero_official/libero
LLaVA-OneVision-COCO -> /data/wangqiwei/ICLR2026/data/libero_official/LLaVA-OneVision-COCO    (sharegpt4v_coco unzipped)
LEROBOT_LIBERO_STEREO_DATA -> /data/wangqiwei/ICLR2026/data/libero_stereo_selfrender  (Phase 2 stereo 自渲，当前不用)
LEROBOT_LIBERO_DATA.bak_user_mono_20260519 -> libero_mono/libero    (历史备份)
```

### 数据 prep — 为什么用 `scripts/4090d/data_preparation.sh` 而不是上游

**结论先放**：`scripts/4090d/data_preparation.sh` **行为跟上游 `examples/LIBERO/data_preparation.sh` 等价**（产出目录结构 / `modality.json` / 符号链接完全一样），只是下载 transport 换了能在 4090d 工作的方式。上游脚本在 4090d 上**走不通**，三个具体卡点：

**a. 上游脚本第一行 `pip install -U "huggingface-hub==0.35.3"` 会把 venv 弄废**
   - 当前 venv 是 `huggingface-hub 1.14.0`
   - 降级到 0.35.3 会**连带破坏** transformers / accelerate 等吃 hf-hub 新 API 的依赖
   - 更荒谬：**0.35.3 根本没有 `hf` CLI binary**（v1+ 才加的），而脚本紧接着就调 `hf download "$repo" --local-dir ...` —— 脚本自我矛盾

**b. 即使跳过 pip install 直接用 `hf download`，4090d 网络下崩在 mirror 分页 bug**
   - 4090d 在国内，`huggingface.co` 不稳定 → 必须走 `hf-mirror.com`（系统 env 已设 `HF_ENDPOINT`）
   - 但 huggingface_hub 1.14.0 的 `snapshot_download` 调 `list_repo_tree` 翻第二页时，server 返的 Link header 指向 `https://huggingface.co/api/...`（**不**用 HF_ENDPOINT 重写）
   - 触发 `Errno 101 Network is unreachable` → 内部 httpx client 被 close → `RuntimeError: Cannot send a request` → 整个 thread_map 崩
   - 试过 4 种调 hf-hub 的方式（snapshot_download max_workers=1 / git-lfs clone / list_repo_files + per-file hf_hub_download / 第一版 aria2c），全崩在同一个分页步骤

**c. 我们的 v6 = 彻底不碰 `huggingface_hub` 库**
   - `curl` 拿每个 suite 的 `meta/info.json` → 知道 episode 总数 (spatial=432, object=454, goal=428, 10=379)
   - 按 LeRobot 标准目录结构 (`data/chunk-000/episode_NNNNNN.parquet` + `videos/chunk-000/<view>/episode_NNNNNN.mp4`) **自己生成全部 URL list**（约 6900 个 file URL）
   - 喂给 `aria2c`：8 路并行 + resume + retry-wait=15 + max-tries=20 + max-connection-per-server=2（对 mirror rate-limit 友好）
   - 自动 skip 已完整 suite（parquet ≥ N + video ≥ N + meta ≥ 4 文件）→ 断点重启零成本
   - 完整复刻上游脚本尾部的"建符号链接 + cp modality.json"

**禁止**：在 4090d 上直接跑 `bash examples/LIBERO/data_preparation.sh` —— 第一步就会破坏 venv。同理任何"跑 starvla 自带 prep"的指令都改成走我们这个。

**未来其他集群迁移**：如果新集群网络能直连 hf.co 且 venv 能容忍 hf-hub 降级，可以直接跑上游；否则套用我们 v6 的思路（拿 info.json → 生成 URL list → aria2c）写一份 `scripts/<new_cluster>/data_preparation.sh`。

---

## 7. 当前进行中（持续更新见 `~/.claude/projects/.../memory/project_active_session.md`）

**正在跑**：`fast_smoke_goal_qwenpi_0519`（pid 3484798 launcher，6 子进程，2026-05-19 08:28 UTC 起）
- log: `/tmp/train_fast.log` (旧 launch 时写的，下次重启 log 走 `playground/Checkpoints/*_resume.log`)
- ckpt: `playground/Checkpoints/fast_smoke_goal_qwenpi_0519/`
- 单 suite `libero_goal` × 5000 step, save 1000, ~1.5h 全跑

**模型 (QwenPI)**：1.22B 总，全 trainable，无 freeze
- `qwen_vl_interface` (Qwen3.5-0.8B VLM): 852.99M, LR 1e-5
- `action_model` (DiT + encoders): 362.38M, LR 1e-4

---

## 8. 工作流 / 协作规则（项目级）

1. **代码改远程，本地是只读快照**。改完 `bash /Users/agiuser/Documents/ICLR2026/sync-starvla-review.sh` 同步到本地以便 codex / claude review。
2. **大修后自动后台跑 codex adversarial review**（用户全局规则）；命令是 `node ~/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs adversarial-review --background --scope working-tree`，**在 sync 后的本地快照目录里跑**。
3. **GPU 调度**：0,2-6 给训练；1, 7 通常另有占用者，eval / 渲染前先 `nvidia-smi`。eval server 需 ~15GB，渲染纯 CPU。
4. **多个 ssh 并行或合并到一条**（避免 connection 被拒）。
5. **任何项目产出文件都放项目内**（远程：`/data/wangqiwei/ICLR2026/starVLA/`；本地：`/Users/agiuser/Documents/ICLR2026/`）；**不准 /tmp**，除非用户明说。
6. **长任务 launch 后立刻 update memory 里的 pid / log / resume 命令**（用户 Mac 不稳定）。

---

## 9. 速查

| 任务 | 命令 |
|---|---|
| 同步远程 → 本地快照 | `bash /Users/agiuser/Documents/ICLR2026/sync-starvla-review.sh` |
| 列正在跑的训练 | `ssh 4090d 'ps -ef \| grep train_starvla \| grep -v grep'` |
| 看 GPU | `ssh 4090d 'nvidia-smi'` |
| 启 fast smoke 训练 | `ssh 4090d 'cd /data/wangqiwei/ICLR2026/starVLA && nohup bash scripts/4090d/run_libero_train_fast.sh > playground/Checkpoints/fast_launch.log 2>&1 &'` |
| 启完整 4-suite 80k | 同上换 `run_libero_train.sh` |
| 重新 prep 数据 | `ssh 4090d 'cd /data/wangqiwei/ICLR2026/starVLA && bash scripts/4090d/data_preparation.sh'` (DEST 在脚本内) |
| 打印模型参数 breakdown | `ssh 4090d 'CUDA_VISIBLE_DEVICES=1 /data/wangqiwei/ICLR2026/starVLA/.venv/bin/python /data/wangqiwei/ICLR2026/starVLA/scripts/4090d/model_param_breakdown.py'` |
| 起 mono eval server | 见 §5 server |
| 跑 mono eval (2 trial smoke) | 见 §5 client + `--args.num-trials-per-task 2 --args.max-tasks 1` |
| 把 stereo work mv 回来（Phase 2） | `ssh 4090d 'mv /data/wangqiwei/ICLR2026/stereo_work_archive/20260519/examples/LIBERO_STEREO /data/wangqiwei/ICLR2026/starVLA/examples/ && mv /data/wangqiwei/ICLR2026/stereo_work_archive/20260519/SSF/render_scripts /data/wangqiwei/ICLR2026/starVLA/SSF/'` |

---

## 10. ⭐ StereoWorld 官方代码深读结论（cam_rope 复活依据, 2026-06-10）

> 详细逐文件 walkthrough（全部 file:line）: `notes/stereoworld_code_walkthrough_20260610.md`（本地 ICLR2026/notes/ + 4090d notes/）。
> 仓库 clone: 本地 `ICLR2026/notes/code_refs/stereoworld/`（HEAD 7856c29, 2026-06-08 放权重）。

**背景**: 我们的 stereo cam_rope 参考 StereoWorld 论文（arXiv 2603.17375）实现, 但 q_cam_proj/k_cam_proj 双零初始化 → 梯度死锁, 从未学习（摆设, 见 memory `project_cam_rope_inert_bug`）; 且扩维 256→272 撞 FlashAttention 上限 → 慢 4 倍（2026-06-10 已用 CAM_ROPE=0 旁路修掉）。

**深读后的 4 个关键事实**:
1. **发布代码 ≠ 论文叙述**: 论文的 "Camera-Frame RoPE 扩维" 和 "分解式 Stereo Attention" 在官方代码里都不存在。实际实现 = 主注意力完全不动 + 每层一条**并行相机注意力支路**（自带 q/k/v/out 普通投影）, 支路输出残差加回主注意力。
2. **几何零参数**: 支路内用 PRoPE 官方变换（Q 乘投影矩阵转置、K/V 乘逆、输出乘回; 同相机打分不变, 跨相机注入相对几何）, 不可学 → 不存在"学不动"问题。
3. **防死锁配方 = 单边零初始化**: 整条链只把支路出口 out_proj 置零（"zero-initialize out_proj for stable residual training"）, q/k/v 保持 xavier → step-0 == baseline 且第一步就有梯度。我们的双零双线性是自己发明的, 参考代码里 grep 不到。
4. **不撞 FA 上限**: PRoPE 变换在 attention kernel 外面做, head_dim 不变; 他们 head_dim 才 128。变换完直接喂 FlashAttention。

**我们的旧实现还有 3 个独立偏差**（修也救不回来, 建议放弃旧机制）: Q/K 用同一个 P 旋转（他们 Q 用转置、K 用逆 → 同相机=单位阵; 我们同相机也被扰动）; V 不变换、输出不乘回; 内参归一化常数差 2 倍。

**复活路线（推荐路线 1）**:
- 路线 1 ⭐ 并行相机支路: 选定层挂窄支路（宽度/层数都有旋钮）, 支路内 PRoPE 变换 + out_proj 零初始化。step-0==baseline、FA2 不掉、免死锁、有 production 参考。混合序列（文本+图像）用 per-token 矩阵分支（文本=单位阵）。与 #5/#7 "零出口残差支路"方法族同构。
- 路线 2 ✗ 只修双零 init: 扩维慢 4 倍的问题会回来, 不推荐。
- 路线 3 零新参数 in-place PRoPE（alpha-gate 保 step-0）: 最轻但直接扰动 Qwen 已训练 M-RoPE 维度, 风险高一档。
- 考场提醒: 饱和 LIBERO 4-suite 大概率测不出收益（FFS 全系教训）; 真正考场 = 非饱和深度敏感 suite。

---

_最后更新: 2026-06-10 by Claude（增 §10 StereoWorld 代码深读 + cam_rope 复活路线; 上次 2026-05-19）_
