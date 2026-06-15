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

## 11. ⭐ FastWAM 官方代码深读结论（"它到底做没做预训练", 2026-06-12）

论文 arXiv 2603.16666（IIIS 清华 + Galaxea，与 DepthVLA 同组）。仓库 github yuantianyuan01/FastWAM，本地 clone `code_refs/fastwam`。完整逐文件 walkthrough = `notes/fastwam_code_walkthrough.md`；论文笔记 = `notes/2603.16666_FastWAM.md`。方法：8 读码 + 3 对抗代理（11 Opus），结论带 file:line。

**核心问题：FastWAM 做没做预训练？→ 没有自己的预训练（3/3 对抗代理判 False）。** 拆三类：
- (a) **继承的预训练**：加载现成 **Wan2.2-TI2V-5B**（视频 DiT 3072/30层/24头 + VAE + T5），阿里 Wan 团队训的，不是它训的（wan22.py:50-51；ModelScope/HF 按 hash 加载）。
- (b) **权重手术（非训练）**：动作专家骨干 = 把 Wan2.2 DiT 权重线性插值+alpha 缩放初始化（`preprocess_action_dit_backbone.py`，无梯度无数据）；action_encoder/head/proprio 保持随机。
- (c) **它自己的预训练阶段**：**没有**。全仓只有一个训练阶段 = benchmark 上单次 finetune（run_training→一个 Wan22Trainer.train()）。搜遍 pretrain/stage/OpenX/DROID/OXE 全无。

**⭐ 关键纠正（别误读）："不预训练" ≠ "训练轻"**：训练时那个 **5B 视频 DiT 是被训练的,不是冻结的**——`self.dit=self.mot`（含视频+动作两专家），trainer 全冻后只 `model.dit.requires_grad_(True)`，优化器 = `list(model.dit.parameters())`(+proprio)（trainer.py:85-90,289-291）；只冻 VAE+T5。"without embodied pretraining" 的真义 = 不用机器人语料/不分预训练阶段,但整个 6B end-to-end finetune 了 → 这才是 RoboTwin 要 64 卡的原因。

**训练配方（实锤）**：AdamW lr 1e-4 cosine+5%warmup，wd 1e-2，bf16，DeepSpeed ZeRO-1；batch 16/卡；**LIBERO 8 卡 10 epochs（全局 128）/ RoboTwin 64 卡 5 epochs（全局 1024）**（README.md:259）。数据 = LIBERO 4 套件 lerobot + RoboTwin 27,500 demos（HF 下载），33 帧窗口=32 动作+9 视频帧（4:1）。

**对我们（2 H100）**：① "省具身预训练 + 借大模型先验" 学得了（同 DepthVLA 套路）；② 但它端到端训 6B（视频 DiT 没冻）→ **Wan2.2-5B 我们 2 卡放不下,要换轻量 video DiT 或改"冻骨干只训小专家"**（偏离其 co-train 核心，而 ablation 证删 co-train 真机 90%→10%）；③ "训练 co-train 塑表征、推理砍未来" 实证支持我们 stereo-4D 的"训练预测未来 4D、推理一次前向"。


## §12 ControlVLA 官方代码深读 + 我们 #7 一致性核对

> **一句话结论**：我们的 `_our_controlvla_impl/controlvla_branch.py`（#7 并行 K/V 控制支路）在**所有受力轴上忠实复刻**了 ControlVLA 官方"活路径"的 K/V 控制机制；唯二的实质差异都是**有意为之的适配**（控制条件从物体掩码 token 换成 FFS 立体 token；条件编码器从微调改成硬冻结），不破坏"ControlVLA 式并行 K/V 支路"这个 claim。**跑 #7 前没有必须修的 mechanism 级 divergence**，只有一个可选的小加固（给支路 K/V 也清零 bias，但我们本来 bias=False，所以其实已经干净）。

### (a) ControlVLA 是什么 + 它的 K/V 控制机制

**ControlVLA**（论文 "ControlVLA: Few-shot Object-centric Adaptation for Pre-trained VLA"）= 在一个**预训练好的 VLA**（官方基于 diffusion_policy 的 action-diffusion transformer）上，加一条 **ControlNet 风格的零初始化控制支路**，用极少样本把**物体中心信号**注入冻结的策略里。核心思想跟 ControlNet 一样：冻结预训练主干 + 训练一条"出生即零贡献、慢慢长出来"的旁支。

**⚠️ 文件陷阱（核对官方时第一坑）**：仓里有两套同名的 "control" 实现，**只有一套是真货**：
- ❌ `transformer/control_transformer.py` + `diffusion/transformer_for_action_control_diffusion.py` = **废弃的旧变体**。它的控制 attention 整段被注释掉，实际跑的是 `memory = torch.cat([memory, control_memory])` 再做**一次普通 softmax**（= 把控制 token 拼进 memory 做单 softmax）。这**不是** ControlVLA 出货的机制，对比时引用它会误判。
- ✅ **真正出货的 K/V 机制 = 这三个文件**：
  - `transformer/modules.py` — 核心 attention 数学（`KVControlMultiheadAttention` + `control_multi_head_attention_forward` + `control_scaled_dot_product_attention`）
  - `transformer/kvcontrol_transformer.py` — decoder 层把上面那个 attention 接进每个 cross-attn block
  - `diffusion/transformer_for_action_kvcontrol_diffusion.py` — 外层 action-diffusion 包装 + 零初始化 + 冻结/训练划分

**官方 K/V 支路怎么工作（核心数学，`modules.py` L311-338）**：每个 cross-attn decoder 层里，给一个**共享的 Query Q**（来自 action/去噪 token），跑**两个独立 softmax 再相加**：

```
对每个 cross-attn 层:
    Q       = 主干 in_proj 投出的 query                       # 支路不另投 Q（control-Q 被丢弃）
    K_t,V_t = in_proj 投 memory（VL/obs 条件）               # 主干键值
    K_z,V_z = control_in_proj 投 control_memory（物体 token）  # 控制键值，独立投影
    主干输出   = softmax(Q·K_tᵀ/√d) · V_t
    控制输出   = softmax(Q·K_zᵀ/√d) · V_z                    # 第二个独立 softmax，不与主干拼键
    融合      = 主干输出 + 控制输出                            # 在 out_proj 之前相加
    最终      = 同一个共享 out_proj(融合)                      # 整条只有一个输出投影
```

关键点（每条都是后面核对的"受力轴"）：
1. **共享 Q**：控制支路复用主干 Q，自己不投 query（`modules.py:552` 把 control-Q 丢弃 `_, control_k, control_v = ...`）。
2. **两个独立 softmax 再求和**，不是"把控制键拼进主干键做一次 softmax"。`modules.py:553-554` 那两行 `cat([k, control_k])` 是**被注释掉的**——作者特意选了 separate-softmax-then-sum。
3. **融合在 out_proj 之前，且共享同一个 out_proj**（`modules.py:709` 只有一次 `linear(..., out_proj_weight)`）。没有可学门控、没有 concat。
4. **零初始化 = 清零控制输入投影，不是 ZeroConv**。`transformer_for_action_kvcontrol_diffusion.py` 的 `_zero_init_control_weights`（L71-78，由 `zeroinit_control=True` 触发）对每个 `KVControlMultiheadAttention` 做 `constant_(control_in_proj_weight, 0)` + `constant_(control_in_proj_bias, 0)`。这把打包的 [Q;K;V] 控制投影全清零 → 出生时 K_z=V_z=0 → 控制输出 = softmax(Q·0)·0 = 0 → step-0 策略输出 == 预训练策略。
   - **ZeroConv1d 是红鲱鱼**：`common/zeroconv_layer.py` 里确实有个真零初始化的 `ZeroConv1d`，但它在 K/V 活路径里的引用（`kvcontrol_transformer.py` L243/248）**整段被注释掉**。谁要是说"必须用 ZeroConv 才忠实"，是错的——官方活路径根本没用它。
5. **冻结/训练**：忠实配方是 `get_control_optim_groups`（按 `if 'control' in pn` 过滤参数），只训练 `control_in_proj_weight/bias` + 控制条件位置嵌入 + 物体编码器，主干被排除出优化器 = 冻结。
   - ⚠️ **官方出货的 UMI workspace 自己都没严格冻**：`kvcontrol_finetune_..._workspace.py` L61 实际调的是**全量** `get_optimizer`，control-only 那行（L60）被注释掉了。所以出货配置是"全量微调 + warm-start + 零初始化控制支路"，比论文 claim 松。论文真正的"只训控制支路"在那条被注释的路径上。
6. **条件来源 = 物体中心掩码**：`vision/oc_obs_encoder.py`（默认 config 选的 `OCObsEncoder`）把每个物体的二值分割掩码（B×num_objs×H×W bool）编成**每物体 1 个稀疏 token**（掩码质心的 2D 位置嵌入 + 可学 global_object_embedding，物体缺席则用 empty_object_embedding，可选 graycnn 形状 token）。即"物体在哪 + 在不在"。这是 few-shot **object-centric** adaptation 的本体。
7. **控制支路无 mask**：`control_scaled_dot_product_attention` 若传入 `is_causal`/`attn_mask` 直接报错（L318/325）——控制注意力永远是全可见无遮挡。
8. **每层都有**：decoder 是 `_get_clones` 的同一层，每个 `KVControlTransformerDecoderLayer` 各持一个 `KVControlMultiheadAttention`，同一份 control_memory 广播到每层。

### (b) 一致性裁决：逐轴对比 我们的 `controlvla_branch.py`

我们的实现（`ControlVLAAttention`，per cross-attn DiT block 把 `block.attn1` **替换**为 wrapper，原 `Attention` 存为 `self.base`）：

| 轴 | 官方（kvcontrol 活路径） | 我们的 #7 | 裁决 |
|---|---|---|---|
| **(1) K_z/V_z 来源 + 共享 Q** | 专用 `control_in_proj` 投 control_memory，复用主干 Q，丢弃 control-Q | `to_k_z(ffs)`/`to_v_z(ffs)` 两个独立无偏 Linear，复用 `base.to_q(hidden_states)` 的同一 Q，不另投 Q（L111/142-146） | **一致** |
| **(2) 零初始化形式** | 清零打包 `control_in_proj_weight`+`bias`（含没用的 Q 片）→ K_z=V_z=0 | `nn.init.zeros_` 清零 `to_k_z.weight` + `to_v_z.weight`（bias=False，无 bias 可漏，L95-98）。任一清零即足，两个都清是双保险 | **一致** |
| **(3) 融合点（前/后 out_proj，sum/gate/concat）** | sum 两个值聚合 **在** out_proj **之前**，过**一个共享** out_proj；两个独立 softmax（concat 行注释掉） | `fused = attn_t + attn_z`（L149）在 `base.to_out[0]/[1]` 之前过**一次共享** to_out（L151-154）；两次独立 `F.scaled_dot_product_attention` = 两个独立 softmax | **一致** |
| **(4) 每层 vs 单点** | 每个 cross-attn decoder 层各一个（clone 广播同一 control_memory） | `install_controlvla_branches` 给**每个** cross-attn block 换 attn1，`require_all_cross_attn=True` 断言全覆盖 + 总块数 guard，所有块读同一 `_STATE.ffs_tokens` | **一致** |
| **(5) 冻结 vs 训练** | 忠实配方只训 `control*` + 微调物体编码器（低 LR）；**出货 UMI workspace 实际全量微调**（control-only 注释掉） | 训练 = 每层 `to_k_z/to_v_z` + 框架级 `ffs_pos_emb`；**硬冻结 FFS 立体编码器**（`.eval()` + `requires_grad=False` + no_grad）；主干冻结由 launcher FREEZE 管 | **部分**（见下） |
| **(6) 条件来源** | 稀疏物体中心掩码 token（质心位置 + 物体身份） | 稠密 FFS 立体几何 token（`backbone` concat[L,R] 单目，或 `gru_hidden` net[0] 视差感知；AdaptiveAvgPool 到 8×8=64 token + `ffs_pos_emb`） | **部分**（见下） |
| **(7) batch/mask/位置嵌入** | 控制支路无 mask；time_emb 拼到 cond 和 control_cond 双流；control_cond 加可学位置嵌入 | 支路 SDPA 不传 attn_mask（主干保留 mask）→ 一致；`ffs_pos_emb` 对应 control_cond_pos_emb；**HIGH-1**：`ffs.repeat(B//B_ffs,1,1)` = [s0,s1,s0,s1] 匹配 `QwenPI.forward` 的 `actions.repeat`（不是 repeat_interleave，否则 [s0,s0,s1,s1] 串扰）；time_emb 没拼进 FFS token（主干仍带 time） | **一致** |

**总裁决：在所有受力轴（机制本身 = 轴 1/2/3/4/7）上 term-for-term 忠实。** 对抗 reviewer 无法在"ControlVLA 式并行 K/V 支路"这个 claim 上挑出机制级破绽：共享 Q、对独立条件流做零初始化独立 K/V 投影、两个独立 softmax 求和、在单个共享输出投影之前融合、每层放置、控制注意力无 mask——全对得上，且对的是**活的 kvcontrol 路径**而非废弃的 concat 路径（如果我们当初是"把 FFS 键拼进 K_t 再做一次 SDPA"，那才会匹配死路径 = 不忠实）。

### (c) 跑 #7 前的 divergence 处理

**必须修的（mechanism 级）：无。** 机制完全一致，可以直接跑。

**可选小加固（不是 bug）**：
- 官方零清了打包投影的 weight **和** bias；我们的 `to_k_z/to_v_z` 是 `bias=False`，所以根本没有非零 bias 会泄漏 → step-0 已经是干净的字节级零贡献，**不需要额外动作**。（如果将来有人把它们改成 `bias=True`，记得补 `nn.init.zeros_(...bias)`，否则 step-0 parity 破。）

**有意为之、可接受的适配（论文里要明说，别藏）**：
1. **条件来源换内容（轴 6）**：官方 = 稀疏物体中心掩码 token；我们 = 稠密 FFS 立体几何 token（`backbone` 单目 concat 或 `gru_hidden` net[0] 真视差）。这是往**同一个结构插槽**（action query 经零初始化支路额外注意的一条独立条件流）里换内容——K/V 机制本身 source-agnostic，所以是合法适配，**正是我们工作的 novelty**，论文应明确陈述而非淡化。
2. **冻结条件编码器（轴 5）**：官方**微调**物体编码器（低 LR）；我们**硬冻结** FFS 立体编码器（它是蒸馏出来的几何 teacher）。这是可辩护的设计选择，要明说它跟 ControlVLA 的"微调条件编码器"配方不同。**反讽点**：官方出货的 UMI workspace 实际是全量微调 + warm-start（control-only 优化器被注释掉），所以连官方代码都比它论文 claim 松——我们靠 launcher FREEZE 冻主干，反而**更接近 ControlVLA 论文声称的"冻预训练、只训控制支路"原意**。
3. **time_emb 没拼进 FFS 控制 token**：官方把 time 也拼进 control_cond；我们没拼。无所谓——两边主干 cond 流都还带 time，action query 照样看得到 time，不影响控制机制。
4. **零初始化基底形式**：官方清一个打包 control_in_proj（含没用的 Q 片）；我们清两个独立无偏 K/V Linear。step-0 no-op 功能等价。
5. **两次 SDPA vs 官方单函数双 softmax**：数学完全相同（两个 softmax 求和）。

**论文撰写建议**：(1) 引用 `modules.py` + `kvcontrol_transformer.py` + `transformer_for_action_kvcontrol_diffusion.py` 作为 spec，**别引** `control_transformer.py`（死路径）；(2) 把"条件源替换"和"冻结编码器"作为有意设计明说；(3) 说明 step-0 字节级 parity 就是 ControlNet 式 warm start，对应官方 `zeroinit_control=True`。

**关键文件锚点**：我们 `_our_controlvla_impl/controlvla_branch.py`（机制，L97-98 零初始化 / L149 融合 / L153-154 共享 to_out）+ `QwenPI_ControlVLA_FFS.py`（条件源 + 冻结）；官方 spec `controlvla_official/diffusion_policy/model/transformer/modules.py`（L311-338 双 softmax 求和 / L552 丢弃 control-Q / L709 共享 out_proj）+ `.../diffusion/transformer_for_action_kvcontrol_diffusion.py`（L71-78 零初始化）。

## §13 3D-VLA 官方代码深读 + 我们点云注入(Utonia)落地评估 (2026-06-14)

> 一句话: 3D-VLA(ICML2024, UMass) = **BLIP-2(LAVIS 框架)+ 冻结 FlanT5-XL**。它的"点云"**不是用点云编码器编的**——是**离线预算好的 1408-d EVA-CLIP/BLIP 特征 lift 到 SAM 分割的体素点云**(3D-LLM 那套"2D 特征抬到 3D");一个 **Q-Former(32 learnable query)** cross-attend 这些点特征压成 32 token → Linear → 拼进 T5 输入嵌入的 `<scene>` 占位符位置。**动作=离散文本 token**(无连续动作头);**goal 点云=独立的 Point-E 扩散**(跟 LLM 只靠指令文本连,不可微耦合)。clone: `code_refs/3dvla`;论文 `papers/2403.09631.pdf`(15 页全读)。

### (a) 3D-VLA 注入机制(受力轴,file:line)
1. **3D 编码器=离线,不在训练图里**。dataset 直接从磁盘 load `pc_feat (N,1408)` + `pc (N,3)`(`threedvqa_datasets.py:58-61,129-132`)。1408=EVA-ViT-g 宽度(`eva_vit.py:522`);特征是 EVA-CLIP/BLIP 图像特征经 SAM 分割+体素化 lift 到点(路径名 `voxelized_features_sam_nonzero`/`nps_blip`, `:35-43`)。**训练时没有 live ViT/点网**。N=sample_num 固定(默认 8000;config 6400)。
2. **几何只以"小幅相加正弦 PE"进入**:整数体素 XYZ(clamp 0..255)查固定 `PositionalEncoding1D(1408//3=469)` → reshape 1407 → pad 1408 → `pc_embeds = pc_feat + pos*0.1`(`blip2_t5.py:132-135,146-151`)。这是 LLM 分支里**唯一**的 3D 几何来源(粗、不可学)。
3. **Q-Former 压缩**:32 个 learnable query(`blip2.py:55-66`, encoder_width=1408, cross_attention_freq=2)先自注意再 cross-attend N 个点特征 → `(B*T,32,768)`(`blip2_t5.py:153-160`; cross-attn K/V=Linear(1408→768))。
4. **投影 + 拼接进 LLM**:`t5_proj=Linear(768→2048)`(`:130,161`)→ 每帧 32 token;`insert_3d_feats` 把这 32 token splice 进 T5 ENCODER 输入嵌入序列里**每个 `<scene>` 标记之后**(`:167-198`, 倒序遍历 cat 左/特征/右)。T5 当普通词嵌入吃(`inputs_embeds`, `:257-263`)。**单点、嵌入层注入,无逐层 adapter、无注入 LLM 内部 cross-attn**。
5. **冻结/可训**:FlanT5-XL transformer 块**全冻**,只解冻 {Q-Former, 32 query, t5_proj, T5 输入/输出 embedding 表(因加了新 special token)}(`:121-126`)。从 BLIP2-FlanT5-XL 权重 init(strict=False),**不是**从 3D-LLM ckpt。
6. **动作=离散文本 token**:tokenizer 加 256×`<aloc>` + 256×`<arot>` + `<gripper0/1>` + `<ACT_SEP>` + 256×`<loc>`(`:104-113`);整模型一个 T5 seq2seq 交叉熵 loss(`forward :200-266`)。**没有连续动作头**(跟我们 GR00T DiT 完全不同范式)。
7. **goal 点云/图像=独立扩散,非张量耦合**:goal 点云=Point-E `GoalPointDiffusionTransformer`(`pointe/transformer.py:627-706`),把 start 点云**通道拼接**(6→12 ch, `modify_layer` 重建 input_proj, output_proj 零初始化)给噪声 goal,CLIP 文本 condition;DDPM 1024 步 ε-MSE;独立 `train_pe_goal_pcd.py` 训。goal 图像=SD-2 InstructPix2Pix。LLM 只发 `<image>`/`<pcd>` 标记,两个扩散头各自 render。三组件分开训(`launcher/train_{llm,ldm,pe}.sh`),不可微联训。

### (b) 对我们: "左图+深度+内参→点云→Utonia→注入" 落地评估
**结论: 方案可行且比 3D-VLA 更扎实, 但注入要走 Utonia 自家的【相加】法、不要照搬 3D-VLA 的 Q-Former, 且决定性实验必须上非饱和/OOD 单变量对照。**

- **比 3D-VLA 更扎实**: 3D-VLA 根本没用真点云编码器(它的"点特征"=2D CLIP 特征贴 3D 位置 + 0.1 正弦 PE)。我们用 **真反投影点云(左图+左深度+K)+ 冻结 Utonia(137M PTv3,真几何编码器)**,几何信息比 3D-VLA 强。
- **注入法选 Utonia 自家的相加, 不照搬 3D-VLA Q-Former**: Utonia 论文 A.3 自己就做了我们要做的事——per image-patch 3D 坐标→伪点云→冻结 Utonia→特征**逐元素加到 2D 视觉 token**(Video-3D LLM 上),操作 Tab.9 Utonia 82.1>Concerto 80.0>Sonata 74.7。对 0.8B 小模型,相加(对齐到左图 token 网格)比 3D-VLA 的"Q-Former 32 token+splice 占位符+离散动作 token+FlanT5"整套轻得多,且我们已有 #2(ControlNet 残差→vl_embs)/#8(残差→action Value)注入机器可复用。**3D-VLA 真正可借的只有"encoder→resampler(Q-Former/Perceiver)→投影→marker splice"这个 pattern**(若点数多需 resampler 压 token);`pointe/perceiver.py SimplePerceiver` 是现成 cross-attn resampler。
- **注入点**: 跟我们已验证一致——VLM 端加到 vl_embs 的**左图图像 token 位置**(几何对齐,因 Utonia 点特征与左图同源),或动作端。PointVLA/PointACT 证据: 别注进冻结 VLM trunk 深处, action 端/边界更稳。
- **命门(别忘 6pp 噪声底)**: 我们 ~10 个注入变体在饱和 LIBERO 全噪声带内; 点云/几何收益在文献里**只在 OOD 显**(PointVLA 高度/GeoVLA 视角/Utonia cluttered)。所以**决定性实验 = Utonia 点云特征 vs 我们现有的立体深度特征(FoundationStereo net[0])单变量对照, 上非饱和/OOD split(RoboCasa+右目 / 高度尺度视角扰动)**。若 Utonia 不能在 OOD 上超我们已有的立体注入 + 超 6pp, 就**不值得**加 137M 冻结编码器 + 反投影 + PTv3 forward 的推理开销。
- **接入步骤(starVLA)**: ① 左深度+K 反投影成点云(右先左后约定下用左/primary 帧); ② 冻结 Utonia forward(channel 整除 6, 注意 granularity rescale); ③ Utonia per-point 特征对齐到左图 token 网格(scatter, 仿 #2/#8 的 primary-token 对齐); ④ 零初始化相加到 vl_embs(step-0=baseline)。⑤ Utonia 权重从 Pointcept 取(pointcept.github.io/Utonia)。
- 参见 Utonia 笔记 `notes/2603.03283_Utonia.md`、点云 VLA 清单 `notes/pointcloud_vla_litscan.md`、`[[project_experiment_registry]]`(6pp 噪声底)、`[[project_stereo_consistent_4d_proposal]]`(Utonia future work 也指向 4D)。


## §14 Utonia 仓库深读 + #4 上下文 token 形式注入点云特征到 starVLA (2026-06-14)

> 仓库 `code_refs/utonia`(github Pointcept/Utonia, 小: 23 .py, 全模型在 `utonia/utonia/model.py`)。论文 `papers/2603.03283.pdf` + 笔记 `notes/2603.03283_Utonia.md`。一句话: Utonia = **RoPE-PTv3 冻结点编码器**, `utonia.load()` 从 HF 下权重(架构 config 在 .pth 里), forward 吃一个 grid-sample 后的 point dict, 返回带 per-point `.feat` 的 Point 对象。我们要把它的 per-point 特征经 **#4 上下文 token 形式**(project→token→插 VLM 序列)塞进 starVLA。

### (a) 核心模型(model.py 一个文件)
- **Point3DRoPE**(model.py:51-107): head_dim 必须**整除 6**(assert %3==0 + arange step2 + rotate_half), 把 head_dim 三等分对 x/y/z 各加 1D rotary, **作用在原始连续 coord 上**(几何进 attention 的唯一机制)。
- **SerializedAttention**(model.py:147-333): 序列化(z-order/hilbert)patch 注意力, 每层建一个 rope, `q,k=rope(q,k,point.coord[order])`(model.py:294,298), flash-attn varlen 或手写 softmax。
- **PointTransformerV3**(model.py:648-849): embedding stem + 5 段 encoder(GridPooling 下采样 + Block) + 4 段 decoder(GridUnpooling 上采样, enc_mode 时跳过)。`forward(data_dict)` 返回 **Point(addict.Dict), 不是裸 tensor**; per-point 特征在 `out.feat`; encoder 每次 pool 存 `pooling_parent`/`pooling_inverse` 供多尺度读出。
- **⚠️ 真架构 config 不在源码, 在 ckpt["config"] 里**(model.py:885 `PointTransformerV3(**ckpt["config"])`)。源码默认 args 是**坏占位符**(enc_channels/heads 给 head_dim=16 不整除3 → Point3DRoPE assert 崩)。**绝不能用默认实例化**; 必先 `ckpt=utonia.load("utonia",ckpt_only=True); print(ckpt["config"])` 拿真 channels/heads/depths/in_channels/enc_mode。
- 3 个论文设计里**只有 RoPE-on-coords 在推理仓实现**; Causal Modality Blinding(无 mask) + Granularity Rescale(shift/jitter/rescale_coords 入参但 forward 没用) 是训练期/stub。推理就当 plain RoPE-PTv3。
- ⚠️ 两 agent 分歧: 一个说释放权重 enc+dec(全分辨率输出), 一个说 enc_mode=True(纯编码器)。**以 ckpt["config"]["enc_mode"] 为准**(pre-flight 打印)。

### (b) 集成 API(要在 starVLA 复刻的配方)
1. **load+冻结**: `import utonia; model=utonia.load("utonia",repo_id="Pointcept/Utonia").cuda().eval(); [p.requires_grad_(False) for p in model.parameters()]`。CPU/无 flash: `custom_config=dict(enable_flash=False, enc_patch_size=[1024]*5)`。权重落 ~/.cache/utonia/ckpt。
2. **建点云(我们的左图+深度+K)**: 逐像素反投影 `X=(u-cx)*d/fx, Y=(v-cy)*d/fy, Z=d`(相机系)→ coord(N,3); color(N,3)=左图 RGB 0-255; normal(N,3)= 没有就 `np.zeros_like(coord)`(Utonia 把缺模态当零)。坐标用**米制**。
3. **transform(强制)**: `t=utonia.transform.default(scale=4.0, apply_z_positive=True, normalize_coord=False)`(scale=4.0=操作粒度, 是**唯一**粒度旋钮, grid_size 固定 0.01 scaled 单位 → 有效体素=0.01/scale)。`point=t({"coord","color","normal"})` → 张量 dict: `coord/grid_coord/color/feat(=cat[coord,color,normal]=9维,须等于 in_channels)/offset/inverse(N_orig→grid 映射)`。多样本 `utonia.data.collate_fn([...])`。
4. **forward**: `with torch.inference_mode(): out=model(point)`。
5. **读 per-point 特征**: 走 pooling 链拼多尺度(demo: `for _ in range(2 or 4): parent=out.pop("pooling_parent"); inv=out.pop("pooling_inverse"); parent.feat=cat([parent.feat,out.feat[inv]]); out=parent`)→ 再 `per_point=out.feat[out.inverse]` 映回**原始 N 点(跟像素 1:1)**。**输出维度从 ckpt["config"]/linear-probe head config 读, 别信占位符**(占位 96/512)。
- **依赖(重, 接入风险)**: py3.10 / CUDA12.4 / **torch 2.5.0**(我们 starVLA 是 torch **2.6.0**+cu124, 版本差一档要验) / spconv-cu124 / torch-scatter / flash-attn(或 enable_flash=False) / huggingface_hub / addict / timm / numpy<=1.26.4。**spconv + torch_scatter 是承重原生依赖**, 4090d/h100b 要先装通。

### (c) #4 上下文 token 形式接入(用户要的)
**可行, 复用 #4(QwenGR00T_DepthTokenFFS)机器, 换特征源**。流程:
```
左图深度+K → 反投影点云 → 冻结 Utonia forward → per-point 特征 out.feat[out.inverse] (N≈像素数, 跟左图 1:1)
   → reshape 回左图 H×W 网格(因每点=一像素) → AdaptiveAvgPool 到 8×8=64 token(同 #4 对 net[0] 的做法)
   → 零初始化 Linear 投到 Qwen 宽度 → 插入 VLM 输入序列(keep/strip, 同 #4)
```
- **关键: 这一步解决了"Utonia 输出不规则点 vs #4 要规则网格"的矛盾** —— 因为点是从左图反投影来的, per-point 特征能直接 scatter 回图像网格, 再 AdaptiveAvgPool, 既复用 #4、又保图像对齐(也=Utonia A.3 "per image-patch" 的精神)。
- **落地 = 新框架 `QwenGR00T_UtoniaPointTokenFFS`**(镜像 #4 DepthTokenFFS, 加 "反投影+Utonia forward+多尺度读出+scatter回网格" 前处理, 换掉 FoundationStereo net[0])。零初始化投影 → step-0=baseline。冻结整个 Utonia(eval+requires_grad False+inference_mode)。
- **pre-flight(接入前必做)**: ① `print(ckpt["config"])` 拿真 in_channels/输出维度/enc_mode; ② 4090d/h100b 装通 spconv+torch_scatter+flash-attn(torch 2.6 vs 2.5 验兼容); ③ 确认反投影点云的 N 和左图网格能对齐 reshape; ④ smoke step-0==baseline。
- **命门(还是 6pp 噪声底, 决定建不建)**: 这是**第 11 种几何注入**。决定性实验 = **Utonia 点云特征 vs 我们现有 FoundationStereo net[0] 立体特征, 单变量, 上非饱和/OOD split**(RoboCasa+右目 / 高度尺度视角扰动)。Utonia 在 OOD 超不过已有立体注入 + 超不过 6pp → 不值得加这套重点云管线(反投影+137M PTv3+spconv 推理开销)。**真问题不是"能不能用#4装Utonia"(能), 是"它在OOD能不能超已有立体注入"。**
- 参见 `[[reference_3dvla_utonia_pointcloud]]`(裁决) / §13(3D-VLA, 对比: 3D-VLA 没用真点云编码器) / `notes/2603.03283_Utonia.md` / `[[project_experiment_registry]]`(6pp)。


## §15 4D 点云怎么进 VLM(token 预算)+ 注入法对比 + 为何 3D-VLA 整套不适合(2026-06-14)

> 背景: 用户问"输入 4D 帧→4D 点云, 庞大点云怎么喂 VLM", 以及"能不能用固定长度 Query(跟单图 token 等长)学点云特征再 cat 进 VLM 序列"。结论先放: **能, query-resampler 是对的, 且是 3D-VLA 唯一可复用的部分; "3D-VLA 不适合"指它的整套 stack(冻结 FlanT5 + 离散动作文本 token + 离线 2D-lift 假点特征 + 重 Q-Former), 不是 resampler 概念。**

### (a) 铁律: 永远不把【原始点】当 token 喂 —— 点编码器 + 固定 token 预算扛
原始 4D(T 帧 × N 点)几十万点, 直接当 token 必爆。三层压缩:
- **空间维(单帧 N 点 → 固定 K token), 三选一**:
  1. **per-patch 对齐 + 相加(Utonia 自家法, 最省, 新增 0 token)**: 一个图像 patch 一个伪点 → Utonia 特征跟视觉 token 1:1 → 逐元素加。token 数 = patch 数不变。代价: 几何分辨率被 patch 网格限。
  2. **网格池化(#4 法)**: per-point 特征 scatter 回图像网格 → AdaptiveAvgPool 8×8=64 token/帧。
  3. **学习式 resampler(Q-Former/Perceiver, = 用户的提议)**: K 个 learnable query cross-attend N 点 → K token, **token 数与点数解耦**。
- **时间维(T 帧, 别堆 T 份完整 token 否则 T×K 爆)**:
  - (a) **帧聚合**: 多帧位姿对齐聚成一个规范点云再编码一次(Utonia frame-aug 即此)。
  - (b) **关键帧 + 运动 token**: 少数帧编空间 + 紧凑 scene-flow 编时间。
  - (c) **跨 4D 固定预算 resampler**: K query cross-attend 整个 4D 点集(加时空 PE)→ 全程 K token, 不随 T/N 长。
  - (d) token merging/pruning(ToMe 式)编码后合冗余。
- **0.8B 账**: Qwen 一帧图像 token ~64-256; naive 4D = T×(图像+几何) token, T>2-3 爆。Utonia-add 新增 0; resampler 固定 K(32-64)不随 4D 长。

### (b) 注入法三选一 + 取舍(给我们)
| 法 | token 增长 | 几何对齐 | 可训练量 | 适用 |
|---|---|---|---|---|
| **Utonia-add(per-patch 相加)** | 0(骑在图像 token) | 强(1:1 像素对齐) | 最少(一个零初始化投影) | 单帧/可 per-patch 对齐; 小模型最省; Utonia 验证过 |
| **#4 网格 token insert(keep/strip)** | +64/帧 | 中(scatter 回网格) | 少 | 单帧, 复用我们现成 #4 机器 |
| **Query-resampler → cat(用户提议=3D-VLA 可复用核)** | +K(固定, 不随点/帧长) | 弱(需 VLM 学 query↔图像 对应) | 中(resampler 跨注意力 + K query) | **庞大/不可对齐的 4D 点云首选** |

- **关键洞察**: 单帧可对齐 → Utonia-add 最省; **庞大 4D 不可 per-patch 对齐 → resampler(用户提议)才是对的工具**(token 与点数/帧数解耦)。所以用户的直觉对——4D 场景就该用 resampler。
- K 不必 = 单图 token 长(query 是"摘要"不需对齐图像数量); K=32-64 通常够(3D-VLA 用 32)。等长也行, 只是更贵。
- ⚠️ resampler 的 cat token 跟图像 token **不天然对齐**, 冻结 VLM 要靠注意力**学**它们的对应(BLIP-2/Flamingo 能做到, 但小模型/饱和数据下比 aligned-add 难收敛)→ 小模型先试 Utonia-add, 4D/大点云再上 resampler。

### (c) 为何 3D-VLA【整套】不适合, 但 resampler【概念】可复用
3D-VLA 可复用的**只有** "learnable query cross-attend 点特征 → 固定 token → splice 进 LLM 序列" 这个 resampler 机制(= 用户提议)。**不适合的是它整套 stack**:
1. **骨干 = 冻结 FlanT5-XL(encoder-decoder T5)**; 我们 = Qwen3.5(decoder)+ **GR00T DiT 连续动作头**。范式不同。
2. **动作 = 离散文本 token**(`<aloc>/<arot>` 由 T5 解码), 无连续动作头; 我们是连续 diffusion 动作。直接搬 = 错范式。
3. **它的"点特征"不是真点编码器**, 是离线 lift 的 2D EVA-CLIP 特征贴 3D 位置; 我们用**真 Utonia 点特征**, 更扎实。
4. **Q-Former = 完整 BERT(重)+ 需 BLIP-2 init**; 我们用更轻的 Perceiver/小 cross-attn resampler 就够。
→ 所以: **借 resampler 概念(用户提议), 接到我们 Qwen+GR00T-DiT + 真 Utonia 特征上**; 不搬 3D-VLA 的 T5/离散动作/Q-Former/2D-lift。

### (d) 给我们的落地建议(按场景)
- **单帧立体深度/点云(现在)**: Utonia-add(per-patch, 零新增 token) 或 #4 insert —— 最省, 复用现成机器, 先做。
- **未来 4D 点云**: **resampler(用户提议)固定 K token + 时间维聚合或 scene-flow** —— token 与 4D 规模解耦。但记住 **4D 真价值在运动**: 与其堆 4D 点, 不如编/预测 scene-flow(接 stereo-4D / LaMP)。
- 命门不变(6pp 噪声底): 不管哪种注入, 决定性实验 = Utonia/点云特征 vs 现有立体 net[0], 单变量, 上 OOD; 超不过 6pp 不值得。
- 参见 §13(3D-VLA 走查)/ §14(Utonia + #4 接入)/ `notes/2603.03283_Utonia.md` / `[[project_stereo_consistent_4d_proposal]]`。
