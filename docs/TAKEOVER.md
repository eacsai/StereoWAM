# StereoWAM 研究项目接手入口（2026-10-29）

本文件是新同学和新 Codex 的首读入口。研究仓库为
https://github.com/eacsai/StereoWAM ，工作分支为 ffs_controlnet_dev。
公司仓库 eacsai/StereoVLA 不属于本项目。旧 CLAUDE.md、2026-07-08
handoff 和 E2-E5 roadmap 是历史材料；其中旧服务器、正在运行、计划中等描述
不能覆盖本文和当前进程/日志证据。

## 本次接手要做什么

用户最新目标是：Primary 双目图像 + 单目 Wrist 图像 + 由 Primary 双目
重建的 Utonia 点云特征，并让三台相机的外参通过几何位置编码进入网络。
用户明确指定 Primary 双目 rig 坐标系作为世界系 W。
实现约定：用原 primary 光学相机作为该 rig 的参考原点/轴向，
T_W_from_primary = I。另一目与 wrist 都使用相对此参考的外参。
这是一项待实施的新方案，当前代码尚未支持真实三相机动态外参。

先读 [三视图方案](experiments/20261029_stereo_wrist_utonia_plan.md)，
再读 [实验结果](experiments/20261029_l40_results.md) 和
[资产与环境恢复](ASSET_MANIFEST.md)。

## 最短恢复方式

~~~bash
git clone --branch ffs_controlnet_dev --single-branch \
  https://github.com/eacsai/StereoWAM.git starVLA
cd starVLA
bash scripts/codex_restore_context.sh --read-only
python3 scripts/verify_handoff.py --source-only
git status --short
git log -5 --oneline
~~~

前两个恢复命令只读当前 checkout，不安装环境、不下载权重、不启动 GPU。
本仓库内的交接资料足够开始设计/读码，不依赖私有 starvla-project-archive。
完整历史私有归档当前未在接手主机恢复，当前 GitHub 账号也无法确认访问；
--with-archive 才会尝试可选归档流程。不存在的材料不能标记为已恢复。

如继续使用原 L40，由负责人单独提供 SSH 权限和 PROJECT_ROOT；
不要在公开文件中提交登录凭据。进入原项目后先核对 branch/status，
不要用 fresh clone 覆盖原工作树。新机器 clone 为它自己的权威工作树；
在原 L40 上继续时，以原 L40 项目为权威源。

## 当前实际状态

- 2026-10-29 只读核对：分支仍为 ffs_controlnet_dev，交接前 HEAD 为 925a28f。
  本次交接提交后的 Git HEAD 才是代码快照身份。
- 训练框架：Qwen3.5-0.8B + GR00T action DiT。Qwen 配置为 24 层，
  顺序重复三层 linear_attention 加一层 full_attention，共六个 full 层。
- 普通 stereo 输入为 primary,left_view；新生成的额外图像是逻辑 left_view。
  磁盘历史字段 right_view 经 loader alias 映射为 left_view，不要据字段名再次交换。
- 双目数据保留了 wrist 视频，但没有本方案所需的逐帧三相机标定契约。
- Qwen 权重、FFS/Utonia 权重、四套双目数据、Utonia 历史缓存和已有 checkpoint
  在原主机仍可定位；没有放进 GitHub。
- 最后两个 visual_causal / visual_image_bidir 的 legacyaccum_retry_20260723
  run 已于 2026-07-23 按用户要求停止。两者最新已保存模型为
  steps_14000_pytorch_model.pt；没有这两个 run 的最终 30k 评估结论。
- 2026-10-29 没有发现这两组训练重新运行。GPU 当前另有负载；
  不能把历史 GPU 分配视为当前可用授权。
- .pt 保存的是模型 state_dict；不能称为包含 optimizer/RNG/sampler 的完整断点。
  恢复权重不等于精确续训，见结果记录。

## 代码地图

| 位置 | 已有功能 / 本次接手用途 |
|---|---|
| starVLA/model/framework/VLM4A/QwenGR00T.py | Qwen+GR00T 主框架、Camera Branch 接线 |
| starVLA/model/modules/stereo/cam_branch_attention.py | 双相机 PRoPE 分支、E3 初始化、E5 稀疏路由；固定双目几何仍是限制 |
| starVLA/model/modules/stereo/visual_attention.py | full-attention 的因果/图像双向对照；不隔离 linear state |
| starVLA/model/framework/VLM4A/QwenGR00T_UtoniaPromptTokenFFS.py | 两视图 prompt-token 框架，当前 builder 只写两幅图，wrist 会被遗漏 |
| starVLA/model/modules/stereo/utonia_pointcloud.py | FFS 视差、left_view 反投影、冻结 Utonia、缓存验证 |
| starVLA/model/modules/stereo/depth_token_inject.py | 插入 64 个点云 token、位置与 mask 延长 |
| starVLA/model/modules/stereo/cam_rope_hook.py | 图像 token 到相机 ID 的映射；历史命名，不代表恢复旧 Camera-RoPE |
| starVLA/dataloader/gr00t_lerobot/data_config.py | video_keys 和 observation_indices |
| starVLA/dataloader/gr00t_lerobot/mixtures.py | data_mix 到 robot config 的实际路由 |
| SSF/render_scripts/、scripts/4090d/regenerate_libero_stereo.py | 历史三路图像渲染，需要扩展逐帧位姿导出并校验图像方向 |
| examples/LIBERO/eval_files/、deployment/model_server/ | rollout 客户端与策略服务端，必须同步传三视图和当前时刻标定 |
| scripts/tools/precompute_utonia_cache.py | 历史 left_view-frame 网格缓存；不能直接冒充 W-frame 缓存 |

历史评估工具还有已确认的操作风险：scripts/a800/eval_one_ckpt_l40.sh
会按端口正则执行 pkill，eval_qwen2p5vl_4suite_l40.sh 会删除既有评估目录，
并包含进程清理。它们为保存历史复现路径而保留，不是本次只读恢复入口。
接手者在正式使用前先改成新输出目录、明确进程所有权及精确 PID 清理，
不能根据“过去跑过”直接执行。此次交接未执行这些清理/覆盖操作。

## 接手后的顺序

1. 完成只读恢复，列出自己机器缺少的资产和源码依赖。
2. 在同一 episode/frame 上验证 primary、left_view、wrist 及三组 K/T。
   相机外参缺失时先补导出，不开始长训。
3. 先做纯三视图 RGB baseline，再分别加入 Utonia 和相机几何，
   最后做两者组合；不要只与少一路 wrist 的旧 stereo baseline 比。
4. 用点重投影/坐标变换 round-trip 和实际 token 分组检查接线。
   新三视图 smoke 通过后才按当前资源批准启动训练。
5. E3-full 是前任选择的几何分支起点，不是已证明优于 baseline 的方案。
   E4 是新方案的数据标定前置工作；E6/E7 严格状态隔离保留为后续消融。
   Scene Flow / past-flow 和 GAN 不纳入第一批三视图实验。

接手请求授权只读检查和与明确接手任务相关的可逆准备，不授权自动开始训练、
评估、大下载、sudo、覆盖配置、删除 checkpoint 或终止其他任务。
不要把旧文件里的“自动排队”或“仍在运行”当成当前指令。

## 给同学直接粘贴到 Codex 的说明

> 请接手 eacsai/StereoWAM 的 ffs_controlnet_dev 研究分支。先读根 AGENTS.md，
> 运行 bash scripts/codex_restore_context.sh --read-only，并完整阅读
> docs/TAKEOVER.md 和它列出的方案/实验/资产文件。目标是
> primary + left_view + wrist 三视图、双目生成 Utonia 特征，
> Primary 双目 rig 参考系作为世界系 W，三相机外参通过几何位置编码进入网络。
> 当前原 primary 光学相机定义为 rig 参考，T_W_from_primary=I；
> wrist 外参必须逐帧更新。请先核对现有实现与缺口，汇报首批小规模验收方案。
> 不依赖私有历史归档，不恢复旧 Camera-RoPE，不自动启动长训或覆盖已有文件。
