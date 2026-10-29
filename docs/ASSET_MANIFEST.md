# 外部资产与环境恢复

GitHub 保存代码、方案、精简结果与依赖元数据；不保存数据、模型权重、
checkpoint、缓存、虚拟环境、完整日志或大批可视化。
当前源树的 playground 和 external 是指向主机数据盘的链接；
fresh clone 不会恢复这些链接的目标内容。

本机目录由接手者设置 PROJECT_ROOT 和 ASSET_ROOT，不能照抄旧服务器绝对路径。
原 L40 的实际登录和目录由负责人单独提供。原始旧路径只作为历史 provenance，
不是新机器恢复步骤。

## 依赖源码

[environment.json](handoff/environment.json) 保存 2026-10-29 观察到的版本。
以下完整 SHA 是本次从已有 checkout 读取，clone 后必须 checkout 到对应 SHA：

| 目录（相对项目） | 来源 | 完整 revision |
|---|---|---|
| external/Fast-FoundationStereo | https://github.com/NVlabs/Fast-FoundationStereo | a290ba04c1b3ad1ec41a33974a157b2917b624d4 |
| external/Utonia | https://github.com/Pointcept/Utonia | da776a0bd3a48c6df83ac2ae0e27b26141cc7e31 |
| SSF/libero_sim_env/LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 8f1084e3132a39270c3a13ebe37270a43ece2a01 |
| code_refs/StereoWorld（仅参考，可选） | https://github.com/SunYangtian/StereoWorld | 7856c295b22121f4f5ae2160ef687ea6983635dd |

下面是首次恢复命令，目标目录存在时先检查 remote/HEAD/dirty，不覆盖：

~~~bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git external/Fast-FoundationStereo
git -C external/Fast-FoundationStereo checkout a290ba04c1b3ad1ec41a33974a157b2917b624d4
git clone https://github.com/Pointcept/Utonia.git external/Utonia
git -C external/Utonia checkout da776a0bd3a48c6df83ac2ae0e27b26141cc7e31
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git SSF/libero_sim_env/LIBERO
git -C SSF/libero_sim_env/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
~~~

不 vendoring 第三方 .git/源码/权重。许可证以各固定 revision 的 LICENSE、
model_card 与数据许可为准；本交接没有重新审定其再分发许可，不将未知许可资产上传。

## 权重

| 资产 | 逻辑目标路径 | 当前身份 / 恢复来源 |
|---|---|---|
| Qwen3.5-0.8B | playground/Pretrained_models/Qwen3.5-0.8B | 官方 HF Qwen/Qwen3.5-0.8B；历史 snapshot revision UNKNOWN。单权重文件 1,746,942,600 bytes |
| Fast-FoundationStereo 20-30-48 | external/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth | 固定源码 readme.md 的官方权重下载入口，选择 20-30-48，约 646 MB |
| Utonia | playground/Pretrained_models/Utonia/utonia.pth | HF Pointcept/utonia 的 utonia.pth；历史下载 revision=main，确切 snapshot UNKNOWN；目录约 524 MiB |
| 项目实验 checkpoint | playground/Checkpoints/<run_id>/checkpoints/ | 未上传；只能经负责人授权复制，或按记录重新训练，不能从 GitHub 直接恢复 |

已观察到的 SHA-256：

~~~text
Qwen model.safetensors-00001-of-00001.safetensors
04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

FFS model_best_bp2_serialize.pth
98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692

Utonia utonia.pth
a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353
~~~

Qwen 摘要在本次重新计算；FFS/Utonia 摘要来自现有缓存 meta，
接手时必须重新计算实际文件核对。不得仅凭文件名宣布一致。

获得下载许可并确认空间后，可以在新目录执行（不是恢复脚本的自动步骤）：

~~~bash
hf download Qwen/Qwen3.5-0.8B --local-dir "$PROJECT_ROOT/playground/Pretrained_models/Qwen3.5-0.8B"
hf download Pointcept/utonia utonia.pth --local-dir "$PROJECT_ROOT/playground/Pretrained_models/Utonia"
~~~

main 可能已变化，下载后若摘要不同应记录新 revision 并停止“严格历史复现”的说法。
FFS 下载以该固定源码的 readme.md 官方链接为准；需要登录/接受许可时由负责人完成。
不自动执行 gated/付费/大体积下载。

## 数据

- 官方 primary+wrist：HF dataset IPEC-COMMUNITY/libero，目标
  playground/Datasets/LEROBOT_LIBERO_DATA，现有目录约 1.8 GiB。
  旧 scripts/4090d/data_preparation.sh 能说明目录/文件校验协议，但 CUR 是旧机器路径，
  且还下载 LLaVA 数据，不能在新机器原样自动运行。
- 双目+腕部：原 L40 的 playground/Datasets/LIBERO_STEREO_DATA/<suite>，
  四个 suite 为 libero_spatial/object/goal/10。
  每个 suite 的 meta/info.json 包含 observation.images.image、right_view、
  wrist_image、action、state、timestamp/episode/frame 等字段。
  训练入口 LEROBOT_LIBERO_STEREO_LEFTPRIMARY/<suite>_no_noops_1.0.0_lerobot
  是指向上述四个 suite 的链接。完整 raw-data revision 与全文件 checksum
  尚未保存，不能声称 bit-exact 重建。
- 历史右目磁盘名不等于几何右眼：loader 将历史 right_view alias 成 left_view；
  VLM 用 [primary,left_view]，FFS 用 [left_view,primary]。
- 新三视图方案仍需导出/验证每帧三组 K/T。当前视频存在并不意味着外参已准备好。

若负责人无法提供现有数据副本，先从固定 LIBERO 项目的官方说明获取
原始 HDF5 demonstrations 与任务资源（遵守原数据许可）。
之后用仓库内 renderer 生成到新的空目标目录，例如先做一个 demo：

~~~bash
python scripts/4090d/regenerate_libero_stereo.py \
  --suite libero_goal --libero_raw_data_dir "$ASSET_ROOT/libero_raw" \
  --libero_target_dir "$ASSET_ROOT/libero_stereo_rebuild" \
  --baseline 0.06 --num_tasks 1 --num_demos 1
~~~

该脚本当前生成三图/动作，但没有新方案的逐帧 W-frame 外参；
先按方案扩展导出、检验图像翻转及 action/gripper 约定，再扩到完整四套件。
不使用 --overwrite 指向现有数据，也不把新生成数据冒充旧版本。

## Utonia 缓存

目标：playground/Datasets/utonia_cache_perpatch_leftprimary_0719，
现有约 46 GiB。每个 suite 有 meta.json、grid.f16.npy、done.npy、index.pkl。
meta 包含模型摘要、图像摘要抽样、count、预处理和 left_view 参考系。
不要反序列化未知来源的 pickle。先验证可信来源和摘要。

历史缓存维度 [count,1387,8,8]，其中 1386 维 Utonia 特征加 occupancy；
旧参考系是 left_view，world_frame_definition 不满足新的 W-frame 契约。
同一旧数据/旧设置可按下列入口重新生成，正式 GPU 使用需另行安排：

~~~bash
python scripts/tools/precompute_utonia_cache.py \
  --cache-dir "$ASSET_ROOT/utonia_cache_new" \
  --data-root "$PROJECT_ROOT/playground/Datasets/LEROBOT_LIBERO_STEREO_LEFTPRIMARY" \
  --data-mix libero_all_sfstereo_leftprimary \
  --ffs-repo-dir "$PROJECT_ROOT/external/Fast-FoundationStereo" \
  --ffs-model-path "$PROJECT_ROOT/external/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth" \
  --utonia-ckpt-path "$PROJECT_ROOT/playground/Pretrained_models/Utonia/utonia.pth"
~~~

这条是历史 left_view-frame generator，不是新 W-frame generator。
新方案修改输入坐标后必须另建缓存版本并同时改 live inference。

## 环境

2026-10-29 实际主环境：Python 3.10.20、torch 2.6.0、torchvision 0.21.0、
transformers 5.3.0、accelerate 1.5.2、deepspeed 0.16.9、numpy 1.26.4、
mujoco 2.3.7、robosuite 1.4.0、av 12.3.0。
还需要 spconv-cu124 2.3.8、cumm-cu124 0.7.11、torch_scatter 2.1.2+pt26cu124。
现有环境没有 flash-attn；使用实际支持的 attention backend，
不要因为配置名写 flash_attention_2 就假设它已生效。

新机器单独建 Python 3.10 venv，再安装匹配 torch/CUDA ABI 的依赖。
environment.json 是观察记录，不是完整的可移植 lockfile；
spconv/scatter 驱动与 wheel 兼容、MuJoCo/EGL 和 editable LIBERO/Utonia 源码
均需在目标机器验证。不能复制旧 .venv 来声称新环境恢复。
历史训练、仿真可能用不同 venv；本次 L40 eval 入口使用主 .venv，
NumPy 2.x 曾导致旧评估异常，务必核对实际 client/server 的解释器。

## 验收与缺失项

~~~bash
python3 scripts/verify_handoff.py --source-only
python3 scripts/verify_handoff.py --asset-root "$PROJECT_ROOT/playground"
~~~

第二条仅检查核心资产入口与元数据文件存在，不加载权重、不跑 GPU，
也不替代全量数据/缓存校验。缺资源时输出明确路径，不能创建空目录伪装成功。
完整私有记忆归档、原实验模型及其完整优化器状态不在 GitHub；
逐帧三相机 K/T、新 W-frame Utonia 缓存和三视图网络尚待实施。
