# Primary 双目 + Wrist + Utonia + 相机几何：接手研究方案

状态：PROPOSED / NOT IMPLEMENTED。记录日期：2026-10-29。
目标来自用户本次接手要求；Primary 双目 rig 作为世界坐标系也是用户明确要求。
下面的接口、token 布局和消融是建议设计，不是已经运行的模型或已验证的提升。

## 1. 输入与坐标系

三路图像固定按 [primary, left_view, wrist] 标识，不能复用历史 right/left 名字猜相机。
“Primary 双目”指 primary 与其配对的 left_view，并非多加一个 primary 图像。
W 固定在该双目 rig 上；本方案以原 primary 光学相机为 rig 参考原点与轴向，
光学轴约定为 x 向图像右、y 向图像下、z 向前，长度单位米。
T_A_from_B 是将 B 坐标转换为 A 坐标的 4x4 齐次变换：

~~~text
X_A = T_A_from_B @ [X_B, 1]
T_W_from_primary = I
T_W_from_left = 双目标定的固定相对位姿
T_W_from_wrist(t) = 随当前机器人姿态变化的相对位姿

如果从仿真全局系 S 导出：
T_W_from_C(t) = inverse(T_S_from_primary(t)) @ T_S_from_C(t)
T_C_from_W(t) = inverse(T_W_from_C(t))

如果从机器人手眼标定和 FK 计算：
T_W_from_wrist(t) =
    T_W_from_base @ T_base_from_EE(t) @ T_EE_from_wrist
~~~

S 只用于导出相对变换，不把物体 GT、目标位姿、未来动作或未来帧的位姿作为输入。
每个 RGB、K、T、action 必须对应同一个观测时刻 t，图像要在执行 action_t 前采样。
静态外部 rig 只是 W 固定的常见情形，不能因此把腕部相机也当成静态相机。
只给 wrist 单目图像和外参不提供它的像素深度；可以构造射线，
不能假装知道 wrist 每个像素的 3D 点。

几何特征统一到 W，不改变现有 action/proprio 的标签坐标、归一化或 gripper 约定。
若以后改变动作坐标，必须作为单独设计并在训练/rollout 中成对实现逆变换。

## 2. 建议的第一版网络

复用 Qwen3.5-0.8B 与 GR00T flow-matching action head，
先冻结 FFS 和 Utonia，训练投影器、几何支路与既有策略可训练部分。
不增加独立的大型 stereo transformer，不加入 GAN/scene-flow loss。

~~~text
RGB P,L: [B,2,3,H,W]     RGB Wrist: [B,3,H,W]
       |                        |
       +--> Qwen 视觉编码 --> [B,Np+Nl+Nw,D] ----+
       |                                         |
       +--> 冻结 FFS(L,P)                         |
                | d_L [B,1,H,W]                   |
                v                                 |
       left_view 光学系 XYZ_L [B,N,3]             |
                | T_W_from_left                   |
                v                                 |
       XYZ_W + RGB_L --> 冻结 Utonia               |
                | per-point features              |
                v                                 |
       网格/occupancy + projector --> [B,64,D] ----+
                                                  |
Instruction --> text tokens [B,Nt,D] -------------+
                                                  v
                     Qwen 混合骨干 + 三相机几何分支
                     hidden [B,Np+Nl+Nw+64+Nt+special,D]
                                                  |
                               既有 conditioning/mask
                                                  v
                           GR00T action DiT --> action chunk
~~~

B 是 batch；D 在当前 0.8B 配置为 1024。
Np/Nl/Nw 由各图像经 processor 得到的实际 grid 决定，不能硬编码为 64。
现有 Utonia 网格是 [B,1387,8,8]，其中 1386 是 feature channel，
另 1 channel 是 occupancy；映射后是 64 个点云 token。
1386 不是点数，64 也不是所有相机必须具有的视觉 token 数。

为减少第一版改动，建议保留现有插入位置，再加 wrist：
[Primary][既有点云提示与 Utonia token][Left][Wrist][Instruction]。
具体 role/position/mask 必须由结构化 token span 决定，不能依赖解析一段自然语言 marker。
历史 checkpoint 评估要保留它训练时的提示字符串；去掉或移动 marker/prefix
是后续单独消融，不能只在 eval 改文字后宣称模型本体变化。

这是输入层融合的设计图；action head 实际还经过框架内 conditioning/mask 处理，
不能把图中的合流误写成已实现的直接 raw-token 拼接给 DiT。

## 3. 点云参考系是首个关键缺口

当前有效 stereo 顺序是 FFS(image1=left_view, image2=primary)。
视差与 RGB 反投影对应 left_view 光学系；VLM 图像顺序则是 primary,left_view。
不能因为用户称其为“Primary 双目点云”，就说历史缓存已在原 primary 坐标系。

新方案先对有效点做 X_W = T_W_from_left X_left，再按记录的 Utonia
transform/scale 编码。scatter 回参考 left_view 像素网格仍保存逐点像素索引，
世界系坐标与像素归属是两个不同字段。
建议保存每个网格的有效点位置/occupancy，必要时给点云 token 加 W-frame xyz 编码，
使“点云在哪个位置”与“点云语义特征”同时可解释。

旧缓存是 left_view-frame 的编码结果，不能直接改 meta 名字充当 W-frame 缓存。
Utonia 特征的 1386 个通道不是 xyz，不能对它们乘 3x3 旋转矩阵。
如果点坐标在 Utonia 前发生变换、标准化或尺度变化，应新建缓存并重编码，
除非有独立实验证明编码器的相应等变/不变性。保留旧缓存用于历史重现。

新缓存 identity 至少包含：数据版本、episode/frame、图像摘要、图像顺序、
K 与像素变换、T_W_from_left、world_frame_definition、长度单位、FFS/Utonia
权重摘要、预处理参数与代码 revision。训练 cache 与 rollout live 路径必须一致。

## 4. 外参如何作为隐式位置编码

优先沿用 Camera Branch 的 PRoPE 思路，扩展成逐样本三相机接口：

~~~text
camera_to_world: [B,3,4,4]    # 按 P,L,Wrist，映射到 W
intrinsics:      [B,3,3,3]    # 与送给视觉编码器的像素一致
viewmats = inverse(camera_to_world)
P_i = lift(K_norm_i) @ viewmat_i
image_token_camera_id: [B,Nimage]
~~~

用对应相机的 P/P transpose/P inverse 变换支路 Q/K/V 与输出，
几何通过 attention 进入学习；不是把外参数字写进 instruction。
K 虽不是本次创新变量，仍需正确提供：PRoPE 的投影矩阵不能只靠外参构造。
点云 token 使用 W 下的位置和角色，不冒充 wrist 图像 token，也不因插入而挤乱相机 ID。

当前 compute_libero_camera_prope_triples 只接受两台相机，使用固定 baseline/fovy，
且保留了历史兼容符号。新三相机路径必须独立接受实际 K/T；
不能把 num_cameras 改成 3 就认为完成，也不能沿用旧固定 +baseline 注释来解释实际 viewmat。
保留 legacy path 用于重现旧实验。

首版只扩展六个 full-attention 层中的相机支路：
层索引 [3,7,11,15,19,23]（0-based）。
E3-full 的 out_proj std=0.02 是前任选择的起点；不是有统计证明的最优值。
Qwen 主干仍有 linear attention，所以只改 full 的 mask 不等于整个骨干双向，
更不等于已经实现 Utonia/image 的严格信息隔离。

先选定并记录一种主干/支路方向规则，所有对照保持一致。
全因果与图像双向是后续方向消融；2026-07-23 的两组对照停在 14k，
目前没有它们的最终胜负结论。需要比较时两组使用同一实际 SDPA 后端。
E5 的 same-row/top-k 只适用于已整流双目，不用于相机朝向不同的 wrist；
wrist 跨视图边默认走一般几何 attention，不能硬套同一图像行。

## 5. 数据与 rollout 接口建议（尚未实现）

在独立的新 data_mix/robot config 中显式定义：
- video_keys = [video.primary_image, video.left_view, video.wrist_image]
- camera_order = [primary,left_view,wrist]
- camera_to_world[B,3,4,4]、intrinsics[B,3,3,3]
- world_frame_definition = primary_optical_rig_reference_v1
- frame/episode ID、有效标记、像素变换与相机单位

实际 wrist key 由 loader alias 验证，不能仅把上述设计字符串抄入 YAML。
不要修改旧 libero_all_sfstereo_leftprimary 的两视图意义。
将 stereo_pair_size=2 与 total_rgb_views=3 分开；不能把 FFS 的双目数量也改成 3。
当前 PromptToken builder 只构造两图 message，即使 dataloader 多给一幅 wrist
也不会自动被网络使用；必须改 builder、图像 token 映射和对应 hook。

训练端：在同一仿真状态导出三台相机的世界位置/旋转，明确 MuJoCo 的 -Z
与所选光学 +Z 之间的变换，再形成 W 下的相对 K/T。
导出不使用策略不可能获得的 object GT，也不根据训练未来信息算输入。
评估端：客户端发送当前三图及标定，服务端接受并传到骨干，
reset episode 时清理缓存/状态，不能用上一 episode 的 wrist 位姿。

图像 180 度旋转、crop、resize 或 flip 必须联动像素几何。
可以记录像素变换 A 并用 K_effective=A K，也可等价改变光学基；
不能重复补偿。现有只读 K(fovy) 函数不足以表示所有新变换，
应通过射线与点重投影验证最终实现。

## 6. 最小实验矩阵与公平性

| ID | RGB 输入 | Utonia | 三相机几何 PE | 回答的问题 |
|---|---|---|---|---|
| B-PW | primary,wrist | 无 | 无 | 原版参考；历史均值 96.00% |
| B-PL | primary,left_view | 无 | 无 | 旧 stereo 参考；历史均值 90.25% |
| T0 | primary,left_view,wrist | 无 | 无 | 仅补 wrist 的三视图基线 |
| T1 | 同 T0 | W-frame | 无 | 点云本身是否有增益 |
| T2 | 同 T0 | 无 | 有 | 相机几何本身是否有增益 |
| T3 | 同 T0 | W-frame | 有 | 两者结合是否有增益 |

T0-T3 用同一数据/seed/视图预处理/模型起点/优化预算/rollout协议，
逐项比较，不把多一路 wrist 的收益归因于点云或 PE。
优先完成小规模确定性接线，再决定正式训练；本交接不启动任何实验。

有效 batch 需要实际 micro_batch * world_size * engine_GA。
历史 L40 是 8*4*4=128，但日志 30k 不等于 30k optimizer updates：
旧 Accelerate GA=1 与 DeepSpeed GA=4 的计数不同，详见结果文档。
新代码必须记录 micro_step、engine global_steps、scheduler step、samples_seen；
预算对齐实际 optimizer updates/样本量，不只对齐 run 名字里的 30k。

先按同一 LIBERO 四套件、相同任务列表与 trials/task 评估，报告逐 suite、
总计成功数/试验数、种子和重复评估范围。历史单次结果不能当置信区间。
如果常规任务已饱和，再独立设计深度/视角/遮挡 OOD；不能把自选 OOD 的胜出
混进原四套件均值。

## 7. 接手第一批验收（完成后再长训）

- 三图同帧：检查至少两个 episode 的起/中/末帧，图像、action、wrist pose 对齐。
- 几何：T_inverse round-trip、旋转正交/det=1、基线长度、点重投影、
  resize/180度变换一致性；W 的 primary 位姿为 I。
- 动态腕部：不同时间 wrist 相对位姿确实变化；常量回退必须报错。
- token：三个图像 span、相机 ID 和实际 grid 一致；64 个点云 token 单独有角色；
  batch>1、padding、不同图像 token 长度均不串样本。
- 网络：证明 wrist 像素扰动能改变对应视觉/策略输出；
  Utonia/PE 的开关和梯度可观测；缓存与 live 结果在容差内一致。
- 元数据：缺 K/T 不静默填 identity；禁用几何时必须真正旁路。
- 参数错配负对照：只交换 K/T 或打乱 wrist pose，几何测试应失败。
  图像与 K/T 一起交换只验证几何绑定；因果序列有顺序效应，不能要求动作输出完全不变。
- 真实小规模训练：先验证 no_sync、NaN、checkpoint key/shape、真实 step 计数和
  server/client 一致性，再由当前资源授权决定长训。

## 8. 代码拆分建议

1. 数据导出/读取：扩展 renderer、data_config/mixtures 与 collate，建立新三视图 schema。
2. 几何模块：增加通用的逐样本 K/T 三相机路径与坐标测试，保留旧双目路径。
3. Utonia：W-frame 建点与缓存版本，保留像素索引/occupancy/位置。
4. 骨干：三图 builder、role spans、position/mask hook 与相机支路接线。
5. rollout：client/server 标定传输、episode reset 和训练/推理一致性。
6. 实验：先 T0，后 T1/T2，最后 T3；E6/E7 状态隔离另起消融。

这六项可按接口分工；共享 schema 与最终集成由一个负责人管理。
没有完整论文/manuscript 可供本次校对，本文件不作论文一致性或新颖性保证。
