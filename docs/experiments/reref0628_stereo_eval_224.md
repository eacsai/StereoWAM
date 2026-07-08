# reref0628 stereo — 224 (Fix#2 bicubic) 重评

这是 **Fix#2 修好后**（eval resize → 224、用 PIL 默认 bicubic 对齐训练）对 ① stereo_plain / ② cam_branch 的**重评**。

- **协议**：leftprimary, from-scratch, 30k, eff-batch 128 (BS16×GA8), video_keys=primary,left_view, **eval 图 224×224 bicubic**。
- **取代**：`reref0628_stereo_autoeval.md` 里 ①② 的旧 **256** 数（那批 eval 喂 256 未 resize，是 train/eval 分辨率错配；旧 per-run 文件已挪到 `playground/Checkpoints/<rid>_eval_results.txt.256_pre_fix2`）。
- **③ utonia**：eff32 探索版的 224 数在 `reref0628_stereo_autoeval.md`（用的是 Fix#2-bilinear，已废）；矫正版（eff128 + Fix#1 补 Utonia 路径）训完后在此追加（同 224 bicubic 协议），届时 ①②③ 三者统一可比。

---

## 旧 256 数（仅供对照，已废）
- ① stereo_plain @256: spatial 0.97 / object 0.92 / goal 0.93 / l10 0.58 / **MEAN 0.85**
- ② cam_branch  @256: spatial 0.89 / object 0.89 / goal 0.92 / l10 0.58 / **MEAN 0.82**

## 224 (bicubic) 重评结果
（autoeval 守护自动追加在下方）

===== qwen0p8_groot_stereo_plain_leftprimary_fromscratch_reref0628_30k step=30000 vk=primary,left_view (Mon Jun 29 02:55:43 PM UTC 2026) =====
  libero_spatial: SR=0.91
  libero_object: SR=0.93
  libero_goal: SR=0.92
  libero_10: SR=0.55
  MEAN: 0.8275
===== qwen3p5_0p8b_4suite_monoprimary_ourrender_fromscratch_reref0628_30k step=30000 vk=primary (Mon Jun 29 03:38:46 PM UTC 2026) =====
  libero_spatial: SR=0.9
  libero_object: SR=0.95
  libero_goal: SR=0.85
  libero_10: SR=0.63
  MEAN: 0.8325
===== qwen0p8_groot_stereo_cambranch_leftprimary_fromscratch_reref0628_30k step=30000 vk=primary,left_view (Mon Jun 29 04:30:33 PM UTC 2026) =====
  libero_spatial: SR=0.91
  libero_object: SR=0.95
  libero_goal: SR=0.9
  libero_10: SR=0.64
  MEAN: 0.8500
===== qwen0p8_groot_depthtoken_keep_sceneflow_all4_lambda130_leftprimary_fromscratch_fullft_eff128_maskfix_30k step=30000 vk=primary,left_view (Tue Jun 30 06:37:51 PM UTC 2026) =====
  libero_spatial: SR=0.88
  libero_object: SR=0.95
  libero_goal: SR=0.89
  libero_10: SR=0.5
  MEAN: 0.8050
===== qwen3p5_0p8b_utonia_prompttoken_leftprimary_fromscratch_eff128_maskfix_30k step=30000 vk=primary,left_view (Wed Jul  1 01:03:33 AM UTC 2026) =====
  libero_spatial: SR=0.91
  libero_object: SR=0.98
  libero_goal: SR=0.88
  libero_10: SR=0.69
  MEAN: 0.8650
===== qwen3p5_0p8b_utonia_prompttoken_cambranch_leftprimary_fromscratch_eff128_maskfix_30k step=30000 vk=primary,left_view (Wed Jul  1 01:53:40 PM UTC 2026) =====
  libero_spatial: SR=0.93
  libero_object: SR=1.0
  libero_goal: SR=0.9
  libero_10: SR=0.7
  MEAN: 0.8825
===== qwen3p5_0p8b_utonia_prompttoken_sceneflow_lambda130_leftprimary_fromscratch_fullft_eff128_maskfix_30k step=30000 vk=primary,left_view (Wed Jul  1 06:23:24 PM UTC 2026) =====
  libero_spatial: SR=0.87
  libero_object: SR=0.84
  libero_goal: SR=0.82
  libero_10: SR=0.5
  MEAN: 0.7575
