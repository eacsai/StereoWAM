===== qwen0p8_groot_depthtoken_keep_sceneflow_all4_gradnorm_leftprimary_fullft_30k step=30000 vk=primary,left_view (Sat Jun 27 03:35:05 AM UTC 2026) =====
  libero_spatial: SR=0.89
  libero_object: SR=0.88
  libero_goal: SR=0.85
  libero_10: SR=0.62
  MEAN: 0.8100
===== qwen0p8_groot_depthtoken_keep_sceneflow_all4_lambda130_leftprimary_fullft_30k step=30000 vk=primary,left_view (Sat Jun 27 04:14:40 AM UTC 2026) =====
  libero_spatial: SR=0.85
  libero_object: SR=0.85
  libero_goal: SR=0.83
  libero_10: SR=0.57
  MEAN: 0.7750
===== qwen3p5_0p8b_utonia_prompttoken_liveffs_leftprimary_fullft_30k step=30000 vk=primary,left_view (Sat Jun 27 02:33:02 PM UTC 2026) =====
  libero_spatial: SR=0.88
  libero_object: SR=0.9
  libero_goal: SR=0.87
  libero_10: SR=0.56
  MEAN: 0.8025
===== qwen0p8_groot_stereo_plain_leftprimary_fullft_30k step=30000 vk=primary,left_view (Sat Jun 27 03:19:46 PM UTC 2026) =====
  libero_spatial: SR=0.87
  libero_object: SR=0.94
  libero_goal: SR=0.74
  libero_10: SR=0.6
  MEAN: 0.7875
===== qwen0p8_groot_stereo_cambranch_prope_leftprimary_fullft_30k step=30000 vk=primary,left_view (Sat Jun 27 03:55:19 PM UTC 2026) =====
  libero_spatial: SR=0.92
  libero_object: SR=0.95
  libero_goal: SR=0.94
  libero_10: SR=0.55
  MEAN: 0.8400
===== qwen0p8_groot_stereo_orthogrid_leftprimary_fromscratch_fullft_eff128_maskfix_30k step=30000 vk=primary,left_view (Fri Jul  4 2026, eval on 4090d GPU0/2/6) =====
  libero_spatial: SR=0.73
  libero_object: SR=0.83
  libero_goal: SR=0.78
  libero_10: SR=0.33
  MEAN: 0.6675
  # orthogrid third view = live server-side render (deployment/model_server/ortho_render_live.py).
  # Eval pipeline validated: render_grid==sqlite cache byte-exact (maxdiff=0), wiring smoke
  # [ortho-eval-inject] n_images=3 sizes=[(224,224),(224,224),(454,454)] prompt_md5=36b1a455.
  # VERDICT: orthogrid HURTS -> WORST leftprimary run (below stereo_plain 0.7875 / cambranch 0.84).
  # libero_10 0.33 (vs 0.55-0.62 others) = long-horizon collapse. Likely live(FFS)-vs-offline
  # render domain gap on the rendered 3rd view (plain-stereo/cambranch consume camera params, no
  # rendered 3rd image, so they don't eat this gap). position=last; ortho-first variant TBD.
