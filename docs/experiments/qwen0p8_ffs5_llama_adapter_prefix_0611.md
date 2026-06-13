# #5 LLaMA-Adapter prefix FFS injection (frozen-VLM arm) — 2026-06-11

## Question
Can FFS stereo-depth features (`net[0]`, the post-GRU disparity hidden state) help the policy
when injected the **LLaMA-Adapter way** — a small set of learnable prompt tokens that absorb the
stereo features and are gated-prepended into the VLM attention, with the VLM itself **frozen**?
This is method **#5** of the 8-method FFS-injection port. Anchors on the saturated LIBERO 4-suite:
mono primary **0.913** (ceiling), depthtoken_keep **0.910** (best FFS), B stereo baseline **0.908**.

## Method (faithful to LLaMA-Adapter v1, 2303.16199)
- A few **learnable prompt tokens** cross-attend to a summary of the FFS `net[0]` stereo features
  (true post-GRU disparity, not the monocular pre-cost-volume features), then are **gated-prepended**
  to the key/value stream of the **last 6 softmax layers** of the Qwen3.5-0.8B VLM.
- **Zero-init `tanh` gate** (so step-0 ≡ baseline B; the injection starts as a no-op and the gate
  opens only if it helps) — verified faithful to LLaMA-Adapter v1 (`alpaca_finetuning_v1` per-head
  `gate.tanh()`); V2-multimodal dropped the tanh, we keep it.
- Prefix tokens are **not RoPE-rotated** (per v1).
- **Only the adapter trains** (frozen arm): warm-start from B, freeze the VLM
  (`FREEZE=qwen_vl_interface`), so the new module is the only thing learning.
- `o_proj` is param-detached on the injection path (`_linear_with_detached_params`) — faithful to
  LLaMA-Adapter's fixed-backbone philosophy; for the frozen arm this is harmless (VLM frozen anyway).

### 5 deviations from the original (documented for the paper)
1. **single-scalar gate** (this frozen arm) instead of per-head — the per-head variant is the
   separate from-scratch arm (see below).
2. injected at **6 softmax layers** (the last 6) rather than the top-L prompt-tuning layers.
3. a **shared summary** of the stereo features feeds all prefix tokens.
4. **cross-attention absorption** of the stereo features into the prompt tokens.
5. a **new `kv_proj`** for the prefix stream.

参见 [[project_method5_perhead_ablation]] [[project_ffs_8methods_port_status]]
[[project_ffs_features_are_monocular]] (net[0] = true disparity).

## Arms
| arm | run_id | gate | init | trains |
|---|---|---|---|---|
| **frozen** (this doc) | `qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k` | single scalar | warm-start B, freeze VLM | adapter only |
| perhead | `qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k` | per-head | from-scratch | full finetune |

> The **perhead** from-scratch arm was SIGKILLed in the 2026-06-13 host-OOM event (h100b, 3 heavy
> runs stacked) before producing a usable ckpt; **not yet rerun** (deferred pending user). This doc
> covers the **frozen** arm, which completed 30k. Code/launcher/smoke for both arms committed
> (`feat(ffs): LLaMA-Adapter prefix injection #5`). Confound note: frozen↔perhead differ in TWO
> variables at once (gate granularity AND frozen-vs-finetune) — not a clean single-variable contrast;
> user accepted this.

## Results (frozen arm, 4-suite, our-render, right_view+primary eval)
| step | spatial | object | goal | libero_10 | mean |
|---|---|---|---|---|---|
| 20k | 0.92 | 0.95 | 0.85 | 0.71 | **0.858** |
| 30k | 0.91 | 0.96 | 0.95 | 0.68 | **0.875** |

20k→30k: goal jumped 0.85→0.95 and object 0.95→0.96, but libero_10 (long) slipped 0.71→0.68 and
spatial 0.92→0.91 — net +1.7pp, all within the suite's noise band.

## Conclusion
**#5 frozen 30k = 0.875 does NOT beat mono (0.913), and trails the best FFS variant
(depthtoken_keep 0.910) and plain stereo B (0.908).** It sits squarely in the 0.85–0.91 band that
every depth/stereo/geometry injection method occupies on this saturated suite. Consistent with the
project verdict: **on saturated LIBERO the second-view / depth signal is not measurable** — mono is
the ceiling, no injection scheme (VLM-side or action-side) clears it. Freezing the VLM and learning
only a LLaMA-Adapter prefix is no exception. To make any stereo/depth contribution measurable, move
off saturated LIBERO to a 3D-stressing generalization split (height/scale/occlusion), per the
stereo-4D proposal.

参见 [[project_experiment_registry]] [[project_stereo_consistent_4d_proposal]]
[[project_ffs_comparison_notes]]
