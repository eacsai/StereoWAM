> Historical roadmap (July 20). Its "planned" labels for E2/E3/E5 are stale:
> these experiments were later trained/evaluated. E6/E7 remain proposals.
> Current results and the new tri-view takeover plan are in
> [docs/TAKEOVER.md](../docs/TAKEOVER.md).

# Camera Branch E2-E5 roadmap

Project: /home/wangqw/ICLR27/starVLA
Branch: ffs_controlnet_dev
Created: 2026-07-20

## Baseline reference

Plain stereo baseline:
- input: primary + left_view
- CAM_BRANCH=0, CAM_ROPE=0
- effective batch size: 128
- max steps: 30000
- seed: 42
- reported mean success rate: 90.25%

The existing Camera Branch run reached 86.50% mean, but its config has is_resume=true and the run resumed after 10k steps. This is not a clean causal comparison.

## E2: bidirectional image-only Camera Branch

Status: planned, not launched yet.

Change only the Camera Branch attention mask:
- current: F.scaled_dot_product_attention(..., is_causal=True)
- E2: image-only branch is non-causal, so both stereo views can read each other
- keep the main Qwen language attention causal
- keep PRoPE geometry, six selected layers, 4 heads x 128 dim, data mix, seed, batch size, warmup, and 30k steps unchanged
- run continuously from scratch; do not resume from the previous Camera Branch 10k checkpoint

Primary hypothesis: the current causal image-only branch lets the later view attend to the earlier view but blocks the reverse direction, weakening stereo correspondence.

Required diagnostics:
- step-0 output parity against plain stereo
- nonzero branch output-projection and QKV gradients
- per-layer branch output norm
- fixed-subset evaluation, especially libero_10

Current resource note: GPUs 4-7 are occupied by the existing Utonia training; do not stop it automatically.

## E3: gated / small-nonzero Camera Branch initialization

Reminder: run after E2.

Candidate:
- keep the residual branch safe at initialization
- add per-layer scalar alpha
- compare zero output projection against small nonzero output initialization around 1e-3
- optionally warm-start from the plain stereo checkpoint for a short branch-only phase

Goal: remove delayed QKV learning caused by an exactly zero output projection without destabilizing the pretrained VLA path.

## E4: camera-token and calibration alignment

Reminder: run after E3.

Required checks:
- verify primary + left_view order against camera IDs and right_first
- swap images and camera IDs together; outputs should remain consistent
- swap images without IDs; outputs should change
- use sample-level K/viewmats when available
- update K after resize/crop/flip
- compare fixed LIBERO K/T against dynamic metadata

Goal: ensure the PRoPE branch uses the correct view semantics and geometry.

## E5: geometry-routed cross-view attention

Reminder: run after E4.

Candidate:
- add a small gated cross-view residual branch in middle/late layers
- restrict cross-view links using rectified same-row or K/T-derived epipolar top-k candidates
- optionally use FFS disparity/Utonia point-cloud features to route or gate correspondences
- keep self-attention and primary action path unchanged

Goal: move from camera-conditioned attention to explicit stereo correspondence and depth-aware information routing.

## E6: partial linear-state isolation with instruction as the readout sink

Status: planned, not launched yet.

Input order:
- `[Utonia][Primary-Image][Left-View][Instruction]`

Attention contract:
- full-attention layers: keep the non-causal Camera Branch for `Primary <-> Left-View`
- full-attention layers: block direct Utonia-to-image links; Instruction reads Utonia and both image views
- linear-attention layers: split Utonia into `S_U` and Primary/Left into a shared `S_STEREO`
- linear-attention layers: Instruction reads the merged Utonia/stereo states, while earlier input tokens do not read Instruction
- Primary/Left remain causal inside the linear layers; only the full-attention Camera Branch is non-causal

Primary hypothesis: isolating Utonia from image tokens removes harmful geometry/appearance interference while retaining the pretrained Qwen3.5 backbone and the existing Camera Branch.

Required diagnostics:
- role/state leakage test with sentinel tokens for Utonia, Primary, Left-View, and Instruction
- verify that Instruction hidden states receive all three input groups
- verify that Utonia hidden states do not receive Primary/Left-View signal
- compare P/L causal behavior in linear layers against the full-attention non-causal branch
- fixed-subset evaluation with the same setup as the plain stereo baseline

## E7: strict multi-state linear isolation with bidirectional stereo state

Status: planned, not launched yet.

Input order:
- `[Utonia][Primary-Image][Left-View][Instruction]`

Attention contract:
- full-attention layers: same non-causal `Primary <-> Left-View` Camera Branch as E6
- linear-attention layers: maintain a separate Utonia state `S_U`
- linear-attention layers: maintain bidirectional stereo states `S_STEREO_forward` and `S_STEREO_reverse`
- Primary and Left-View can read each other in both scan directions
- Utonia never reads Primary/Left-View; Primary/Left-View never reads Utonia or Instruction
- Instruction reads the Utonia and both stereo states; Instruction does not feed back to earlier input tokens

Primary hypothesis: strict state isolation plus bidirectional stereo information flow is a cleaner realization of the grouped-attention design, without introducing a standalone stereo encoder.

Implementation risk:
- requires a Qwen3.5 GatedDeltaNet state-routing change rather than only a padding/attention mask
- requires forward/reverse stereo scans or an equivalent bidirectional linear-attention implementation
- must preserve gradient-checkpointing, distributed training, and inference-cache behavior
- should reuse pretrained Q/K/V/gating weights and use a controlled warm-up; do not assume direct checkpoint continuation is behaviorally identical

Required diagnostics:
- exact token-role state-isolation unit tests
- forward/reverse P/L information-flow test
- instruction readout test for Utonia + Primary + Left-View
- memory/throughput comparison against E6
- fixed-subset evaluation with matched seed, batch size, steps, and checkpoint policy


## Ordered reminder

E2 -> E3 -> E4 -> E5 -> E6 -> E7

Do not interpret a result as a Camera Branch improvement unless training setup, checkpoint continuity, evaluation resolution, suite/task list, and seed are matched to the plain stereo baseline.

E6 and E7 are reminders for later experiments only. Do not launch either experiment automatically.
