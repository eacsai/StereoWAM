#!/usr/bin/env python3
"""Standalone rollout-FIFO smoke: build the past-flow model, call predict_action() a few times
(simulating action-chunks within one episode), and assert the FIFO mechanics + sample_field +
past-flow-fed extraction all run with NO crash under bf16 (mimics eval --use_bf16). Then check
reset_past_flow() clears the FIFO. Read-only w.r.t. training. Run ON 4090d."""
import sys, numpy as np, torch
from omegaconf import OmegaConf

CFG = "playground/Checkpoints/gpusmoke_pastflow_cambranch/config.full.yaml"
cfg = OmegaConf.load(CFG)
# point base VLM at the local pretrained dir (config may have a training-time path)
try:
    from starVLA.model.framework.base_framework import build_framework
except Exception:
    from starVLA.model.framework.share_tools import build_framework  # fallback

print("[smoke] building framework (fresh weights; code-path test) ...", flush=True)
model = build_framework(cfg)
model = model.to("cuda").to(torch.bfloat16)   # mimic eval --use_bf16
model.eval()
print("[smoke] scene_predictor_enabled=", getattr(model, "scene_predictor_enabled", None),
      "past_flow_enabled=", getattr(getattr(model, "scene_predictor", None), "past_flow_enabled", None),
      "K=", getattr(model, "_pf_K", None), flush=True)

# dummy example in the format predict_action expects: image = LIST of view PILs (stereo
# leftprimary = 2 views), lang str, state vec
from PIL import Image
H = 256
def mk_example():
    views = [Image.fromarray(np.random.randint(0, 255, (H, H, 3), dtype=np.uint8)) for _ in range(2)]
    return {
        "image": views,
        "lang": "pick up the object and place it",
        "state": np.zeros((1, 7), dtype=np.float32),
    }

print("[smoke] calling predict_action x3 (3 chunks, one episode) ...", flush=True)
for i in range(3):
    out = model.predict_action([mk_example()])
    fifo = getattr(model, "_pf_fifo", None)
    na = out["normalized_actions"]
    print(f"[smoke] chunk {i}: actions_shape={np.asarray(na).shape} fifo_len={len(fifo)} "
          f"fifo0_shape={(tuple(fifo[0].shape) if fifo else None)} fifo0_dtype={(fifo[0].dtype if fifo else None)}",
          flush=True)

assert len(model._pf_fifo) == model._pf_K, f"FIFO should saturate at K={model._pf_K}, got {len(model._pf_fifo)}"
assert tuple(model._pf_fifo[0].shape)[-2:] == (16, 16), f"FIFO field grid should be 16x16, got {model._pf_fifo[0].shape}"
model.reset_past_flow()
assert len(model._pf_fifo) == 0, "reset_past_flow did not clear the FIFO"
print("[smoke] FIFO_SMOKE_PASS: predict_action ran 3x, FIFO saturated at K then reset cleanly, no crash", flush=True)
