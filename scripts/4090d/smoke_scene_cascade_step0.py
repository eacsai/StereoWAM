#!/usr/bin/env python3
"""Unit smoke for the scene-flow dual-DiT cascade NEW modules (no VLM load).

Verifies the load-bearing claims cheaply on dummy tensors:
  (1) STEP-0 IDENTITY: the DiT with the zero-init motion coupler ACTIVE (motion_hidden
      passed) produces BYTE-identical output to coupler-OFF (motion_hidden=None), in
      BOTH eval() and train() (RNG reset before each) -> zero_proj=0 + dropout=0.
  (2) SANITY: perturbing zero_proj makes the coupled output differ (coupler CAN act).
  (3) SceneFieldMatchingHead.extract_conditioning_hidden -> right shape, GT-free.
  (4) SceneFieldMatchingHead.forward -> finite scalar flow_loss.
Run in the starVLA .venv on a free GPU. READ-ONLY (builds tiny modules from scratch).
"""
import torch

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.flow_matching_head.scene_flow_head import SceneFieldMatchingHead

dev = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL.append(name)


# ---------------- (1)(2) coupler step-0 identity ----------------
heads, hd, num_layers = 4, 32, 4
inner = heads * hd  # 128
motion_dim, cross_dim = 96, 48
B, T, S, N = 2, 10, 6, 64
dit = DiT(
    num_attention_heads=heads, attention_head_dim=hd, output_dim=64, num_layers=num_layers,
    dropout=0.1, final_dropout=True, interleave_self_attention=True,
    cross_attention_dim=cross_dim, motion_coupler_dim=motion_dim,
    norm_type="ada_norm", positional_embeddings=None,
).to(dev)
n_couplers = len(dit.motion_coupler) if dit.motion_coupler is not None else 0
check("coupler built on even/cross-attn blocks only (2 of 4 layers)", n_couplers == 2)

hs = torch.randn(B, T, inner, device=dev)
enc = torch.randn(B, S, cross_dim, device=dev)
ts = torch.randint(0, 1000, (B,), device=dev)
motion = torch.randn(B, N, motion_dim, device=dev)


def run(mode, motion_hidden):
    getattr(dit, mode)()
    torch.manual_seed(1234)
    if dev == "cuda":
        torch.cuda.manual_seed_all(1234)
    with torch.no_grad():
        return dit(hs, enc, timestep=ts, motion_hidden=motion_hidden)


for mode in ("eval", "train"):
    out_off = run(mode, None)
    out_on = run(mode, motion)
    md = (out_off - out_on).abs().max().item()
    check(f"step-0 identity [{mode}]: coupler-on == coupler-off (maxabs={md:.3e})", md == 0.0)

# (2) sanity: perturb a coupler's zero_proj -> output must change
with torch.no_grad():
    list(dit.motion_coupler.values())[0].zero_proj.weight.add_(1.0)
out_pert = run("eval", motion)
out_off_eval = run("eval", None)
check("sanity: perturbed coupler changes output", (out_off_eval - out_pert).abs().max().item() > 0)

# ---------------- (3)(4) scene head ----------------
sub_cfg = {
    "action_model_type": "DiT-B",
    "grid_size": 8,
    "field_dim": 3,
    "tap_hidden_index": 1,
    "diffusion_model_cfg": {"num_layers": 2, "cross_attention_dim": cross_dim},
    "target_key": "flow_gt", "valid_key": "flow_valid", "dynamic_key": "flow_dynamic",
    "dynamic_fallback_to_valid": False,
}
head = SceneFieldMatchingHead(full_config=None, sub_cfg=sub_cfg).to(dev)
vl = torch.randn(B, S, cross_dim, device=dev)

mh = head.extract_conditioning_hidden(vl)
check(f"extract_conditioning_hidden shape (B,{8*8},D)={tuple(mh.shape)}", mh.shape[0] == B and mh.shape[1] == 64)
check("extract hidden is detached (v1)", not mh.requires_grad)

gt = {
    "flow_gt": torch.randn(B, 3, 16, 16, device=dev),
    "flow_valid": torch.ones(B, 16, 16, device=dev),
    "flow_dynamic": (torch.rand(B, 16, 16, device=dev) > 0.5).float(),
}
out = head(vl, gt)
fl = out["flow_loss"]
check(f"flow_loss finite scalar (={fl.item():.4f})", fl.dim() == 0 and torch.isfinite(fl).item())

print("STEP0_EQ_BASELINE=PASS" if not FAIL else f"SMOKE_FAILED: {FAIL}")
raise SystemExit(1 if FAIL else 0)
