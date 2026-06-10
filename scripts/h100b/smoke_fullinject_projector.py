#!/usr/bin/env python
"""Smoke for FFSPerTokenProjector gate_init={zero,identity} (CPU, no GPU/FFS needed).

Validates the from-scratch full-injection change:
  - gate_init="zero"     -> step-0 residual is exactly 0 (warm-start parity, unchanged behaviour)
  - gate_init="identity" -> step-0 residual = spatial_proj(net[0]) at full strength, finite & sane
  - bad gate_init        -> raises ValueError

Run (CPU):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/mnt/data/wangqiwei/wangqiwei/starVLA \
    /opt/conda/envs/starvla/bin/python scripts/h100b/smoke_fullinject_projector.py
"""
import os
import sys

os.environ.setdefault("FFS_REPO_DIR", "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo")

import torch  # noqa: E402

from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import FFSPerTokenProjector  # noqa: E402

IN_CH, LLM_DIM, HIDDEN, H_TOK, W_TOK = 16, 1024, 256, 16, 16


def run() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, IN_CH, 64, 64)

    # zero-init: residual must be identically zero at step 0
    p_zero = FFSPerTokenProjector(IN_CH, LLM_DIM, HIDDEN, gate_init="zero")
    o_zero = p_zero(x, H_TOK, W_TOK)
    assert o_zero.shape == (H_TOK * W_TOK, LLM_DIM), f"bad shape {tuple(o_zero.shape)}"
    z_max = float(o_zero.abs().max())
    assert z_max == 0.0, f"zero-init must output 0, got max_abs={z_max}"

    # identity-init: residual = spatial_proj(x) (zero_proj is identity), non-zero, finite, sane scale
    p_id = FFSPerTokenProjector(IN_CH, LLM_DIM, HIDDEN, gate_init="identity")
    o_id = p_id(x, H_TOK, W_TOK)
    i_mean, i_max = float(o_id.abs().mean()), float(o_id.abs().max())
    assert torch.isfinite(o_id).all(), "identity-init output has non-finite values"
    assert i_max > 0.0, "identity-init must output non-zero residual"
    # identity gate == pass-through of spatial_proj: confirm zero_proj is exactly identity
    sp = p_id.spatial_proj(x.to(dtype=next(p_id.parameters()).dtype))
    sp = torch.nn.functional.interpolate(sp, size=(H_TOK, W_TOK), mode="bilinear", align_corners=False)
    sp = sp.flatten(2).transpose(1, 2).contiguous().squeeze(0)
    passthrough_err = float((o_id - sp).abs().max())
    assert passthrough_err < 1e-4, f"identity gate not pass-through: err={passthrough_err}"

    # bad value must raise
    try:
        FFSPerTokenProjector(IN_CH, LLM_DIM, HIDDEN, gate_init="full")
        raise AssertionError("bad gate_init should have raised ValueError")
    except ValueError:
        pass

    print(f"[smoke] zero-init  max_abs={z_max}")
    print(f"[smoke] identity   mean_abs={i_mean:.4f} max_abs={i_max:.4f} passthrough_err={passthrough_err:.2e}")
    # token embeddings are ~O(1); a step-0 residual mean_abs <~ 2 is sane (won't blow up from scratch)
    if i_mean > 2.0:
        print(f"[smoke] WARNING: identity-init residual mean_abs={i_mean:.3f} is large; watch early loss")
    print("PROJECTOR_SMOKE_OK")


if __name__ == "__main__":
    run()
    sys.exit(0)
