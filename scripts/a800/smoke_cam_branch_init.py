"""Focused CPU checks for the Camera Branch E3 small-init change."""
from __future__ import annotations

import torch

from starVLA.model.modules.stereo.cam_branch_attention import CamBranchAttention


def main() -> None:
    kwargs = dict(hidden_dim=16, branch_heads=1, branch_head_dim=4)

    torch.manual_seed(20260721)
    default = CamBranchAttention(**kwargs)
    default_rng = torch.random.get_rng_state()

    torch.manual_seed(20260721)
    small = CamBranchAttention(**kwargs, out_proj_init_std=1e-3)
    small_rng = torch.random.get_rng_state()

    assert torch.count_nonzero(default.out_proj.weight) == 0
    assert torch.count_nonzero(default.out_proj.bias) == 0
    assert torch.count_nonzero(small.out_proj.weight) > 0
    assert torch.count_nonzero(small.out_proj.bias) == 0
    observed_std = float(small.out_proj.weight.detach().std())
    assert 5e-4 < observed_std < 1.5e-3, observed_std
    assert torch.equal(default_rng, small_rng), "small-init must preserve the global RNG stream"
    for name in ("q_proj", "k_proj", "v_proj"):
        assert torch.equal(getattr(default, name).weight, getattr(small, name).weight), name

    eye = torch.eye(4).reshape(1, 1, 4, 4).expand(2, 2, 4, 4).contiguous()
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = small(x, eye, eye, eye)
    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    x_a = torch.zeros(1, 4, 16)
    x_b = x_a.clone()
    x_b[:, 3, 0] = 1.0
    y_a = small(x_a, eye[:1], eye[:1], eye[:1])
    y_b = small(x_b, eye[:1], eye[:1], eye[:1])
    assert not torch.allclose(
        y_a[:, 0], y_b[:, 0]
    ), "bidirectional branch did not read a later image token"
    print(f"cam_branch E3 checks passed; out_proj_std={observed_std:.6g}")


if __name__ == "__main__":
    main()
