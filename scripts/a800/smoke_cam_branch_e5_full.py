#!/usr/bin/env python
"""CPU-first E5 route/parameter/checkpoint smoke; no dataset or model download."""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path

import torch

from starVLA.model.modules.stereo.cam_branch_attention import (
    CamBranchAttention,
    _build_stereo_grid_metadata,
    build_same_row_opposite_indices,
)


def _sha_state(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fixture(batch: int, height: int, width: int):
    n = 2 * height * width
    cams = torch.cat(
        [torch.zeros(height * width, dtype=torch.long), torch.ones(height * width, dtype=torch.long)]
    ).repeat(batch, 1)
    rows_one = torch.arange(height, dtype=torch.long).repeat_interleave(width)
    rows = torch.cat([rows_one, rows_one]).repeat(batch, 1)
    cols_one = torch.arange(width, dtype=torch.long).repeat(height)
    cols = torch.cat([cols_one, cols_one]).repeat(batch, 1)
    return cams, rows, cols, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cams, rows, cols, n = _fixture(2, 2, 8)
    candidates, effective_k, row_width = build_same_row_opposite_indices(cams, rows, topk=4)
    assert candidates.shape == (2, n, 8)
    assert effective_k == 4 and row_width == 8
    assert candidates[0, 0].tolist() == list(range(16, 24))
    assert candidates[0, 16].tolist() == list(range(0, 8))

    small_cams, small_rows, _, _ = _fixture(2, 2, 2)
    _, small_k, small_width = build_same_row_opposite_indices(small_cams, small_rows, topk=4)
    assert small_width == 2 and small_k == 2

    # Explicit record_index=2*b+run_slot camera binding oracle.
    image_positions = torch.arange(n, dtype=torch.long).view(1, n).expand(2, -1)
    per_token_cam_id = cams.clone()
    grid = torch.tensor([[1, 4, 16], [1, 4, 16], [1, 4, 16], [1, 4, 16]], dtype=torch.long)
    actual_cams, actual_rows, actual_cols = _build_stereo_grid_metadata(
        per_token_cam_id=per_token_cam_id,
        image_positions=image_positions,
        image_grid_thw=grid,
        spatial_merge_size=2,
        num_cameras=2,
    )
    assert torch.equal(actual_cams, cams)
    assert torch.equal(actual_rows, rows)
    assert torch.equal(actual_cols, cols)

    hidden_dim, heads, head_dim = 32, 2, 16
    eye = torch.eye(4).view(1, 1, 4, 4).expand(2, n, -1, -1)
    x = torch.randn(2, n, hidden_dim, requires_grad=True)
    torch.manual_seed(args.seed)
    e3 = CamBranchAttention(hidden_dim, heads, head_dim, out_proj_init_std=0.02)
    torch.manual_seed(args.seed)
    e3_e5 = CamBranchAttention(
        hidden_dim, heads, head_dim, out_proj_init_std=0.02,
        e5_enabled=True, e5_topk=4, e5_gate_init=0.1,
        e5_out_proj_init_std=0.02, e5_init_seed=args.seed + 11,
    )
    common = set(e3.state_dict()).intersection(e3_e5.state_dict())
    assert all(torch.equal(e3.state_dict()[k], e3_e5.state_dict()[k]) for k in common), "E3 init changed"
    assert all("e5" not in k for k in e3.state_dict())

    out = e3_e5.forward_composite(x, eye, eye, eye, candidates, effective_k)
    loss = out.float().square().mean()
    loss.backward()
    assert e3_e5.e5_branch is not None and e3_e5.e5_gate.grad is not None
    e5_params = [p for name, p in e3_e5.named_parameters() if "e5_" in name or name.startswith("e5_branch.")]
    assert e5_params, "E5 parameters were not registered"
    optimizer = torch.optim.AdamW(e3_e5.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in e5_params]
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, e5_params)), "E5 params did not update"
    assert len({id(p) for group in optimizer.param_groups for p in group["params"]}) == len(
        list(e3_e5.parameters())
    )

    buf = io.BytesIO()
    torch.save(e3_e5.state_dict(), buf)
    restored = CamBranchAttention(
        hidden_dim, heads, head_dim, out_proj_init_std=0.02,
        e5_enabled=True, e5_topk=4, e5_gate_init=0.1,
        e5_out_proj_init_std=0.02, e5_init_seed=args.seed + 11,
    )
    restored.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=True), strict=True)
    excepted = False
    try:
        e3_e5(x, eye, eye, eye)
    except RuntimeError:
        excepted = True
    assert excepted
    with torch.no_grad():
        got = e3_e5.forward_composite(x, eye, eye, eye, candidates, effective_k)
        expected = restored.forward_composite(x, eye, eye, eye, candidates, effective_k)
    assert torch.equal(got, expected), "checkpoint round-trip changed fixed output"
    assert _sha_state(e3_e5) == _sha_state(restored)
    print("PASS e5 route/metadata/gradient/optimizer/checkpoint/global-init smoke")


if __name__ == "__main__":
    main()
