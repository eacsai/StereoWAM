#!/usr/bin/env python
"""Four-GPU synthetic E3-vs-E5 smoke with memory/time gates."""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.stereo.cam_branch_attention import (
    CamBranchAttention,
    build_same_row_opposite_indices,
)


def _init_dist() -> tuple[int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank = int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", str(rank)))
        return rank, local
    return 0, 0


def _make_route(batch: int, height: int, width: int, device: torch.device):
    n_per = height * width
    cams = torch.cat(
        [torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)]
    ).repeat(batch, 1).to(device)
    rows_one = torch.arange(height, device=device, dtype=torch.long).repeat_interleave(width)
    rows = torch.cat([rows_one, rows_one]).repeat(batch, 1)
    return build_same_row_opposite_indices(cams, rows, topk=4)


def _run(model, x, eye, candidates, effective_k, steps: int):
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps + 2):
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        y = model.forward_composite(x, eye, eye, eye, candidates, effective_k)
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if len(times) >= 0:
            times.append(elapsed)
    return statistics.median(times[2:]), torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


class _SyntheticQwenFull(nn.Module):
    """Qwen-like QKV + SDPA + o_proj cost for a full-layer matched gate."""

    def __init__(self, hidden_dim: int = 1024, heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, self.head_dim).unbind(dim=2)
        q, k, v = (z.transpose(1, 2).contiguous() for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, s, -1))


def _run_full_stack(branches, full_layers, image_x, full_x, eye, candidates, effective_k, steps: int):
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps + 2):
        for module in list(branches) + list(full_layers):
            module.zero_grad(set_to_none=True)
        image_x.grad = None
        full_x.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = full_x
        for layer, branch in zip(full_layers, branches):
            x = layer(x)
            delta = branch.forward_composite(x[:, : image_x.shape[1]], eye, eye, eye, candidates, effective_k)
            x = torch.cat((x[:, : image_x.shape[1]] + delta, x[:, image_x.shape[1] :]), dim=1)
        x.float().square().mean().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return statistics.median(times[2:]), torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    rank, local = _init_dist()
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(20260721)
    B, H, W, D, heads, head_dim = 2, 8, 8, 1024, 4, 128
    N = 2 * H * W
    candidates, effective_k, row_width = _make_route(B, H, W, device)
    assert candidates.shape == (B, N, W) and effective_k == 4 and row_width == 8
    eye = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).expand(B, N, -1, -1)
    x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    full_x = torch.randn(B, 256, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    torch.manual_seed(42)
    e3 = CamBranchAttention(D, heads, head_dim, out_proj_init_std=0.02).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(42)
    e5 = CamBranchAttention(
        D, heads, head_dim, out_proj_init_std=0.02, e5_enabled=True,
        e5_topk=4, e5_gate_init=0.1, e5_out_proj_init_std=0.02, e5_init_seed=53,
    ).to(device=device, dtype=torch.bfloat16)
    e3_stack = nn.ModuleList([
        CamBranchAttention(D, heads, head_dim, out_proj_init_std=0.02).to(device=device, dtype=torch.bfloat16)
        for _ in range(6)
    ])
    e5_stack = nn.ModuleList([
        CamBranchAttention(
            D, heads, head_dim, out_proj_init_std=0.02,
            e5_enabled=(idx >= 2), e5_topk=4, e5_gate_init=0.1,
            e5_out_proj_init_std=0.02, e5_init_seed=53 + idx,
        ).to(device=device, dtype=torch.bfloat16)
        for idx in range(6)
    ])
    e3_full_layers = nn.ModuleList([_SyntheticQwenFull().to(device=device, dtype=torch.bfloat16) for _ in range(6)])
    e5_full_layers = nn.ModuleList([_SyntheticQwenFull().to(device=device, dtype=torch.bfloat16) for _ in range(6)])
    e3_ms, e3_alloc, e3_reserved = _run(e3, x, eye, None, None, args.steps)
    e5_ms, e5_alloc, e5_reserved = _run(e5, x, eye, candidates, effective_k, args.steps)
    e3_stack_ms, e3_stack_alloc, e3_stack_reserved = _run_full_stack(
        e3_stack, e3_full_layers, x, full_x, eye, None, None, args.steps
    )
    e5_stack_ms, e5_stack_alloc, e5_stack_reserved = _run_full_stack(
        e5_stack, e5_full_layers, x, full_x, eye, candidates, effective_k, args.steps
    )
    metrics = (rank, e3_ms, e5_ms, e3_alloc, e5_alloc, e3_reserved, e5_reserved)
    gathered = [None for _ in range(dist.get_world_size() if dist.is_initialized() else 1)]
    if dist.is_initialized():
        dist.all_gather_object(gathered, metrics)
    else:
        gathered = [metrics]
    stack_metrics = (rank, e3_stack_ms, e5_stack_ms, e3_stack_alloc, e5_stack_alloc, e3_stack_reserved, e5_stack_reserved)
    stack_gathered = [None for _ in range(dist.get_world_size() if dist.is_initialized() else 1)]
    if dist.is_initialized():
        dist.all_gather_object(stack_gathered, stack_metrics)
    else:
        stack_gathered = [stack_metrics]
    if rank == 0:
        e3_time = max(x[1] for x in gathered)
        e5_time = max(x[2] for x in gathered)
        max_alloc = max(x[4] for x in gathered)
        max_reserved = max(x[6] for x in gathered)
        stack_e3_time = max(x[1] for x in stack_gathered)
        stack_e5_time = max(x[2] for x in stack_gathered)
        stack_max_alloc = max(x[4] for x in stack_gathered)
        stack_max_reserved = max(x[6] for x in stack_gathered)
        print(f"branch-only per-rank={gathered}")
        print(f"full-stack per-rank={stack_gathered}")
        print(f"branch-only ratio={e5_time/max(e3_time,1e-9):.3f}; full-stack e3_max_ms={stack_e3_time*1000:.3f} e5_max_ms={stack_e5_time*1000:.3f} ratio={stack_e5_time/max(stack_e3_time,1e-9):.3f} max_alloc_gib={stack_max_alloc/2**30:.3f} max_reserved_gib={stack_max_reserved/2**30:.3f}")
        assert stack_max_alloc < 44 * 2**30 and stack_max_reserved < 44 * 2**30
        assert stack_e5_time <= 1.5 * max(stack_e3_time, 1e-9)
        print("PASS 4-GPU E5 synthetic memory/time gate")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
