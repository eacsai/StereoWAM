"""Print module breakdown + parameter counts for the current QwenPI framework
as configured in examples/LIBERO/train_files/starvla_cotrain_libero.yaml + the
CLI overrides used by run_libero_train_fast.sh."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/data/wangqiwei/ICLR2026/starVLA")
os.chdir("/data/wangqiwei/ICLR2026/starVLA")

import yaml
import torch
from omegaconf import OmegaConf

# Load yaml + apply the CLI overrides
cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
cfg.framework.name = "QwenPI"
cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/Qwen3.5-0.8B"
cfg.datasets.vla_data.data_root_dir = "playground/Datasets/LEROBOT_LIBERO_DATA"

print("=== Framework name:", cfg.framework.name)
print("=== Base VLM     :", cfg.framework.qwenvl.base_vlm)
print("=== Action model :", dict(cfg.framework.action_model))
print()

# Import the framework module dynamically (matches train_starvla.py logic)
from starVLA.model.framework.base_framework import build_framework  # type: ignore

print("[info] building framework (this loads VLM weights from disk, may take ~30 s)...")
model = build_framework(cfg)
print("[info] framework built.")
print()


def fmt(n: int) -> str:
    for u, d in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= d:
            return f"{n/d:.2f}{u}"
    return str(n)


def count_params(m: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


# Top-level modules
total, trainable = count_params(model)
print(f"=== Top-level total = {fmt(total)} ({total:,}), trainable = {fmt(trainable)} ({trainable:,})")
print()
print(f"{'Module':<60s} {'Total':>12s} {'Trainable':>12s}")
print("-" * 86)

# Direct children first
direct_children = list(model.named_children())
for name, child in direct_children:
    t, tr = count_params(child)
    print(f"  {name:<58s} {fmt(t):>12s} {fmt(tr):>12s}")

# Drill down one more level for big modules (>50M params)
print()
print("=== Drill-down (modules > 50M params) ===")
for top_name, top_child in direct_children:
    t_top, _ = count_params(top_child)
    if t_top < 50_000_000:
        continue
    print(f"\n[{top_name}]  ({fmt(t_top)})")
    for name, child in top_child.named_children():
        t, tr = count_params(child)
        if t > 0:
            print(f"  {name:<58s} {fmt(t):>12s} {fmt(tr):>12s}")
