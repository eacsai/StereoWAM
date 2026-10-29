#!/usr/bin/env python
"""Distributed pre-model GPU UUID/PCI assignment check for E5 launchers."""
from __future__ import annotations

import argparse
import os
import subprocess

import torch
import torch.distributed as dist


def _query(local_index: int) -> tuple[str, str]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(local_index),
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, pci = [part.strip() for part in raw.split(",", 1)]
    return uuid, pci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-uuids", required=True)
    parser.add_argument("--forbidden-uuids", required=True)
    args = parser.parse_args()
    expected = [x.strip() for x in args.expected_uuids.split(",") if x.strip()]
    forbidden = set(x.strip() for x in args.forbidden_uuids.split(",") if x.strip())
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4 or len(expected) != 4 or not 0 <= local_rank < 4:
        raise RuntimeError(f"invalid distributed preflight env rank={local_rank} world={world_size}")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected 4 visible GPUs, got {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    uuid, pci = _query(local_rank)
    if uuid != expected[local_rank]:
        raise RuntimeError(f"local rank {local_rank} mapped to {uuid}, expected {expected[local_rank]}")
    if uuid in forbidden:
        raise RuntimeError(f"local rank {local_rank} unexpectedly overlaps forbidden E3 UUID {uuid}")

    dist.init_process_group(backend="nccl")
    try:
        records = [None for _ in range(world_size)]
        dist.all_gather_object(records, (local_rank, uuid, pci))
        if sorted(records) != [(idx, expected[idx], records[idx][2]) for idx in range(4)]:
            raise RuntimeError(f"rank/UUID all-gather mismatch: {records}")
        if len({record[1] for record in records}) != 4:
            raise RuntimeError(f"duplicate GPU UUIDs across ranks: {records}")
        if dist.get_rank() == 0:
            print(f"PASS distributed GPU preflight: {records}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
