#!/usr/bin/env python3
"""Fail-closed preflight for the matched visual-attention arms."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _parse_exact_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected exact true/false, got {raw!r}")


def _parse_gpus(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values or any(not part.isdigit() for part in values):
        raise ValueError(f"GPUS must be a comma-separated list of integers, got {raw!r}")
    gpus = [int(part) for part in values]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPUS contains duplicates: {raw!r}")
    if gpus != sorted(gpus):
        raise ValueError(f"GPUS must be in strictly ascending order: {raw!r}")
    return gpus


def _fuser_pids(gpu: int) -> list[int]:
    device = Path(f"/dev/nvidia{gpu}")
    if not device.exists():
        raise RuntimeError(f"target GPU device does not exist: {device}")
    result = subprocess.run(
        ["fuser", "-v", str(device)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(
            f"fuser failed for {device}: exit={result.returncode} stderr={result.stderr.strip()!r}"
        )
    pids: set[int] = set()
    for token in f"{result.stdout}\n{result.stderr}".split():
        if token.isdigit():
            pid = int(token)
            if pid > 0:
                pids.add(pid)
    if result.returncode == 0 and not pids:
        raise RuntimeError(f"fuser reported an occupied {device} but returned no PID")
    return sorted(pids)


def _pid_owner(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-o", "user=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "<unknown>"


def check_gpu_ownership(gpus: list[int]) -> None:
    current_pid = os.getpid()
    occupied: list[str] = []
    for gpu in gpus:
        for pid in _fuser_pids(gpu):
            if pid == current_pid:
                continue
            occupied.append(f"GPU{gpu}:pid={pid}:user={_pid_owner(pid)}")
    if occupied:
        raise RuntimeError(
            "target GPUs are occupied; refusing to kill/reset or race another task: "
            + ", ".join(occupied)
        )


def validate(args: argparse.Namespace) -> dict[str, object]:
    gpus = _parse_gpus(args.gpus)
    if args.mode not in {"causal", "image_bidir"}:
        raise ValueError(f"new arms require mode=causal or image_bidir, got {args.mode!r}")
    expected_branch = args.mode == "causal"
    if args.branch_causal is not expected_branch:
        raise ValueError(
            f"mismatched pair: mode={args.mode} requires branch_causal={str(expected_branch).lower()}"
        )
    if args.attention_implementation != "sdpa":
        raise ValueError("matched visual arms require attention_implementation=sdpa")
    if args.per_device_batch != args.expected_per_device_batch:
        raise ValueError(
            f"per_device_batch={args.per_device_batch} must equal expected "
            f"{args.expected_per_device_batch}"
        )
    if args.grad_accum != args.expected_grad_accum:
        raise ValueError(f"grad_accum={args.grad_accum} must equal expected {args.expected_grad_accum}")
    if args.world_size != args.expected_world_size:
        raise ValueError(f"world_size={args.world_size} must equal expected {args.expected_world_size}")
    effective_batch = args.per_device_batch * len(gpus) * args.grad_accum
    if effective_batch != args.expected_effective_batch:
        raise ValueError(
            f"effective batch mismatch: {args.per_device_batch}*{len(gpus)}*{args.grad_accum}="
            f"{effective_batch}, expected {args.expected_effective_batch}"
        )
    if args.world_size != len(gpus):
        raise ValueError(f"world_size={args.world_size} must equal number of GPUs={len(gpus)}")
    if args.check_gpus:
        check_gpu_ownership(gpus)
    return {
        "gpus": gpus,
        "mode": args.mode,
        "branch_causal": args.branch_causal,
        "attention_implementation": args.attention_implementation,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "world_size": args.world_size,
        "effective_batch": effective_batch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--branch-causal", type=_parse_exact_bool, required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--per-device-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--expected-effective-batch", type=int, default=128)
    parser.add_argument("--expected-per-device-batch", type=int, default=8)
    parser.add_argument("--expected-grad-accum", type=int, default=4)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--check-gpus", action="store_true")
    args = parser.parse_args()
    summary = validate(args)
    print("PASS visual-attention preflight")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
