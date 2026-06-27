#!/usr/bin/env python3
"""Run the Method #10A cached bf16/DeepSpeed two-step dry-run smoke.

This wrapper intentionally invokes the real h100b launcher. It does not build a
model directly, so checkpoint saving goes through train_starvla.py's
accelerator.get_state_dict path under the selected DeepSpeed config.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCHER = ROOT / "scripts/h100b/run_qwen0p8_groot_utonia_perpatch.sh"
LOSS_RE = re.compile(r"Step\s+\d+,\s+Loss:\s+(\{.*?\})\)")
AUDIT_RE = re.compile(r"METHOD10_CKPT_AUDIT_OK path=(\S+)")
CACHE_TELEMETRY_RE = re.compile(
    r"Utonia cache telemetry .*?hits=(\d+) row_misses=(\d+) "
    r"whole_batch_live_samples=(\d+) hit_rate=([0-9.]+)"
)


def _first_loss(output: str) -> float:
    for match in LOSS_RE.finditer(output):
        metrics = ast.literal_eval(match.group(1))
        if "action_dit_loss" in metrics:
            return float(metrics["action_dit_loss"])
    raise RuntimeError("could not find logged action_dit_loss; set LOGGING_FREQUENCY=1 in the launcher env")


def _assert_no_frozen_keys(path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint state_dict must be dict, got {type(state).__name__}: {path}")
    bad = [
        key for key in state.keys()
        if key.startswith(("utonia.", "ffs.")) or ".utonia." in key or ".ffs." in key
    ]
    if bad:
        raise RuntimeError(f"frozen Utonia/FFS keys leaked into checkpoint: {bad[:8]}")


def _assert_cache_hits_without_live_fallback(output: str) -> None:
    matches = list(CACHE_TELEMETRY_RE.finditer(output))
    if not matches:
        raise RuntimeError("cached run did not log Utonia cache telemetry")
    hits = max(int(match.group(1)) for match in matches)
    row_misses = max(int(match.group(2)) for match in matches)
    whole_batch_live = max(int(match.group(3)) for match in matches)
    if hits <= 0:
        raise RuntimeError("cached run reported zero Utonia cache hits")
    if row_misses != 0 or whole_batch_live != 0:
        raise RuntimeError(
            "cached run used live fallback: "
            f"max_row_misses={row_misses} max_whole_batch_live_samples={whole_batch_live}"
        )
    forbidden = (
        "falling back to live deterministic compute",
        "serving whole batch live",
    )
    for needle in forbidden:
        if needle in output:
            raise RuntimeError(f"cached run logged live fallback: {needle}")


def _run_launcher(args: argparse.Namespace, *, run_id: str, cache_dir: str | None) -> str:
    env = os.environ.copy()
    env.update(
        {
            "METHOD10_DRYRUN": "1",
            "MAX_STEPS": "2",
            "SAVE_INTERVAL": "1",
            "LOGGING_FREQUENCY": "1",
            "BS": str(args.batch_size),
            "NUM_PROCESSES": str(args.num_processes),
            "GPUS": args.gpus,
            "DS_CONFIG": args.ds_config,
            "DATA_MIX": args.data_mix,
            "RUN_ID": run_id,
            "RESUME": "0",
            "WANDB_MODE": "disabled",
            "SKIP_UTONIA_R5_BENCH": "1",
            "METHOD10_CKPT_AUDIT_POLL_SEC": "2",
            "METHOD10_CKPT_AUDIT_STABILITY_SEC": "1",
            "METHOD10_DRYRUN_SEED_BEFORE_BUILD": "1",
        }
    )
    if cache_dir:
        env["UTONIA_CACHE_DIR"] = cache_dir
    else:
        env.pop("UTONIA_CACHE_DIR", None)

    proc = subprocess.run(
        ["bash", str(args.launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )
    output = proc.stdout
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"{run_id}.log").write_text(output)
    if proc.returncode != 0:
        raise RuntimeError(f"launcher failed run_id={run_id} status={proc.returncode}; log={args.log_dir / (run_id + '.log')}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.environ.get("UTONIA_CACHE_DIR"))
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ds-config", default="starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml")
    parser.add_argument("--data-mix", default=os.environ.get("DATA_MIX", "libero_all_sfstereo_leftprimary"))
    parser.add_argument("--loss-atol", type=float, default=1e-5)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "tmp/utonia_cache_train_dryrun_smoke")
    parser.add_argument("--run-prefix", default=f"method10_dryrun_{int(time.time())}")
    args = parser.parse_args()

    if not args.cache_dir:
        print("UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=missing_cache_dir")
        return 2
    cache_path = Path(args.cache_dir).expanduser()
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache_dir = str(cache_path)
    if not cache_path.is_dir():
        print(f"UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=cache_dir_not_found path={cache_dir}")
        return 3

    baseline_id = f"{args.run_prefix}_baseline"
    cached_id = f"{args.run_prefix}_cached"
    baseline_out = _run_launcher(args, run_id=baseline_id, cache_dir=None)
    cached_out = _run_launcher(args, run_id=cached_id, cache_dir=cache_dir)
    try:
        _assert_cache_hits_without_live_fallback(cached_out)
    except RuntimeError as exc:
        print(f"UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=cache_hit_telemetry error={exc}")
        return 7

    baseline_loss = _first_loss(baseline_out)
    cached_loss = _first_loss(cached_out)
    allowed = float(args.loss_atol) + float(args.loss_rtol) * abs(baseline_loss)
    diff = abs(cached_loss - baseline_loss)
    print(
        "UTONIA_CACHE_TRAIN_DRYRUN_LOSS "
        f"baseline={baseline_loss:.9f} cached={cached_loss:.9f} diff={diff:.9f} allowed={allowed:.9f}"
    )
    if diff > allowed:
        print("UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=step0_loss_mismatch")
        return 4

    audit_match = AUDIT_RE.search(cached_out)
    if not audit_match:
        print("UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=missing_checkpoint_audit_ok")
        return 5
    _assert_no_frozen_keys(Path(audit_match.group(1)))
    if "METHOD10_CKPT_AUDIT_FAIL" in cached_out:
        print("UTONIA_CACHE_TRAIN_DRYRUN_FAIL reason=checkpoint_audit_fail")
        return 6
    print("UTONIA_CACHE_TRAIN_DRYRUN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
