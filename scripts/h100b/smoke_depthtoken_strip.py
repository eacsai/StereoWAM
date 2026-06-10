#!/usr/bin/env python3
"""Smoke: keep vs strip equivalence for the depth-token action-head-access ablation.

Proves DYNAMICALLY (no literal seq lengths), on the SAME model + input + weights:
  - keep  : encode seq len == original_len + num_depth_tokens  (depth rows reach the action head)
  - strip : encode seq len == original_len  AND  strip_hidden == keep_hidden[keep_mask]
            (strip removes EXACTLY the depth rows, nothing else; gather mirrors QwenPI)
  - strip with no hook run (keep_mask is None) -> raises RuntimeError

Run on h100b (needs Qwen3.5-0.8B + FFS weights present). Uses 64 depth tokens (pool 8x8),
matching the new keep/strip runs. Weights are random-init (structural test) -> the asserts are
weight-independent.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/h100b/smoke_depthtoken_strip.py
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.depth_token_inject import (  # noqa: E402
    clear_state as clear_depth_state,
    get_state as get_depth_state,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402

from scripts.h100b.smoke_groot_ffs import (  # noqa: E402
    DEFAULT_BASE_VLM,
    DEFAULT_DATA_MIX,
    DEFAULT_DATA_ROOT,
    DEFAULT_FFS_MODEL,
    DEFAULT_FFS_REPO_DIR,
    DEFAULT_FFS_SHA256,
    DEFAULT_PRETRAINED_CKPT,
    batch_images,
    build_framework_model,
    install_ffs_repo,
    instructions,
    load_ckpt_config,
    make_examples,
    move_model,
    run_check,
    update_cfg,
)
from scripts.h100b.smoke_groot_depthtoken import FRAMEWORK, build_depth_cfg, ffs_depth_cfg  # noqa: E402


def build_strip_model(args: argparse.Namespace, ckpt_cfg, device: torch.device):
    # cam_rope default OFF here: the #4 relaunch trains with CAM_ROPE=0 (inert-cam_rope
    # bypass), so the gate must smoke the configuration that actually gets launched.
    cfg = build_depth_cfg(
        args,
        ckpt_cfg,
        pretrained_ckpt="",
        freeze_modules="",
        cam_rope_enabled=bool(args.cam_rope_enabled),
    )
    depth_cfg = ffs_depth_cfg(args)
    depth_cfg["num_depth_tokens"] = args.num_depth_tokens
    depth_cfg["pool_hw"] = args.pool_hw
    depth_cfg["strip_depth_tokens"] = False  # flipped at runtime per check
    update_cfg(cfg, "framework.ffs_depth_token", depth_cfg)
    model = build_framework_model(cfg)
    model = TrainerUtils.freeze_backbones(model, freeze_modules="")
    model = move_model(model, device)
    model.eval()
    return model


def check_keep_vs_strip(model, examples, args) -> str:
    imgs, instr = batch_images(examples), instructions(examples)
    qwen_inputs = model._build_depthtoken_qwenvl_inputs(imgs, instr)
    orig_len = int(qwen_inputs["input_ids"].shape[1])
    n_depth = int(model.num_depth_tokens)

    # --- keep arm (default GR00T behavior): depth rows kept ---
    model.strip_depth_tokens = False
    clear_depth_state()
    with torch.inference_mode():
        keep_hidden = model._encode_last_hidden_with_ffs(imgs, instr)
    keep_mask = get_depth_state().keep_mask
    if keep_mask is None:
        raise AssertionError("keep run did not populate keep_mask")
    keep_mask = keep_mask.clone()
    if int(keep_hidden.shape[1]) != orig_len + n_depth:
        raise AssertionError(
            f"keep seq len {int(keep_hidden.shape[1])} != original {orig_len} + depth {n_depth}"
        )

    # --- strip arm: depth rows removed before the action head ---
    model.strip_depth_tokens = True
    clear_depth_state()
    with torch.inference_mode():
        strip_hidden = model._encode_last_hidden_with_ffs(imgs, instr)
    if int(strip_hidden.shape[1]) != orig_len:
        raise AssertionError(f"strip seq len {int(strip_hidden.shape[1])} != original {orig_len}")

    # strip output must equal the kept rows of the keep output (same deterministic VLM forward)
    bsz, hdim = int(keep_hidden.shape[0]), int(keep_hidden.shape[-1])
    expected = keep_hidden[keep_mask].view(bsz, orig_len, hdim)
    max_diff = float((strip_hidden.float() - expected.float()).abs().max().cpu())
    if max_diff > args.strip_atol:
        raise AssertionError(
            f"strip_hidden != keep_hidden[keep_mask]: max|diff|={max_diff:.3g} > atol {args.strip_atol} "
            "(strip removed the wrong rows or perturbed the kept ones)"
        )
    # reset to default for any later checks
    model.strip_depth_tokens = False
    return (
        f"orig={orig_len}, keep={orig_len + n_depth}, strip={orig_len}, "
        f"depth_tokens={n_depth}, max|keep[mask]-strip|={max_diff:.3g}"
    )


def check_strip_requires_hook(model, args) -> str:
    clear_depth_state()  # no forward ran -> keep_mask is None
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 4, 8, device=device)
    try:
        model._strip_depth_tokens(dummy)
    except RuntimeError as exc:
        return f"raised as expected: {str(exc)[:70]}"
    raise AssertionError("_strip_depth_tokens did not raise when keep_mask is None")


def check_train_step_timing(model, examples, args) -> str:
    if args.timing_steps <= 0:
        return "skipped (--timing-steps <= 0)"
    device = next(model.parameters()).device
    if device.type != "cuda":
        return "skipped (CUDA timing gate only)"

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise AssertionError("no trainable parameters for timing gate")

    model.train()
    model.strip_depth_tokens = False
    optimizer = torch.optim.AdamW(params, lr=args.timing_lr)
    durations = []
    total_steps = int(args.timing_warmup_steps + args.timing_steps)
    try:
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            clear_depth_state()
            torch.manual_seed(args.timing_seed + step)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(examples=examples)
            loss = out["action_loss"]
            if not torch.isfinite(loss.detach()).all():
                raise AssertionError(f"step {step}: non-finite loss")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.timing_warmup_steps:
                durations.append(elapsed)
    finally:
        model.eval()

    if not durations:
        raise AssertionError("timing gate produced no measured steps")
    baseline = min(durations)
    threshold = max(float(args.timing_min_step_sec), baseline * float(args.timing_max_slowdown))
    slow = [duration for duration in durations if duration > threshold]
    if slow:
        raise AssertionError(
            "depth-token train-step timing spike: "
            f"durations={[round(d, 3) for d in durations]}, "
            f"baseline={baseline:.3f}s, threshold={threshold:.3f}s"
        )
    return (
        f"timed_steps={len(durations)}, durations={[round(d, 3) for d in durations]}, "
        f"baseline={baseline:.3f}s, threshold={threshold:.3f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--ffs-model-path", default=DEFAULT_FFS_MODEL)
    parser.add_argument("--ffs-repo-dir", default=DEFAULT_FFS_REPO_DIR)
    parser.add_argument("--ffs-expected-sha256", default=DEFAULT_FFS_SHA256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ffs-image-size", type=int, default=256)
    parser.add_argument("--num-depth-tokens", type=int, default=64)
    parser.add_argument("--pool-hw", type=int, default=8)
    parser.add_argument("--cam-rope-enabled", type=int, choices=(0, 1), default=0)
    parser.add_argument("--strip-atol", type=float, default=1e-3)
    parser.add_argument("--timing-steps", type=int, default=3)
    parser.add_argument("--timing-warmup-steps", type=int, default=1)
    parser.add_argument("--timing-lr", type=float, default=1e-5)
    parser.add_argument("--timing-seed", type=int, default=456)
    parser.add_argument("--timing-max-slowdown", type=float, default=8.0)
    parser.add_argument("--timing-min-step-sec", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_ffs_repo(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    print(f"[setup] config={cfg_path}")
    print(
        f"[setup] framework={FRAMEWORK} num_depth_tokens={args.num_depth_tokens} "
        f"pool_hw={args.pool_hw} cam_rope_enabled={bool(args.cam_rope_enabled)}"
    )

    ok = True
    try:
        model = build_strip_model(args, ckpt_cfg, device)
        examples = make_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
        ok = run_check(FRAMEWORK, "keep_vs_strip_equivalence", lambda: check_keep_vs_strip(model, examples, args)) and ok
        ok = run_check(FRAMEWORK, "strip_requires_hook", lambda: check_strip_requires_hook(model, args)) and ok
        ok = run_check(FRAMEWORK, "train_step_timing_gate", lambda: check_train_step_timing(model, examples, args)) and ok
    except Exception as exc:  # noqa: BLE001
        print(f"[{FRAMEWORK}] FAIL fatal: {exc}")
        traceback.print_exc()
        ok = False
    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
