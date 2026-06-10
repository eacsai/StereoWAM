#!/usr/bin/env python3
"""Smoke: disabling the inert stereo cam_rope is output-equivalent (and faster).

Premise being verified ON GPU (not just statically): cam_rope's q_cam_proj AND
k_cam_proj are both zero-initialized (cam_rope.py), so their bilinear attention
contribution AND their gradients are identically zero forever -> the patch is a
no-op that only costs speed (head_dim 256->272 kills FlashAttention2). Therefore
training the #4 arms with stereo_cam_rope_enabled=false (CAM_ROPE=0) must produce
the same outputs as the current slow runs, modulo attention-backend bf16 noise.

Checks (QwenGR00T_DepthTokenFFS, 64 depth tokens / pool 8x8 / keep, matching the
relaunch config; weights random-init then TRANSPLANTED so all three models share
identical shared weights — seeded-identical init is impossible because the ON
config consumes extra RNG draws creating the cam projections):
  1. premise_cam_params_zero : every q_cam_proj/k_cam_proj weight is exactly 0
  2. same_kernel_equivalence : ON vs OFF, BOTH forced onto the SDPA-MATH kernel
     -> last_hidden allclose atol=2e-3 (repo's bf16 trunk-parity precedent) and
     action_loss relative diff < 5e-3. This is the mathematical-equivalence claim.
  3. flash_backend_parity    : OFF-flash (production FA2) vs OFF-sdpa-math, same
     weights -> loose scale-invariant gate (hidden rel RMS diff < 0.25; measured
     pure backend noise on random-init weights is ~0.1, a real semantics bug gives
     ~O(1); loss rel diff < 2e-2; absolute max|diff| printed but not gated).
     Pure known backend noise; NOT the headline claim.
  4. inert_grads_backward    : forward+backward on ON and OFF (same kernel, same
     seed): ON's cam projection grads are EXACTLY zero (re-proves inertness on a
     real backward); shared-parameter grad cosine per module group > 0.999; and
     OFF-flash forward+backward produces a finite loss (de-risks the production
     backward path). Production batch size itself needs no de-risk here: the keep
     arm already ran 5k real steps at BS=32, and CAM_ROPE=0 does not touch FFS.

Run on h100b (needs Qwen3.5-0.8B + FFS weights + the B-run config dir):
  CUDA_VISIBLE_DEVICES=0 python scripts/h100b/smoke_camrope_disable_equivalence.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.depth_token_inject import (  # noqa: E402
    clear_state as clear_depth_state,
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

# Both key families under which the SAME StereoCamRoPELayer modules are registered:
# as the framework-level ModuleList and as a child of each patched attention.
CAM_KEY_MARKERS = ("stereo_cam_rope_layers.", ".stereo_cam_layer.")


def is_cam_key(name: str) -> bool:
    return name.startswith(CAM_KEY_MARKERS[0]) or CAM_KEY_MARKERS[1] in name


def build_model(args, ckpt_cfg, device, *, cam_rope_enabled: bool, attn_impl: str):
    cfg = build_depth_cfg(
        args,
        ckpt_cfg,
        pretrained_ckpt="",
        freeze_modules="",
        cam_rope_enabled=cam_rope_enabled,
    )
    update_cfg(cfg, "framework.qwenvl.attn_implementation", attn_impl)
    depth_cfg = ffs_depth_cfg(args)
    depth_cfg["num_depth_tokens"] = args.num_depth_tokens
    depth_cfg["pool_hw"] = args.pool_hw
    depth_cfg["strip_depth_tokens"] = False
    update_cfg(cfg, "framework.ffs_depth_token", depth_cfg)
    model = build_framework_model(cfg)
    model = TrainerUtils.freeze_backbones(model, freeze_modules="")
    model = move_model(model, device)
    model.eval()
    return model


def transplant(model_on, model_off) -> int:
    """Load ON weights minus the cam projections into OFF. strict=True proves the
    remaining key sets are identical (no silent drift between the two configs)."""
    src = model_on.state_dict()
    filtered = {k: v for k, v in src.items() if not is_cam_key(k)}
    dropped = len(src) - len(filtered)
    if dropped == 0:
        raise AssertionError("ON state_dict contains no cam_rope keys — premise broken")
    model_off.load_state_dict(filtered, strict=True)
    return dropped


def cam_named_params(model_on):
    params = [(n, p) for n, p in model_on.named_parameters() if is_cam_key(n)]
    if not params:
        raise AssertionError("ON model has no cam projection parameters")
    return params


def forward_hidden(model, examples, seed: int):
    imgs, instr = batch_images(examples), instructions(examples)
    clear_depth_state()
    torch.manual_seed(seed)
    with torch.inference_mode():
        return model._encode_last_hidden_with_ffs(imgs, instr)


def forward_loss(model, examples, seed: int, *, backward: bool):
    clear_depth_state()
    torch.manual_seed(seed)
    if backward:
        model.zero_grad(set_to_none=True)
        out = model(examples=examples)
        loss = out["action_loss"]
        loss.backward()
        return loss.detach()
    with torch.inference_mode():
        out = model(examples=examples)
        return out["action_loss"].detach()


def grouped_grads(model) -> dict[str, torch.Tensor]:
    """Flatten grads per top-level module group, skipping cam keys and grad-less
    params (frozen FFS). Name-keyed so ON/OFF group contents align exactly."""
    groups: dict[str, list[torch.Tensor]] = {}
    for name, param in sorted(model.named_parameters()):
        if is_cam_key(name) or param.grad is None:
            continue
        groups.setdefault(name.split(".")[0], []).append(param.grad.detach().float().flatten())
    return {g: torch.cat(ts) for g, ts in groups.items()}


def check_premise_cam_params_zero(model_on) -> str:
    n = 0
    for name, param in cam_named_params(model_on):
        if not torch.all(param.detach() == 0):
            raise AssertionError(f"{name} is not all-zero — inertness premise broken")
        n += 1
    return f"{n} cam projection tensors verified exactly zero"


def check_same_kernel_equivalence(model_on, model_off_sdpa, examples, args) -> str:
    with sdpa_kernel([SDPBackend.MATH]):
        hidden_on = forward_hidden(model_on, examples, args.seed)
        hidden_off = forward_hidden(model_off_sdpa, examples, args.seed)
        loss_on = forward_loss(model_on, examples, args.seed, backward=False)
        loss_off = forward_loss(model_off_sdpa, examples, args.seed, backward=False)
    if hidden_on.shape != hidden_off.shape:
        raise AssertionError(f"hidden shape mismatch: {hidden_on.shape} vs {hidden_off.shape}")
    max_diff = float((hidden_on.float() - hidden_off.float()).abs().max().cpu())
    if max_diff > args.tight_atol:
        raise AssertionError(
            f"same-kernel last_hidden max|ON-OFF|={max_diff:.3g} > atol {args.tight_atol} "
            "— disabling cam_rope is NOT output-equivalent, do not relaunch"
        )
    rel = float((loss_on - loss_off).abs() / loss_on.abs().clamp_min(1e-8))
    if rel > args.tight_loss_rtol:
        raise AssertionError(
            f"same-kernel loss rel diff {rel:.3g} > {args.tight_loss_rtol} "
            f"(ON={float(loss_on):.6f} OFF={float(loss_off):.6f})"
        )
    return f"hidden max|diff|={max_diff:.3g} (atol {args.tight_atol}), loss rel diff={rel:.3g}"


def check_flash_backend_parity(model_off_sdpa, model_off_flash, examples, args) -> str:
    # Known bf16 backend noise, NOT the equivalence claim (that's the same-kernel
    # check). Gate on SCALE-INVARIANT metrics only: random-init hiddens have large
    # magnitudes, so an absolute element-wise max-diff gate flakes meaninglessly
    # (observed: max|diff| 12.7 while loss rel diff was 0.145% and grad cosine
    # >0.999). Absolute max-diff is printed for reference, not gated.
    with sdpa_kernel([SDPBackend.MATH]):
        hidden_math = forward_hidden(model_off_sdpa, examples, args.seed)
        loss_math = forward_loss(model_off_sdpa, examples, args.seed, backward=False)
    hidden_flash = forward_hidden(model_off_flash, examples, args.seed)
    loss_flash = forward_loss(model_off_flash, examples, args.seed, backward=False)
    diff = hidden_math.float() - hidden_flash.float()
    max_diff = float(diff.abs().max().cpu())
    rel_rms = float(diff.norm() / hidden_math.float().norm().clamp_min(1e-8))
    rel = float((loss_math - loss_flash).abs() / loss_math.abs().clamp_min(1e-8))
    if rel_rms > args.loose_hidden_rel_rms or rel > args.loose_loss_rtol:
        raise AssertionError(
            f"flash-vs-math backend gap unexpectedly large: hidden rel RMS diff={rel_rms:.3g} "
            f"(gate {args.loose_hidden_rel_rms}), loss rel diff={rel:.3g} (gate {args.loose_loss_rtol})"
        )
    return (
        f"hidden rel RMS diff={rel_rms:.3g}, loss rel diff={rel:.3g}, "
        f"max|diff|={max_diff:.3g} (informational; bf16 backend noise)"
    )


def check_inert_grads_backward(model_on, model_off_sdpa, model_off_flash, examples, args) -> str:
    with sdpa_kernel([SDPBackend.MATH]):
        loss_on = forward_loss(model_on, examples, args.seed, backward=True)
        loss_off = forward_loss(model_off_sdpa, examples, args.seed, backward=True)
    for name, param in cam_named_params(model_on):
        grad = param.grad
        if grad is None:
            # None would mean the cam branch never participated in the forward at
            # all — the inertness claim would then be vacuous, not proven.
            raise AssertionError(f"{name} has grad=None — ON model did not exercise the cam branch")
        if not torch.all(grad == 0):
            raise AssertionError(
                f"{name} got a NON-zero gradient — cam_rope is not inert, bypass is unsafe"
            )
    grads_on, grads_off = grouped_grads(model_on), grouped_grads(model_off_sdpa)
    if set(grads_on) != set(grads_off):
        raise AssertionError(f"grad group mismatch: {sorted(grads_on)} vs {sorted(grads_off)}")
    cosines = {}
    for group in sorted(grads_on):
        a, b = grads_on[group], grads_off[group]
        if a.numel() != b.numel():
            raise AssertionError(f"group {group} grad size mismatch")
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        cosines[group] = round(cos, 5)
        if cos < args.grad_cos_min:
            raise AssertionError(f"grad cosine for {group} = {cos:.5f} < {args.grad_cos_min}")
    loss_flash = forward_loss(model_off_flash, examples, args.seed, backward=True)
    if not torch.isfinite(loss_flash):
        raise AssertionError("OFF-flash backward produced non-finite loss")
    model_on.zero_grad(set_to_none=True)
    model_off_sdpa.zero_grad(set_to_none=True)
    model_off_flash.zero_grad(set_to_none=True)
    return (
        f"cam grads exactly 0; grad cosines={cosines}; "
        f"losses ON={float(loss_on):.5f} OFF={float(loss_off):.5f} OFF-flash={float(loss_flash):.5f}"
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
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--tight-atol", type=float, default=2e-3)
    parser.add_argument("--tight-loss-rtol", type=float, default=5e-3)
    # 0.25: measured pure backend noise on RANDOM-INIT weights is ~0.1 rel RMS
    # (chaotic layer-to-layer amplification; loss rel diff stays <1e-3 and grad
    # cosine >0.9999). A real semantics bug (wrong mask/ordering) gives ~O(1)
    # (~1.4 for uncorrelated vectors) — 0.25 separates the two regimes cleanly.
    parser.add_argument("--loose-hidden-rel-rms", type=float, default=0.25)
    parser.add_argument("--loose-loss-rtol", type=float, default=2e-2)
    parser.add_argument("--grad-cos-min", type=float, default=0.999)
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
    print(f"[setup] framework={FRAMEWORK} ntok={args.num_depth_tokens} pool={args.pool_hw}")

    ok = True
    try:
        model_on = build_model(args, ckpt_cfg, device, cam_rope_enabled=True, attn_impl="sdpa")
        model_off_sdpa = build_model(args, ckpt_cfg, device, cam_rope_enabled=False, attn_impl="sdpa")
        model_off_flash = build_model(
            args, ckpt_cfg, device, cam_rope_enabled=False, attn_impl="flash_attention_2"
        )
        dropped = transplant(model_on, model_off_sdpa)
        transplant(model_on, model_off_flash)
        print(f"[setup] transplanted ON->OFF weights ({dropped} cam keys dropped, strict load OK)")
        examples = make_examples(model_on, batch_size=args.batch_size, image_size=args.ffs_image_size)

        ok = run_check(FRAMEWORK, "premise_cam_params_zero", lambda: check_premise_cam_params_zero(model_on)) and ok
        ok = run_check(
            FRAMEWORK, "same_kernel_equivalence",
            lambda: check_same_kernel_equivalence(model_on, model_off_sdpa, examples, args),
        ) and ok
        ok = run_check(
            FRAMEWORK, "flash_backend_parity",
            lambda: check_flash_backend_parity(model_off_sdpa, model_off_flash, examples, args),
        ) and ok
        ok = run_check(
            FRAMEWORK, "inert_grads_backward",
            lambda: check_inert_grads_backward(model_on, model_off_sdpa, model_off_flash, examples, args),
        ) and ok
    except Exception as exc:  # noqa: BLE001
        print(f"[{FRAMEWORK}] FAIL fatal: {exc}")
        traceback.print_exc()
        ok = False
    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
