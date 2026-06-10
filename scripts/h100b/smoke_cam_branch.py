#!/usr/bin/env python3
"""Smoke test for QwenGR00T parallel PRoPE cam_branch.

Run on h100b:

    python scripts/h100b/smoke_cam_branch.py

The checks are intentionally synthetic-data based so they do not depend on the
LIBERO dataloader, but they build the real model and use the real checkpoint
warm-start path.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h100b.smoke_groot_ffs import (  # noqa: E402
    DEFAULT_BASE_VLM,
    DEFAULT_DATA_MIX,
    DEFAULT_DATA_ROOT,
    DEFAULT_PRETRAINED_CKPT,
    batch_images,
    build_framework_model,
    clone_cfg,
    instructions,
    load_ckpt,
    load_ckpt_config,
    load_with_trainer_path,
    make_examples,
    move_model,
    update_cfg,
)
# Configure logging BEFORE the cam_branch module-level logger is created; otherwise
# overwatch's later dictConfig(disable_existing_loggers=True) silences [cam_branch] lines.
from starVLA.training.trainer_utils import overwatch as _overwatch  # noqa: E402,F401

from starVLA.model.modules.stereo.cam_branch_attention import (  # noqa: E402
    _apply_tiled_projmat,
    compute_libero_camera_prope_triples,
)
from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id  # noqa: E402


PLAIN_FRAMEWORK = "QwenGR00T"
CAM_BRANCH_PREFIX = "stereo_cam_branch_layers_modules."


def build_cam_cfg(args: argparse.Namespace, ckpt_cfg, *, cam_branch_enabled: bool):
    cfg = clone_cfg(ckpt_cfg)
    update_cfg(cfg, "framework.name", PLAIN_FRAMEWORK)
    update_cfg(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_enabled", False)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_d_c", 16)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_num_cameras", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_baseline_m", 0.06)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_fovy_degrees", 45.0)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_width", args.image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_height", args.image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_spatial_merge", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_init_mode", "zero")
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_right_first", True)
    update_cfg(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_enabled", bool(cam_branch_enabled))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_heads", args.branch_heads)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_head_dim", args.branch_head_dim)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_layers", None)
    update_cfg(cfg, "datasets.vla_data.data_root_dir", args.data_root)
    update_cfg(cfg, "datasets.vla_data.data_mix", args.data_mix)
    update_cfg(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    update_cfg(cfg, "trainer.pretrained_checkpoint", args.pretrained_ckpt)
    update_cfg(cfg, "trainer.freeze_modules", "qwen_vl_interface")
    update_cfg(cfg, "trainer.logging_frequency", 20)
    return cfg


def iter_branch_modules(model: nn.Module) -> list[nn.Module]:
    modules = getattr(model, "stereo_cam_branch_layers_modules", None)
    if modules is None:
        return []
    return list(modules)


def language_model_layers(hf_model) -> nn.ModuleList:
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("could not locate language_model.layers")
    return lm.layers


def first_cam_branch_attention(model: nn.Module) -> nn.Module:
    for layer in language_model_layers(model.qwen_vl_interface.model):
        for child_name in ("self_attn", "attention", "attn"):
            if hasattr(layer, child_name):
                attn = getattr(layer, child_name)
                if getattr(attn, "_stereo_cam_branch_installed", False):
                    return attn
    raise AssertionError("no wrapped cam_branch attention layer found")


def qwen_hidden(model: nn.Module, examples: list[dict], *, grad: bool = False) -> torch.Tensor:
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images(examples),
        instructions=instructions(examples),
    )
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
    return outputs.hidden_states[-1]


def finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor.detach()).all():
        raise AssertionError(f"{label} contains non-finite values")


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu().item())


def grad_abs_sum(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().cpu())
    return total


def rng_snapshot(device: torch.device):
    cpu = torch.get_rng_state().clone()
    cuda = None
    if device.type == "cuda":
        cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return cpu, cuda


def assert_rng_same(before, after) -> None:
    if not torch.equal(before[0], after[0]):
        raise AssertionError("CPU RNG state changed during cam_branch VLM forward")
    if before[1] is None:
        return
    for idx, (a, b) in enumerate(zip(before[1], after[1])):
        if not torch.equal(a, b):
            raise AssertionError(f"CUDA RNG state changed during cam_branch VLM forward on device {idx}")


class CamBranchContext:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        ckpt_cfg,
        ckpt: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.args = args
        self.ckpt_cfg = ckpt_cfg
        self.ckpt = ckpt
        self.device = device
        self.branch_model: nn.Module | None = None
        self.baseline_model: nn.Module | None = None
        self.examples: list[dict] | None = None
        self.transplant_detail: str | None = None

    def get_examples(self, batch_size: int | None = None) -> list[dict]:
        if batch_size is None:
            batch_size = self.args.batch_size
        if self.examples is None or len(self.examples) != batch_size:
            self.examples = make_examples(
                self.get_branch_model(),
                batch_size=batch_size,
                image_size=self.args.image_size,
            )
        return self.examples

    def _build_branch_model_cpu(self) -> nn.Module:
        cfg = build_cam_cfg(self.args, self.ckpt_cfg, cam_branch_enabled=True)
        model = build_framework_model(cfg)
        model = load_with_trainer_path(model, self.args.pretrained_ckpt)
        missing = sorted(set(model.state_dict()) - set(self.ckpt))
        bad_missing = [key for key in missing if not key.startswith(CAM_BRANCH_PREFIX)]
        if bad_missing:
            raise AssertionError(
                f"warm-start left non-cam_branch keys missing: count={len(bad_missing)}, "
                f"first={bad_missing[:20]}"
            )
        return model

    def get_branch_model(self) -> nn.Module:
        if self.branch_model is None:
            self.branch_model = self._build_branch_model_cpu()
            self.branch_model = move_model(self.branch_model, self.device)
            self.branch_model.eval()
        return self.branch_model

    def get_baseline_model(self) -> nn.Module:
        if self.baseline_model is None:
            branch = self.get_branch_model()
            cfg = build_cam_cfg(self.args, self.ckpt_cfg, cam_branch_enabled=False)
            baseline = build_framework_model(cfg)
            branch_sd = branch.state_dict()
            filtered = {
                key: value.detach().cpu()
                for key, value in branch_sd.items()
                if not key.startswith(CAM_BRANCH_PREFIX)
            }
            baseline_keys = set(baseline.state_dict().keys())
            filtered_keys = set(filtered.keys())
            if filtered_keys != baseline_keys:
                missing = sorted(baseline_keys - filtered_keys)[:20]
                extra = sorted(filtered_keys - baseline_keys)[:20]
                raise AssertionError(
                    "filtered branch state_dict key set does not exactly match baseline; "
                    f"missing_first={missing}, extra_first={extra}"
                )
            baseline.load_state_dict(filtered, strict=True)
            self.transplant_detail = (
                f"branch_keys={len(branch_sd)}, filtered_keys={len(filtered)}, "
                f"removed_cam_branch={len(branch_sd) - len(filtered)}"
            )
            self.baseline_model = move_model(baseline, self.device)
            self.baseline_model.eval()
        return self.baseline_model

    def cleanup(self) -> None:
        self.branch_model = None
        self.baseline_model = None
        self.examples = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def capture_branch_inputs(model: nn.Module, examples: list[dict]):
    attn = first_cam_branch_attention(model)
    branch = iter_branch_modules(model)[0]
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module, args, kwargs):
        hidden = kwargs.get("hidden_states", None)
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            raise AssertionError("attention hook did not receive hidden_states")
        captured["hidden"] = hidden.detach()

    handle = attn.register_forward_pre_hook(_capture, with_kwargs=True)
    try:
        _ = qwen_hidden(model, examples, grad=False)
    finally:
        handle.remove()

    state = getattr(model, "_stereo_cam_branch_state", None)
    if state is None or state.image_positions is None:
        raise AssertionError("cam_branch state did not cache image positions")
    positions = state.image_positions.to(captured["hidden"].device)
    gather_index = positions.unsqueeze(-1).expand(-1, -1, captured["hidden"].shape[-1])
    hidden_img = captured["hidden"].gather(dim=1, index=gather_index)
    return (
        branch,
        hidden_img,
        state.P_img.to(captured["hidden"].device),
        state.P_T_img.to(captured["hidden"].device),
        state.P_inv_img.to(captured["hidden"].device),
    )


def check_step0_equivalence(ctx: CamBranchContext) -> str:
    branch = ctx.get_branch_model()
    baseline = ctx.get_baseline_model()
    examples = ctx.get_examples()

    before = rng_snapshot(ctx.device)
    hidden_branch = qwen_hidden(branch, examples, grad=False).detach().cpu()
    after = rng_snapshot(ctx.device)
    assert_rng_same(before, after)
    # Self-contained vacuity guard: with a broken pre-hook the wrapper short-circuits
    # to baseline behavior and "bitwise equal" would pass while the feature is dead.
    if branch._stereo_cam_branch_state.image_positions is None:
        raise AssertionError(
            "pre-hook did not populate cam_branch state — bitwise equivalence would be vacuous"
        )
    hidden_base = qwen_hidden(baseline, examples, grad=False).detach().cpu()
    if not torch.equal(hidden_branch, hidden_base):
        raise AssertionError(f"last_hidden is not bitwise equal; max_abs_diff={max_abs_diff(hidden_branch, hidden_base):.6g}")

    with torch.inference_mode():
        torch.manual_seed(ctx.args.loss_seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed)
        loss_branch = branch(examples=examples)["action_loss"].detach().cpu()
        torch.manual_seed(ctx.args.loss_seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed)
        loss_base = baseline(examples=examples)["action_loss"].detach().cpu()
    if not torch.equal(loss_branch, loss_base):
        raise AssertionError(f"action_loss is not bitwise equal; max_abs_diff={max_abs_diff(loss_branch, loss_base):.6g}")
    return f"{ctx.transplant_detail}; last_hidden_shape={tuple(hidden_branch.shape)}; action_loss={float(loss_branch.float()):.6g}; rng_unchanged=True"


def check_no_deadlock_grads(ctx: CamBranchContext) -> str:
    model = ctx.get_branch_model()
    examples = ctx.get_examples()
    for param in model.parameters():
        param.requires_grad = False
    branches = iter_branch_modules(model)
    for branch in branches:
        for param in branch.parameters():
            param.requires_grad = True
    trainable = [param for branch in branches for param in branch.parameters()]
    if not trainable:
        raise AssertionError("no cam_branch trainable parameters found")

    model.train()
    model.qwen_vl_interface.eval()
    optimizer = torch.optim.SGD(trainable, lr=ctx.args.grad_lr)

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(ctx.args.loss_seed)
    if ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(ctx.args.loss_seed)
    loss1 = model(examples=examples)["action_loss"]
    loss1.backward()
    out_grad = grad_abs_sum(branch.out_proj.weight for branch in branches)
    if out_grad <= 0.0:
        raise AssertionError("first backward produced zero out_proj.weight grad")
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(ctx.args.loss_seed + 1)
    if ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(ctx.args.loss_seed + 1)
    loss2 = model(examples=examples)["action_loss"]
    loss2.backward()
    qkv_grad = grad_abs_sum(
        param
        for branch in branches
        for param in (branch.q_proj.weight, branch.k_proj.weight, branch.v_proj.weight)
    )
    if qkv_grad <= 0.0:
        raise AssertionError("second backward produced zero q/k/v projection grad")
    optimizer.step()
    model.eval()
    return f"loss1={float(loss1.detach().float().cpu()):.6g}, loss2={float(loss2.detach().float().cpu()):.6g}, out_grad={out_grad:.6g}, qkv_grad={qkv_grad:.6g}"


def check_matrix_triple_correctness(ctx: CamBranchContext) -> str:
    P, P_T, P_inv = compute_libero_camera_prope_triples(
        num_cameras=2,
        baseline_m=0.06,
        fovy_degrees=45.0,
        image_width=ctx.args.image_size,
        image_height=ctx.args.image_size,
        right_first=True,
    )
    eye = torch.eye(4, dtype=torch.float32).expand_as(P)
    prod = torch.einsum("cij,cjk->cik", P, P_inv)
    err = float((prod - eye).abs().max().item())
    if err > 1e-4:
        raise AssertionError(f"P times P_inv is not identity enough: max_err={err:.6g}")
    if not (float(P[0, 0, 3]) > 0.0 and float(P[1, 0, 3]) == 0.0):
        raise AssertionError("right-first P_stack order is not [right, left]")

    model = ctx.get_branch_model()
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images(ctx.get_examples()),
        instructions=instructions(ctx.get_examples()),
    )
    cam = compute_per_token_cam_id(
        input_ids=qwen_inputs["input_ids"],
        image_token_id=int(model.qwen_vl_interface.model.config.image_token_id),
        image_grid_thw=qwen_inputs.get("image_grid_thw", None),
        num_cameras=2,
        spatial_merge_size=2,
    )
    img_ids = cam[0][cam[0] >= 0]
    if int(img_ids.numel()) == 0:
        raise AssertionError("no image tokens found for cam_id check")
    transitions = torch.nonzero(img_ids[1:] != img_ids[:-1], as_tuple=True)[0]
    if int(img_ids[0].item()) != 0 or int(img_ids[-1].item()) != 1 or int(transitions.numel()) != 1:
        raise AssertionError(f"expected right-camera tokens then left-camera tokens, got ids={img_ids.tolist()[:20]}...")

    torch.manual_seed(777)
    q = torch.randn(1, 1, 2, ctx.args.branch_head_dim, dtype=torch.float32)
    k = torch.randn(1, 1, 2, ctx.args.branch_head_dim, dtype=torch.float32)
    same_cam_P_T = P_T[0:1].expand(1, 2, 4, 4)
    same_cam_P_inv = P_inv[0:1].expand(1, 2, 4, 4)
    q_t = _apply_tiled_projmat(q, same_cam_P_T)
    k_t = _apply_tiled_projmat(k, same_cam_P_inv)
    score_ref = torch.einsum("bhid,bhjd->bhij", q, k)
    score_new = torch.einsum("bhid,bhjd->bhij", q_t, k_t)
    inv_err = float((score_ref - score_new).abs().max().item())
    if inv_err > 1e-4:
        raise AssertionError(f"same-camera PRoPE score changed: max_err={inv_err:.6g}")

    src = (ROOT / "starVLA/model/modules/stereo/cam_branch_attention.py").read_text()
    forbidden = ["torch." + "inverse", "torch.linalg." + "inv"]
    present = [needle for needle in forbidden if needle in src]
    if present or re.search(r"_invert_SE3\s*\(\s*P", src):
        raise AssertionError(f"forbidden matrix inversion path present: {present}")
    return f"P_identity_err={err:.3g}, same_camera_score_err={inv_err:.3g}, image_tokens={int(img_ids.numel())}"


def check_causal_no_leak(ctx: CamBranchContext) -> str:
    model = ctx.get_branch_model()
    branch, hidden_img, P_img, P_T_img, P_inv_img = capture_branch_inputs(model, ctx.get_examples())
    n_img = int(hidden_img.shape[1])
    if n_img < 4:
        raise AssertionError(f"need at least 4 image tokens for causal probe, got {n_img}")
    cut = n_img // 2
    with torch.inference_mode():
        raw_a = branch.forward_raw(hidden_img, P_img, P_T_img, P_inv_img).detach().cpu()
        hidden_b = hidden_img.clone()
        perturb = torch.linspace(
            0.01,
            0.02,
            steps=hidden_b[:, cut:, :].numel(),
            device=hidden_b.device,
            dtype=hidden_b.dtype,
        ).view_as(hidden_b[:, cut:, :])
        hidden_b[:, cut:, :] = hidden_b[:, cut:, :] + perturb
        raw_b = branch.forward_raw(hidden_b, P_img, P_T_img, P_inv_img).detach().cpu()
    early_a = raw_a[:, :cut]
    early_b = raw_b[:, :cut]
    if not torch.equal(early_a, early_b):
        raise AssertionError(f"early raw branch output changed from future-token perturbation; max_abs_diff={max_abs_diff(early_a, early_b):.6g}")
    moved = max_abs_diff(raw_a[:, cut:], raw_b[:, cut:])
    if moved <= 0.0:
        raise AssertionError("future-token perturbation did not move later raw outputs")
    return f"n_img={n_img}, protected_prefix={cut}, future_diff={moved:.6g}"


def check_leftpad_no_nan(ctx: CamBranchContext) -> str:
    model = ctx.get_branch_model()
    examples = make_examples(model, batch_size=2, image_size=ctx.args.image_size)
    examples[0]["lang"] = "pick"
    examples[1]["lang"] = "pick up the object after moving around the long obstacle and aligning the wrist carefully"
    branch, hidden_img, P_img, P_T_img, P_inv_img = capture_branch_inputs(model, examples)
    with torch.inference_mode():
        raw = branch.forward_raw(hidden_img, P_img, P_T_img, P_inv_img)
        finite_tensor(raw, "raw branch output")
        hidden = qwen_hidden(model, examples, grad=False)
        finite_tensor(hidden, "last_hidden")
        torch.manual_seed(ctx.args.loss_seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed)
        out = model(examples=examples)
        loss = out["action_loss"]
        finite_tensor(loss, "action_loss")
    return f"raw_shape={tuple(raw.shape)}, hidden_shape={tuple(hidden.shape)}, loss={float(loss.detach().float().cpu()):.6g}"


def set_speed_trainable(model: nn.Module, *, branch_trainable: bool) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.action_model.parameters():
        param.requires_grad = True
    if branch_trainable:
        for branch in iter_branch_modules(model):
            for param in branch.parameters():
                param.requires_grad = True
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("speed model has no trainable parameters")
    model.train()
    model.qwen_vl_interface.eval()
    return trainable


def timed_train_steps(
    model: nn.Module,
    examples: list[dict],
    *,
    steps: int,
    lr: float,
    seed: int,
    branch_trainable: bool,
    device: torch.device,
) -> list[float]:
    trainable = set_speed_trainable(model, branch_trainable=branch_trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    times: list[float] = []
    for step in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(seed + step)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + step)
        loss = model(examples=examples)["action_loss"]
        if not torch.isfinite(loss.detach()).all():
            raise AssertionError(f"speed step {step}: non-finite loss")
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def check_speed_gate(ctx: CamBranchContext) -> str:
    branch = ctx.get_branch_model()
    baseline = ctx.get_baseline_model()
    examples = make_examples(branch, batch_size=2, image_size=ctx.args.image_size)
    # Silence the per-layer norm logging during timing: its norm().cpu() is a GPU
    # sync that would pollute the measured step times.
    branch_state = getattr(branch, "_stereo_cam_branch_state", None)
    saved_logging_frequency = branch_state.logging_frequency if branch_state is not None else None
    if branch_state is not None:
        branch_state.logging_frequency = 0
    try:
        base_times = timed_train_steps(
            baseline,
            examples,
            steps=ctx.args.speed_steps,
            lr=ctx.args.speed_lr,
            seed=ctx.args.loss_seed + 1000,
            branch_trainable=False,
            device=ctx.device,
        )
        branch_times = timed_train_steps(
            branch,
            examples,
            steps=ctx.args.speed_steps,
            lr=ctx.args.speed_lr,
            seed=ctx.args.loss_seed + 1000,
            branch_trainable=True,
            device=ctx.device,
        )
    finally:
        if branch_state is not None and saved_logging_frequency is not None:
            branch_state.logging_frequency = saved_logging_frequency
    # Drop the first 2 steps of each series: first-step CUDA kernel compilation /
    # autotune (and SM contention when the smoke shares a GPU with live training)
    # dominates there and is not branch overhead. Same warmup convention as the
    # other smokes' timing gates.
    warmup = 2
    if len(base_times) <= warmup or len(branch_times) <= warmup:
        raise AssertionError(f"speed gate needs > {warmup} steps, got {len(branch_times)}")
    base_steady = base_times[warmup:]
    branch_steady = branch_times[warmup:]
    fastest = min(base_steady)
    max_branch = max(branch_steady)
    limit = max(20.0, 8.0 * fastest)
    if max_branch > limit:
        raise AssertionError(
            f"cam_branch speed gate failed: max_branch={max_branch:.3f}s limit={limit:.3f}s "
            f"base_times={base_times} branch_times={branch_times}"
        )
    # Relative gate: min-vs-min catches a cam_rope-class (~4x) structural slowdown
    # that the 20s absolute floor would miss. ONLY valid on an exclusive GPU: under
    # co-location (FORCE=1 next to a live training) the two minima are not sampled
    # under equal contention — the baseline can luck into a contention-free step the
    # branch never gets (observed: 0.121s vs 1.407s, pure interference asymmetry).
    rel_ratio = min(branch_steady) / max(min(base_steady), 1e-9)
    if os.environ.get("FORCE", "0") == "1":
        print(
            f"[cam_branch] speed_gate INFO (shared GPU, relative gate not enforced): "
            f"best branch {min(branch_steady):.3f}s vs best base {min(base_steady):.3f}s "
            f"(ratio {rel_ratio:.1f}x)"
        )
    elif rel_ratio > 3.0:
        raise AssertionError(
            f"cam_branch structural slowdown: best branch step {min(branch_steady):.3f}s > "
            f"3x best baseline step {min(base_steady):.3f}s"
        )
    mean_base = sum(base_steady) / len(base_steady)
    mean_branch = sum(branch_steady) / len(branch_steady)
    return (
        f"steps={ctx.args.speed_steps} (warmup {warmup} dropped), base_mean={mean_base:.3f}s, "
        f"branch_mean={mean_branch:.3f}s, overhead={mean_branch - mean_base:.3f}s, "
        f"branch_max={max_branch:.3f}s, limit={limit:.3f}s"
    )


def check_eval_path_predict(ctx: CamBranchContext) -> str:
    """predict_action must fire the cam_branch pre-forward hook too (eval path)."""
    branch = ctx.get_branch_model()
    examples = ctx.get_examples(batch_size=1)
    state = getattr(branch, "_stereo_cam_branch_state", None)
    if state is None:
        raise AssertionError("branch model has no _stereo_cam_branch_state")
    state.clear()
    with torch.inference_mode():
        pred = branch.predict_action(examples)
    if state.image_positions is None:
        raise AssertionError(
            "predict_action did not populate cam_branch state — the pre-forward hook "
            "did not fire on the eval path"
        )
    actions = pred["normalized_actions"]
    import numpy as _np

    if not _np.isfinite(actions).all():
        raise AssertionError("predict_action produced non-finite actions")
    return f"pred_shape={actions.shape}, image_positions={tuple(state.image_positions.shape)}"


def run_check(name: str, fn: Callable[[], str]) -> bool:
    try:
        detail = fn()
    except Exception as exc:
        print(f"[cam_branch] FAIL {name}: {exc}")
        traceback.print_exc()
        return False
    print(f"[cam_branch] PASS {name}: {detail}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--branch-heads", type=int, default=4)
    parser.add_argument("--branch-head-dim", type=int, default=128)
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--grad-lr", type=float, default=1e-3)
    parser.add_argument("--speed-steps", type=int, default=10)
    parser.add_argument("--speed-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    ckpt = load_ckpt(args.pretrained_ckpt)
    print(f"[setup] config={cfg_path}")
    print(f"[setup] checkpoint={args.pretrained_ckpt}")
    print(f"[setup] base_vlm={args.base_vlm}")
    print(f"[setup] data_mix={args.data_mix} image_order=right_first")

    ctx = CamBranchContext(args=args, ckpt_cfg=ckpt_cfg, ckpt=ckpt, device=device)
    try:
        ok = True
        ok = run_check("step0_equivalence", lambda: check_step0_equivalence(ctx)) and ok
        # NOTE: no_deadlock_grads MUTATES the cached branch model (two optimizer steps
        # move out_proj off zero; requires_grad flags flipped). Keep it AFTER
        # step0_equivalence and never add later checks that assume zero-init weights.
        ok = run_check("no_deadlock_grads", lambda: check_no_deadlock_grads(ctx)) and ok
        ok = run_check("matrix_triple_correctness", lambda: check_matrix_triple_correctness(ctx)) and ok
        ok = run_check("causal_no_leak", lambda: check_causal_no_leak(ctx)) and ok
        ok = run_check("leftpad_no_nan", lambda: check_leftpad_no_nan(ctx)) and ok
        ok = run_check("eval_path_predict", lambda: check_eval_path_predict(ctx)) and ok
        ok = run_check("speed_gate", lambda: check_speed_gate(ctx)) and ok
    finally:
        ctx.cleanup()

    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
