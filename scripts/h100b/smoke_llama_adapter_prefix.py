#!/usr/bin/env python3
"""Smoke test for GR00T FFS #5 LLaMA-Adapter prefix injection.

Run on h100b:

    python scripts/h100b/smoke_llama_adapter_prefix.py

The script builds the real QwenGR00T_LlamaAdapterPrefixFFS framework, loads the
real warm-start B checkpoint through the trainer loader, and uses synthetic
leftprimary stereo examples so it does not depend on the LIBERO dataloader.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    clone_cfg,
    instructions,
    install_ffs_repo,
    load_ckpt,
    load_ckpt_config,
    load_with_trainer_path,
    make_examples,
    max_abs_diff,
    missing_key_report,
    move_model,
    update_cfg,
)
from starVLA.training.trainer_utils import overwatch as _overwatch  # noqa: E402,F401
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups  # noqa: E402


PREFIX_FRAMEWORK = "QwenGR00T_LlamaAdapterPrefixFFS"
PLAIN_FRAMEWORK = "QwenGR00T"
PREFIX_KEY = "ffs_prefix_adapter."
FFS_KEY = "ffs."


def build_prefix_cfg(args: argparse.Namespace, ckpt_cfg, *, framework_name: str):
    cfg = clone_cfg(ckpt_cfg)
    update_cfg(cfg, "framework.name", framework_name)
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
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_enabled", False)
    update_cfg(cfg, "framework.ffs_llama_adapter_prefix", {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.image_size,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "n_prompts": 10,
        "absorb_dim": 256,
        "gate_per_head": False,
    })
    update_cfg(cfg, "datasets.vla_data.data_root_dir", args.data_root)
    update_cfg(cfg, "datasets.vla_data.data_mix", args.data_mix)
    update_cfg(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    update_cfg(cfg, "trainer.pretrained_checkpoint", args.pretrained_ckpt)
    update_cfg(cfg, "trainer.freeze_modules", "qwen_vl_interface")
    update_cfg(cfg, "trainer.logging_frequency", 20)
    return cfg


def finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor.detach()).all():
        raise AssertionError(f"{label} contains non-finite values")


def grad_abs_sum(params: Iterable[torch.nn.Parameter] | nn.Module) -> float:
    if isinstance(params, nn.Module):
        params = params.parameters()
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
        raise AssertionError("CPU RNG state changed during prefix VLM forward")
    if before[1] is None:
        return
    for idx, (a, b) in enumerate(zip(before[1], after[1])):
        if not torch.equal(a, b):
            raise AssertionError(f"CUDA RNG state changed during prefix VLM forward on device {idx}")


def prefix_adapters(model: nn.Module) -> list[nn.Module]:
    parent = getattr(model, "ffs_prefix_adapter", None)
    if parent is None:
        return []
    return list(parent.adapters)


def adapter_params(model: nn.Module) -> list[torch.nn.Parameter]:
    parent = getattr(model, "ffs_prefix_adapter", None)
    if parent is None:
        return []
    return [param for param in parent.parameters()]


def set_all_prefix_gates(model: nn.Module, value: float) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    saved = []
    with torch.no_grad():
        for adapter in prefix_adapters(model):
            saved.append((adapter.gate, adapter.gate.detach().clone()))
            adapter.gate.fill_(float(value))
    return saved


def restore_params(saved: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for param, value in saved:
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def qwen_hidden_plain(model: nn.Module, examples: list[dict]) -> torch.Tensor:
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images(examples),
        instructions=instructions(examples),
    )
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
    return outputs.hidden_states[-1]


def qwen_hidden_prefix(model: nn.Module, examples: list[dict], *, grad: bool = False) -> torch.Tensor:
    ctx = torch.enable_grad() if grad else torch.inference_mode()
    with ctx:
        return model._encode_last_hidden_with_ffs(batch_images(examples), instructions(examples))


class PrefixContext:
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
        self.prefix_model: nn.Module | None = None
        self.baseline_model: nn.Module | None = None
        self.examples: list[dict] | None = None
        self.transplant_detail: str | None = None

    def get_examples(self, batch_size: int | None = None, *, long_pad: bool = False) -> list[dict]:
        if batch_size is None:
            batch_size = self.args.batch_size
        if long_pad:
            examples = make_examples(self.get_prefix_model(), batch_size=batch_size, image_size=self.args.image_size)
            if len(examples) >= 2:
                examples[0]["lang"] = "pick"
                examples[1]["lang"] = (
                    "pick up the object after moving around the long obstacle and "
                    "aligning the wrist carefully"
                )
            return examples
        if self.examples is None or len(self.examples) != batch_size:
            self.examples = make_examples(
                self.get_prefix_model(),
                batch_size=batch_size,
                image_size=self.args.image_size,
            )
        return self.examples

    def _build_prefix_model_cpu(self) -> nn.Module:
        cfg = build_prefix_cfg(self.args, self.ckpt_cfg, framework_name=PREFIX_FRAMEWORK)
        model = build_framework_model(cfg)
        model = load_with_trainer_path(model, self.args.pretrained_ckpt)
        missing, suspicious = missing_key_report(model, self.ckpt)
        suspicious = [
            key for key in suspicious
            if not (key.startswith(PREFIX_KEY) or key.startswith(FFS_KEY))
        ]
        if suspicious:
            raise AssertionError(
                f"warm-start left non-prefix/non-FFS keys missing: count={len(suspicious)}, "
                f"first={suspicious[:20]}"
            )
        model.ffs_prefix_adapter.assert_all_gates_zero("[smoke_prefix]")
        return model

    def get_prefix_model(self) -> nn.Module:
        if self.prefix_model is None:
            self.prefix_model = self._build_prefix_model_cpu()
            self.prefix_model = move_model(self.prefix_model, self.device)
            self.prefix_model.eval()
        return self.prefix_model

    def get_baseline_model(self) -> nn.Module:
        if self.baseline_model is None:
            prefix = self.get_prefix_model()
            cfg = build_prefix_cfg(self.args, self.ckpt_cfg, framework_name=PLAIN_FRAMEWORK)
            baseline = build_framework_model(cfg)
            prefix_sd = prefix.state_dict()
            filtered = {
                key: value.detach().cpu()
                for key, value in prefix_sd.items()
                if not (key.startswith(PREFIX_KEY) or key.startswith(FFS_KEY))
            }
            baseline_keys = set(baseline.state_dict().keys())
            filtered_keys = set(filtered.keys())
            if filtered_keys != baseline_keys:
                missing = sorted(baseline_keys - filtered_keys)[:20]
                extra = sorted(filtered_keys - baseline_keys)[:20]
                raise AssertionError(
                    "filtered prefix state_dict key set does not exactly match baseline; "
                    f"missing_first={missing}, extra_first={extra}"
                )
            baseline.load_state_dict(filtered, strict=True)
            self.transplant_detail = (
                f"prefix_keys={len(prefix_sd)}, filtered_keys={len(filtered)}, "
                f"removed_prefix_or_ffs={len(prefix_sd) - len(filtered)}"
            )
            self.baseline_model = move_model(baseline, self.device)
            self.baseline_model.eval()
        return self.baseline_model

    def cleanup(self) -> None:
        self.prefix_model = None
        self.baseline_model = None
        self.examples = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def check_step0_bitwise(ctx: PrefixContext) -> str:
    prefix = ctx.get_prefix_model()
    baseline = ctx.get_baseline_model()
    examples = ctx.get_examples()

    before = rng_snapshot(ctx.device)
    hidden_prefix = qwen_hidden_prefix(prefix, examples, grad=False).detach().cpu()
    after = rng_snapshot(ctx.device)
    assert_rng_same(before, after)
    state = prefix._ffs_prefix_state
    if state.summary is None or state.summary_ready_count <= 0:
        raise AssertionError("summary cache was not filled; step-0 equality would be vacuous")
    hidden_base = qwen_hidden_plain(baseline, examples).detach().cpu()
    if not torch.equal(hidden_prefix, hidden_base):
        raise AssertionError(
            f"last_hidden is not bitwise equal; max_abs_diff={max_abs_diff(hidden_prefix.float(), hidden_base.float()):.6g}"
        )

    with torch.inference_mode():
        torch.manual_seed(ctx.args.loss_seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed)
        loss_prefix = prefix(examples=examples)["action_loss"].detach().cpu()
        torch.manual_seed(ctx.args.loss_seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed)
        loss_base = baseline(examples=examples)["action_loss"].detach().cpu()
    if not torch.equal(loss_prefix, loss_base):
        raise AssertionError(
            f"action_loss is not bitwise equal; max_abs_diff={max_abs_diff(loss_prefix.float(), loss_base.float()):.6g}"
        )
    return (
        f"{ctx.transplant_detail}; hidden_shape={tuple(hidden_prefix.shape)}; "
        f"summary_shape={state.summary_shape}; rng_unchanged=True"
    )


def check_real_b_full_loader(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    if getattr(model, "stereo_cam_rope_layers", None) is not None:
        raise AssertionError("CAM_ROPE=0 loader left stereo_cam_rope_layers installed")
    missing, suspicious = missing_key_report(model, ctx.ckpt)
    allowed = [key for key in missing if key.startswith(PREFIX_KEY) or key.startswith(FFS_KEY)]
    if len(allowed) != len(missing) or suspicious:
        raise AssertionError(
            f"unexpected warm-start missing keys: missing_first={missing[:20]}, suspicious_first={suspicious[:20]}"
        )
    model.ffs_prefix_adapter.assert_all_gates_zero("[smoke_prefix]")
    legacy = [key for key in ctx.ckpt if key.startswith("stereo_cam_rope_layers.") or ".stereo_cam_layer." in key]
    return f"missing_allowed={len(allowed)}, legacy_cam_rope_keys_in_B={len(legacy)}, gates_zero=True"


def check_no_deadlock_grads(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    examples = ctx.get_examples()
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter_params(model):
        param.requires_grad = True
    for param in model.action_model.parameters():
        param.requires_grad = True
    trainable = adapter_params(model) + [p for p in model.action_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=ctx.args.grad_lr)
    model.train()
    model.qwen_vl_interface.eval()

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(ctx.args.loss_seed)
    if ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(ctx.args.loss_seed)
    loss1 = model(examples=examples)["action_loss"]
    loss1.backward()
    gate_grad = grad_abs_sum(adapter.gate for adapter in prefix_adapters(model))
    if gate_grad <= 0.0:
        raise AssertionError("zero-gate backward produced zero gate grad")
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(ctx.args.loss_seed + 1)
    if ctx.device.type == "cuda":
        torch.cuda.manual_seed_all(ctx.args.loss_seed + 1)
    loss2 = model(examples=examples)["action_loss"]
    loss2.backward()
    absorber_grad = grad_abs_sum(model.ffs_prefix_adapter.absorber)
    kv_grad = grad_abs_sum(adapter.kv_proj.weight for adapter in prefix_adapters(model))
    if absorber_grad <= 0.0 or kv_grad <= 0.0:
        raise AssertionError(
            f"post-gate microstep did not reach full adapter chain: absorber_grad={absorber_grad}, kv_grad={kv_grad}"
        )
    optimizer.step()
    model.eval()
    return (
        f"loss1={float(loss1.detach().float().cpu()):.6g}, loss2={float(loss2.detach().float().cpu()):.6g}, "
        f"gate_grad={gate_grad:.6g}, absorber_grad={absorber_grad:.6g}, kv_grad={kv_grad:.6g}"
    )


def check_gradient_checkpointing_grads(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    examples = ctx.get_examples()
    gc_enable = getattr(model.qwen_vl_interface.model, "gradient_checkpointing_enable", None)
    gc_disable = getattr(model.qwen_vl_interface.model, "gradient_checkpointing_disable", None)
    input_grads_enable = getattr(model.qwen_vl_interface.model, "enable_input_require_grads", None)
    input_grads_disable = getattr(model.qwen_vl_interface.model, "disable_input_require_grads", None)
    if not callable(gc_enable):
        raise AssertionError("qwen_vl_interface.model has no gradient_checkpointing_enable")
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter_params(model):
        param.requires_grad = True
    for param in model.action_model.parameters():
        param.requires_grad = True
    saved = set_all_prefix_gates(model, 1.0e-3)
    try:
        gc_enable()
        if callable(input_grads_enable):
            input_grads_enable()
        model.train()
        model.ffs.eval()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(ctx.args.loss_seed + 20)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed_all(ctx.args.loss_seed + 20)
        loss = model(examples=examples)["action_loss"]
        loss.backward()
        prompt_grad = grad_abs_sum([model.ffs_prefix_adapter.absorber.prompts])
        absorber_grad = grad_abs_sum(model.ffs_prefix_adapter.absorber)
        kv_grad = grad_abs_sum(adapter.kv_proj.weight for adapter in prefix_adapters(model))
        if prompt_grad <= 0.0 or absorber_grad <= 0.0 or kv_grad <= 0.0:
            raise AssertionError(
                "gradient-checkpointing backward did not reach prefix adapter: "
                f"prompt_grad={prompt_grad}, absorber_grad={absorber_grad}, kv_grad={kv_grad}"
            )
    finally:
        if callable(gc_disable):
            gc_disable()
        if callable(input_grads_disable):
            input_grads_disable()
        restore_params(saved)
        model.eval()
    return (
        f"loss={float(loss.detach().float().cpu()):.6g}, prompt_grad={prompt_grad:.6g}, "
        f"absorber_grad={absorber_grad:.6g}, kv_grad={kv_grad:.6g}"
    )


def check_gqa_shapes(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    _ = qwen_hidden_prefix(model, ctx.get_examples(), grad=False)
    details = []
    for adapter in prefix_adapters(model):
        dbg = adapter.last_debug
        if not dbg:
            raise AssertionError(f"layer {adapter.layer_idx} did not record prefix debug shapes")
        expected_pk = (ctx.args.batch_size, 2, 10, 256)
        expected_rep = (ctx.args.batch_size, 8, 10, 256)
        if tuple(dbg["pk_shape"]) != expected_pk or tuple(dbg["pv_shape"]) != expected_pk:
            raise AssertionError(f"layer {adapter.layer_idx} bad pk/pv shape: {dbg}")
        if tuple(dbg["pk_repeat_shape"]) != expected_rep or tuple(dbg["pv_repeat_shape"]) != expected_rep:
            raise AssertionError(f"layer {adapter.layer_idx} bad repeated shape: {dbg}")
        projected = tuple(dbg["projected_shape"])
        if projected[0] != ctx.args.batch_size or projected[-1] != 1024:
            raise AssertionError(f"layer {adapter.layer_idx} projected shape must end in 1024, got {projected}")
        details.append(f"L{adapter.layer_idx}:{projected}")
    return ", ".join(details)


def check_nonzero_gate_pad_invariance(ctx: PrefixContext) -> str:
    prefix = ctx.get_prefix_model()
    baseline = ctx.get_baseline_model()
    examples = ctx.get_examples(batch_size=2, long_pad=True)
    saved = set_all_prefix_gates(prefix, 0.25)
    try:
        hidden_prefix = qwen_hidden_prefix(prefix, examples, grad=False).detach().cpu()
    finally:
        restore_params(saved)
    hidden_base = qwen_hidden_plain(baseline, examples).detach().cpu()
    state = prefix._ffs_prefix_state
    valid = state.valid_query_mask
    if valid is None:
        raise AssertionError("attention_mask was not cached; cannot check pad-row invariance")
    valid_cpu = valid.detach().cpu().bool()
    pad = ~valid_cpu
    if not bool(pad.any()):
        raise AssertionError("synthetic long-pad examples produced no pad rows")
    diff = (hidden_prefix.float() - hidden_base.float()).abs()
    moved_valid = float(diff[valid_cpu].max().item()) if bool(valid_cpu.any()) else 0.0
    moved_pad = float(diff[pad].max().item())
    if moved_valid <= 0.0:
        raise AssertionError("nonzero gates did not move any valid row")
    if moved_pad != 0.0:
        raise AssertionError(f"pad rows changed under nonzero gates: max_pad_diff={moved_pad:.6g}")
    return f"valid_max_diff={moved_valid:.6g}, pad_max_diff={moved_pad:.6g}, pad_rows={int(pad.sum().item())}"


def check_summary_bookkeeping_and_indices(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    if (model.left_ref_idx, model.primary_view_idx, model.inject_cam_id) != (1, 0, 1):
        raise AssertionError(
            f"bad FFS view constants: left_ref={model.left_ref_idx}, primary_view={model.primary_view_idx}, "
            f"inject_cam_id={model.inject_cam_id}"
        )
    _ = qwen_hidden_prefix(model, ctx.get_examples(), grad=False)
    state = model._ffs_prefix_state
    if state.summary is None or tuple(state.summary.shape[1:]) != (10, 1024):
        raise AssertionError(f"bad summary shape: {None if state.summary is None else tuple(state.summary.shape)}")
    cam = state.per_token_cam_id
    if cam is None:
        raise AssertionError("per_token_cam_id was not captured")
    img_ids = cam[0][cam[0] >= 0].detach().cpu()
    if int(img_ids.numel()) == 0:
        raise AssertionError("no image tokens found")
    transitions = torch.nonzero(img_ids[1:] != img_ids[:-1], as_tuple=True)[0]
    if int(img_ids[0].item()) != 0 or int(img_ids[-1].item()) != 1 or int(transitions.numel()) != 1:
        raise AssertionError(f"expected right-camera tokens then left-camera tokens, got {img_ids.tolist()[:20]}...")
    return f"summary_shape={tuple(state.summary.shape)}, image_tokens={int(img_ids.numel())}, right_first=True"


def check_eval_path_predict(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    examples = ctx.get_examples(batch_size=1)
    before = int(model._ffs_prefix_state.summary_ready_count)
    with torch.inference_mode():
        pred = model.predict_action(examples)
    after = int(model._ffs_prefix_state.summary_ready_count)
    if after <= before or model._ffs_prefix_state.summary is None:
        raise AssertionError("predict_action did not prepare prefix summary")
    actions = pred["normalized_actions"]
    if not np.isfinite(actions).all():
        raise AssertionError("predict_action produced non-finite actions")
    return f"pred_shape={actions.shape}, summary_ready_count={after}"


def check_optimizer_grouping(ctx: PrefixContext) -> str:
    model = ctx.get_prefix_model()
    cfg = build_prefix_cfg(ctx.args, ctx.ckpt_cfg, framework_name=PREFIX_FRAMEWORK)
    for param in model.qwen_vl_interface.parameters():
        param.requires_grad = True
    for param in model.action_model.parameters():
        param.requires_grad = True
    for param in model.ffs_prefix_adapter.parameters():
        param.requires_grad = True
    for param in model.ffs.parameters():
        param.requires_grad = False
    groups = build_param_lr_groups(model, cfg)
    param_to_group = {}
    for group in groups:
        for param in group["params"]:
            param_to_group[id(param)] = group["name"]
    adapter_group_names = {param_to_group.get(id(param)) for param in adapter_params(model)}
    adapter_group_names.discard(None)
    if adapter_group_names != {"base"}:
        raise AssertionError(f"adapter params must be in base optimizer group, got {adapter_group_names}")
    action_group_names = {param_to_group.get(id(param)) for param in model.action_model.parameters()}
    action_group_names.discard(None)
    if not action_group_names:
        raise AssertionError("action_model params were absent from optimizer groups")
    qwen_leak = any(id(param) in param_to_group for param in model.qwen_vl_interface.parameters())
    if qwen_leak:
        raise AssertionError("freeze_modules=qwen_vl_interface leaked Qwen params into optimizer groups")
    ffs_leak = any(id(param) in param_to_group for param in model.ffs.parameters())
    if ffs_leak:
        raise AssertionError("requires_grad=False FFS params leaked into optimizer groups")
    TrainerUtils.freeze_backbones(model, freeze_modules="qwen_vl_interface")
    return f"groups={[g['name'] for g in groups]}, adapter_group=base, action_groups={sorted(action_group_names)}"


def set_speed_trainable(model: nn.Module, *, prefix_trainable: bool) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.action_model.parameters():
        param.requires_grad = True
    if prefix_trainable and hasattr(model, "ffs_prefix_adapter"):
        for param in model.ffs_prefix_adapter.parameters():
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
    prefix_trainable: bool,
    device: torch.device,
) -> list[float]:
    trainable = set_speed_trainable(model, prefix_trainable=prefix_trainable)
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
        finite_tensor(loss, f"speed step {step} loss")
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def check_speed_gate(ctx: PrefixContext) -> str:
    prefix = ctx.get_prefix_model()
    baseline = ctx.get_baseline_model()
    examples = make_examples(prefix, batch_size=2, image_size=ctx.args.image_size)
    saved_freq = prefix._ffs_prefix_state.logging_frequency
    prefix._ffs_prefix_state.logging_frequency = 0
    try:
        base_times = timed_train_steps(
            baseline,
            examples,
            steps=ctx.args.speed_steps,
            lr=ctx.args.speed_lr,
            seed=ctx.args.loss_seed + 1000,
            prefix_trainable=False,
            device=ctx.device,
        )
        prefix_times = timed_train_steps(
            prefix,
            examples,
            steps=ctx.args.speed_steps,
            lr=ctx.args.speed_lr,
            seed=ctx.args.loss_seed + 1000,
            prefix_trainable=True,
            device=ctx.device,
        )
    finally:
        prefix._ffs_prefix_state.logging_frequency = saved_freq

    warmup = 2
    if len(base_times) <= warmup or len(prefix_times) <= warmup:
        raise AssertionError(f"speed gate needs > {warmup} steps, got {len(prefix_times)}")
    base_steady = base_times[warmup:]
    prefix_steady = prefix_times[warmup:]
    fastest = min(base_steady)
    max_prefix = max(prefix_steady)
    limit = max(20.0, 8.0 * fastest)
    if max_prefix > limit:
        raise AssertionError(
            f"prefix speed gate failed: max_prefix={max_prefix:.3f}s limit={limit:.3f}s "
            f"base_times={base_times} prefix_times={prefix_times}"
        )
    rel_ratio = min(prefix_steady) / max(min(base_steady), 1e-9)
    if os.environ.get("FORCE", "0") == "1":
        print(
            f"[llama_prefix] speed_gate INFO (shared GPU, relative gate not enforced): "
            f"best prefix {min(prefix_steady):.3f}s vs best base {min(base_steady):.3f}s "
            f"(ratio {rel_ratio:.1f}x)"
        )
    elif rel_ratio > 3.0:
        raise AssertionError(
            f"prefix structural slowdown: best prefix step {min(prefix_steady):.3f}s > "
            f"3x best baseline step {min(base_steady):.3f}s"
        )
    mean_base = sum(base_steady) / len(base_steady)
    mean_prefix = sum(prefix_steady) / len(prefix_steady)
    return (
        f"steps={ctx.args.speed_steps} (warmup {warmup} dropped), base_mean={mean_base:.3f}s, "
        f"prefix_mean={mean_prefix:.3f}s, overhead={mean_prefix - mean_base:.3f}s, "
        f"prefix_max={max_prefix:.3f}s, limit={limit:.3f}s"
    )


def run_check(name: str, fn: Callable[[], str]) -> bool:
    try:
        detail = fn()
    except Exception as exc:
        print(f"[llama_prefix] FAIL {name}: {exc}")
        traceback.print_exc()
        return False
    print(f"[llama_prefix] PASS {name}: {detail}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--ffs-repo-dir", default=DEFAULT_FFS_REPO_DIR)
    parser.add_argument("--ffs-model-path", default=DEFAULT_FFS_MODEL)
    parser.add_argument("--ffs-expected-sha256", default=DEFAULT_FFS_SHA256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--grad-lr", type=float, default=1e-3)
    parser.add_argument("--speed-steps", type=int, default=10)
    parser.add_argument("--speed-lr", type=float, default=1e-5)
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
    ckpt = load_ckpt(args.pretrained_ckpt)
    print(f"[setup] config={cfg_path}")
    print(f"[setup] checkpoint={args.pretrained_ckpt}")
    print(f"[setup] base_vlm={args.base_vlm}")
    print(f"[setup] ffs_model_path={args.ffs_model_path}")
    print(f"[setup] data_mix={args.data_mix} image_order=leftprimary")

    ctx = PrefixContext(args=args, ckpt_cfg=ckpt_cfg, ckpt=ckpt, device=device)
    try:
        ok = True
        ok = run_check("01_step0_bitwise", lambda: check_step0_bitwise(ctx)) and ok
        ok = run_check("02_real_b_full_loader", lambda: check_real_b_full_loader(ctx)) and ok
        ok = run_check("03_no_deadlock_grads", lambda: check_no_deadlock_grads(ctx)) and ok
        ok = run_check("04_gradient_checkpointing_grads", lambda: check_gradient_checkpointing_grads(ctx)) and ok
        ok = run_check("05_gqa_shapes", lambda: check_gqa_shapes(ctx)) and ok
        ok = run_check("06_nonzero_gate_pad_invariance", lambda: check_nonzero_gate_pad_invariance(ctx)) and ok
        ok = run_check("07_summary_bookkeeping_and_indices", lambda: check_summary_bookkeeping_and_indices(ctx)) and ok
        ok = run_check("08_eval_path_predict", lambda: check_eval_path_predict(ctx)) and ok
        ok = run_check("09_optimizer_grouping", lambda: check_optimizer_grouping(ctx)) and ok
        ok = run_check("10_speed_gate", lambda: check_speed_gate(ctx)) and ok
    finally:
        ctx.cleanup()

    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
