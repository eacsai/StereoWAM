#!/usr/bin/env python3
"""Smoke checks for GR00T FFS #4 depth-token.

This is intentionally separate from smoke_groot_ffs.py because #4 inserts new
tokens, so step-0 parity with baseline is not expected.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id
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
    check_net0_left_sanity,
    clone_cfg,
    grad_sum,
    install_ffs_repo,
    instructions,
    load_ckpt_config,
    load_with_trainer_path,
    make_examples,
    move_model,
    run_check,
    update_cfg,
)


FRAMEWORK = "QwenGR00T_DepthTokenFFS"


def ffs_depth_cfg(args: argparse.Namespace) -> dict:
    return {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.ffs_image_size,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "num_depth_tokens": 16,
        "pool_hw": 4,
    }


def build_depth_cfg(
    args: argparse.Namespace,
    ckpt_cfg,
    *,
    pretrained_ckpt: str,
    freeze_modules: str,
    cam_rope_enabled: bool = True,
):
    cfg = clone_cfg(ckpt_cfg)
    update_cfg(cfg, "framework.name", FRAMEWORK)
    update_cfg(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_enabled", bool(cam_rope_enabled))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_d_c", 16)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_num_cameras", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_baseline_m", 0.06)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_fovy_degrees", 45.0)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_width", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_height", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_spatial_merge", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_init_mode", "zero")
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_right_first", True)
    update_cfg(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False)
    update_cfg(cfg, "framework.ffs_depth_token", ffs_depth_cfg(args))
    update_cfg(cfg, "datasets.vla_data.data_root_dir", args.data_root)
    update_cfg(cfg, "datasets.vla_data.data_mix", args.data_mix)
    update_cfg(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    update_cfg(cfg, "trainer.pretrained_checkpoint", pretrained_ckpt)
    update_cfg(cfg, "trainer.freeze_modules", freeze_modules)
    return cfg


def build_model(
    args: argparse.Namespace,
    ckpt_cfg,
    *,
    pretrained_ckpt: str,
    freeze_modules: str,
    device: torch.device,
) -> nn.Module:
    cfg = build_depth_cfg(
        args,
        ckpt_cfg,
        pretrained_ckpt=pretrained_ckpt,
        freeze_modules=freeze_modules,
    )
    model = build_framework_model(cfg)
    if pretrained_ckpt:
        model = load_with_trainer_path(model, pretrained_ckpt)
    model = TrainerUtils.freeze_backbones(model, freeze_modules=freeze_modules)
    model = move_model(model, device)
    return model


def find_subsequence(haystack: torch.Tensor, needle: list[int], end: int) -> int:
    if not needle:
        return -1
    values = haystack[:end].detach().cpu().tolist()
    n = len(needle)
    for idx in range(0, max(len(values) - n + 1, 0)):
        if values[idx : idx + n] == needle:
            return idx
    return -1


def pooled_cell_source(primary_pos: torch.Tensor, num_insert: int) -> torch.Tensor:
    n_prim = int(primary_pos.numel())
    k = int(round(num_insert ** 0.5))
    g = int(round(n_prim ** 0.5))
    if k * k == num_insert and g * g == n_prim and g >= k:
        sel = []
        for i in range(k):
            r = min(int((i + 0.5) * g / k), g - 1)
            for j in range(k):
                c = min(int((j + 0.5) * g / k), g - 1)
                sel.append(r * g + c)
        return primary_pos.index_select(0, torch.tensor(sel, dtype=torch.long, device=primary_pos.device))
    if n_prim == 1:
        return primary_pos.expand(num_insert)
    lin = torch.linspace(0, n_prim - 1, steps=num_insert, device=primary_pos.device)
    return primary_pos.index_select(0, lin.round().long())


def position_columns(position_ids: torch.Tensor, batch_idx: int, cols: torch.Tensor) -> torch.Tensor:
    pos_b = position_ids[..., batch_idx, :]
    return pos_b.index_select(-1, cols.to(position_ids.device))


def check_sequence_contract(model: nn.Module, examples: list[dict], args: argparse.Namespace) -> str:
    model.eval()
    clear_depth_state()
    qwen_inputs = model._build_depthtoken_qwenvl_inputs(batch_images(examples), instructions(examples))
    input_ids = qwen_inputs["input_ids"]
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    spatial_merge = int(model.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
    image_token_id = int(model.qwen_vl_interface.model.config.image_token_id)
    original_cam = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=image_token_id,
        image_grid_thw=image_grid_thw,
        num_cameras=2,
        spatial_merge_size=spatial_merge,
    )
    prompt_ids = model.qwen_vl_interface.processor.tokenizer(
        model.depth_token_prompt,
        add_special_tokens=False,
    )["input_ids"]

    with torch.inference_mode():
        hidden = model._encode_last_hidden_with_ffs(batch_images(examples), instructions(examples))

    state = get_depth_state()
    if state.keep_mask is None or state.per_token_cam_id is None or state.insert_idx is None:
        raise AssertionError("depth-token hook did not populate keep_mask/per_token_cam_id/insert_idx")
    if hidden.shape[1] != input_ids.shape[1] + model.num_depth_tokens:
        raise AssertionError(
            f"hidden seq len {hidden.shape[1]} != original {input_ids.shape[1]} + {model.num_depth_tokens}"
        )
    if tuple(state.keep_mask.shape) != tuple(state.per_token_cam_id.shape):
        raise AssertionError("keep_mask and expanded cam_id shape mismatch")

    details = []
    for b in range(input_ids.shape[0]):
        left_pos = (original_cam[b] == 1).nonzero(as_tuple=True)[0]
        right_pos = (original_cam[b] == 0).nonzero(as_tuple=True)[0]
        if int(left_pos.numel()) == 0 or int(right_pos.numel()) == 0:
            raise AssertionError(f"sample {b}: missing left/right image token run")
        left_start = int(left_pos.min().item())
        right_end = int(right_pos.max().item()) + 1
        insert_idx = int(state.insert_idx[b].item())
        if insert_idx != left_start:
            raise AssertionError(f"sample {b}: insert_idx={insert_idx}, expected left_start={left_start}")
        prompt_at = find_subsequence(input_ids[b], prompt_ids, left_start)
        if prompt_at < right_end:
            raise AssertionError(
                f"sample {b}: depth prompt tokens not found between right image and left image "
                f"(right_end={right_end}, left_start={left_start}, found={prompt_at})"
            )
        depth_slice = slice(insert_idx, insert_idx + model.num_depth_tokens)
        if not torch.equal(
            state.per_token_cam_id[b, depth_slice],
            torch.full_like(state.per_token_cam_id[b, depth_slice], -1),
        ):
            raise AssertionError(f"sample {b}: inserted depth token cam_id segment is not all -1")
        if bool(state.keep_mask[b, depth_slice].any()):
            raise AssertionError(f"sample {b}: keep_mask did not mark depth-token segment as inserted")
        if int((~state.keep_mask[b]).sum().item()) != model.num_depth_tokens:
            raise AssertionError(f"sample {b}: keep_mask false count is not {model.num_depth_tokens}")
        details.append(f"b{b}:orig={input_ids.shape[1]},expanded={hidden.shape[1]},insert={insert_idx},prompt={prompt_at}")

    if state.position_ids_before is None or state.position_ids_after is None:
        raise AssertionError("depth-token hook did not observe/expand position_ids")
    before = state.position_ids_before
    after = state.position_ids_after
    if int(after.shape[-1]) != int(before.shape[-1]) + model.num_depth_tokens:
        raise AssertionError("position_ids did not expand by num_depth_tokens")
    for b in range(input_ids.shape[0]):
        insert_idx = int(state.insert_idx[b].item())
        left_pos = (original_cam[b] == 1).nonzero(as_tuple=True)[0]
        source_cols = pooled_cell_source(left_pos.to(before.device), model.num_depth_tokens)
        expected = position_columns(before, b, source_cols)
        got = after[..., b, insert_idx : insert_idx + model.num_depth_tokens]
        if not torch.equal(got.cpu(), expected.cpu()):
            raise AssertionError(f"sample {b}: depth position_ids do not match left-image pooled cells")

    if state.depth_tokens is None:
        raise AssertionError("state.depth_tokens is empty")
    depth_max = float(state.depth_tokens.detach().abs().max().cpu().item())
    inserted_max = (
        float(state.inserted_depth_embeds.detach().abs().max().cpu().item())
        if state.inserted_depth_embeds is not None
        else depth_max
    )
    if depth_max > args.zero_atol or inserted_max > args.zero_atol:
        raise AssertionError(
            f"zero-init depth tokens not near zero: projector={depth_max:.6g}, inserted={inserted_max:.6g}"
        )
    return "; ".join(details) + f"; depth_zero_max={depth_max:.6g}; inserted_zero_max={inserted_max:.6g}"


def capture_head_seq_lens(model: nn.Module, examples: list[dict]) -> str:
    forward_lens: list[int] = []
    predict_lens: list[int] = []
    orig_forward = model.action_model.forward
    orig_predict = model.action_model.predict_action

    def wrapped_forward(vl_embs, *args, **kwargs):
        forward_lens.append(int(vl_embs.shape[1]))
        return orig_forward(vl_embs, *args, **kwargs)

    def wrapped_predict(vl_embs, *args, **kwargs):
        predict_lens.append(int(vl_embs.shape[1]))
        return orig_predict(vl_embs, *args, **kwargs)

    model.action_model.forward = wrapped_forward
    model.action_model.predict_action = wrapped_predict
    try:
        clear_depth_state()
        model._ffs_captured_net0 = None
        with torch.inference_mode():
            torch.manual_seed(123)
            out = model(examples=examples)
        if model._ffs_captured_net0 is None:
            raise AssertionError("forward did not trigger FFS net[0] hook")
        if not torch.isfinite(out["action_loss"].detach()).all():
            raise AssertionError("forward action_loss is not finite")
        forward_state = get_depth_state()
        if not forward_lens or forward_lens[-1] != int(forward_state.keep_mask.shape[1]):
            raise AssertionError("forward head did not receive expanded depth-token sequence")

        clear_depth_state()
        model._ffs_captured_net0 = None
        with torch.inference_mode():
            torch.manual_seed(123)
            pred = model.predict_action(examples=examples)
        if model._ffs_captured_net0 is None:
            raise AssertionError("predict_action did not trigger FFS net[0] hook")
        predict_state = get_depth_state()
        if not predict_lens or predict_lens[-1] != int(predict_state.keep_mask.shape[1]):
            raise AssertionError("predict_action head did not receive expanded depth-token sequence")
        pred_shape = np.asarray(pred["normalized_actions"]).shape
    finally:
        model.action_model.forward = orig_forward
        model.action_model.predict_action = orig_predict

    return f"forward_head_seq={forward_lens[-1]}, predict_head_seq={predict_lens[-1]}, pred_shape={pred_shape}"


def trainable_params(module: nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def param_grad_sum(params: Iterable[torch.nn.Parameter]) -> float:
    return grad_sum(params)


def assert_no_grad(module: nn.Module, label: str) -> None:
    leaked = []
    for name, param in module.named_parameters():
        if param.grad is not None and float(param.grad.detach().abs().sum().cpu()) != 0.0:
            leaked.append(name)
            if len(leaked) >= 5:
                break
    if leaked:
        raise AssertionError(f"{label} received grads despite freeze: {leaked}")


def cam_rope_grad_sum(model: nn.Module) -> float:
    module = getattr(model, "stereo_cam_rope_layers", None)
    if module is None:
        return 0.0
    return param_grad_sum(param for param in module.parameters() if param.requires_grad)


def run_train_steps(
    model: nn.Module,
    examples: list[dict],
    args: argparse.Namespace,
    *,
    expect_vlm_frozen: bool,
    require_cam_rope_grad: bool,
) -> str:
    model.train()
    optimizer = torch.optim.AdamW(trainable_params(model), lr=args.train_lr)
    last_loss = None
    depth_grad = head_grad = vlm_grad = cam_grad = 0.0
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(args.loss_seed + step)
        out = model(examples=examples)
        loss = out["action_loss"]
        if not torch.isfinite(loss.detach()).all():
            raise AssertionError(f"step {step}: non-finite loss {loss.detach().float().cpu().item()}")
        loss.backward()
        depth_grad = param_grad_sum(model.depth_token_projector.parameters())
        head_grad = param_grad_sum(model.action_model.parameters())
        vlm_grad = param_grad_sum(param for param in model.qwen_vl_interface.parameters() if param.requires_grad)
        cam_grad = cam_rope_grad_sum(model)
        if depth_grad <= 0.0:
            raise AssertionError(f"step {step}: depth_token_projector grad is zero")
        if head_grad <= 0.0:
            raise AssertionError(f"step {step}: action head grad is zero")
        if expect_vlm_frozen:
            assert_no_grad(model.qwen_vl_interface, "qwen_vl_interface")
        elif vlm_grad <= 0.0:
            raise AssertionError(f"step {step}: trainable VLM grad is zero")
        if require_cam_rope_grad and cam_grad <= 0.0:
            raise AssertionError(f"step {step}: trainable cam_rope grad is zero")
        optimizer.step()
        last_loss = float(loss.detach().float().cpu().item())
    if last_loss is None:
        raise AssertionError("train_steps was zero")
    return (
        f"steps={args.train_steps}, last_loss={last_loss:.6g}, "
        f"depth_grad={depth_grad:.6g}, head_grad={head_grad:.6g}, "
        f"vlm_grad={vlm_grad:.6g}, cam_rope_grad={cam_grad:.6g}"
    )


def check_freeze_boundary(model: nn.Module, *, expect_vlm_frozen: bool) -> str:
    qwen_trainable = any(param.requires_grad for param in model.qwen_vl_interface.parameters())
    cam_module = getattr(model, "stereo_cam_rope_layers", None)
    cam_trainable = cam_module is not None and any(param.requires_grad for param in cam_module.parameters())
    depth_trainable = any(param.requires_grad for param in model.depth_token_projector.parameters())
    head_trainable = any(param.requires_grad for param in model.action_model.parameters())
    ffs_trainable = any(param.requires_grad for param in model.ffs.parameters())
    if ffs_trainable:
        raise AssertionError("raw frozen FFS net has trainable params")
    # #4a warm-start: 只冻 qwen_vl_interface(VLM); cam_rope 挂 framework 顶层
    # (stereo_cam_rope_layers), FREEZE_MODULES=qwen_vl_interface 冻不到它 → 保持 trainable,
    # 跟 #1/#2/#3 + head-only 对照一致(它们 warm-start 下 cam_rope 也都 trainable)。所以这里
    # 只断言 VLM 冻, 允许 cam_rope trainable(M1 fix, codex spec/code review 2026-06-08)。
    if expect_vlm_frozen and qwen_trainable:
        raise AssertionError("warm-start #4a expected qwen_vl_interface (VLM) frozen")
    if not expect_vlm_frozen and not (qwen_trainable and cam_trainable):
        raise AssertionError("from-scratch #4b expected VLM + cam_rope trainable")
    if not depth_trainable or not head_trainable:
        raise AssertionError("depth_token_projector and action head must be trainable")
    return (
        f"qwen_trainable={qwen_trainable}, cam_rope_trainable={cam_trainable}, "
        f"depth_projector_trainable={depth_trainable}, head_trainable={head_trainable}, ffs_trainable={ffs_trainable}"
    )


def check_no_checkpoint_config(model: nn.Module) -> str:
    ckpt = str(model.config.trainer.get("pretrained_checkpoint", ""))
    freeze_modules = str(model.config.trainer.get("freeze_modules", ""))
    if ckpt:
        raise AssertionError(f"from-scratch config still has pretrained_checkpoint={ckpt}")
    if freeze_modules:
        raise AssertionError(f"from-scratch config still has freeze_modules={freeze_modules}")
    return "trainer.pretrained_checkpoint='', trainer.freeze_modules=''"


class Net0Context:
    def __init__(self, model: nn.Module, args: argparse.Namespace) -> None:
        self._model = model
        self.args = args

    def get_model(self) -> nn.Module:
        return self._model


def run_warmstart(args: argparse.Namespace, ckpt_cfg, device: torch.device) -> bool:
    model = build_model(
        args,
        ckpt_cfg,
        pretrained_ckpt=args.pretrained_ckpt,
        freeze_modules="qwen_vl_interface",
        device=device,
    )
    examples = make_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
    ok = True
    ok = run_check(FRAMEWORK, "warm_start_boundary", lambda: check_freeze_boundary(model, expect_vlm_frozen=True)) and ok
    ok = run_check(FRAMEWORK, "sequence_prompt_depth_position", lambda: check_sequence_contract(model, examples, args)) and ok
    ok = run_check(FRAMEWORK, "head_retains_depth_tokens", lambda: capture_head_seq_lens(model, examples)) and ok
    ok = run_check(FRAMEWORK, "net0_left_frame_sanity", lambda: check_net0_left_sanity(Net0Context(model, args))) and ok
    ok = run_check(
        FRAMEWORK,
        "warm_start_10step_frozen_vlm",
        lambda: run_train_steps(model, examples, args, expect_vlm_frozen=True, require_cam_rope_grad=False),
    ) and ok
    return ok


def run_fromscratch(args: argparse.Namespace, ckpt_cfg, device: torch.device) -> bool:
    model = build_model(
        args,
        ckpt_cfg,
        pretrained_ckpt="",
        freeze_modules="",
        device=device,
    )
    examples = make_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
    ok = True
    ok = run_check(FRAMEWORK, "fromscratch_boundary", lambda: check_freeze_boundary(model, expect_vlm_frozen=False)) and ok
    ok = run_check(FRAMEWORK, "fromscratch_no_checkpoint_config", lambda: check_no_checkpoint_config(model)) and ok
    ok = run_check(
        FRAMEWORK,
        "fromscratch_10step_all_trainable",
        lambda: run_train_steps(model, examples, args, expect_vlm_frozen=False, require_cam_rope_grad=True),
    ) and ok
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("warmstart", "fromscratch", "both"), default="both")
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--ffs-model-path", default=DEFAULT_FFS_MODEL)
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-expected-sha256", default=DEFAULT_FFS_SHA256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ffs-image-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--train-lr", type=float, default=1e-5)
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--zero-atol", type=float, default=1e-7)
    parser.add_argument("--same-disp-abs-mean-max", type=float, default=1.0)
    parser.add_argument("--same-disp-std-max", type=float, default=0.5)
    parser.add_argument("--swap-sign-cos-min", type=float, default=0.15)
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
    print(f"[setup] checkpoint={args.pretrained_ckpt}")
    print(f"[setup] base_vlm={args.base_vlm}")
    print(f"[setup] ffs_model={args.ffs_model_path}")
    print("[setup] framework=QwenGR00T_DepthTokenFFS image_order=leftprimary inject_cam_id=1")

    ok = True
    try:
        if args.mode in ("warmstart", "both"):
            ok = run_warmstart(args, ckpt_cfg, device) and ok
        if args.mode in ("fromscratch", "both"):
            ok = run_fromscratch(args, ckpt_cfg, device) and ok
    except Exception as exc:
        print(f"[{FRAMEWORK}] FAIL fatal: {exc}")
        traceback.print_exc()
        ok = False
    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
