#!/usr/bin/env python3
"""Smoke checks for QwenPI_DepthTokenLoRAFFS."""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h100b.smoke_depth_token import (  # noqa: E402
    assert_stereo_sensitivity,
    assert_tiny_hook_insertion,
    build_cfg_from_ckpt as build_depth_cfg_from_ckpt,
    clear_depth_state,
    get_depth_state,
    load_ckpt,
    load_ckpt_config,
    make_examples,
)


def is_lora_name(name: str) -> bool:
    return (
        ".lora_A." in name
        or ".lora_B." in name
        or ".lora_embedding_A" in name
        or ".lora_embedding_B" in name
    )


def build_lora_cfg_from_ckpt(args, ckpt_cfg):
    from omegaconf import OmegaConf

    cfg = build_depth_cfg_from_ckpt(args, ckpt_cfg, "QwenPIDepthTokenFFS")
    OmegaConf.update(cfg, "framework.name", "QwenPIDepthTokenLoRAFFS", force_add=True)
    OmegaConf.update(cfg, "framework.ffs_depth_token.vlm_lora", True, force_add=True)
    OmegaConf.update(cfg, "framework.ffs_depth_token.vlm_lora_r", args.lora_r, force_add=True)
    OmegaConf.update(cfg, "framework.ffs_depth_token.vlm_lora_alpha", args.lora_alpha, force_add=True)
    OmegaConf.update(cfg, "framework.ffs_depth_token.vlm_lora_dropout", args.lora_dropout, force_add=True)
    OmegaConf.update(
        cfg,
        "framework.ffs_depth_token.vlm_lora_target_modules",
        [x.strip() for x in args.lora_targets.split(",") if x.strip()],
        force_add=True,
    )
    OmegaConf.update(cfg, "trainer.freeze_modules", "", force_add=True)
    OmegaConf.update(cfg, "trainer.learning_rate.base", 2.5e-5, force_add=True)
    OmegaConf.update(cfg, "trainer.learning_rate.qwen_vl_interface", 1e-4, force_add=True)
    OmegaConf.update(cfg, "trainer.learning_rate.depth_token_projector", 1e-4, force_add=True)
    OmegaConf.update(cfg, "trainer.learning_rate.action_model", 1e-4, force_add=True)
    return cfg


def remap_ckpt_key_for_lora(model, ckpt_key: str):
    own_keys = set(model.state_dict().keys())
    prefix = "qwen_vl_interface.model."
    peft_prefix = prefix + "base_model.model."
    if not ckpt_key.startswith(prefix):
        return None
    remapped = peft_prefix + ckpt_key[len(prefix):]
    if remapped in own_keys:
        return remapped
    for target in model.vlm_lora_target_modules:
        needle = f".{target}."
        if needle in remapped and (remapped.endswith(".weight") or remapped.endswith(".bias")):
            candidate = remapped.replace(needle, f".{target}.base_layer.", 1)
            if candidate in own_keys:
                return candidate
    return None


def assert_lora_setup(model) -> None:
    lora_trainable = []
    base_trainable = []
    for name, param in model.qwen_vl_interface.named_parameters():
        if not param.requires_grad:
            continue
        if is_lora_name(name):
            lora_trainable.append((name, param))
        else:
            base_trainable.append(name)
            if len(base_trainable) >= 10:
                break
    if not lora_trainable:
        raise AssertionError("no trainable LoRA parameters found under qwen_vl_interface")
    if base_trainable:
        raise AssertionError(f"base VLM params are trainable: {base_trainable}")


def assert_warmstart_remap_loaded(model, ckpt) -> None:
    lora_keys = [k for k in model.state_dict().keys() if is_lora_name(k)]
    if not lora_keys:
        raise AssertionError("model state_dict has no LoRA adapter keys")
    if any(k in ckpt for k in lora_keys):
        raise AssertionError("warm-start checkpoint unexpectedly contains LoRA adapter keys")

    pair = None
    for ckpt_key, value in ckpt.items():
        if not ckpt_key.startswith("qwen_vl_interface.model.model.language_model.layers."):
            continue
        if not torch.is_tensor(value) or not (ckpt_key.endswith(".weight") or ckpt_key.endswith(".bias")):
            continue
        own_key = remap_ckpt_key_for_lora(model, ckpt_key)
        if own_key is not None and tuple(model.state_dict()[own_key].shape) == tuple(value.shape):
            pair = (ckpt_key, own_key)
            break
    if pair is None:
        raise AssertionError("could not find a baseline VLM key that remaps into the PEFT model")

    ckpt_key, own_key = pair
    expected = ckpt[ckpt_key].detach().cpu().float()
    got = model.state_dict()[own_key].detach().cpu().float()
    if not torch.allclose(got, expected, rtol=1e-3, atol=1e-3):
        diff = float((got - expected).abs().max())
        raise AssertionError(f"remapped VLM key did not load from ckpt: {ckpt_key} -> {own_key}, max_diff={diff}")


def assert_peft_resume_and_stereo_optional(model) -> None:
    current = model.state_dict()
    model.load_state_dict(current, strict=True, init_from_baseline=False)

    stereo_keys = [
        key
        for key in current.keys()
        if ".language_model.layers." in key and ".self_attn.stereo_cam_layer." in key
    ]
    if not stereo_keys:
        raise AssertionError("no PEFT-prefixed stereo cam-rope keys found in LoRA model state_dict")

    missing_key = stereo_keys[0]
    partial = OrderedDict((key, value) for key, value in current.items() if key != missing_key)
    model.load_state_dict(partial, strict=True, init_from_baseline=True)
    model.load_state_dict(current, strict=True, init_from_baseline=False)


def assert_lora_projector_backward(model) -> None:
    model.train()
    for param in model.ffs.parameters():
        param.requires_grad = False
    for param in model.depth_token_projector.parameters():
        param.requires_grad = True

    examples = make_examples(model)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    loss = model(examples=examples)["action_loss"]
    if not torch.isfinite(loss.detach()):
        raise AssertionError(f"non-finite action loss: {loss.detach().cpu().item()}")
    loss.backward()

    lora_grad_sum = 0.0
    base_grad_leaks = []
    for name, param in model.qwen_vl_interface.named_parameters():
        grad_sum = 0.0
        if param.grad is not None:
            grad_sum = float(param.grad.detach().abs().sum().cpu())
        if is_lora_name(name):
            lora_grad_sum += grad_sum
        elif grad_sum != 0.0:
            base_grad_leaks.append(name)
            if len(base_grad_leaks) >= 5:
                break
    if lora_grad_sum <= 0.0:
        raise AssertionError("LoRA parameters received zero gradient")
    if base_grad_leaks:
        raise AssertionError(f"frozen base VLM params received non-zero grads: {base_grad_leaks}")

    projector_grad_sum = 0.0
    for param in model.depth_token_projector.parameters():
        if param.grad is not None:
            projector_grad_sum += float(param.grad.detach().abs().sum().cpu())
    if projector_grad_sum <= 0.0:
        raise AssertionError("depth_token_projector received zero gradient")


def assert_lr_groups(model, cfg) -> None:
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    groups = build_param_lr_groups(model=model, cfg=cfg)
    by_name = {group["name"]: group for group in groups}
    for name in ("qwen_vl_interface", "depth_token_projector"):
        if name not in by_name:
            raise AssertionError(f"missing optimizer LR group: {name}")

    qwen_param_ids = {id(param) for param in by_name["qwen_vl_interface"]["params"]}
    depth_param_ids = {id(param) for param in by_name["depth_token_projector"]["params"]}
    if qwen_param_ids & depth_param_ids:
        raise AssertionError("qwen_vl_interface and depth_token_projector LR groups overlap")

    lora_param_ids = {
        id(param)
        for name, param in model.qwen_vl_interface.named_parameters()
        if param.requires_grad and is_lora_name(name)
    }
    if not lora_param_ids or not lora_param_ids.issubset(qwen_param_ids):
        raise AssertionError("qwen_vl_interface LR group does not contain all trainable LoRA params")


def assert_real_model(args) -> None:
    from starVLA.model.framework.base_framework import build_framework

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    ckpt = load_ckpt(args.pretrained_ckpt)
    plain_cfg = build_depth_cfg_from_ckpt(args, ckpt_cfg, "QwenPI")
    lora_cfg = build_lora_cfg_from_ckpt(args, ckpt_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    plain = build_framework(plain_cfg)
    plain.load_state_dict(ckpt, strict=False)
    plain.to(device).float().eval()
    examples_for_plain = make_examples(plain)
    with torch.no_grad():
        ref_vl = plain._encode_vl_hidden_states(
            [examples_for_plain[0]["image"]], [examples_for_plain[0]["lang"]]
        )
    baseline_s = int(ref_vl[-1].shape[1])
    del ref_vl, plain
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    depth = build_framework(lora_cfg)
    if type(depth).__name__ != "QwenPI_DepthTokenLoRAFFS":
        raise AssertionError(f"wrong framework class: {type(depth).__name__}")
    assert_lora_setup(depth)
    depth.load_state_dict(ckpt, strict=True)
    assert_warmstart_remap_loaded(depth, ckpt)
    assert_peft_resume_and_stereo_optional(depth)
    depth.to(device).float().eval()

    examples = make_examples(depth)
    clear_depth_state()
    with torch.no_grad():
        got_vl = depth._encode_vl_hidden_states([examples[0]["image"]], [examples[0]["lang"]])
    state = get_depth_state()
    if state.keep_mask is None:
        raise AssertionError("real LoRA depth-token forward did not write keep_mask")
    if state.keep_mask.shape[1] != baseline_s + 16:
        raise AssertionError(
            f"LM expanded length {state.keep_mask.shape[1]} != baseline S+16 ({baseline_s + 16})"
        )
    false_per_sample = (~state.keep_mask).sum(dim=1).detach().cpu().tolist()
    if false_per_sample != [16]:
        raise AssertionError(f"keep_mask false counts are not [16]: {false_per_sample}")
    for idx, hidden in enumerate(got_vl):
        if int(hidden.shape[1]) != baseline_s:
            raise AssertionError(
                f"action expert hidden state {idx} sees seq {hidden.shape[1]}, expected {baseline_s}"
            )

    assert_lora_projector_backward(depth)
    assert_lr_groups(depth, lora_cfg)
    assert_stereo_sensitivity(depth)
    clear_depth_state()

    print(f"real LoRA model smoke used config: {cfg_path}")


def run(name, fn) -> bool:
    try:
        fn()
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        traceback.print_exc()
        return False
    print(f"PASS {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vlm", default="./playground/Pretrained_models/Qwen3.5-0.8B")
    parser.add_argument(
        "--pretrained-ckpt",
        default="./playground/Checkpoints/goal_phase3b_camrope_0523/checkpoints/steps_30000_pytorch_model.pt",
    )
    parser.add_argument(
        "--ffs-model-path",
        default="/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--ffs-expected-sha256",
        default="98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692",
    )
    parser.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_STEREO_DATA")
    parser.add_argument("--data-mix", default="libero_goal_stereo")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--tiny-hooks-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ok = run("tiny_depth_token_hooks", assert_tiny_hook_insertion)
    if not args.tiny_hooks_only:
        ok = run("real_depthtoken_lora_framework", lambda: assert_real_model(args)) and ok
    clear_depth_state()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
