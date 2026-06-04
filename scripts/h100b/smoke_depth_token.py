#!/usr/bin/env python3
"""Smoke checks for QwenPI_DepthTokenFFS.

Default mode builds the real QwenPI_DepthTokenFFS framework with the real FFS
checkpoint and the warm-start run config, then checks that depth tokens are
inserted inside the VLM sequence and sliced out before the action head. Use
--tiny-hooks-only for a fast local hook bookkeeping check.
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.depth_token_inject import (  # noqa: E402
    clear_state as clear_depth_state,
    get_state as get_depth_state,
    install_depth_token_hooks,
    set_depth_tokens,
)

IMAGE_TOKEN_ID = 32000


class TinyLanguageModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim)])
        self.last_inputs_embeds = None
        self.last_position_ids = None
        self.last_attention_mask = None

    def forward(self, *, inputs_embeds, position_ids=None, attention_mask=None, **kwargs):
        self.last_inputs_embeds = inputs_embeds
        self.last_position_ids = position_ids
        self.last_attention_mask = attention_mask
        return self.layers[0](inputs_embeds)


class TinyHFModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = SimpleNamespace(language_model=TinyLanguageModel(dim))

    def forward(self, *, input_ids, image_grid_thw, inputs_embeds, position_ids, attention_mask):
        return self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _ckpt_run_dir(ckpt_path: str) -> Path:
    path = Path(ckpt_path).expanduser().resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def load_ckpt_config(ckpt_path: str):
    from omegaconf import OmegaConf

    run_dir = _ckpt_run_dir(ckpt_path)
    for name in ("config.full.yaml", "config.yaml"):
        cfg_path = run_dir / name
        if cfg_path.is_file():
            return OmegaConf.load(cfg_path), cfg_path
    raise FileNotFoundError(
        f"Could not find config.full.yaml or config.yaml next to checkpoint run dir: {run_dir}"
    )


def _clone_cfg(cfg):
    from omegaconf import OmegaConf

    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def build_cfg_from_ckpt(args, ckpt_cfg, framework_name: str):
    from omegaconf import OmegaConf

    cfg = _clone_cfg(ckpt_cfg)
    OmegaConf.update(cfg, "framework.name", framework_name, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", args.base_vlm, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.attn_implementation", "eager", force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_enabled", True, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False, force_add=True)
    OmegaConf.update(
        cfg,
        "framework.action_model.diffusion_model_cfg.interleave_self_attention",
        True,
        force_add=True,
    )
    OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", args.data_root, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.data_mix", args.data_mix, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.per_device_batch_size", 1, force_add=True)
    if framework_name == "QwenPIDepthTokenFFS":
        OmegaConf.update(
            cfg,
            "framework.ffs_depth_token",
            {
                "ffs_model_path": args.ffs_model_path,
                "ffs_expected_sha256": args.ffs_expected_sha256,
                "ffs_feature_source": "gru_hidden",
                "gru_hidden_dim": 16,
                "ffs_image_size": 256,
                "primary_idx": 0,
                "right_view_idx": 1,
                "num_cameras": 2,
                "num_depth_tokens": 16,
                "pool_hw": 4,
            },
            force_add=True,
        )
    return cfg


def make_pattern_images(seed: int = 11):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    primary = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    right_a = np.roll(primary, shift=5, axis=1)
    right_b = np.flip(primary, axis=1).copy()
    left = Image.fromarray(primary, mode="RGB")
    return left, Image.fromarray(right_a, mode="RGB"), Image.fromarray(right_b, mode="RGB")


def make_examples(model, right_variant: int = 0):
    import numpy as np

    left, right_a, right_b = make_pattern_images()
    right = right_a if right_variant == 0 else right_b
    action_dim = int(model.config.framework.action_model.action_dim)
    horizon = int(model.action_horizon)
    action = np.zeros((horizon, action_dim), dtype=np.float32)
    state_dim = int(model.config.framework.action_model.get("state_dim", action_dim))
    state = np.zeros((1, state_dim), dtype=np.float32)
    return [
        {
            "image": [left, right],
            "lang": "pick up the object",
            "action": action,
            "state": state,
        }
    ]


def freeze_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def assert_tiny_hook_insertion() -> None:
    torch.manual_seed(5)
    clear_depth_state()
    dim = 8
    model = TinyHFModel(dim)
    cam_state = SimpleNamespace(
        per_token_cam_id=None,
        per_token_row_id=None,
        epipolar_mask_enabled=False,
    )
    install_depth_token_hooks(
        model,
        lm=model.model.language_model,
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=0,
        cam_rope_state=cam_state,
        image_token_id=IMAGE_TOKEN_ID,
    )
    input_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    embeds = torch.randn(1, input_ids.shape[1], dim)
    position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1).clone()
    attention_mask = torch.ones_like(input_ids)
    depth = torch.randn(1, 16, dim, requires_grad=True)
    set_depth_tokens(depth)
    out = model(
        input_ids=input_ids,
        image_grid_thw=grid,
        inputs_embeds=embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    loss = out.square().mean()
    loss.backward()
    lm = model.model.language_model
    state = get_depth_state()
    if lm.last_inputs_embeds.shape[1] != input_ids.shape[1] + 16:
        raise AssertionError("tiny LM did not see S+16 inputs_embeds")
    if lm.last_position_ids.shape[-1] != input_ids.shape[1] + 16:
        raise AssertionError("tiny LM did not see S+16 position_ids")
    if lm.last_attention_mask.shape[1] != input_ids.shape[1] + 16:
        raise AssertionError("tiny LM did not see S+16 attention_mask")
    if state.keep_mask is None or state.keep_mask.shape != (1, input_ids.shape[1] + 16):
        raise AssertionError("keep_mask was not written with expanded length")
    if int((~state.keep_mask).sum().item()) != 16:
        raise AssertionError("keep_mask does not mark exactly 16 inserted tokens")
    # HIGH-2 (2026-06-03): depth tokens now inserted BEFORE all image tokens. Tiny input is
    # [text, IMG x4 (primary), text, IMG x4 (right), text]; first image token is index 1, so the
    # 16 inserted depth tokens occupy [1:17].
    if state.keep_mask[0, 1:17].any():
        raise AssertionError("inserted token span is not right before the first image token")
    if cam_state.per_token_cam_id.shape[1] != input_ids.shape[1] + 16:
        raise AssertionError("cam_rope state was not expanded to S+16")
    if not torch.equal(cam_state.per_token_cam_id[0, 1:17], torch.full((16,), -1)):
        raise AssertionError("inserted depth tokens are not text-like cam_id=-1")
    if depth.grad is None or float(depth.grad.abs().sum()) <= 0.0:
        raise AssertionError("tiny inserted depth tokens did not receive gradient")
    clear_depth_state()


def assert_projector_grads(model) -> None:
    model.train()
    freeze_module_params(model.qwen_vl_interface)
    for param in model.ffs.parameters():
        param.requires_grad = False
    for param in model.depth_token_projector.parameters():
        param.requires_grad = True
    if any(param.requires_grad for param in model.qwen_vl_interface.parameters()):
        raise AssertionError("VLM freeze failed")
    if not all(param.requires_grad for param in model.depth_token_projector.parameters()):
        raise AssertionError("depth_token_projector is not trainable")

    examples = make_examples(model)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    loss = model(examples=examples)["action_loss"]
    if not torch.isfinite(loss.detach()):
        raise AssertionError(f"non-finite action loss: {loss.detach().cpu().item()}")
    loss.backward()
    grad_sum = 0.0
    for param in model.depth_token_projector.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.detach().abs().sum().cpu())
    if grad_sum <= 0.0:
        raise AssertionError("depth_token_projector received zero gradient")
    leaked = []
    for name, param in model.qwen_vl_interface.named_parameters():
        if param.grad is not None and float(param.grad.detach().abs().sum().cpu()) != 0.0:
            leaked.append(name)
            if len(leaked) >= 5:
                break
    if leaked:
        raise AssertionError(f"frozen VLM params received non-zero grads: {leaked}")


def assert_stereo_sensitivity(model) -> None:
    examples_a = make_examples(model, right_variant=0)
    examples_b = make_examples(model, right_variant=1)
    images_a = [examples_a[0]["image"]]
    images_b = [examples_b[0]["image"]]
    net_a = model._compute_ffs_feature(images_a)
    net_b = model._compute_ffs_feature(images_b)
    net_diff = float((net_a - net_b).abs().max().detach().cpu())
    if net_diff <= 0.0:
        raise AssertionError("FFS net[0] did not change when only the right view changed")

    saved = {k: v.detach().clone() for k, v in model.depth_token_projector.state_dict().items()}
    try:
        with torch.no_grad():
            model.depth_token_projector.proj.weight.fill_(1.0 / float(model.ffs_feat_dim))
            model.depth_token_projector.proj.bias.zero_()
        tok_a = model.depth_token_projector(net_a)
        tok_b = model.depth_token_projector(net_b)
        tok_diff = float((tok_a - tok_b).abs().max().detach().cpu())
        if tok_diff <= 0.0:
            raise AssertionError("projected depth tokens did not change under a non-zero projector")
    finally:
        model.depth_token_projector.load_state_dict(saved, strict=True)


def assert_real_model(args) -> None:
    from starVLA.model.framework.base_framework import build_framework

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    ckpt = load_ckpt(args.pretrained_ckpt)
    plain_cfg = build_cfg_from_ckpt(args, ckpt_cfg, "QwenPI")
    depth_cfg = build_cfg_from_ckpt(args, ckpt_cfg, "QwenPIDepthTokenFFS")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    plain = build_framework(plain_cfg)
    plain.load_state_dict(ckpt, strict=False)
    plain.to(device).float().eval()
    examples_for_plain = make_examples(plain)
    with torch.no_grad():
        ref_vl = plain._encode_vl_hidden_states(
            [examples_for_plain[0]["image"]], [examples_for_plain[0]["lang"]]
        )
        torch.manual_seed(args.loss_seed)
        ref_loss = plain(examples=examples_for_plain)["action_loss"].detach().float().cpu()
    baseline_s = int(ref_vl[-1].shape[1])
    del ref_vl, plain
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    depth = build_framework(depth_cfg)
    if type(depth).__name__ != "QwenPI_DepthTokenFFS":
        raise AssertionError(f"wrong framework class: {type(depth).__name__}")
    depth.load_state_dict(ckpt, strict=False)
    depth.to(device).float().eval()
    examples = make_examples(depth)
    with torch.no_grad():
        got_vl = depth._encode_vl_hidden_states([examples[0]["image"]], [examples[0]["lang"]])
    state = get_depth_state()
    if state.keep_mask is None:
        raise AssertionError("real depth-token forward did not write keep_mask")
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

    with torch.no_grad():
        torch.manual_seed(args.loss_seed)
        got_loss = depth(examples=examples)["action_loss"].detach().float().cpu()
    loss_diff = float((got_loss - ref_loss).abs().max())
    print(
        "step-0 note: inserted zero depth tokens can still perturb attention; "
        f"action-loss abs diff vs no-depth reference = {loss_diff:.6g}"
    )

    assert_projector_grads(depth)
    assert_stereo_sensitivity(depth)
    clear_depth_state()

    print(f"real model smoke used config: {cfg_path}")


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
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--tiny-hooks-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ok = run("tiny_depth_token_hooks", assert_tiny_hook_insertion)
    if not args.tiny_hooks_only:
        ok = run("real_depthtoken_framework", lambda: assert_real_model(args)) and ok
    clear_depth_state()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
