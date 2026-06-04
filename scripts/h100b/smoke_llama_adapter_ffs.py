#!/usr/bin/env python3
"""Smoke checks for QwenPILlamaAdapterFFS.

The step-0 baseline is the same frozen VLM path with gated injection disabled.
That matters for reverse_image_order=True because the baseline must still use the
view-correct swapped cam_id that cam_rope will see in the real run.
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.multilayer_gated_inject import assert_gates_zero  # noqa: E402


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
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


def _cfg_select(cfg, key, default=None):
    from omegaconf import OmegaConf

    return OmegaConf.select(cfg, key, default=default)


def _parse_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got {text!r}")


def action_arch_from_cfg(cfg, label: str) -> tuple[bool, int]:
    interleave = _cfg_select(
        cfg,
        "framework.action_model.diffusion_model_cfg.interleave_self_attention",
        default=None,
    )
    num_layers = _cfg_select(
        cfg,
        "framework.action_model.diffusion_model_cfg.num_layers",
        default=None,
    )
    if interleave is None:
        raise AssertionError(f"{label}: missing action_model interleave_self_attention in config")
    if num_layers is None:
        raise AssertionError(f"{label}: missing action_model diffusion_model_cfg.num_layers in config")
    return bool(interleave), int(num_layers)


def assert_runtime_arch_matches(args, ckpt_arch: tuple[bool, int]) -> None:
    ckpt_interleave, ckpt_num_layers = ckpt_arch
    if ckpt_interleave is not True:
        raise AssertionError(
            "warm-start ckpt config says interleave_self_attention is not true; "
            "this smoke is for goal_phase3b_camrope_0523 and should fail closed."
        )
    runtime_interleave = _parse_bool(args.runtime_interleave_self_attention)
    runtime_num_layers = args.runtime_num_layers
    if runtime_num_layers is None:
        runtime_num_layers = ckpt_num_layers
    if runtime_interleave != ckpt_interleave or int(runtime_num_layers) != ckpt_num_layers:
        raise AssertionError(
            "runtime/launcher action_model config does not match ckpt: "
            f"runtime interleave={runtime_interleave}, num_layers={runtime_num_layers}; "
            f"ckpt interleave={ckpt_interleave}, num_layers={ckpt_num_layers}. "
            "Align the launcher with the warm-start checkpoint."
        )


def build_cfg_from_ckpt(args, ckpt_cfg, reverse_image_order: bool):
    from omegaconf import OmegaConf

    cfg = _clone_cfg(ckpt_cfg)
    OmegaConf.update(cfg, "framework.name", "QwenPILlamaAdapterFFS", force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", args.base_vlm, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.attn_implementation", args.attn_implementation, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_enabled", True, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_d_c", 16, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_num_cameras", 2, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_baseline_m", 0.06, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_fovy_degrees", 45.0, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_image_width", 256, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_image_height", 256, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_spatial_merge", 2, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_cam_rope_init_mode", "zero", force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False, force_add=True)
    OmegaConf.update(
        cfg,
        "framework.action_model.diffusion_model_cfg.interleave_self_attention",
        _parse_bool(args.runtime_interleave_self_attention),
        force_add=True,
    )
    if args.runtime_num_layers is not None:
        OmegaConf.update(
            cfg,
            "framework.action_model.diffusion_model_cfg.num_layers",
            int(args.runtime_num_layers),
            force_add=True,
        )
    OmegaConf.update(
        cfg,
        "framework.ffs_llama_adapter",
        {
            "ffs_model_path": args.ffs_model_path,
            "ffs_expected_sha256": args.ffs_expected_sha256,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
            "num_cameras": 2,
            "hint_hidden_dim": 256,
            "reverse_image_order": bool(reverse_image_order),
        },
        force_add=True,
    )
    OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", args.data_root, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.data_mix", args.data_mix, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.per_device_batch_size", 1, force_add=True)
    return cfg


def make_images_and_instruction():
    left = Image.new("RGB", (256, 256), color=(80, 120, 160))
    right = Image.new("RGB", (256, 256), color=(84, 120, 160))
    return [[left, right]], ["pick up the object"]


def make_forward_examples(model):
    batch_images, instructions = make_images_and_instruction()
    action_dim = int(model.config.framework.action_model.action_dim)
    horizon = int(model.action_horizon)
    action = [[0.0 for _ in range(action_dim)] for _ in range(horizon)]
    return [{"image": batch_images[0], "lang": instructions[0], "action": action}]


def freeze_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def grad_abs_sum(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


def assert_action_model_key_parity(model, ckpt) -> None:
    own = {k: v for k, v in model.state_dict().items() if k.startswith("action_model.")}
    provided = {k: v for k, v in ckpt.items() if k.startswith("action_model.")}
    missing = sorted(set(own) - set(provided))
    unexpected = sorted(set(provided) - set(own))
    if missing or unexpected:
        raise AssertionError(
            "action_model checkpoint key mismatch; "
            f"missing_first={missing[:20]}, unexpected_first={unexpected[:20]}. "
            "This usually means the runtime action-DiT architecture does not match the warm-start ckpt."
        )
    bad_shapes = []
    for key in sorted(own):
        if tuple(own[key].shape) != tuple(provided[key].shape):
            bad_shapes.append((key, tuple(own[key].shape), tuple(provided[key].shape)))
            if len(bad_shapes) >= 10:
                break
    if bad_shapes:
        raise AssertionError(f"action_model shape mismatch: {bad_shapes}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def assert_ffs_physical_input_order(model) -> None:
    batch_images, _ = make_images_and_instruction()
    calls = []
    device = next(model.parameters()).device
    orig_imgs_to_ffs_tensor = model._imgs_to_ffs_tensor
    orig_ffs_forward = model.ffs.forward

    def fake_imgs_to_ffs_tensor(batch, view_idx):
        calls.append(int(view_idx))
        return torch.full(
            (len(batch), 3, int(model.ffs_image_size), int(model.ffs_image_size)),
            fill_value=float(view_idx),
            device=device,
            dtype=torch.float32,
        )

    def fake_ffs_forward(image1, image2, *args, **kwargs):
        if float(image1.flatten()[0].item()) != float(model.primary_idx):
            raise AssertionError("FFS image1 is not the physical primary/left view")
        if float(image2.flatten()[0].item()) != float(model.right_view_idx):
            raise AssertionError("FFS image2 is not the physical right view")
        model._ffs_captured_net0 = torch.zeros(
            (int(image1.shape[0]), int(model.ffs_feat_dim), 4, 4),
            device=image1.device,
            dtype=torch.float32,
        )
        return None

    model._imgs_to_ffs_tensor = fake_imgs_to_ffs_tensor
    model.ffs.forward = fake_ffs_forward
    try:
        feat = model._compute_ffs_feature(batch_images)
    finally:
        model._imgs_to_ffs_tensor = orig_imgs_to_ffs_tensor
        model.ffs.forward = orig_ffs_forward

    expected = [int(model.primary_idx), int(model.right_view_idx)]
    if calls != expected:
        raise AssertionError(f"FFS view order changed: got {calls}, expected physical {expected}")
    if int(feat.shape[1]) != int(model.ffs_feat_dim):
        raise AssertionError(f"fake FFS feature channel mismatch: {tuple(feat.shape)}")


def assert_reverse_and_cam_state(model, expect_reverse: bool) -> None:
    state = model._llama_adapter_state
    cam = state.per_token_cam_id
    if cam is None:
        raise AssertionError("adapter state did not capture per_token_cam_id")
    rope_cam = getattr(model._stereo_cam_rope_state, "per_token_cam_id", None)
    if rope_cam is None:
        raise AssertionError("cam_rope_state.per_token_cam_id is None")
    if not torch.equal(cam.detach().cpu(), rope_cam.detach().cpu()):
        raise AssertionError("cam_rope_state did not receive the adapter-swapped cam_id tensor")

    cam0 = cam[0].detach().cpu()
    left_pos = (cam0 == int(model.primary_idx)).nonzero(as_tuple=True)[0]
    right_pos = (cam0 == int(model.right_view_idx)).nonzero(as_tuple=True)[0]
    if int(left_pos.numel()) == 0 or int(right_pos.numel()) == 0:
        raise AssertionError(f"missing left/right image tokens: left={left_pos.tolist()} right={right_pos.tolist()}")
    if expect_reverse:
        if int(right_pos.max()) >= int(left_pos.min()):
            raise AssertionError(
                f"reverse order broken: right block {right_pos.tolist()} is not before left block {left_pos.tolist()}"
            )
    else:
        if int(left_pos.max()) >= int(right_pos.min()):
            raise AssertionError(
                f"normal order broken: left block {left_pos.tolist()} is not before right block {right_pos.tolist()}"
            )

    h_tok, w_tok = state.primary_grid[0]
    if int(left_pos.numel()) != h_tok * w_tok:
        raise AssertionError(
            f"primary-grid count mismatch: left_tokens={int(left_pos.numel())}, h*w={h_tok * w_tok}"
        )


def assert_step0_parity(model, args, expect_reverse: bool) -> None:
    assert_gates_zero(model.gates)
    batch_images, instructions = make_images_and_instruction()
    model.eval()

    model._llama_adapter_state.enabled = False
    with torch.no_grad():
        ref = model._encode_vl_hidden_states(batch_images, instructions)
    model._llama_adapter_state.enabled = True
    with torch.no_grad():
        got = model._encode_vl_hidden_states(batch_images, instructions)

    if len(ref) != len(got):
        raise AssertionError(f"hidden-state count mismatch: {len(ref)} vs {len(got)}")
    for idx, (a, b) in enumerate(zip(ref, got)):
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"sequence/hidden shape changed at hidden state {idx}: {tuple(a.shape)} vs {tuple(b.shape)}")
        if not torch.allclose(a, b, atol=args.parity_atol, rtol=0.0):
            diff = float((a - b).abs().max().item())
            raise AssertionError(f"step-0 VLM parity failed at hidden state {idx}: max abs diff {diff}")

    assert_reverse_and_cam_state(model, expect_reverse=expect_reverse)

    examples = make_forward_examples(model)
    model._llama_adapter_state.enabled = False
    with torch.no_grad():
        torch.manual_seed(args.loss_seed)
        ref_loss = model(examples=examples)["action_loss"]
    model._llama_adapter_state.enabled = True
    with torch.no_grad():
        torch.manual_seed(args.loss_seed)
        got_loss = model(examples=examples)["action_loss"]
    if not torch.allclose(ref_loss, got_loss, atol=max(args.parity_atol, 1e-4), rtol=0.0):
        diff = float((ref_loss - got_loss).abs().max().item())
        raise AssertionError(f"step-0 action-loss parity failed: max abs diff {diff}")


def assert_direct_residual_left_only(model) -> None:
    batch_images, instructions = make_images_and_instruction()
    model.eval()
    with torch.no_grad():
        for gate in model.gates:
            gate.weight.zero_()
            gate.bias.fill_(0.125)
        model._llama_adapter_state.enabled = True
        _ = model._encode_vl_hidden_states(batch_images, instructions)

    state = model._llama_adapter_state
    cam = state.per_token_cam_id.detach().cpu()
    left_mask = cam == int(model.primary_idx)
    right_or_text = ~left_mask
    masks = state.last_residual_token_mask
    if masks is None:
        raise AssertionError("no residual masks were recorded")
    for idx, mask in enumerate(masks):
        if mask is None:
            raise AssertionError(f"gate[{idx}] did not record a residual mask")
        if bool(mask[right_or_text].any()):
            raise AssertionError(f"gate[{idx}] direct residual touched right-view or text tokens")
        if not torch.equal(mask[left_mask], torch.ones_like(mask[left_mask], dtype=torch.bool)):
            raise AssertionError(f"gate[{idx}] did not touch every left image token under non-zero bias")


def assert_backward_grad_flow(model) -> None:
    freeze_module_params(model.qwen_vl_interface)
    for param in model.ffs.parameters():
        param.requires_grad = False
    for param in model.hint.parameters():
        param.requires_grad = True
    for param in model.gates.parameters():
        param.requires_grad = True
    for param in model.action_model.parameters():
        param.requires_grad = True

    with torch.no_grad():
        for gate in model.gates:
            gate.weight.fill_(1.0e-5)
            gate.bias.fill_(1.0e-3)

    model.train()
    if model.ffs.training:
        raise AssertionError("frozen FFS module switched to train mode after model.train()")
    model.zero_grad(set_to_none=True)
    torch.manual_seed(777)
    loss = model(examples=make_forward_examples(model))["action_loss"]
    loss.backward()

    if grad_abs_sum(model.hint) <= 0.0:
        raise AssertionError("hint received no gradient after non-zero gate backward")
    if grad_abs_sum(model.gates) <= 0.0:
        raise AssertionError("gates received no gradient")
    if grad_abs_sum(model.action_model) <= 0.0:
        raise AssertionError("action_model received no gradient")

    leaked = []
    for name, param in model.qwen_vl_interface.named_parameters():
        if param.grad is not None and float(param.grad.detach().abs().sum().item()) != 0.0:
            leaked.append(name)
            if len(leaked) >= 5:
                break
    if leaked:
        raise AssertionError(f"frozen VLM params received non-zero grads: {leaked}")

    ffs_leaked = []
    for name, param in model.ffs.named_parameters():
        if param.grad is not None and float(param.grad.detach().abs().sum().item()) != 0.0:
            ffs_leaked.append(name)
            if len(ffs_leaked) >= 5:
                break
    if ffs_leaked:
        raise AssertionError(f"frozen FFS params received non-zero grads: {ffs_leaked}")


def build_loaded_model(args, ckpt_cfg, ckpt, reverse_image_order: bool):
    from starVLA.model.framework.base_framework import build_framework

    cfg = build_cfg_from_ckpt(args, ckpt_cfg, reverse_image_order=reverse_image_order)
    if action_arch_from_cfg(cfg, "runtime cfg") != action_arch_from_cfg(ckpt_cfg, "ckpt cfg"):
        raise AssertionError("runtime action architecture differs from checkpoint config")
    model = build_framework(cfg)
    if not bool(getattr(model, "_ffs_sha256_verified", False)):
        raise AssertionError("FFS SHA256 was not verified during model build")
    assert_action_model_key_parity(model, ckpt)
    model.load_state_dict(ckpt, strict=False)
    assert_gates_zero(model.gates)
    device = resolve_device(args.device)
    model.to(device)
    return model


def run_case(args, ckpt_cfg, ckpt, reverse_image_order: bool, deep_checks: bool) -> None:
    model = build_loaded_model(args, ckpt_cfg, ckpt, reverse_image_order=reverse_image_order)
    label = "reverse=true" if reverse_image_order else "reverse=false"
    print(f"[smoke] built {label} on {next(model.parameters()).device}")
    assert_ffs_physical_input_order(model)
    assert_step0_parity(model, args, expect_reverse=reverse_image_order)
    if deep_checks:
        assert_direct_residual_left_only(model)
        assert_backward_grad_flow(model)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    parser.add_argument("--runtime-interleave-self-attention", default="true")
    parser.add_argument("--runtime-num-layers", type=int, default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--parity-atol", type=float, default=0.0)
    args = parser.parse_args()

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    ckpt_arch = action_arch_from_cfg(ckpt_cfg, str(cfg_path))
    assert_runtime_arch_matches(args, ckpt_arch)
    ckpt = load_ckpt(args.pretrained_ckpt)

    ok = run("reverse_true_step0_reverse_direct_backward", lambda: run_case(args, ckpt_cfg, ckpt, True, True))
    ok = run("reverse_false_step0_direct_backward", lambda: run_case(args, ckpt_cfg, ckpt, False, True)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
