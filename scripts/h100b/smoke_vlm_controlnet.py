#!/usr/bin/env python3
"""Smoke checks for QwenPIVLMControlNetFFS.

The real-model path loads the warm-start checkpoint's saved config from the run
folder next to the .pt file, builds the plain QwenPI reference from that config,
and checks zero-conv step-0 parity through both the VLM hidden states and a full
action-loss forward. This catches action-DiT architecture mismatches such as an
interleave_self_attention value that differs from the checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.vlm_controlnet import (
    FFSControlNetHint,
    VLMControlNetBranch,
    clear_state,
    get_state,
    install_vlm_controlnet_hooks,
    set_ffs_feature,
)

IMAGE_TOKEN_ID = 32000


class TinyLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return self.self_attn(hidden_states), None


class TinyLanguageModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(dim), TinyLayer(dim)])

    def forward(self, *, inputs_embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, **kwargs)[0]
        return h


class TinyHFModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = SimpleNamespace(language_model=TinyLanguageModel(dim))

    def forward(self, *, input_ids, image_grid_thw, inputs_embeds, **kwargs):
        return self.model.language_model(inputs_embeds=inputs_embeds, **kwargs)


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


def _parse_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got {text!r}")


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


def build_cfg_from_ckpt(args, ckpt_cfg, framework_name: str):
    from omegaconf import OmegaConf

    cfg = _clone_cfg(ckpt_cfg)
    OmegaConf.update(cfg, "framework.name", framework_name, force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", args.base_vlm, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", args.data_root, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.data_mix", args.data_mix, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.per_device_batch_size", 1, force_add=True)
    if framework_name == "QwenPIVLMControlNetFFS":
        OmegaConf.update(
            cfg,
            "framework.ffs_vlm_controlnet",
            {
                "ffs_model_path": args.ffs_model_path,
                "ffs_expected_sha256": args.ffs_expected_sha256,
                "ffs_feature_source": "gru_hidden",
                "gru_hidden_dim": 16,
                "ffs_scale": 0,
                "ffs_image_size": 256,
                "primary_idx": 0,
                "right_view_idx": 1,
                "num_cameras": 2,
                "hint_hidden_dim": 256,
            },
            force_add=True,
        )
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


def assert_action_model_key_parity(model, ckpt, label: str) -> None:
    own = {k: v for k, v in model.state_dict().items() if k.startswith("action_model.")}
    provided = {k: v for k, v in ckpt.items() if k.startswith("action_model.")}
    missing = sorted(set(own) - set(provided))
    unexpected = sorted(set(provided) - set(own))
    if missing or unexpected:
        raise AssertionError(
            f"{label}: action_model checkpoint key mismatch; "
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
        raise AssertionError(f"{label}: action_model shape mismatch: {bad_shapes}")


def assert_action_model_param_equal(plain, control) -> None:
    ref = {k: v for k, v in plain.state_dict().items() if k.startswith("action_model.")}
    got = {k: v for k, v in control.state_dict().items() if k.startswith("action_model.")}
    if set(ref) != set(got):
        raise AssertionError("plain/control action_model key sets differ after load")
    for key in sorted(ref):
        if not torch.equal(ref[key].detach().cpu(), got[key].detach().cpu()):
            raise AssertionError(f"plain/control action_model param differs after load: {key}")


def assert_copy_init(model) -> None:
    trunk_layers = model._vlm_controlnet_trunk_layers()
    branch_layers = model.ffs_controlnet_branch.branch_layers
    for idx, (trunk, branch) in enumerate(zip(trunk_layers, branch_layers)):
        trunk_params = {
            k: v for k, v in trunk.named_parameters() if ".stereo_cam_layer." not in k
        }
        branch_params = {
            k: v for k, v in branch.named_parameters() if ".stereo_cam_layer." not in k
        }
        if set(trunk_params) != set(branch_params):
            raise AssertionError(f"copy-init param-name mismatch at branch layer {idx}")
        for name in trunk_params:
            b = branch_params[name]
            t = trunk_params[name].to(dtype=b.dtype, device=b.device)
            if not torch.equal(b, t):
                raise AssertionError(f"branch layer {idx} param {name} differs from trunk")
            if b.data_ptr() == trunk_params[name].data_ptr():
                raise AssertionError(f"branch layer {idx} param {name} shares storage with trunk")


def freeze_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def assert_backward_grad_flow(model, batch_images, instructions) -> None:
    model.train()
    freeze_module_params(model.qwen_vl_interface)
    for param in model.ffs.parameters():
        param.requires_grad = False
    model.zero_grad(set_to_none=True)
    vl = model._encode_vl_hidden_states(batch_images, instructions)
    loss = sum(h.float().square().mean() for h in vl[-2:])
    loss.backward()
    for idx, zero in enumerate(model.ffs_controlnet_branch.zero_convs):
        grad = zero.weight.grad
        if grad is None or float(grad.abs().sum()) <= 0.0:
            raise AssertionError(f"zero_conv[{idx}].weight grad is missing or zero")
    leaked = []
    for name, param in model.qwen_vl_interface.named_parameters():
        if param.grad is not None and float(param.grad.abs().sum()) != 0.0:
            leaked.append(name)
            if len(leaked) >= 5:
                break
    if leaked:
        raise AssertionError(f"frozen VLM params received non-zero grads: {leaked}")


def assert_step0_parity(args) -> None:
    from starVLA.model.framework.base_framework import build_framework

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    ckpt_arch = action_arch_from_cfg(ckpt_cfg, str(cfg_path))
    assert_runtime_arch_matches(args, ckpt_arch)

    plain_cfg = build_cfg_from_ckpt(args, ckpt_cfg, "QwenPI")
    control_cfg = build_cfg_from_ckpt(args, ckpt_cfg, "QwenPIVLMControlNetFFS")
    # CPU smoke: flash_attention_2 has no CPU kernel. Force eager attention
    # (math-equivalent; both models use the SAME impl so the control-vs-plain
    # step-0 parity is unaffected — the zero-convs still add exactly 0).
    from omegaconf import OmegaConf as _OC
    _OC.update(plain_cfg, "framework.qwenvl.attn_implementation", "eager", force_add=True)
    _OC.update(control_cfg, "framework.qwenvl.attn_implementation", "eager", force_add=True)
    if action_arch_from_cfg(plain_cfg, "plain runtime cfg") != ckpt_arch:
        raise AssertionError("plain runtime action architecture differs from ckpt config")
    if action_arch_from_cfg(control_cfg, "control runtime cfg") != ckpt_arch:
        raise AssertionError("control runtime action architecture differs from ckpt config")

    ckpt = load_ckpt(args.pretrained_ckpt)
    plain = build_framework(plain_cfg)
    control = build_framework(control_cfg)
    assert_action_model_key_parity(plain, ckpt, "plain QwenPI")
    assert_action_model_key_parity(control, ckpt, "ControlNet QwenPI")

    plain.load_state_dict(ckpt, strict=False)
    control.load_state_dict(ckpt, strict=False)
    assert_action_model_param_equal(plain, control)

    plain.eval()
    control.eval()
    # CPU smoke: training uses autocast(bf16) on GPU; here there is no autocast
    # and the built model is mixed-dtype (VLM bf16 / action_model fp32), which
    # mismatches on a bare CPU forward. Cast both models fully to fp32 — the
    # parity check is control-vs-plain (both fp32), so it is unaffected, and the
    # zero-convs still add exactly 0 at step 0.
    plain.float()
    control.float()
    batch_images, instructions = make_images_and_instruction()
    with torch.no_grad():
        ref = plain._encode_vl_hidden_states(batch_images, instructions)
        got = control._encode_vl_hidden_states(batch_images, instructions)
    if len(ref) != len(got):
        raise AssertionError(f"hidden-state count mismatch: {len(ref)} vs {len(got)}")
    for idx, (a, b) in enumerate(zip(ref, got)):
        if not torch.allclose(a, b, atol=args.parity_atol, rtol=0.0):
            diff = float((a - b).abs().max())
            raise AssertionError(f"step-0 VLM parity failed at hidden state {idx}: max abs diff {diff}")

    examples = make_forward_examples(plain)
    with torch.no_grad():
        torch.manual_seed(args.loss_seed)
        ref_loss = plain(examples=examples)["action_loss"]
        torch.manual_seed(args.loss_seed)
        got_loss = control(examples=examples)["action_loss"]
    if not torch.allclose(ref_loss, got_loss, atol=max(args.parity_atol, 1e-4), rtol=0.0):
        diff = float((ref_loss - got_loss).abs().max())
        raise AssertionError(f"step-0 action-loss parity failed: max abs diff {diff}")

    assert_copy_init(control)
    assert_backward_grad_flow(control, batch_images, instructions)


def assert_tiny_primary_token_coverage() -> None:
    torch.manual_seed(7)
    dim = 4
    model = TinyHFModel(dim)
    source = [copy.deepcopy(model.model.language_model.layers[0])]
    branch = VLMControlNetBranch(source, [0], dim, torch.float32)
    hint = FFSControlNetHint(in_ch=2, llm_dim=dim, hidden_dim=8)
    install_vlm_controlnet_hooks(
        model,
        branch=branch,
        hint=hint,
        inject_depths=[0],
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        image_token_id=IMAGE_TOKEN_ID,
    )
    input_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    embeds = torch.randn(1, input_ids.shape[1], dim)
    ffs_feat = torch.randn(1, 2, 3, 3)
    clear_state()
    set_ffs_feature(ffs_feat)
    _ = model(input_ids=input_ids, image_grid_thw=grid, inputs_embeds=embeds)
    state = get_state()
    primary = (state.per_token_cam_id[0] == 1).nonzero(as_tuple=True)[0]
    non_primary = (state.per_token_cam_id[0] == 0).nonzero(as_tuple=True)[0]
    if primary.tolist() != [6, 7, 8, 9]:
        raise AssertionError(f"primary rows wrong: {primary.tolist()}")
    if non_primary.tolist() != [1, 2, 3, 4]:
        raise AssertionError(f"non-primary rows wrong: {non_primary.tolist()}")

    bad_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    try:
        clear_state()
        set_ffs_feature(ffs_feat)
        model(input_ids=bad_ids, image_grid_thw=grid, inputs_embeds=torch.randn(1, bad_ids.shape[1], dim))
    except RuntimeError as exc:
        if "refusing to silently" not in str(exc):
            raise AssertionError(f"mismatch RuntimeError was not specific enough: {exc}") from exc
        return
    raise AssertionError("malformed primary-token grid did not fail loud")


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
    parser.add_argument("--pretrained-ckpt", default="./playground/Checkpoints/goal_phase3b_camrope_0523/checkpoints/steps_30000_pytorch_model.pt")
    parser.add_argument("--ffs-model-path", default="/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth")
    parser.add_argument("--ffs-expected-sha256", default="98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692")
    parser.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_STEREO_DATA")
    parser.add_argument("--data-mix", default="libero_goal_stereo")
    parser.add_argument("--runtime-interleave-self-attention", default="true")
    parser.add_argument("--runtime-num-layers", type=int, default=None)
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--parity-atol", type=float, default=0.0)
    parser.add_argument("--tiny-hooks-only", action="store_true")
    args = parser.parse_args()

    ok = run("tiny_primary_token_coverage", assert_tiny_primary_token_coverage)
    if not args.tiny_hooks_only:
        ok = run("real_step0_copyinit_backward", lambda: assert_step0_parity(args)) and ok
    clear_state()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
