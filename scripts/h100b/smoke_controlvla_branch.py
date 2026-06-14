#!/usr/bin/env python3
"""Smoke tests for GR00T #7 ControlVLA K/V branch.

The default path is CPU-safe and uses a tiny GR00T DiT to validate the branch
mechanics without loading Qwen or Fast-FoundationStereo weights. Full model
training/eval still needs the real h100b environment.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def make_tiny_dit(interleave: bool = True):
    from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

    return DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",
        max_num_positional_embeddings=32,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=interleave,
        cross_attention_dim=6,
        norm_type="ada_norm",
    )


def fixed_inputs(batch: int = 4):
    torch.manual_seed(1234)
    hidden = torch.randn(batch, 5, 8)
    encoder = torch.randn(batch, 7, 6)
    timestep = torch.arange(batch, dtype=torch.long) + 3
    ffs = torch.randn(2, 4, 3)
    return hidden, encoder, timestep, ffs


def test_import_register_and_key_prefixes():
    from starVLA.model.framework.VLM4A.QwenGR00T_ControlVLAFFS import QwenGR00T_ControlVLAFFS
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    from starVLA.model.modules.stereo.controlvla_branch import install_controlvla_branches

    assert "QwenGR00T_ControlVLAFFS" in FRAMEWORK_REGISTRY._registry

    fake = object.__new__(QwenGR00T_ControlVLAFFS)
    nn.Module.__init__(fake)
    fake.action_model = nn.Module()
    fake.action_model.model = make_tiny_dit(interleave=True)
    install_controlvla_branches(
        fake.action_model.model,
        ffs_token_dim=3,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    expected = (
        "ffs.",
        "ffs_pos_emb",
        "action_model.model.transformer_blocks.0.attn1.to_k_z.",
        "action_model.model.transformer_blocks.0.attn1.to_v_z.",
        "action_model.model.transformer_blocks.2.attn1.to_k_z.",
        "action_model.model.transformer_blocks.2.attn1.to_v_z.",
    )
    assert fake._ffs_key_prefixes() == expected


def test_dit_step0_branch_and_alignment():
    from starVLA.model.modules.stereo.controlvla_branch import (
        clear_ffs_tokens,
        install_controlvla_branches,
        set_ffs_tokens,
    )

    torch.manual_seed(1)
    baseline = make_tiny_dit(interleave=True).eval()
    branch = make_tiny_dit(interleave=True).eval()
    branch.load_state_dict(baseline.state_dict())

    n_patched = install_controlvla_branches(
        branch,
        ffs_token_dim=3,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    cross_attn_count = sum(1 for block in branch.transformer_blocks if block.cross_attention_dim is not None)
    assert n_patched == cross_attn_count == 2

    for block in branch.transformer_blocks:
        if hasattr(block.attn1, "to_k_z"):
            assert torch.count_nonzero(block.attn1.to_k_z.weight).item() == 0
            assert torch.count_nonzero(block.attn1.to_v_z.weight).item() == 0

    hidden, encoder, timestep, ffs = fixed_inputs(batch=4)
    captured = {}

    def capture_to_k_z(_module, inputs):
        captured["ffs"] = inputs[0].detach().clone()

    first_branch = next(block.attn1 for block in branch.transformer_blocks if hasattr(block.attn1, "to_k_z"))
    handle = first_branch.to_k_z.register_forward_pre_hook(capture_to_k_z)
    try:
        clear_ffs_tokens()
        set_ffs_tokens(ffs)
        torch.manual_seed(999)
        out_branch = branch(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            timestep=timestep,
        )
        clear_ffs_tokens()
        torch.manual_seed(999)
        out_baseline = baseline(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            timestep=timestep,
        )
    finally:
        handle.remove()
        clear_ffs_tokens()

    assert torch.equal(out_branch, out_baseline), "zero-init branch must be bit-identical to baseline"
    assert "ffs" in captured
    expected_repeat = ffs.repeat(2, 1, 1)
    assert torch.equal(captured["ffs"], expected_repeat), "FFS repeat must be [s0,s1,s0,s1]"

    for block in branch.transformer_blocks:
        if hasattr(block.attn1, "to_k_z"):
            torch.manual_seed(200 + id(block) % 97)
            block.attn1.to_k_z.weight.data.normal_(0.0, 0.2)
            block.attn1.to_v_z.weight.data.normal_(0.0, 0.2)

    set_ffs_tokens(ffs)
    out_live = branch(hidden_states=hidden, encoder_hidden_states=encoder, timestep=timestep)
    clear_ffs_tokens()
    assert not torch.equal(out_live, out_baseline), "non-zero branch weights must change the DiT output"

    try:
        branch(hidden_states=hidden, encoder_hidden_states=encoder, timestep=timestep)
    except RuntimeError as exc:
        assert "ffs_tokens is None" in str(exc)
    else:
        raise AssertionError("missing set_ffs_tokens() did not raise")


def test_remap_preserves_trunk():
    from starVLA.model.framework.VLM4A.QwenGR00T_ControlVLAFFS import _remap_legacy_attn1_keys
    from starVLA.model.modules.stereo.controlvla_branch import install_controlvla_branches

    torch.manual_seed(11)
    baseline = make_tiny_dit(interleave=True)
    branch = make_tiny_dit(interleave=True)
    install_controlvla_branches(
        branch,
        ffs_token_dim=3,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    remapped, n_legacy = _remap_legacy_attn1_keys(
        baseline.state_dict(),
        own_keys=set(branch.state_dict().keys()),
    )
    missing, unexpected = branch.load_state_dict(remapped, strict=False)
    assert all(("to_k_z" in key or "to_v_z" in key) for key in missing), missing
    assert not unexpected, unexpected
    assert n_legacy > 0

    for idx, block in enumerate(branch.transformer_blocks):
        if hasattr(block.attn1, "base"):
            legacy_prefix = f"transformer_blocks.{idx}.attn1"
            assert torch.equal(
                block.attn1.base.to_q.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_q.weight"],
            )
            assert torch.equal(
                block.attn1.base.to_k.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_k.weight"],
            )
            assert torch.equal(
                block.attn1.base.to_v.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_v.weight"],
            )
            assert torch.equal(
                block.attn1.base.to_out[0].weight,
                baseline.state_dict()[f"{legacy_prefix}.to_out.0.weight"],
            )
            assert torch.equal(
                block.attn1.base.to_out[0].bias,
                baseline.state_dict()[f"{legacy_prefix}.to_out.0.bias"],
            )
        else:
            legacy_prefix = f"transformer_blocks.{idx}.attn1"
            assert torch.equal(
                block.attn1.to_q.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_q.weight"],
            )
            assert torch.equal(
                block.attn1.to_k.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_k.weight"],
            )
            assert torch.equal(
                block.attn1.to_v.weight,
                baseline.state_dict()[f"{legacy_prefix}.to_v.weight"],
            )
            assert torch.equal(
                block.attn1.to_out[0].weight,
                baseline.state_dict()[f"{legacy_prefix}.to_out.0.weight"],
            )
            assert torch.equal(
                block.attn1.to_out[0].bias,
                baseline.state_dict()[f"{legacy_prefix}.to_out.0.bias"],
            )


def fake_framework_instance(raise_in_action: bool = False, predict: bool = False):
    from starVLA.model.framework.VLM4A.QwenGR00T_ControlVLAFFS import QwenGR00T_ControlVLAFFS

    fake = object.__new__(QwenGR00T_ControlVLAFFS)
    nn.Module.__init__(fake)
    fake.action_horizon = 2
    fake.config = types.SimpleNamespace(
        framework=types.SimpleNamespace(action_model={"repeated_diffusion_steps": 1}),
        datasets=types.SimpleNamespace(vla_data=types.SimpleNamespace(obs_image_size=None)),
    )
    fake._encode_last_hidden_plain = lambda batch_images, instructions: torch.zeros(2, 3, 6)
    fake._compute_ffs_tokens = lambda batch_images: torch.ones(2, 4, 3)

    class ActionModel:
        def __call__(self, *args, **kwargs):
            from starVLA.model.modules.stereo.controlvla_branch import get_state

            assert get_state().ffs_tokens is not None
            if raise_in_action:
                raise RuntimeError("synthetic forward failure")
            return torch.tensor(0.0)

        def predict_action(self, last_hidden, state):
            from starVLA.model.modules.stereo.controlvla_branch import get_state

            assert get_state().ffs_tokens is not None
            if raise_in_action:
                raise RuntimeError("synthetic predict failure")
            return torch.zeros(last_hidden.shape[0], 2, 1)

    fake.action_model = ActionModel()
    return fake


def test_framework_try_finally_clears_state():
    from starVLA.model.framework.VLM4A.QwenGR00T_ControlVLAFFS import QwenGR00T_ControlVLAFFS
    from starVLA.model.modules.stereo.controlvla_branch import get_state

    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
    examples = [
        {
            "image": image,
            "lang": "test",
            "action": np.zeros((2, 1), dtype=np.float32),
            "state": np.zeros((1, 1), dtype=np.float32),
        },
        {
            "image": image,
            "lang": "test",
            "action": np.ones((2, 1), dtype=np.float32),
            "state": np.zeros((1, 1), dtype=np.float32),
        },
    ]

    ok = fake_framework_instance(raise_in_action=False)
    QwenGR00T_ControlVLAFFS.forward(ok, examples)
    assert get_state().ffs_tokens is None

    bad = fake_framework_instance(raise_in_action=True)
    try:
        QwenGR00T_ControlVLAFFS.forward(bad, examples)
    except RuntimeError as exc:
        assert "synthetic forward failure" in str(exc)
    else:
        raise AssertionError("synthetic action failure did not propagate")
    assert get_state().ffs_tokens is None

    bad_predict = fake_framework_instance(raise_in_action=True, predict=True)
    try:
        QwenGR00T_ControlVLAFFS.predict_action(bad_predict, examples)
    except RuntimeError as exc:
        assert "synthetic predict failure" in str(exc)
    else:
        raise AssertionError("synthetic predict failure did not propagate")
    assert get_state().ffs_tokens is None


def test_train_only_optimizer_exact_allowlist():
    from starVLA.model.modules.stereo.controlvla_branch import install_controlvla_branches
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    class TinyWarmStart(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffs_pos_emb = nn.Parameter(torch.zeros(1, 4, 3))
            self.action_model = nn.Module()
            self.action_model.model = make_tiny_dit(interleave=True)
            install_controlvla_branches(
                self.action_model.model,
                ffs_token_dim=3,
                expected_n_total_blocks=4,
                require_all_cross_attn=False,
            )
            self.trunk = nn.Linear(3, 3)

    model = TinyWarmStart()
    train_only = "ffs_pos_emb,.attn1.to_k_z.,.attn1.to_v_z."
    cfg = types.SimpleNamespace(
        trainer=DotDict(
            {
                "learning_rate": DotDict({"base": 1e-4}),
                "freeze_modules": "",
                "train_only": train_only,
            }
        )
    )
    groups = build_param_lr_groups(model, cfg)
    opt_names = {name for group in groups for name in group.get("param_names", [])}
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    allowed_tokens = ("to_k_z", "to_v_z", "ffs_pos_emb")
    for name, _param in model.named_parameters():
        if any(token in name for token in allowed_tokens):
            assert name in opt_names, f"{name} should be in train_only optimizer groups"
            assert name in trainable, f"{name} should remain trainable by train_only"
        else:
            assert name not in opt_names, f"{name} should not be in train_only optimizer groups"
            assert name not in trainable, f"{name} should remain frozen by train_only"

    trunk_k_names = [
        name for name, _param in model.named_parameters() if name.endswith(".attn1.base.to_k.weight")
    ]
    assert trunk_k_names
    assert all(name not in opt_names for name in trunk_k_names)
    assert "trunk.weight" not in opt_names
    assert any(name.endswith("attn1.to_k_z.weight") for name in opt_names)
    assert any(name.endswith("attn1.to_v_z.weight") for name in opt_names)
    assert "ffs_pos_emb" in opt_names


def main() -> None:
    tests = [
        test_import_register_and_key_prefixes,
        test_dit_step0_branch_and_alignment,
        test_remap_preserves_trunk,
        test_framework_try_finally_clears_state,
        test_train_only_optimizer_exact_allowlist,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("[PASS] ControlVLA branch smoke complete")


if __name__ == "__main__":
    main()
