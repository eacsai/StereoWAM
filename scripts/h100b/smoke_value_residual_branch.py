#!/usr/bin/env python3
"""Smoke tests for GR00T #8 ValueResidual FFS branch.

The default path is CPU-safe and uses tiny DiT/action-head instances. Full
Qwen + Fast-FoundationStereo execution still belongs on h100b with real weights.
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


def make_tiny_action_head():
    import starVLA.model.modules.action_model.GR00T_ActionHeader as ah

    old = dict(ah.DiTConfig["DiT-B"])
    ah.DiTConfig["DiT-B"] = {
        "input_embedding_dim": 8,
        "attention_head_dim": 4,
        "num_attention_heads": 2,
    }
    try:
        cfg = types.SimpleNamespace(
            framework=types.SimpleNamespace(
                action_model=DotDict(
                    {
                        "action_model_type": "DiT-B",
                        "diffusion_model_cfg": {
                            "cross_attention_dim": 6,
                            "dropout": 0.0,
                            "final_dropout": False,
                            "interleave_self_attention": True,
                            "norm_type": "ada_norm",
                            "num_layers": 4,
                            "output_dim": 8,
                            "positional_embeddings": None,
                        },
                        "action_horizon": 3,
                        "action_dim": 2,
                        "state_dim": 2,
                        "num_inference_timesteps": 3,
                        "num_target_vision_tokens": 2,
                        "hidden_size": 8,
                        "add_pos_embed": True,
                        "max_seq_len": 16,
                        "noise_beta_alpha": 1.5,
                        "noise_beta_beta": 1.0,
                        "noise_s": 0.999,
                        "num_timestep_buckets": 1000,
                    }
                )
            )
        )
        return ah.FlowmatchingActionHead(cfg)
    finally:
        ah.DiTConfig["DiT-B"] = old


def fixed_inputs(batch: int = 4):
    torch.manual_seed(1234)
    hidden = torch.randn(batch, 5, 8)
    encoder = torch.randn(batch, 7, 6)
    timestep = torch.arange(batch, dtype=torch.long) + 3
    aligned = torch.randn(2, 7, 16)
    return hidden, encoder, timestep, aligned


def test_import_register_and_key_prefixes():
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import QwenGR00T_ValueResidualFFS
    from starVLA.model.modules.stereo.value_residual_branch import install_value_residual_branches
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    assert "QwenGR00T_ValueResidualFFS" in FRAMEWORK_REGISTRY._registry

    fake = object.__new__(QwenGR00T_ValueResidualFFS)
    nn.Module.__init__(fake)
    fake.action_model = nn.Module()
    fake.action_model.model = make_tiny_dit(interleave=True)
    install_value_residual_branches(
        fake.action_model.model,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    expected = (
        "ffs.",
        "action_model.model.transformer_blocks.0.attn1.to_v_resid.",
        "action_model.model.transformer_blocks.2.attn1.to_v_resid.",
    )
    assert fake._ffs_key_prefixes() == expected


def test_dit_step0_repeat_residual_and_missing_state():
    from starVLA.model.modules.stereo.value_residual_branch import (
        clear_aligned_ffs,
        install_value_residual_branches,
        set_aligned_ffs,
    )

    torch.manual_seed(1)
    baseline = make_tiny_dit(interleave=True).eval()
    branch = make_tiny_dit(interleave=True).eval()
    branch.load_state_dict(baseline.state_dict())

    n_patched = install_value_residual_branches(
        branch,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    cross_attn_count = sum(1 for block in branch.transformer_blocks if block.cross_attention_dim is not None)
    assert n_patched == cross_attn_count == 2

    for block in branch.transformer_blocks:
        if hasattr(block.attn1, "to_v_resid"):
            assert torch.count_nonzero(block.attn1.to_v_resid.weight).item() == 0

    hidden, encoder, timestep, aligned = fixed_inputs(batch=4)
    captured = {}

    def capture_to_v_resid(_module, inputs):
        captured["aligned"] = inputs[0].detach().clone()

    first_branch = next(block.attn1 for block in branch.transformer_blocks if hasattr(block.attn1, "to_v_resid"))
    handle = first_branch.to_v_resid.register_forward_pre_hook(capture_to_v_resid)
    try:
        clear_aligned_ffs()
        set_aligned_ffs(aligned)
        torch.manual_seed(999)
        out_branch = branch(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            timestep=timestep,
        )
        clear_aligned_ffs()
        torch.manual_seed(999)
        out_baseline = baseline(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            timestep=timestep,
        )
    finally:
        handle.remove()
        clear_aligned_ffs()

    assert torch.equal(out_branch, out_baseline), "zero-init ValueResidual must match baseline"
    assert "aligned" in captured
    assert torch.equal(captured["aligned"], aligned.repeat(2, 1, 1)), "repeat must be [s0,s1,s0,s1]"

    for block in branch.transformer_blocks:
        if hasattr(block.attn1, "to_v_resid"):
            torch.manual_seed(200 + id(block) % 97)
            block.attn1.to_v_resid.weight.data.normal_(0.0, 0.2)

    set_aligned_ffs(aligned)
    out_live = branch(hidden_states=hidden, encoder_hidden_states=encoder, timestep=timestep)
    clear_aligned_ffs()
    assert not torch.equal(out_live, out_baseline), "non-zero V residual must change output"

    try:
        branch(hidden_states=hidden, encoder_hidden_states=encoder, timestep=timestep)
    except RuntimeError as exc:
        assert "aligned_ffs is None" in str(exc)
    else:
        raise AssertionError("missing set_aligned_ffs() did not raise")


def test_action_head_forward_and_predict_step0_parity():
    from starVLA.model.modules.stereo.value_residual_branch import (
        clear_aligned_ffs,
        install_value_residual_branches,
        set_aligned_ffs,
    )

    torch.manual_seed(22)
    baseline = make_tiny_action_head().eval()
    branch = make_tiny_action_head().eval()
    branch.load_state_dict(baseline.state_dict())
    install_value_residual_branches(
        branch.model,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )

    torch.manual_seed(333)
    vl = torch.randn(2, 7, 6)
    actions = torch.randn(2, 3, 2)
    state = torch.randn(2, 1, 2)
    aligned = torch.randn(2, 7, 16)
    repeat_n = 3
    vl_rep = vl.repeat(repeat_n, 1, 1)
    actions_rep = actions.repeat(repeat_n, 1, 1)
    state_rep = state.repeat(repeat_n, 1, 1)

    torch.manual_seed(444)
    loss_base = baseline(vl_rep, actions_rep, state_rep)
    set_aligned_ffs(aligned)
    try:
        torch.manual_seed(444)
        loss_branch = branch(vl_rep, actions_rep, state_rep)
    finally:
        clear_aligned_ffs()
    assert torch.equal(loss_branch, loss_base), "forward step-0 parity failed"

    torch.manual_seed(555)
    pred_base = baseline.predict_action(vl, state)
    set_aligned_ffs(aligned)
    try:
        torch.manual_seed(555)
        pred_branch = branch.predict_action(vl, state)
    finally:
        clear_aligned_ffs()
    assert torch.equal(pred_branch, pred_base), "predict_action step-0 parity failed"


def test_alignment_scatter_primary_only_and_mismatch():
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import build_primary_aligned_ffs
    from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id

    image_id = 99
    input_ids = torch.tensor(
        [
            [0, 0, 7, 99, 99, 8, 99, 99, 99, 99, 99, 99, 9, 0],
            [0, 7, 99, 99, 99, 99, 8, 99, 99, 99, 99, 9, 0, 0],
        ],
        dtype=torch.long,
    )
    image_grid = torch.tensor(
        [
            [1, 2, 4],  # sample0 right: 1x2 tokens
            [1, 4, 6],  # sample0 primary: 2x3 tokens
            [1, 4, 4],  # sample1 right: 2x2 tokens
            [1, 4, 4],  # sample1 primary: 2x2 tokens
        ],
        dtype=torch.long,
    )
    last_hidden = torch.zeros(2, input_ids.shape[1], 5, dtype=torch.float32)
    net0 = torch.ones(2, 16, 3, 3)
    aligned = build_primary_aligned_ffs(
        net0=net0,
        input_ids=input_ids,
        image_grid_thw=image_grid,
        last_hidden=last_hidden,
        image_token_id=image_id,
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        label="smoke",
    )
    cam_id = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=image_id,
        image_grid_thw=image_grid,
        num_cameras=2,
        spatial_merge_size=2,
    )
    nonzero = aligned.abs().sum(dim=-1) > 0
    assert torch.equal(nonzero, cam_id == 1), "aligned_ffs must be non-zero only at primary tokens"
    assert torch.count_nonzero(aligned[cam_id != 1]).item() == 0

    bad_grid = image_grid.clone()
    bad_grid[1] = torch.tensor([1, 2, 2])
    try:
        build_primary_aligned_ffs(
            net0=net0,
            input_ids=input_ids,
            image_grid_thw=bad_grid,
            last_hidden=last_hidden,
            image_token_id=image_id,
            num_cameras=2,
            spatial_merge_size=2,
            primary_cam_id=1,
            label="smoke",
        )
    except RuntimeError as exc:
        assert "primary token grid" in str(exc)
    else:
        raise AssertionError("primary grid mismatch did not raise")


def test_k_path_unchanged_and_no_extra_k_params():
    from starVLA.model.modules.stereo.value_residual_branch import install_value_residual_branches

    torch.manual_seed(31)
    baseline = make_tiny_dit(interleave=True).eval()
    branch = make_tiny_dit(interleave=True).eval()
    branch.load_state_dict(baseline.state_dict())
    install_value_residual_branches(
        branch,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    assert not any("to_k_z" in name or "K_z" in name for name, _ in branch.named_parameters())
    # K path unchanged = the to_k projection (weight+bias) is byte-identical to baseline.
    # Compare weights directly (robust to per-block in_features differing between cross-attn
    # blocks (in=cross_attention_dim) and self-attn-only interleave blocks (in=query_dim)).
    for idx, block in enumerate(branch.transformer_blocks):
        base_block = baseline.transformer_blocks[idx]
        tgt = block.attn1.base if hasattr(block.attn1, "base") else block.attn1
        assert torch.equal(tgt.to_k.weight, base_block.attn1.to_k.weight)
        if tgt.to_k.bias is not None:
            assert torch.equal(tgt.to_k.bias, base_block.attn1.to_k.bias)


def test_remap_preserves_trunk_and_missing_keys():
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import _remap_legacy_attn1_keys
    from starVLA.model.modules.stereo.value_residual_branch import install_value_residual_branches

    torch.manual_seed(11)
    baseline = make_tiny_dit(interleave=True)
    branch = make_tiny_dit(interleave=True)
    install_value_residual_branches(
        branch,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )
    remapped, n_legacy = _remap_legacy_attn1_keys(
        baseline.state_dict(),
        own_keys=set(branch.state_dict().keys()),
    )
    missing, unexpected = branch.load_state_dict(remapped, strict=False)
    assert missing and all("to_v_resid" in key for key in missing), missing
    assert not unexpected, unexpected
    assert n_legacy > 0

    for idx, block in enumerate(branch.transformer_blocks):
        legacy_prefix = f"transformer_blocks.{idx}.attn1"
        if hasattr(block.attn1, "base"):
            assert torch.equal(block.attn1.base.to_q.weight, baseline.state_dict()[f"{legacy_prefix}.to_q.weight"])
            assert torch.equal(block.attn1.base.to_k.weight, baseline.state_dict()[f"{legacy_prefix}.to_k.weight"])
            assert torch.equal(block.attn1.base.to_v.weight, baseline.state_dict()[f"{legacy_prefix}.to_v.weight"])
            assert torch.equal(
                block.attn1.base.to_out[0].weight,
                baseline.state_dict()[f"{legacy_prefix}.to_out.0.weight"],
            )
            assert torch.count_nonzero(block.attn1.to_v_resid.weight).item() == 0
        else:
            assert torch.equal(block.attn1.to_q.weight, baseline.state_dict()[f"{legacy_prefix}.to_q.weight"])
            assert torch.equal(block.attn1.to_k.weight, baseline.state_dict()[f"{legacy_prefix}.to_k.weight"])
            assert torch.equal(block.attn1.to_v.weight, baseline.state_dict()[f"{legacy_prefix}.to_v.weight"])


def test_framework_warmstart_audit_allows_only_ffs_and_value_resid_missing():
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import QwenGR00T_ValueResidualFFS
    from starVLA.model.modules.stereo.value_residual_branch import install_value_residual_branches

    torch.manual_seed(17)
    baseline = nn.Module()
    baseline.action_model = nn.Module()
    baseline.action_model.model = make_tiny_dit(interleave=True)

    fake = object.__new__(QwenGR00T_ValueResidualFFS)
    nn.Module.__init__(fake)
    fake.config = types.SimpleNamespace(
        framework=types.SimpleNamespace(
            qwenvl=DotDict({"stereo_cam_branch_enabled": False})
        )
    )
    fake.stereo_cam_rope_layers = None
    fake.ffs = nn.Linear(2, 2)
    fake.action_model = nn.Module()
    fake.action_model.model = make_tiny_dit(interleave=True)
    install_value_residual_branches(
        fake.action_model.model,
        ffs_feat_dim=16,
        expected_n_total_blocks=4,
        require_all_cross_attn=False,
    )

    result = QwenGR00T_ValueResidualFFS.load_state_dict(
        fake,
        baseline.state_dict(),
        strict=False,
        init_from_baseline=True,
    )
    allowed_missing = [key for key in result.missing_keys if key.startswith("ffs.") or ".attn1.to_v_resid." in key]
    assert sorted(allowed_missing) == sorted(result.missing_keys), result.missing_keys
    assert any(key.startswith("ffs.") for key in result.missing_keys), result.missing_keys
    assert any(".attn1.to_v_resid." in key for key in result.missing_keys), result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys

    own_state = fake.state_dict()
    base_state = baseline.state_dict()
    for key, value in base_state.items():
        if ".attn1." in key and any(part in key for part in (".to_q.", ".to_k.", ".to_v.", ".to_out.")):
            remapped_key = key.replace(".attn1.", ".attn1.base.", 1)
            if remapped_key in own_state:
                assert torch.equal(own_state[remapped_key], value), remapped_key
                continue
        assert torch.equal(own_state[key], value), key
    for name, param in fake.named_parameters():
        if ".attn1.to_v_resid." in name:
            assert torch.count_nonzero(param).item() == 0, name


def test_train_only_optimizer_exact_allowlist():
    from starVLA.model.modules.stereo.value_residual_branch import install_value_residual_branches
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    class TinyWarmStart(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffs = nn.Linear(2, 2)
            self.action_model = nn.Module()
            self.action_model.model = make_tiny_dit(interleave=True)
            install_value_residual_branches(
                self.action_model.model,
                ffs_feat_dim=16,
                expected_n_total_blocks=4,
                require_all_cross_attn=False,
            )
            self.trunk = nn.Linear(3, 3)

    model = TinyWarmStart()
    train_only = ".attn1.to_v_resid."
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
    expected = {name for name, _param in model.named_parameters() if ".attn1.to_v_resid." in name}
    assert opt_names == expected
    assert expected
    assert all("ffs." not in name for name in opt_names)
    assert all(".attn1.base.to_v." not in name for name in opt_names)


def fake_framework_instance(raise_in_action: bool = False):
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import QwenGR00T_ValueResidualFFS

    fake = object.__new__(QwenGR00T_ValueResidualFFS)
    nn.Module.__init__(fake)
    fake.action_horizon = 2
    fake.config = types.SimpleNamespace(
        framework=types.SimpleNamespace(action_model={"repeated_diffusion_steps": 1}),
        datasets=types.SimpleNamespace(vla_data=types.SimpleNamespace(obs_image_size=None)),
    )
    fake._encode_last_hidden_with_inputs = lambda batch_images, instructions: (
        {"input_ids": torch.zeros(2, 3, dtype=torch.long), "image_grid_thw": torch.zeros(4, 3, dtype=torch.long)},
        torch.zeros(2, 3, 6),
    )
    fake._compute_aligned_ffs = lambda batch_images, qwen_inputs, last_hidden: torch.ones(2, 3, 16)

    class ActionModel:
        def __call__(self, *args, **kwargs):
            from starVLA.model.modules.stereo.value_residual_branch import get_state

            assert get_state().aligned_ffs is not None
            if raise_in_action:
                raise RuntimeError("synthetic forward failure")
            return torch.tensor(0.0)

        def predict_action(self, last_hidden, state):
            from starVLA.model.modules.stereo.value_residual_branch import get_state

            assert get_state().aligned_ffs is not None
            if raise_in_action:
                raise RuntimeError("synthetic predict failure")
            return torch.zeros(last_hidden.shape[0], 2, 1)

    fake.action_model = ActionModel()
    return fake


def test_framework_try_finally_clears_state():
    from starVLA.model.framework.VLM4A.QwenGR00T_ValueResidualFFS import QwenGR00T_ValueResidualFFS
    from starVLA.model.modules.stereo.value_residual_branch import get_state

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
    QwenGR00T_ValueResidualFFS.forward(ok, examples)
    assert get_state().aligned_ffs is None

    bad = fake_framework_instance(raise_in_action=True)
    try:
        QwenGR00T_ValueResidualFFS.forward(bad, examples)
    except RuntimeError as exc:
        assert "synthetic forward failure" in str(exc)
    else:
        raise AssertionError("synthetic action failure did not propagate")
    assert get_state().aligned_ffs is None

    bad_predict = fake_framework_instance(raise_in_action=True)
    try:
        QwenGR00T_ValueResidualFFS.predict_action(bad_predict, examples)
    except RuntimeError as exc:
        assert "synthetic predict failure" in str(exc)
    else:
        raise AssertionError("synthetic predict failure did not propagate")
    assert get_state().aligned_ffs is None


def main():
    tests = [
        test_import_register_and_key_prefixes,
        test_dit_step0_repeat_residual_and_missing_state,
        test_action_head_forward_and_predict_step0_parity,
        test_alignment_scatter_primary_only_and_mismatch,
        test_k_path_unchanged_and_no_extra_k_params,
        test_remap_preserves_trunk_and_missing_keys,
        test_framework_warmstart_audit_allows_only_ffs_and_value_resid_missing,
        test_train_only_optimizer_exact_allowlist,
        test_framework_try_finally_clears_state,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")


if __name__ == "__main__":
    main()
