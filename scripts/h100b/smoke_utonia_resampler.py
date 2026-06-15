#!/usr/bin/env python3
"""CPU-safe structural smokes for Method #10 Version B.

Real FFS-disparity + Utonia forward parity belongs to the R5/GPU smoke gate.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_TOKEN_ID = 99


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class TinyLanguageModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.proj = nn.Linear(dim, dim)
        self.last_inputs_embeds = None
        self.last_position_ids = None
        self.last_attention_mask = None

    def forward(self, *, inputs_embeds, position_ids=None, attention_mask=None, **kwargs):
        self.last_inputs_embeds = inputs_embeds
        self.last_position_ids = position_ids
        self.last_attention_mask = attention_mask
        return self.proj(inputs_embeds)


class TinyHFModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = types.SimpleNamespace(language_model=TinyLanguageModel(dim))

    def forward(self, *, input_ids, image_grid_thw, inputs_embeds, position_ids, attention_mask):
        return self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )


def tiny_inputs(dim: int = 8):
    input_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    image_grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    embeds = torch.randn(1, input_ids.shape[1], dim)
    position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(4, 1, -1).clone()
    attention_mask = torch.ones_like(input_ids)
    return input_ids, image_grid, embeds, position_ids, attention_mask


def test_import_register_key_prefixes_and_checkpoint_exclusion():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaResamplerFFS import (
        QwenGR00T_UtoniaResamplerFFS,
        UtoniaResampler,
    )
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    assert "QwenGR00T_UtoniaResamplerFFS" in FRAMEWORK_REGISTRY._registry
    fake = object.__new__(QwenGR00T_UtoniaResamplerFFS)
    nn.Module.__init__(fake)
    fake.ffs = nn.Linear(2, 2)
    fake.utonia = nn.Linear(2, 2)
    fake.utonia_resampler = UtoniaResampler(num_queries=4, point_dim=8, d_model=8, n_layers=1, n_heads=2)
    assert fake._ffs_key_prefixes() == ("ffs.", "utonia.", "utonia_resampler.")
    sd = fake.state_dict()
    assert not any(k.startswith("ffs.") or k.startswith("utonia.") for k in sd), sorted(sd)
    assert any(k.startswith("utonia_resampler.") for k in sd), sorted(sd)
    prefixed = fake.state_dict(prefix="outer.")
    assert not any(k.startswith("outer.ffs.") or k.startswith("outer.utonia.") for k in prefixed), sorted(prefixed)
    assert any(k.startswith("outer.utonia_resampler.") for k in prefixed), sorted(prefixed)


def test_resampler_zero_out_proj():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaResamplerFFS import UtoniaResampler

    torch.manual_seed(3)
    resampler = UtoniaResampler(num_queries=64, point_dim=1386, d_model=32, n_layers=1, n_heads=4).eval()
    feats = torch.randn(2, 17, 1386)
    mask = torch.ones(2, 17, dtype=torch.bool)
    out = resampler(feats, mask)
    assert out.shape == (2, 64, 32)
    assert torch.equal(out, torch.zeros_like(out)), "zero out_proj must create zero inserted tokens"


def test_resampler_all_false_mask_is_finite_zero():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaResamplerFFS import UtoniaResampler

    torch.manual_seed(33)
    resampler = UtoniaResampler(num_queries=5, point_dim=16, d_model=8, n_layers=1, n_heads=2).eval()
    feats = torch.zeros(2, 3, 16)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    out = resampler(feats, mask)
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out)), "all-False point mask must stay zero under zero out_proj"


def _run_point_hook_grows_sequence_neutral_positions_and_control_parity(*, use_cam_rope_state: bool):
    from starVLA.model.modules.stereo.depth_token_inject import (
        _insert_batched_2d,
        _insert_batched_3d,
        _neutral_position_columns,
    )
    from starVLA.model.modules.stereo.point_token_inject import (
        clear_point_state,
        get_point_state,
        install_point_token_hooks,
        set_point_tokens,
    )
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaResamplerFFS import UtoniaResampler

    torch.manual_seed(4)
    clear_point_state()
    dim = 8
    pc_cfg = {"num_point_tokens": 5}
    k = int(pc_cfg["num_point_tokens"])
    model = TinyHFModel(dim)
    cam_state = (
        types.SimpleNamespace(
            per_token_cam_id=None,
            per_token_row_id=None,
            epipolar_mask_enabled=False,
        )
        if use_cam_rope_state
        else None
    )
    install_point_token_hooks(
        model,
        lm=model.model.language_model,
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        cam_rope_state=cam_state,
        image_token_id=IMAGE_TOKEN_ID,
    )
    input_ids, image_grid, embeds, position_ids, attention_mask = tiny_inputs(dim)
    resampler = UtoniaResampler(num_queries=k, point_dim=12, d_model=dim, n_layers=1, n_heads=2).eval()
    point_feats = torch.randn(1, 7, 12)
    point_mask = torch.tensor([[True, True, False, True, False, True, True]], dtype=torch.bool)
    point_tokens = resampler(point_feats, point_mask)
    assert point_tokens.shape[1] == k
    set_point_tokens(point_tokens)
    out = model(
        input_ids=input_ids,
        image_grid_thw=image_grid,
        inputs_embeds=embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    state = get_point_state()
    lm = model.model.language_model
    assert lm.last_inputs_embeds.shape[1] == input_ids.shape[1] + k
    assert lm.last_position_ids.shape[-1] == input_ids.shape[1] + k
    assert lm.last_attention_mask.shape[1] == input_ids.shape[1] + k
    assert tuple(state.keep_mask.shape) == (1, input_ids.shape[1] + k)
    assert int((~state.keep_mask).sum().item()) == k
    insert_idx = int(state.insert_idx[0].item())
    inserted = slice(insert_idx, insert_idx + k)
    assert not bool(state.keep_mask[0, inserted].any())
    assert torch.equal(state.per_token_cam_id[0, inserted], torch.full((k,), -1, dtype=torch.long))
    if cam_state is not None:
        assert torch.equal(cam_state.per_token_cam_id[0, inserted], torch.full((k,), -1, dtype=torch.long))

    expected_embeds = _insert_batched_3d(embeds, state.insert_idx, point_tokens)
    expected_pos = _neutral_position_columns(position_ids, state.insert_idx, k)
    expected_mask = _insert_batched_2d(attention_mask, state.insert_idx, torch.ones(1, k, dtype=attention_mask.dtype))
    assert torch.equal(lm.last_inputs_embeds, expected_embeds)
    assert torch.equal(lm.last_position_ids, expected_pos)
    assert torch.equal(lm.last_attention_mask, expected_mask)
    anchor = position_ids[..., 0, max(insert_idx - 1, 0)]
    got_neutral = lm.last_position_ids[..., 0, inserted]
    assert torch.equal(got_neutral, anchor.unsqueeze(-1).expand_as(got_neutral))

    with torch.no_grad():
        control = lm.proj(expected_embeds)
    assert torch.equal(out, control), "hook output must match matched zero-insertion control"
    clear_point_state()


def test_point_hook_grows_sequence_neutral_positions_and_control_parity():
    _run_point_hook_grows_sequence_neutral_positions_and_control_parity(use_cam_rope_state=True)


def test_point_hook_grows_sequence_with_cam_rope_state_none():
    _run_point_hook_grows_sequence_neutral_positions_and_control_parity(use_cam_rope_state=False)


def test_train_only_allowlist_excludes_frozen_encoders_and_trunk():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaResamplerFFS import UtoniaResampler
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    model = nn.Module()
    model.ffs = nn.Linear(2, 2)
    model.utonia = nn.Linear(2, 2)
    for p in model.ffs.parameters():
        p.requires_grad_(False)
    for p in model.utonia.parameters():
        p.requires_grad_(False)
    model.trunk = nn.Linear(3, 3)
    model.utonia_resampler = UtoniaResampler(num_queries=4, point_dim=8, d_model=8, n_layers=1, n_heads=2)
    cfg = types.SimpleNamespace(
        trainer=DotDict(
            {
                "learning_rate": DotDict({"base": 1e-4}),
                "freeze_modules": "",
                "train_only": "utonia_resampler.",
            }
        )
    )
    groups = build_param_lr_groups(model, cfg)
    names = {n for g in groups for n in g["param_names"]}
    assert names
    assert all(n.startswith("utonia_resampler.") for n in names)
    assert not any(n.startswith("ffs.") or n.startswith("utonia.") or n.startswith("trunk.") for n in names)


def test_try_finally_clears_point_state():
    from starVLA.model.modules.stereo.point_token_inject import (
        clear_point_state,
        get_point_state,
        set_point_tokens,
    )

    clear_point_state()
    try:
        set_point_tokens(torch.ones(1, 64, 8))
        raise RuntimeError("synthetic failure")
    except RuntimeError:
        pass
    finally:
        clear_point_state()
    assert get_point_state().depth_tokens is None


def main():
    tests = [
        test_import_register_key_prefixes_and_checkpoint_exclusion,
        test_resampler_zero_out_proj,
        test_resampler_all_false_mask_is_finite_zero,
        test_point_hook_grows_sequence_neutral_positions_and_control_parity,
        test_point_hook_grows_sequence_with_cam_rope_state_none,
        test_train_only_allowlist_excludes_frozen_encoders_and_trunk,
        test_try_finally_clears_point_state,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print("SMOKE_ALL_PASS")


if __name__ == "__main__":
    main()
