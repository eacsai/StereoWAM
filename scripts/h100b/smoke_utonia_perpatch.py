#!/usr/bin/env python3
"""CPU-safe structural smokes for Method #10 Version A.

Real FFS-disparity + Utonia forward parity belongs to the R5/GPU smoke gate.
"""
from __future__ import annotations

import os
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


class Qwen3_5Attention(nn.Module):
    pass


class NotQwenAttention(nn.Module):
    pass


class TinyLayer(nn.Module):
    def __init__(self, dim: int, *, bad_attention: bool = False) -> None:
        super().__init__()
        self.self_attn = NotQwenAttention() if bad_attention else Qwen3_5Attention()
        self.proj = nn.Identity()

    def forward(self, hidden):
        return self.proj(hidden)


class TinyLM(nn.Module):
    def __init__(self, dim: int, layers: int = 2, bad_attention_idx: int | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyLayer(dim, bad_attention=(i == bad_attention_idx)) for i in range(layers)]
        )

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyHF(nn.Module):
    def __init__(self, dim: int = 5, layers: int = 2, bad_attention_idx: int | None = None) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(image_token_id=99, hidden_size=dim)
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.model = types.SimpleNamespace(language_model=TinyLM(dim, layers, bad_attention_idx))

    def forward(self, *, input_ids, image_grid_thw, hidden):
        return self.model.language_model(hidden)


class OnesProjector(nn.Module):
    def __init__(self, dim: int, value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))
        self.dim = int(dim)

    def forward(self, feat, h_tok: int, w_tok: int):
        return torch.ones(h_tok * w_tok, self.dim, dtype=feat.dtype, device=feat.device) * self.scale


def tiny_tokens():
    input_ids = torch.tensor(
        [
            [0, 0, 7, 99, 99, 8, 99, 99, 99, 99, 99, 99, 9, 0],
            [0, 7, 99, 99, 99, 99, 8, 99, 99, 99, 99, 9, 0, 0],
        ],
        dtype=torch.long,
    )
    image_grid = torch.tensor(
        [
            [1, 2, 4],
            [1, 4, 6],
            [1, 4, 4],
            [1, 4, 4],
        ],
        dtype=torch.long,
    )
    return input_ids, image_grid


def tiny_square_tokens():
    input_ids = torch.tensor(
        [
            [0, 7, 99, 99, 99, 99, 8, 99, 99, 99, 99, 9],
            [7, 99, 99, 99, 99, 8, 99, 99, 99, 99, 9, 0],
        ],
        dtype=torch.long,
    )
    image_grid = torch.tensor(
        [
            [1, 4, 4],
            [1, 4, 4],
            [1, 4, 4],
            [1, 4, 4],
        ],
        dtype=torch.long,
    )
    return input_ids, image_grid


def test_import_register_key_prefixes_and_checkpoint_exclusion():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import (
        QwenGR00T_UtoniaPerPatchAddFFS,
        UtoniaPerTokenProjector,
    )
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    assert "QwenGR00T_UtoniaPerPatchAddFFS" in FRAMEWORK_REGISTRY._registry
    fake = object.__new__(QwenGR00T_UtoniaPerPatchAddFFS)
    nn.Module.__init__(fake)
    fake.ffs = nn.Linear(2, 2)
    fake.utonia = nn.Linear(2, 2)
    fake.utonia_pertoken_projectors = nn.ModuleList([UtoniaPerTokenProjector(1387, 8)])
    assert fake._ffs_key_prefixes() == ("ffs.", "utonia.", "utonia_pertoken_projectors.")
    sd = fake.state_dict()
    assert not any(k.startswith("ffs.") or k.startswith("utonia.") for k in sd), sorted(sd)
    assert any(k.startswith("utonia_pertoken_projectors.") for k in sd), sorted(sd)
    prefixed = fake.state_dict(prefix="outer.")
    assert not any(k.startswith("outer.ffs.") or k.startswith("outer.utonia.") for k in prefixed), sorted(prefixed)
    assert any(k.startswith("outer.utonia_pertoken_projectors.") for k in prefixed), sorted(prefixed)


def test_real_init_target_layer_guard():
    from omegaconf import OmegaConf
    import starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS as mod

    cls = mod.QwenGR00T_UtoniaPerPatchAddFFS
    original_base_init = mod.QwenGR00TFFSBase.__init__
    original_sync = cls._sync_actual_vlm_hidden_dim
    original_ffs = cls._init_frozen_ffs_for_disparity
    original_utonia = cls._init_frozen_utonia
    bad_idx = None

    def fake_base_init(self, config=None, **kwargs):
        nn.Module.__init__(self)
        base_config = OmegaConf.create(
            {
                "framework": {
                    "qwenvl": {
                        "stereo_cam_branch_enabled": False,
                        "stereo_cam_rope_spatial_merge": 2,
                    },
                    "utonia_pointcloud": {
                        "inject_hidden_dim": 8,
                        "expected_vlm_layers": 24,
                    },
                },
                "datasets": {"vla_data": {}},
            }
        )
        self.config = OmegaConf.merge(base_config, config or {})
        self.qwen_vl_interface = types.SimpleNamespace(
            model=TinyHF(dim=8, layers=24, bad_attention_idx=bad_idx)
        )
        self.num_cameras = 2
        self.primary_idx = 1
        self.right_view_idx = 0
        self.primary_cam_id = 1

    try:
        mod.QwenGR00TFFSBase.__init__ = fake_base_init
        cls._sync_actual_vlm_hidden_dim = lambda self: 8
        cls._init_frozen_ffs_for_disparity = lambda self, pc_cfg, label: None
        cls._init_frozen_utonia = lambda self, pc_cfg, label: None

        good = cls(config={})
        assert good._utonia_inject_depths == [3, 7, 11, 15, 19, 23]

        try:
            cls(config={"framework": {"utonia_pointcloud": {"inject_depths": [0, 1, 2]}}})
        except (ValueError, RuntimeError) as exc:
            assert "inject_depths" in str(exc)
        else:
            raise AssertionError("wrong inject_depths did not fail closed")

        bad_idx = 7
        try:
            cls(config={})
        except RuntimeError as exc:
            assert "Qwen3_5Attention" in str(exc)
            assert "NotQwenAttention" in str(exc)
        else:
            raise AssertionError("bad target-layer attention class did not fail")
    finally:
        mod.QwenGR00TFFSBase.__init__ = original_base_init
        cls._sync_actual_vlm_hidden_dim = original_sync
        cls._init_frozen_ffs_for_disparity = original_ffs
        cls._init_frozen_utonia = original_utonia


def test_projector_zero_gate_byte_exact():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import UtoniaPerTokenProjector

    torch.manual_seed(7)
    projector = UtoniaPerTokenProjector(in_ch=1387, llm_dim=16, hidden_dim=8).eval()
    grid = torch.randn(1, 1387, 8, 8)
    grid[:, -1] = (torch.rand(1, 8, 8) > 0.25).float()
    out = projector(grid, 8, 8)
    assert torch.equal(out, torch.zeros_like(out)), "zero gate must make residual byte-exact zero"


def test_projector_degenerate_empty_and_square_grid_assert():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import UtoniaPerTokenProjector

    projector = UtoniaPerTokenProjector(in_ch=1387, llm_dim=16, hidden_dim=8).eval()
    empty_grid = torch.zeros(1, 1387, 8, 8)
    out = projector(empty_grid, 8, 8)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out)), "empty occupancy must stay zero under zero-gate"
    try:
        projector(empty_grid, 4, 16)
    except RuntimeError as exc:
        assert "square primary token grid" in str(exc)
    else:
        raise AssertionError("non-square 64-token grid did not fail closed")


def test_residual_primary_only_and_grid_mismatch():
    from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import install_ffs_vlm_layer_residual_hooks
    from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id

    model = TinyHF(dim=5)
    state = install_ffs_vlm_layer_residual_hooks(
        model,
        projectors=nn.ModuleList([OnesProjector(5, 1.0)]),
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        image_token_id=99,
        target_layer_indices=[0],
        label="smoke-utonia-A",
    )
    input_ids, image_grid = tiny_square_tokens()
    hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
    state.ffs_feat = torch.ones(input_ids.shape[0], 1387, 8, 8)
    out = model(input_ids=input_ids, image_grid_thw=image_grid, hidden=hidden)
    cam = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=99,
        image_grid_thw=image_grid,
        num_cameras=2,
        spatial_merge_size=2,
    )
    nonzero = out.abs().sum(dim=-1) > 0
    assert torch.equal(nonzero, cam == 1), "Utonia residual must land only on primary tokens"

    bad_grid = image_grid.clone()
    bad_grid[1] = torch.tensor([1, 2, 2])
    try:
        model(input_ids=input_ids, image_grid_thw=bad_grid, hidden=hidden)
    except RuntimeError as exc:
        assert "primary token grid" in str(exc)
    else:
        raise AssertionError("grid mismatch did not fail closed")


def test_hook_zero_residual_matches_baseline():
    from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import install_ffs_vlm_layer_residual_hooks

    torch.manual_seed(8)
    baseline = TinyHF(dim=5)
    hooked = TinyHF(dim=5)
    state = install_ffs_vlm_layer_residual_hooks(
        hooked,
        projectors=nn.ModuleList([OnesProjector(5, 0.0)]),
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        image_token_id=99,
        target_layer_indices=[0],
        label="smoke-utonia-A-zero",
    )
    input_ids, image_grid = tiny_square_tokens()
    hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 5)
    state.ffs_feat = torch.ones(input_ids.shape[0], 1387, 8, 8)
    assert torch.equal(
        hooked(input_ids=input_ids, image_grid_thw=image_grid, hidden=hidden),
        baseline(input_ids=input_ids, image_grid_thw=image_grid, hidden=hidden),
    )


def test_real_projector_in_real_hook_zero_baseline():
    from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import install_ffs_vlm_layer_residual_hooks
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import UtoniaPerTokenProjector

    torch.manual_seed(9)
    baseline = TinyHF(dim=5)
    hooked = TinyHF(dim=5)
    projector = UtoniaPerTokenProjector(in_ch=1387, llm_dim=5, hidden_dim=8).eval()
    state = install_ffs_vlm_layer_residual_hooks(
        hooked,
        projectors=nn.ModuleList([projector]),
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=1,
        image_token_id=99,
        target_layer_indices=[0],
        label="smoke-utonia-A-real-projector",
    )
    input_ids, image_grid = tiny_square_tokens()
    hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 5)
    grid = torch.randn(input_ids.shape[0], 1387, 8, 8)
    grid[:, -1] = (torch.rand(input_ids.shape[0], 8, 8) > 0.2).float()
    state.ffs_feat = grid
    assert torch.equal(
        hooked(input_ids=input_ids, image_grid_thw=image_grid, hidden=hidden),
        baseline(input_ids=input_ids, image_grid_thw=image_grid, hidden=hidden),
    )


def test_train_only_allowlist_excludes_frozen_encoders_and_trunk():
    from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPerPatchAddFFS import UtoniaPerTokenProjector
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    model = nn.Module()
    model.ffs = nn.Linear(2, 2)
    model.utonia = nn.Linear(2, 2)
    for p in model.ffs.parameters():
        p.requires_grad_(False)
    for p in model.utonia.parameters():
        p.requires_grad_(False)
    model.trunk = nn.Linear(3, 3)
    model.utonia_pertoken_projectors = nn.ModuleList([UtoniaPerTokenProjector(1387, 8, hidden_dim=8)])
    cfg = types.SimpleNamespace(
        trainer=DotDict(
            {
                "learning_rate": DotDict({"base": 1e-4}),
                "freeze_modules": "",
                "train_only": "utonia_pertoken_projectors.",
            }
        )
    )
    groups = build_param_lr_groups(model, cfg)
    names = {n for g in groups for n in g["param_names"]}
    assert names
    assert all(n.startswith("utonia_pertoken_projectors.") for n in names)
    assert not any(n.startswith("ffs.") or n.startswith("utonia.") or n.startswith("trunk.") for n in names)


def test_qwen_processor_primary_64_optional():
    base_vlm = Path(os.environ.get("BASE_VLM", "./playground/Pretrained_models/Qwen3.5-0.8B"))
    if not base_vlm.exists():
        msg = f"Qwen processor primary-token smoke: missing {base_vlm}"
        if os.environ.get("REQUIRE_BASE_VLM_SMOKE", "0") == "1":
            raise AssertionError(msg)
        print(f"[skip] {msg}")
        return
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(base_vlm), trust_remote_code=True)
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "image", "image": img},
                    {"type": "text", "text": "pick up the object"},
                ],
            }
        ]
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    grid = batch["image_grid_thw"]
    h_tok = int(grid[1, 1]) // 2
    w_tok = int(grid[1, 2]) // 2
    assert h_tok * w_tok == 64, f"primary image token count expected 64, got {h_tok*w_tok}"


def main():
    tests = [
        test_import_register_key_prefixes_and_checkpoint_exclusion,
        test_real_init_target_layer_guard,
        test_projector_zero_gate_byte_exact,
        test_projector_degenerate_empty_and_square_grid_assert,
        test_residual_primary_only_and_grid_mismatch,
        test_hook_zero_residual_matches_baseline,
        test_real_projector_in_real_hook_zero_baseline,
        test_train_only_allowlist_excludes_frozen_encoders_and_trunk,
        test_qwen_processor_primary_64_optional,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print("SMOKE_ALL_PASS")


if __name__ == "__main__":
    main()
