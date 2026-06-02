#!/usr/bin/env python3
"""CPU pre-launch gate for the VLM-input FFS injection.

This smoke test avoids the real VLM and FFS weights. It exercises the real
FFSVLMInjector plus the two-hook install path against a tiny HF-shaped module,
checking step-0 no-op behavior, configured-primary camera routing, gradient
flow, and fail-loud layout mismatch handling.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.modules.stereo.ffs_vlm_inject import (
    FFSVLMInjector,
    clear_ffs_state,
    install_ffs_vlm_input_hooks,
    set_ffs_feature,
)


IMAGE_TOKEN_ID = 32000
PRIMARY_CAM_ID = 1


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, *, inputs_embeds: torch.Tensor, **_kwargs) -> torch.Tensor:
        return inputs_embeds


class TinyQwenBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = TinyLanguageModel()


class TinyHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = TinyQwenBody()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.language_model(inputs_embeds=inputs_embeds)


def build_fixture():
    torch.manual_seed(7)
    model = TinyHFModel()
    injector = FFSVLMInjector(in_ch=2, llm_dim=4, hidden_dim=8)
    install_ffs_vlm_input_hooks(
        model,
        injector=injector,
        num_cameras=2,
        spatial_merge_size=2,
        primary_cam_id=PRIMARY_CAM_ID,
    )
    input_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    inputs_embeds = torch.randn(1, input_ids.shape[1], 4)
    ffs_feat = torch.randn(1, 2, 3, 3)
    primary_rows = torch.tensor([6, 7, 8, 9], dtype=torch.long)
    non_primary_rows = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    text_rows = torch.tensor([0, 5, 10], dtype=torch.long)
    return model, injector, input_ids, image_grid_thw, inputs_embeds, ffs_feat, primary_rows, non_primary_rows, text_rows


def forward_once(model, input_ids, image_grid_thw, inputs_embeds, ffs_feat):
    clear_ffs_state()
    set_ffs_feature(ffs_feat)
    return model(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        inputs_embeds=inputs_embeds,
    )


def assert_step0_byte_identical() -> None:
    model, _injector, input_ids, grid, embeds, ffs_feat, *_ = build_fixture()
    out = forward_once(model, input_ids, grid, embeds, ffs_feat)
    if not torch.equal(out, embeds):
        raise AssertionError("gate-init forward must be byte-identical to inputs_embeds")


def assert_nonzero_gate_touches_only_primary() -> None:
    model, injector, input_ids, grid, embeds, ffs_feat, primary_rows, non_primary_rows, text_rows = build_fixture()
    with torch.no_grad():
        injector.gate.fill_(0.5)
    out = forward_once(model, input_ids, grid, embeds, ffs_feat)
    delta = out - embeds
    if float(delta[0, primary_rows].abs().sum()) <= 0.0:
        raise AssertionError("configured primary image-token rows did not change")
    if not torch.equal(out[0, non_primary_rows], embeds[0, non_primary_rows]):
        raise AssertionError("non-primary image-token rows changed")
    if not torch.equal(out[0, text_rows], embeds[0, text_rows]):
        raise AssertionError("text rows changed")


def assert_backward_reaches_injector() -> None:
    model, injector, input_ids, grid, embeds, ffs_feat, primary_rows, *_ = build_fixture()
    with torch.no_grad():
        injector.gate.fill_(0.5)
    injector.zero_grad(set_to_none=True)
    out = forward_once(model, input_ids, grid, embeds, ffs_feat)
    loss = (out[0, primary_rows] ** 2).sum()
    loss.backward()
    gate_grad = 0.0 if injector.gate.grad is None else float(injector.gate.grad.abs().sum())
    spatial_grad = 0.0
    for param in injector.spatial_proj.parameters():
        if param.grad is not None:
            spatial_grad += float(param.grad.abs().sum())
    if gate_grad <= 0.0:
        raise AssertionError("ffs_vlm_injector.gate received zero grad")
    if spatial_grad <= 0.0:
        raise AssertionError("spatial_proj weights received zero grad")


def assert_mismatched_grid_raises() -> None:
    model, _injector, _input_ids, grid, _embeds, ffs_feat, *_ = build_fixture()
    bad_ids = torch.tensor(
        [[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
          11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12]],
        dtype=torch.long,
    )
    bad_embeds = torch.randn(1, bad_ids.shape[1], 4)
    try:
        forward_once(model, bad_ids, grid, bad_embeds, ffs_feat)
    except RuntimeError as exc:
        msg = str(exc)
        if "refusing to silently" not in msg or "primary_cam_id=1" not in msg:
            raise AssertionError(f"RuntimeError message was not specific enough: {msg}") from exc
        return
    raise AssertionError("mismatched image grid did not raise RuntimeError")


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
    tests = [
        ("step0_byte_identical", assert_step0_byte_identical),
        ("nonzero_gate_primary_only", assert_nonzero_gate_touches_only_primary),
        ("backward_reaches_gate_and_spatial_proj", assert_backward_reaches_injector),
        ("mismatched_grid_raises", assert_mismatched_grid_raises),
    ]
    ok = True
    for name, fn in tests:
        ok = run(name, fn) and ok
    clear_ffs_state()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
