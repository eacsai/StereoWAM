#!/usr/bin/env python3
"""Tiny real Qwen3.5 full/linear forward-backward smoke for both visual arms."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from starVLA.model.modules.stereo.cam_branch_attention import (
    _discover_supported_attention_layers,
    install_cam_branch,
)


class _TinyQwenWrapper(nn.Module):
    def __init__(self, text_model: nn.Module, text_config, image_token_id: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = text_model
        self.config = SimpleNamespace(text_config=text_config, image_token_id=image_token_id)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.model.language_model, "is_gradient_checkpointing", False))

    def forward(self, input_ids, image_grid_thw, attention_mask=None):
        # image_grid_thw is consumed by the top-level Camera Branch hook and is
        # intentionally not forwarded to Qwen's text-only model.
        return self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state


def _make_model(device: torch.device, mode: str, branch_causal: bool):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        layer_types=["full_attention", "linear_attention", "full_attention"],
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    text_model = Qwen3_5TextModel(config).to(device=device, dtype=torch.bfloat16)
    text_model.gradient_checkpointing_enable()
    model = _TinyQwenWrapper(text_model, config, image_token_id=99).to(device=device)
    supported = _discover_supported_attention_layers(model)
    assert [idx for idx, _ in supported] == [0, 2], supported
    state, branches = install_cam_branch(
        model,
        branch_heads=1,
        branch_head_dim=16,
        out_proj_init_std=0.02,
        layers=[0],
        visual_attention_mode=mode,
        stereo_cam_branch_is_causal=branch_causal,
        attention_implementation="sdpa",
        num_cameras=2,
        logging_frequency=0,
    )
    model.visual_branches = nn.ModuleList(branches).to(device=device, dtype=torch.bfloat16)
    assert len(branches) == 1
    assert getattr(model.model.language_model.layers[0].self_attn, "_stereo_cam_branch_installed")
    assert getattr(model.model.language_model.layers[0].self_attn, "_stereo_visual_attention_installed")
    assert getattr(model.model.language_model.layers[2].self_attn, "_stereo_visual_attention_installed")
    # The middle GatedDeltaNet layer is intentionally absent from supported and
    # remains unwrapped.
    assert type(model.model.language_model.layers[1].linear_attn).__name__ == "Qwen3_5GatedDeltaNet"
    return model, state, branches


def _assert_finite_nonzero_grads(module: nn.Module, label: str) -> None:
    params = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert params, f"{label} has no trainable parameters"
    missing = [idx for idx, parameter in enumerate(params) if parameter.grad is None]
    assert not missing, f"{label} has missing gradients at indices {missing}"
    assert all(torch.isfinite(parameter.grad).all() for parameter in params)
    assert any(torch.any(parameter.grad != 0) for parameter in params)

def _run_one(device: torch.device, mode: str, branch_causal: bool) -> float:
    torch.manual_seed(20260722)
    model, state, branches = _make_model(device, mode, branch_causal)
    input_ids = torch.tensor(
        [[99, 99, 7, 99, 99, 8]], device=device, dtype=torch.long
    )
    image_grid_thw = torch.tensor(
        [[1, 2, 4], [1, 2, 4]], device=device, dtype=torch.long
    )
    call_counts = [0] * len(model.model.language_model.layers)
    forward_wrappers = []
    for index, layer in enumerate(model.model.language_model.layers):
        original_forward = layer.forward
        def _count_forward(*args, _original=original_forward, _index=index, **kwargs):
            call_counts[_index] += 1
            return _original(*args, **kwargs)
        layer.forward = _count_forward
        forward_wrappers.append((layer, original_forward))

    output = model(input_ids=input_ids, image_grid_thw=image_grid_thw)
    assert output.shape == (1, 6, 64)
    assert torch.isfinite(output).all()
    assert call_counts == [1, 1, 1], f"unexpected initial decoder calls: {call_counts}"
    try:
        model(input_ids=input_ids, image_grid_thw=image_grid_thw)
    except RuntimeError as exc:
        assert "overlapping top-level forward" in str(exc), exc
    else:
        raise AssertionError("overlapping checkpointed forward was not rejected")
    loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert call_counts == [2, 2, 2], f"checkpoint recompute did not run exactly once: {call_counts}"
    _assert_finite_nonzero_grads(branches[0], "Camera Branch")
    _assert_finite_nonzero_grads(model.model.language_model.layers[0].self_attn, "full-attention layer 0")
    _assert_finite_nonzero_grads(model.model.language_model.layers[1].linear_attn, "linear-attention layer 1")
    assert state.image_positions is None, "gradient-finalize hook did not clear visual state"
    assert state.image_grid_thw is None, "gradient-finalize hook did not clear image_grid_thw"
    for layer, original_forward in forward_wrappers:
        layer.forward = original_forward
    model.model.language_model.gradient_checkpointing_disable()
    with torch.no_grad():
        cleared_output = model(input_ids=input_ids, image_grid_thw=image_grid_thw)
    assert cleared_output.shape == (1, 6, 64)
    assert torch.isfinite(cleared_output).all()
    assert state.image_positions is None, "non-checkpointed post-hook did not clear visual state"
    assert state.image_grid_thw is None, "post-hook did not clear image_grid_thw"
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke")
    device = torch.device(args.device)
    causal_loss = _run_one(device, "causal", True)
    bidir_loss = _run_one(device, "image_bidir", False)
    print(f"PASS visual attention GPU smoke causal_loss={causal_loss:.6g} image_bidir_loss={bidir_loss:.6g}")


if __name__ == "__main__":
    main()
