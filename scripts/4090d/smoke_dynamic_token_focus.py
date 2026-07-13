#!/usr/bin/env python3
"""Focused smoke for scene-predictor dynamic-token focus.

Checks the non-leaky minimum experiment wiring:
- static-zero regularizer contributes cells while flow_supervised_cells remains dynamic-only;
- past-motion token gate works with train-style past_flow_valid;
- the gate also works with rollout FIFO style batches that omit past_flow_valid;
- missing/no-history past flow suppresses all tokens when conditioning_static_scale=0.
"""

import torch

from starVLA.model.modules.action_model.flow_matching_head.scene_flow_head import SceneFieldMatchingHead


def check(name, cond):
    if not bool(cond):
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    B, S, cross_dim = 2, 5, 32
    sub_cfg = {
        "action_model_type": "DiT-B",
        "grid_size": 4,
        "field_dim": 3,
        "tap_hidden_index": 1,
        "diffusion_model_cfg": {"num_layers": 2, "cross_attention_dim": cross_dim},
        "target_key": "flow_gt",
        "valid_key": "flow_valid",
        "dynamic_key": "flow_dynamic",
        "dynamic_fallback_to_valid": False,
        "static_zero_loss_weight": 0.05,
        "dynamic_direction_loss_weight": 0.2,
        "dynamic_magnitude_loss_weight": 0.2,
        "conditioning_past_motion_gate": True,
        "conditioning_static_scale": 0.0,
        "conditioning_motion_threshold": 1e-5,
        "past_flow_controlnet": {"enabled": True, "n_past_steps": 2, "dropout_p": 0.0},
    }
    head = SceneFieldMatchingHead(full_config=None, sub_cfg=sub_cfg).to(dev)
    head.train()
    vl = torch.randn(B, S, cross_dim, device=dev)

    flow_gt = torch.zeros(B, 3, 8, 8, device=dev)
    flow_gt[:, 0, 0:4, 0:4] = 0.03
    valid = torch.ones(B, 8, 8, dtype=torch.bool, device=dev)
    dynamic = torch.zeros(B, 8, 8, dtype=torch.bool, device=dev)
    dynamic[:, 0:2, 0:2] = True

    past = torch.zeros(B, 2, 3, 8, 8, device=dev)
    past[:, :, 0, 0:2, 0:2] = 0.02
    past_valid = torch.ones(B, 2, 8, 8, dtype=torch.bool, device=dev)
    batch = {
        "flow_gt": flow_gt,
        "flow_valid": valid,
        "flow_dynamic": dynamic,
        "flow_has_gt": torch.ones(B, dtype=torch.bool, device=dev),
        "past_flow_gt": past,
        "past_flow_valid": past_valid,
        "has_past_flow": torch.ones(B, dtype=torch.bool, device=dev),
    }

    torch.manual_seed(456)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(456)
    out = head(vl, batch)
    check("flow_loss finite scalar", out["flow_loss"].dim() == 0 and torch.isfinite(out["flow_loss"]).item())
    check("dynamic-only supervised cells positive", out["flow_supervised_cells"].item() > 0)
    check("static-zero cells positive", out["flow_static_zero_cells"].item() > 0)
    check(
        "weighted cells include static regularizer",
        out["flow_weighted_cells"].item() > out["flow_supervised_cells"].item(),
    )
    check("direction auxiliary finite", torch.isfinite(out["flow_direction_loss"]).item())
    check("magnitude auxiliary finite", torch.isfinite(out["flow_magnitude_loss"]).item())
    check("fm loss metric finite", torch.isfinite(out["flow_fm_loss"]).item())
    check("effective cells include weighted cells", out["flow_effective_cells"].item() >= out["flow_weighted_cells"].item())
    expected = out["flow_fm_loss"] + 0.2 * out["flow_direction_loss"] + 0.2 * out["flow_magnitude_loss"]
    check("aux losses are included in flow_loss", torch.allclose(out["flow_loss"].detach(), expected.detach(), rtol=1e-5, atol=1e-6))

    zero_cfg = dict(sub_cfg)
    zero_cfg["dynamic_direction_loss_weight"] = 0.0
    zero_cfg["dynamic_magnitude_loss_weight"] = 0.0
    zero_head = SceneFieldMatchingHead(full_config=None, sub_cfg=zero_cfg).to(dev)
    zero_head.load_state_dict(head.state_dict())
    zero_head.train()
    torch.manual_seed(456)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(456)
    zero_out = zero_head(vl, batch)
    check("zero aux weights make flow_loss equal fm_loss", torch.allclose(zero_out["flow_loss"].detach(), zero_out["flow_fm_loss"].detach(), rtol=1e-6, atol=1e-7))

    aux_only_cfg = dict(sub_cfg)
    aux_only_cfg["dynamic_loss_weight"] = 0.0
    aux_only_cfg["static_zero_loss_weight"] = 0.0
    aux_only_head = SceneFieldMatchingHead(full_config=None, sub_cfg=aux_only_cfg).to(dev)
    aux_only_head.load_state_dict(head.state_dict())
    aux_only_head.train()
    torch.manual_seed(456)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(456)
    aux_only_out = aux_only_head(vl, batch)
    check("aux-only has zero weighted cells", aux_only_out["flow_weighted_cells"].item() == 0)
    check("aux-only still has effective dynamic cells", aux_only_out["flow_effective_cells"].item() > 0)
    check("aux-only flow_loss finite", torch.isfinite(aux_only_out["flow_loss"]).item())

    static_batch = dict(batch)
    static_batch["flow_dynamic"] = torch.zeros_like(dynamic)
    static_out = head(vl, static_batch)
    check("all-static batch has zero dynamic cells", static_out["flow_supervised_cells"].item() == 0)
    check("all-static batch still has static-zero weighted cells", static_out["flow_weighted_cells"].item() > 0)
    check("all-static static-zero loss finite", torch.isfinite(static_out["flow_loss"]).item())
    check(
        "static-zero weight is not normalized away",
        torch.allclose(static_out["flow_loss"].detach(), 0.05 * static_out["flow_static_zero_loss"].detach(), rtol=1e-5, atol=1e-6),
    )

    missing_dynamic_batch = dict(batch)
    missing_dynamic_batch.pop("flow_dynamic")
    try:
        head(vl, missing_dynamic_batch)
    except RuntimeError as exc:
        check("missing dynamic mask fails closed for dynamic-focus losses", "flow_dynamic" in str(exc))
    else:
        raise AssertionError("missing dynamic mask should fail closed")

    gate = head._past_motion_token_gate(batch, dev, B)
    check("train-style gate shape", gate.shape == (B, head.num_tokens, 1))
    check("train-style gate has moving and suppressed tokens", gate.max().item() == 1.0 and gate.min().item() == 0.0)

    fifo_batch = {"past_flow_gt": past, "has_past_flow": torch.ones(B, dtype=torch.bool, device=dev)}
    fifo_gate = head._past_motion_token_gate(fifo_batch, dev, B)
    check("FIFO-style missing-valid gate keeps moving tokens", fifo_gate.max().item() == 1.0)

    no_past_gate = head._past_motion_token_gate(None, dev, B)
    check("missing history suppresses all tokens at static_scale=0", no_past_gate.max().item() == 0.0)

    no_hist = {"past_flow_gt": past, "has_past_flow": torch.zeros(B, dtype=torch.bool, device=dev)}
    no_hist_gate = head._past_motion_token_gate(no_hist, dev, B)
    check("has_past_flow=False suppresses all tokens", no_hist_gate.max().item() == 0.0)

    hidden = head.extract_conditioning_hidden(vl, past_flow_batch=batch)
    check("extract hidden shape", hidden.shape == (B, head.num_tokens, head.model.inner_dim))
    check("extract hidden finite", torch.isfinite(hidden).all().item())
    print("DYNAMIC_TOKEN_FOCUS_SMOKE=PASS")


if __name__ == "__main__":
    main()
