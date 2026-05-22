"""Phase 3 B Camera-Frame RoPE — focused unit tests (codex round-1 B-2 fix).

Covers:
  1. per_token_cam_id correctness for mono / stereo / wrist+stereo / 3-image samples.
  2. Zero-init parity vs FA2 baseline within an explicit tolerance.
  3. Signal-flow: bumping q_cam_proj weights causes hidden-state diff.
  4. cam_id assignment robust to padding tokens.

Run:
    .venv/bin/python scripts/4090d/test_phase3b_camrope.py
Exit code: 0 = pass, 1 = fail.
"""
from __future__ import annotations

import sys
import torch
import numpy as np

# Tolerance for zero-init parity (FA2 vs SDPA bf16 numerics).
# Empirically measured ~1.1 max-diff on Qwen3.5-0.8B; set 3x headroom for safety.
ZERO_INIT_PARITY_TOL = 3.0


def _make_model(stereo_cam_rope_enabled: bool, d_c: int = 16):
    from omegaconf import OmegaConf
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
    cfg = OmegaConf.create({
        "framework": {"name": "QwenPI",
            "qwenvl": {"base_vlm": "./playground/Pretrained_models/Qwen3.5-0.8B",
                        "attn_implementation": "flash_attention_2",
                        "stereo_cam_rope_enabled": stereo_cam_rope_enabled,
                        "stereo_cam_rope_d_c": d_c},
            "action_model": {"action_horizon": 8}},
        "datasets": {"vla_data": {}}
    })
    m = Qwen_PI(config=cfg).to("cuda").to(torch.bfloat16).eval()
    return m


def test_per_token_cam_id_layouts():
    """per_token_cam_id correct for several image-count layouts."""
    from starVLA.model.modules.stereo import compute_per_token_cam_id

    IMG = 248056  # Qwen3.5-VL image_token_id

    # Mono (1 image, 16 tokens). grid_thw=[(1, 4, 4)]
    input_ids = torch.tensor([[1, 2, IMG, IMG, IMG, IMG, 5, 6]], dtype=torch.long)  # only 4 image tokens
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)  # 1 * 4 * 4 / 4 = 4 merged tokens (spatial_merge_size=2)
    out = compute_per_token_cam_id(input_ids, IMG, grid, num_cameras=2, spatial_merge_size=2)
    assert out.shape == input_ids.shape
    assert (out == -1).sum().item() == 4  # 4 text tokens
    assert (out == 0).sum().item() == 4   # 4 image tokens, all cam_id=0 (only 1 image)
    assert (out == 1).sum().item() == 0   # no right view
    print("  test_per_token_cam_id mono: PASS")

    # Stereo (primary + right_view, 4 tokens each)
    input_ids = torch.tensor([[1, 2, IMG, IMG, IMG, IMG, 3, 4, IMG, IMG, IMG, IMG, 5]], dtype=torch.long)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    out = compute_per_token_cam_id(input_ids, IMG, grid, num_cameras=2, spatial_merge_size=2)
    assert (out == 0).sum().item() == 4  # first image -> cam_id=0
    assert (out == 1).sum().item() == 4  # second image -> cam_id=1
    print("  test_per_token_cam_id stereo: PASS")

    # Three-image sample (primary + right + wrist): cam_id cycles 0, 1, 0
    input_ids = torch.tensor([[IMG, IMG, IMG, IMG, 1, IMG, IMG, IMG, IMG, 2, IMG, IMG, IMG, IMG]], dtype=torch.long)
    grid = torch.tensor([[1, 4, 4]] * 3, dtype=torch.long)
    out = compute_per_token_cam_id(input_ids, IMG, grid, num_cameras=2, spatial_merge_size=2)
    img_cam_ids = out[out >= 0].tolist()
    # Expect 4 zeros, 4 ones, 4 zeros (cam_id cycles 0,1,0 mod num_cameras=2)
    assert img_cam_ids == [0]*4 + [1]*4 + [0]*4, f"got {img_cam_ids}"
    print("  test_per_token_cam_id 3-image: PASS")

    # Padding tokens at end (zeros - not image_token_id, treated as text)
    input_ids = torch.tensor([[IMG, IMG, IMG, IMG, 99, IMG, IMG, IMG, IMG, 0, 0, 0]], dtype=torch.long)
    grid = torch.tensor([[1, 4, 4]] * 2, dtype=torch.long)
    out = compute_per_token_cam_id(input_ids, IMG, grid, num_cameras=2, spatial_merge_size=2)
    assert (out == 0).sum().item() == 4
    assert (out == 1).sum().item() == 4
    # padding zeros should be text (-1) since they don't equal IMG token id
    assert out[0, 9:].tolist() == [-1, -1, -1]
    print("  test_per_token_cam_id with padding: PASS")


def test_zero_init_parity():
    """Zero-init must produce output within ZERO_INIT_PARITY_TOL of FA2 baseline."""
    from PIL import Image
    m_off = _make_model(stereo_cam_rope_enabled=False)
    m_on = _make_model(stereo_cam_rope_enabled=True, d_c=16)
    m_on.qwen_vl_interface.load_state_dict(m_off.qwen_vl_interface.state_dict(), strict=False)
    m_on.action_model.load_state_dict(m_off.action_model.state_dict(), strict=True)

    img1 = Image.fromarray((np.random.RandomState(0).rand(256, 256, 3) * 255).astype(np.uint8))
    img2 = Image.fromarray((np.random.RandomState(1).rand(256, 256, 3) * 255).astype(np.uint8))

    with torch.no_grad():
        e_off = m_off._encode_vl_hidden_states([[img1, img2]], ["hello"])
        e_on = m_on._encode_vl_hidden_states([[img1, img2]], ["hello"])
    diff = (e_off[-1].float() - e_on[-1].float()).abs().max().item()
    assert diff < ZERO_INIT_PARITY_TOL, f"zero-init diff {diff} exceeds tol {ZERO_INIT_PARITY_TOL}"
    print(f"  test_zero_init_parity: PASS (diff={diff:.3f} < tol={ZERO_INIT_PARITY_TOL})")


def test_signal_flow():
    """Bumping q_cam_proj weights must change output."""
    from PIL import Image
    m = _make_model(stereo_cam_rope_enabled=True, d_c=16)

    img1 = Image.fromarray((np.random.RandomState(0).rand(256, 256, 3) * 255).astype(np.uint8))
    img2 = Image.fromarray((np.random.RandomState(1).rand(256, 256, 3) * 255).astype(np.uint8))

    with torch.no_grad():
        e_zero = m._encode_vl_hidden_states([[img1, img2]], ["hello"])

    for scl in m.stereo_cam_rope_layers:
        scl.q_cam_proj.weight.data.normal_(0, 0.1)
        scl.k_cam_proj.weight.data.normal_(0, 0.1)

    with torch.no_grad():
        e_bump = m._encode_vl_hidden_states([[img1, img2]], ["hello"])

    diff = (e_zero[-1].float() - e_bump[-1].float()).abs().max().item()
    assert diff > 1.0, f"signal-flow diff {diff} too small — gradient may not reach q_cam_proj"
    print(f"  test_signal_flow: PASS (diff={diff:.2f})")


def main():
    print("=== Phase 3 B Camera-Frame RoPE tests ===")
    try:
        test_per_token_cam_id_layouts()
        test_zero_init_parity()
        test_signal_flow()
        print("all tests PASSED")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
