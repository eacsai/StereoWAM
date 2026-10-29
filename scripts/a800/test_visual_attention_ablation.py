from __future__ import annotations

import unittest

import torch

from starVLA.model.modules.stereo.cam_branch_attention import CamBranchAttention
from starVLA.model.modules.stereo.visual_attention import (
    parse_exact_bool,
    build_visual_attention_mask,
    prepare_visual_attention_call,
    resolve_visual_attention_profile,
)


def _blocked(value: torch.Tensor) -> bool:
    return bool(value <= torch.finfo(value.dtype).min / 2)


class VisualAttentionHelpersTest(unittest.TestCase):
    def test_exact_boolean_parser(self) -> None:
        self.assertTrue(parse_exact_bool(True, name="test"))
        self.assertFalse(parse_exact_bool("false", name="test"))
        self.assertTrue(parse_exact_bool(" TRUE ", name="test"))
        with self.assertRaises(ValueError):
            parse_exact_bool("0", name="test")
    def test_profile_pairing_and_backend_gate(self) -> None:
        self.assertEqual(
            resolve_visual_attention_profile(
                "legacy_mixed", False, "flash_attention_2", cam_branch_enabled=False
            ),
            ("legacy_mixed", False, "flash_attention_2"),
        )
        self.assertEqual(
            resolve_visual_attention_profile(
                "causal", True, "sdpa", cam_branch_enabled=True
            ),
            ("causal", True, "sdpa"),
        )
        self.assertEqual(
            resolve_visual_attention_profile(
                "image_bidir", "false", "sdpa", cam_branch_enabled=True
            ),
            ("image_bidir", False, "sdpa"),
        )
        with self.assertRaises(ValueError):
            resolve_visual_attention_profile(
                "causal", False, "sdpa", cam_branch_enabled=True
            )
        with self.assertRaises(ValueError):
            resolve_visual_attention_profile(
                "image_bidir", False, "flash_attention_2", cam_branch_enabled=True
            )
        with self.assertRaises(ValueError):
            resolve_visual_attention_profile(
                "image_bidir", False, "sdpa", cam_branch_enabled=False
            )
        with self.assertRaises(ValueError):
            resolve_visual_attention_profile(
                "legacy_mixed", True, "sdpa", cam_branch_enabled=True
            )

    def test_causal_and_bidirectional_image_rules(self) -> None:
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        causal = build_visual_attention_mask(
            None,
            positions,
            mode="causal",
            batch_size=1,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        bidir = build_visual_attention_mask(
            None,
            positions,
            mode="image_bidir",
            batch_size=1,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Image-image is lower-triangular in the causal arm and fully visible
        # in the bidirectional arm.
        self.assertTrue(_blocked(causal[0, 0, 0, 1]))
        self.assertFalse(_blocked(causal[0, 0, 1, 0]))
        self.assertFalse(_blocked(bidir[0, 0, 0, 1]))
        self.assertFalse(_blocked(bidir[0, 0, 1, 0]))
        # Image queries never read instruction keys; instruction queries retain
        # their normal causal visibility to image and text keys.
        self.assertTrue(_blocked(causal[0, 0, 0, 2]))
        self.assertTrue(_blocked(bidir[0, 0, 1, 3]))
        self.assertFalse(_blocked(causal[0, 0, 2, 0]))
        self.assertFalse(_blocked(causal[0, 0, 3, 1]))

    def test_padding_is_preserved_when_unblocking_images(self) -> None:
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        padding = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        mask = build_visual_attention_mask(
            padding,
            positions,
            mode="image_bidir",
            batch_size=1,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertFalse(_blocked(mask[0, 0, 0, 1]))
        self.assertTrue(_blocked(mask[0, 0, 0, 3]))
        self.assertTrue(_blocked(mask[0, 0, 2, 3]))

    def test_call_rewrite_and_text_or_cache_bypass(self) -> None:
        hidden = torch.randn(1, 4, 8)
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        args, kwargs = prepare_visual_attention_call(
            hidden,
            (),
            {"attention_mask": torch.ones(1, 4, dtype=torch.long)},
            image_positions=positions,
            mode="image_bidir",
            layer_idx=3,
        )
        self.assertEqual(args, [])
        self.assertFalse(kwargs["is_causal"])
        self.assertEqual(tuple(kwargs["attention_mask"].shape), (1, 1, 4, 4))

        original_kwargs = {"attention_mask": None, "sentinel": "text-only"}
        _, bypass_kwargs = prepare_visual_attention_call(
            hidden[:, :1],
            (),
            original_kwargs,
            image_positions=None,
            mode="image_bidir",
            layer_idx=3,
        )
        self.assertEqual(bypass_kwargs, original_kwargs)

        class ActiveCache:
            def get_seq_length(self):
                return 1

        with self.assertRaises(RuntimeError):
            prepare_visual_attention_call(
                hidden[:, :1],
                (),
                {"past_key_values": ActiveCache()},
                image_positions=torch.tensor([[0, 1]]),
                mode="causal",
                layer_idx=3,
            )
        with self.assertRaises(RuntimeError):
            prepare_visual_attention_call(
                hidden,
                (),
                {"attention_mask": torch.ones(1, 5, dtype=torch.long)},
                image_positions=positions,
                mode="causal",
                layer_idx=3,
            )

    def test_two_dimensional_padding_retains_causal_triangle(self) -> None:
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        padding = torch.ones(1, 4, dtype=torch.long)
        mask = build_visual_attention_mask(
            padding,
            positions,
            mode="causal",
            batch_size=1,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(_blocked(mask[0, 0, 2, 3]))
        self.assertFalse(_blocked(mask[0, 0, 3, 2]))

    def test_bool_all_blocked_rows_use_negative_infinity(self) -> None:
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        all_blocked = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
        mask = build_visual_attention_mask(
            all_blocked,
            positions,
            mode="causal",
            batch_size=1,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.isneginf(mask).all())

    def test_cache_history_is_layer_specific(self) -> None:
        hidden = torch.randn(1, 4, 8)
        positions = torch.tensor([[0, 1]], dtype=torch.long)

        class LayerCache:
            def get_seq_length(self, layer_idx=0):
                return 0 if layer_idx == 3 else 1

        _, kwargs = prepare_visual_attention_call(
            hidden,
            (),
            {"past_key_values": LayerCache(), "cache_position": torch.zeros(4, dtype=torch.long)},
            image_positions=positions,
            mode="causal",
            layer_idx=3,
        )
        self.assertFalse(kwargs["is_causal"])
        with self.assertRaises(RuntimeError):
            prepare_visual_attention_call(
                hidden,
                (),
                {"past_key_values": LayerCache(), "cache_position": torch.zeros(4, dtype=torch.long)},
                image_positions=positions,
                mode="causal",
                layer_idx=7,
            )

    def test_camera_branch_direction_is_real(self) -> None:
        torch.manual_seed(7)
        causal = CamBranchAttention(
            hidden_dim=8,
            is_causal=True,
            branch_heads=1,
            branch_head_dim=8,
            out_proj_init_std=0.05,
        )
        bidir = CamBranchAttention(
            hidden_dim=8,
            is_causal=False,
            branch_heads=1,
            branch_head_dim=8,
            out_proj_init_std=0.05,
        )
        bidir.load_state_dict(causal.state_dict())
        hidden = torch.randn(1, 4, 8)
        eye = torch.eye(4).repeat(1, 4, 1, 1)
        causal_out = causal.forward_raw(hidden, eye, eye, eye)
        bidir_out = bidir.forward_raw(hidden, eye, eye, eye)
        self.assertEqual(causal_out.shape, bidir_out.shape)
        self.assertTrue(torch.isfinite(causal_out).all())
        self.assertTrue(torch.isfinite(bidir_out).all())
        self.assertFalse(torch.allclose(causal_out, bidir_out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
