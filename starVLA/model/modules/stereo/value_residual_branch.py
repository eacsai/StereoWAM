"""FFS residual into GR00T action-DiT cross-attention Values.

Method #8 is the action-end sibling of the VLM-side ControlNet residual:
each wrapped cross-attention call keeps the trunk K path/logits/softmax
unchanged for the same action hidden state and memory, and only augments the
Value content at memory-token positions that the framework has aligned to the
primary-view image tokens. Later blocks can still see changed action states once
the residual is non-zero; the per-call invariant is "K unchanged, one softmax".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ValueResidualBranchState:
    aligned_ffs: Optional[torch.Tensor] = None  # (B_ffs, T_enc, ffs_feat_dim)
    enabled: bool = True


_STATE = ValueResidualBranchState()


def set_aligned_ffs(aligned_ffs: torch.Tensor) -> None:
    _STATE.aligned_ffs = aligned_ffs


def clear_aligned_ffs() -> None:
    _STATE.aligned_ffs = None


def set_enabled(enabled: bool) -> None:
    _STATE.enabled = bool(enabled)


def get_state() -> ValueResidualBranchState:
    return _STATE


class ValueResidualAttention(nn.Module):
    """Wrap diffusers.Attention and add zero-init FFS residuals into V only."""

    def __init__(self, base_attn: nn.Module, ffs_feat_dim: int):
        super().__init__()
        self.base = base_attn
        hidden_dim = int(base_attn.to_v.out_features)
        if hidden_dim % int(base_attn.heads) != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by heads {base_attn.heads}"
            )
        self.heads = int(base_attn.heads)
        self.dim_head = hidden_dim // self.heads
        self.to_v_resid = nn.Linear(int(ffs_feat_dim), hidden_dim, bias=False)
        nn.init.zeros_(self.to_v_resid.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T_a, D = hidden_states.shape

        Q_flat = self.base.to_q(hidden_states)
        eh = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        K_t_flat = self.base.to_k(eh)
        V_t_flat = self.base.to_v(eh)

        if _STATE.enabled and encoder_hidden_states is not None:
            if _STATE.aligned_ffs is None:
                raise RuntimeError(
                    "[ValueResidual] cross-attn forward but aligned_ffs is None -- "
                    "framework forgot to set_aligned_ffs() this forward. "
                    "Either set state or set_enabled(False)."
                )
            aligned_ffs = _STATE.aligned_ffs
            B_ffs = int(aligned_ffs.shape[0])
            if B != B_ffs:
                if B % B_ffs != 0:
                    raise RuntimeError(
                        f"[ValueResidual] action batch {B} not divisible by aligned_ffs batch {B_ffs}"
                    )
                # Match QwenGR00T's actions.repeat(N, 1, 1) ordering:
                # [s0, s1, s0, s1], not repeat_interleave's [s0, s0, s1, s1].
                aligned_ffs = aligned_ffs.repeat(B // B_ffs, 1, 1)
            if int(aligned_ffs.shape[1]) != int(V_t_flat.shape[1]):
                raise RuntimeError(
                    "[ValueResidual] aligned_ffs token axis mismatch: "
                    f"aligned_ffs T={aligned_ffs.shape[1]} vs V(memory) T={V_t_flat.shape[1]}"
                )
            aligned_ffs = aligned_ffs.to(device=V_t_flat.device, dtype=V_t_flat.dtype)
            V_t_flat = V_t_flat + self.to_v_resid(aligned_ffs)

        Q = Q_flat.view(B, T_a, self.heads, self.dim_head).transpose(1, 2)
        K_t = K_t_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        V_t = V_t_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K_t, V_t, attn_mask=attention_mask)

        out = out.transpose(1, 2).reshape(B, T_a, D)
        out = self.base.to_out[0](out)
        out = self.base.to_out[1](out)
        return out


def install_value_residual_branches(
    action_dit_model,
    ffs_feat_dim: int,
    expected_n_total_blocks: Optional[int] = None,
    require_all_cross_attn: bool = False,
) -> int:
    """Install ValueResidualAttention on every cross-attn block's attn1."""

    n_total = len(action_dit_model.transformer_blocks)
    n_cross_attn = 0
    n_patched = 0
    n_self_attn = 0
    for _idx, block in enumerate(action_dit_model.transformer_blocks):
        if block.cross_attention_dim is None:
            n_self_attn += 1
            continue
        n_cross_attn += 1
        if isinstance(block.attn1, ValueResidualAttention):
            raise RuntimeError("[ValueResidual] attn1 is already wrapped; refusing double install")
        block.attn1 = ValueResidualAttention(block.attn1, ffs_feat_dim=ffs_feat_dim)
        n_patched += 1

    print(
        f"[ValueResidual] swapped attn1 in {n_patched}/{n_cross_attn} cross-attn blocks "
        f"({n_total} total, self_attn_interleave_blocks={n_self_attn}, "
        f"ffs_feat_dim={ffs_feat_dim})"
    )

    if n_patched == 0:
        raise RuntimeError(
            "[ValueResidual] install patched 0 blocks -- DiT has no cross-attention."
        )
    if n_patched != n_cross_attn:
        raise RuntimeError(
            f"[ValueResidual] patched {n_patched} blocks but DiT exposes "
            f"{n_cross_attn} cross-attn blocks. Refusing partial coverage."
        )
    if expected_n_total_blocks is not None and n_total != expected_n_total_blocks:
        raise RuntimeError(
            f"[ValueResidual] DiT has {n_total} total blocks but caller expected "
            f"{expected_n_total_blocks}. Architecture mismatch."
        )
    if require_all_cross_attn and n_self_attn > 0:
        raise RuntimeError(
            f"[ValueResidual] DiT has {n_self_attn} self-attn-only blocks "
            f"(interleave_self_attention=True). Either disable interleave or call "
            f"install with require_all_cross_attn=False to accept cross-attn-only coverage."
        )
    return n_patched
