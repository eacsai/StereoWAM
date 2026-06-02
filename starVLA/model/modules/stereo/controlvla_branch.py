"""ControlVLA-style parallel K/V branch for the action DiT cross-attention.

Round-2 redesign (CODEX HIGH 1-4 all addressed):
    1. HIGH-1 batch order: use Tensor.repeat (NOT repeat_interleave) to match
       QwenPI.forward which does `actions.repeat(N, 1, 1)` → [s0,s1,s0,s1].
       Old repeat_interleave gave [s0,s0,s1,s1] = cross-contamination.
    2. HIGH-2 coverage assert: install now takes expected_n_total_blocks and
       checks against len(transformer_blocks), not against count of cross-attn
       blocks (was tautological). Framework also asserts interleave is OFF.
    3. HIGH-3 fuse-before-to_out: don't keep the original attn1 untouched.
       Replace block.attn1 with ControlVLAAttention wrapper that runs trunk
       SDPA + branch SDPA, SUMS, then applies trunk's to_out ONCE — matches
       ControlVLA paper Sec 3.2 fusion point + repo `kvcontrol_transformer`.
    4. HIGH-4 ds config: ensured deepspeed_zero2_ga2.yaml + the JSON it wraps
       (ds_config_ga2.yaml) both committed.

Pseudocode per cross-attn block (after install):
    block.attn1 = ControlVLAAttention(original_attn)         # replaces attn1
    block.forward                                            # untouched (no monkey-patch)
        -> calls self.attn1(norm_h, encoder=vl_embs, mask)
            -> Q   = base.to_q(norm_h)
            -> K_t = base.to_k(vl_embs); V_t = base.to_v(vl_embs)
            -> attn_t = SDPA(Q, K_t, V_t)
            -> K_z = to_k_z(ffs); V_z = to_v_z(ffs)          # zero-init
            -> attn_z = SDPA(Q, K_z, V_z)
            -> fused = attn_t + attn_z                       # SUM PRE-to_out
            -> return base.to_out(fused)                     # ONE shared output proj
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared state holder.
# Framework sets `ffs_tokens` once per forward; all wrapped attentions read it.
# --------------------------------------------------------------------------- #


@dataclass
class ControlVLABranchState:
    ffs_tokens: Optional[torch.Tensor] = None  # (B_ffs, T_ffs, ffs_token_dim)
    enabled: bool = True


_STATE = ControlVLABranchState()


def set_ffs_tokens(ffs_tokens: torch.Tensor) -> None:
    _STATE.ffs_tokens = ffs_tokens


def clear_ffs_tokens() -> None:
    _STATE.ffs_tokens = None


def set_enabled(enabled: bool) -> None:
    _STATE.enabled = bool(enabled)


def get_state() -> ControlVLABranchState:
    return _STATE


# --------------------------------------------------------------------------- #
# ControlVLAAttention: wraps the original diffusers.Attention so we can:
#   (a) compute Q/K_t/V_t ourselves (= same math as original)
#   (b) compute parallel K_z/V_z from FFS tokens (zero-init)
#   (c) sum the two SDPA outputs BEFORE applying the shared to_out projection
#       (matches ControlVLA Sec 3.2 + repo kvcontrol_transformer)
#
# The wrapped attn1 keeps the original Attention as `self.base` (so its params
# get loaded/saved via state_dict at `attn1.base.*` instead of `attn1.*`).
# --------------------------------------------------------------------------- #


class ControlVLAAttention(nn.Module):
    def __init__(self, base_attn: nn.Module, ffs_token_dim: int):
        super().__init__()
        self.base = base_attn  # original diffusers.Attention
        hidden_dim = base_attn.to_q.out_features
        if hidden_dim % base_attn.heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by heads {base_attn.heads}"
            )
        self.heads = base_attn.heads
        self.dim_head = hidden_dim // base_attn.heads
        # Parallel K_z / V_z, zero-init (either alone suffices to force branch
        # output to zero at step 0; we zero both for belt-and-suspenders).
        self.to_k_z = nn.Linear(ffs_token_dim, hidden_dim, bias=False)
        self.to_v_z = nn.Linear(ffs_token_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.to_k_z.weight)
        nn.init.zeros_(self.to_v_z.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, T_a, D = hidden_states.shape

        # ---- trunk: replicate diffusers.Attention math (Q/K/V projections,
        # multi-head reshape, SDPA). DiT has no q_norm/k_norm in this config. ----
        Q_flat = self.base.to_q(hidden_states)
        eh = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        K_t_flat = self.base.to_k(eh)
        V_t_flat = self.base.to_v(eh)
        Q   = Q_flat.view(B, T_a, self.heads, self.dim_head).transpose(1, 2)
        K_t = K_t_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        V_t = V_t_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        attn_t = F.scaled_dot_product_attention(Q, K_t, V_t, attn_mask=attention_mask)

        # ---- branch: only fire on cross-attn (encoder present) + state set ----
        # CODEX FIX (round-2 MED-2): fail-closed on missing FFS state when this
        # is a cross-attn forward (encoder_hidden_states present).
        fused = attn_t
        if _STATE.enabled and encoder_hidden_states is not None:
            if _STATE.ffs_tokens is None:
                raise RuntimeError(
                    "[ControlVLA] cross-attn forward but ffs_tokens is None — "
                    "framework forgot to set_ffs_tokens() this forward. "
                    "Either set state or set_enabled(False)."
                )
            ffs = _STATE.ffs_tokens
            B_ffs = ffs.shape[0]
            if B != B_ffs:
                if B % B_ffs != 0:
                    raise RuntimeError(
                        f"[ControlVLA] action batch {B} not divisible by ffs batch {B_ffs}"
                    )
                # CODEX FIX (round-2 HIGH-1): QwenPI.forward repeats with
                # Tensor.repeat(N,1,1) → ordering [s0,s1,s0,s1]. Match it here.
                # repeat_interleave would give [s0,s0,s1,s1] → cross-contamination.
                ffs = ffs.repeat(B // B_ffs, 1, 1)
            K_z_flat = self.to_k_z(ffs.to(Q_flat.dtype))
            V_z_flat = self.to_v_z(ffs.to(Q_flat.dtype))
            K_z = K_z_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
            V_z = V_z_flat.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
            attn_z = F.scaled_dot_product_attention(Q, K_z, V_z)
            # CODEX FIX (round-2 HIGH-3): sum BEFORE to_out so branch goes
            # through the shared output projection exactly like trunk does.
            fused = attn_t + attn_z

        out = fused.transpose(1, 2).reshape(B, T_a, D)
        # diffusers Attention.to_out is ModuleList([Linear, Dropout]).
        out = self.base.to_out[0](out)
        out = self.base.to_out[1](out)
        return out


# --------------------------------------------------------------------------- #
# Install: per-block attn1 swap.
#   Replaces block.attn1 with ControlVLAAttention(original_attn1).
#   No monkey-patching of BasicTransformerBlock.forward needed — the wrapper
#   has the same call signature.
# --------------------------------------------------------------------------- #


def install_controlvla_branches(
    action_dit_model,
    ffs_token_dim: int,
    expected_n_total_blocks: Optional[int] = None,
    require_all_cross_attn: bool = True,
) -> int:
    """Install ControlVLA branches on every cross-attn block of the action DiT.

    Args:
        action_dit_model: LayerwiseFlowmatchingActionHead.model (DiT instance).
        ffs_token_dim: dim of FFS tokens fed to to_k_z/to_v_z.
        expected_n_total_blocks: if given, raise unless len(transformer_blocks) ==
            this. CODEX FIX (round-2 HIGH-2): the prior assertion compared
            n_patched against count(cross_attn != None) — tautological, didn't
            catch interleave_self_attention=True silently halving coverage.
            Caller should pass the architectural truth (e.g. config num_layers).
        require_all_cross_attn: if True (default), raise if any block has
            cross_attention_dim is None (= self-attn interleave). Forces the
            user to explicitly disable interleave_self_attention in DiT config
            before launching, so "per-layer ControlVLA" is honestly per-layer.
    """
    n_total = len(action_dit_model.transformer_blocks)
    n_patched = 0
    n_self_attn = 0
    for idx, block in enumerate(action_dit_model.transformer_blocks):
        if block.cross_attention_dim is None:
            n_self_attn += 1
            continue
        block.attn1 = ControlVLAAttention(block.attn1, ffs_token_dim=ffs_token_dim)
        n_patched += 1

    print(
        f"[ControlVLA] swapped attn1 in {n_patched}/{n_total} cross-attn blocks "
        f"(self_attn_interleave_blocks={n_self_attn}, ffs_token_dim={ffs_token_dim})"
    )

    if n_patched == 0:
        raise RuntimeError(
            "[ControlVLA] install patched 0 blocks — DiT has no cross-attention."
        )
    if expected_n_total_blocks is not None and n_total != expected_n_total_blocks:
        raise RuntimeError(
            f"[ControlVLA] DiT has {n_total} total blocks but caller expected "
            f"{expected_n_total_blocks}. Architecture mismatch."
        )
    if require_all_cross_attn and n_self_attn > 0:
        raise RuntimeError(
            f"[ControlVLA] DiT has {n_self_attn} self-attn-only blocks "
            f"(interleave_self_attention=True). 'Per-layer ControlVLA branch' "
            f"requires every layer to be cross-attn. Either pass "
            f"--framework.action_model.diffusion_model_cfg.interleave_self_attention=false "
            f"in your launcher, or call install with require_all_cross_attn=False to "
            f"explicitly accept partial coverage."
        )
    return n_patched
