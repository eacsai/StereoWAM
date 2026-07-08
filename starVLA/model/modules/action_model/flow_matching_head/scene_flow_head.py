# Copyright 2025 starVLA community. MIT License.
"""Scene-field flow-matching DiT head (spec: sceneflow-dualdit-cascade).

A GR00T-style flow-matching DiT that predicts a per-cell 3D FIELD conditioned on
`vl_embs`:
  - future scene flow (16x16x3, camera-frame meters)     -> the "motion" arm
  - current-frame camera-frame XYZ pointmap (16x16x3)     -> the "static-geometry" twin
Same architecture for both arms; only the target (config `target_key`) differs
-> the single-variable A/B.

Its mid-late hidden is TAPPED (detached, deterministic, GT-free) by
`extract_conditioning_hidden()` and fed to the action DiT's zero-init motion
coupler (see cross_attention_dit.MotionCoupler). The supervised flow-matching
forward (`forward`) and the extraction forward are SEPARATE (round-2 N1: never
leak the target into the conditioning hidden).

Deterministic + dropout-free by construction: this DiT is built with dropout=0
so its extraction forward consumes no RNG (step-0 identity; belt-and-suspenders
with the framework's torch.random.fork_rng around extraction).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FieldEncoder(nn.Module):
    """Encode a noised (B, N, field_dim) field + a flow-matching timestep -> (B, N, D).
    Mirrors action_model.ActionEncoder but over grid cells instead of action steps."""

    def __init__(self, field_dim, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(field_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, field, timesteps):
        B, N, _ = field.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, N)
        else:
            raise ValueError("FieldEncoder expects timesteps of shape (B,).")
        a_emb = self.layer1(field)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


class PastFlowEncoder(nn.Module):
    """Encode the last-K past flow fields (B, K, N, field_dim) -> (B, K*N, D) tokens for the
    scene DiT's zero-init MotionCoupler to cross-attend. Per-cell projection + grid-position
    embedding + per-field time-position embedding; cross-time mixing (velocity/accel) is left
    to the coupler's cross-attention (tokens are time-tagged)."""

    def __init__(self, field_dim, num_tokens, n_past, hidden_size):
        super().__init__()
        self.n_past = n_past
        self.num_tokens = num_tokens
        self.proj = nn.Linear(field_dim, hidden_size)
        self.grid_pos = nn.Embedding(num_tokens, hidden_size)
        self.time_pos = nn.Embedding(n_past, hidden_size)
        nn.init.normal_(self.grid_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_pos.weight, mean=0.0, std=0.02)

    def forward(self, past_fields):
        # past_fields: (B, K, N, field_dim)
        B, K, N, _ = past_fields.shape
        x = self.proj(past_fields)  # (B,K,N,D)
        grid = self.grid_pos(torch.arange(N, device=past_fields.device))  # (N,D)
        tpos = self.time_pos(torch.arange(K, device=past_fields.device))  # (K,D)
        x = x + grid.view(1, 1, N, -1) + tpos.view(1, K, 1, -1)
        return x.reshape(B, K * N, x.shape[-1])  # (B, K*N, D)


class SceneFieldMatchingHead(nn.Module):
    """Flow-matching DiT over a 16x16x3 field. See module docstring."""

    def __init__(self, full_config, sub_cfg):
        super().__init__()
        self.full_config = full_config
        cfg = sub_cfg  # framework.scene_predictor.*

        # --- past-flow ControlNet (temporal disambiguation of scene-flow direction) ---
        # Reuse the DiT's built-in zero-init MotionCoupler by giving THIS scene DiT a
        # motion_coupler_dim and feeding the encoded last-K past flow fields as motion_hidden.
        pf_cfg = _cfg_get(cfg, "past_flow_controlnet", {}) or {}
        self.past_flow_enabled = bool(_cfg_get(pf_cfg, "enabled", False))
        self.n_past_steps = int(_cfg_get(pf_cfg, "n_past_steps", 2))
        self.past_flow_spacing = int(_cfg_get(pf_cfg, "spacing_delta", 1))
        self.past_flow_dropout_p = float(_cfg_get(pf_cfg, "dropout_p", 0.0))
        self.past_flow_noise_std = float(_cfg_get(pf_cfg, "noise_aug_std", 0.0))
        self.past_flow_key = str(_cfg_get(pf_cfg, "past_key", "past_flow_gt"))
        self.past_valid_key = str(_cfg_get(pf_cfg, "past_valid_key", "past_flow_valid"))
        self.has_past_key = str(_cfg_get(pf_cfg, "has_past_key", "has_past_flow"))
        self.sample_num_steps = int(_cfg_get(pf_cfg, "sample_num_steps", 4))  # Euler steps for rollout sampling

        action_model_type = _cfg_get(cfg, "action_model_type", "DiT-B")
        base = dict(DiTConfig[action_model_type])
        self.input_embedding_dim = base["input_embedding_dim"]

        diffusion_model_cfg = dict(_cfg_get(cfg, "diffusion_model_cfg", {}) or {})
        diffusion_model_cfg = {**base, **diffusion_model_cfg}
        # Deterministic / dropout-free (extraction must consume no RNG).
        diffusion_model_cfg["dropout"] = 0.0
        diffusion_model_cfg["final_dropout"] = False
        # Full-width DiT output; a small MLP maps to the 3-channel field.
        diffusion_model_cfg["output_dim"] = self.input_embedding_dim
        # Zero-init couplers on the scene DiT so past-flow can be injected (motion_hidden width
        # = input_embedding_dim, matching PastFlowEncoder output). No-op when past_flow disabled.
        if self.past_flow_enabled:
            diffusion_model_cfg["motion_coupler_dim"] = self.input_embedding_dim
        self.model = DiT(**diffusion_model_cfg)

        self.grid_size = int(_cfg_get(cfg, "grid_size", 16))
        self.field_dim = int(_cfg_get(cfg, "field_dim", 3))
        self.num_tokens = self.grid_size * self.grid_size
        self.hidden_size = int(_cfg_get(cfg, "hidden_size", 1024))
        if self.past_flow_enabled:
            self.past_flow_encoder = PastFlowEncoder(
                self.field_dim, self.num_tokens, self.n_past_steps, self.input_embedding_dim
            )

        self.field_encoder = FieldEncoder(self.field_dim, self.input_embedding_dim)
        self.field_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.field_dim,
        )
        # near-zero (NOT exact-zero) output head: first-step gradient flows, but the
        # initial predicted velocity is ~0 (round-2 code_detail).
        nn.init.xavier_uniform_(self.field_decoder.layer2.weight, gain=0.01)
        nn.init.zeros_(self.field_decoder.layer2.bias)

        self.add_pos_embed = bool(_cfg_get(cfg, "add_pos_embed", True))
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.num_tokens, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # flow-matching noise schedule (same as action head)
        self.beta_dist = Beta(
            float(_cfg_get(cfg, "noise_beta_alpha", 1.5)),
            float(_cfg_get(cfg, "noise_beta_beta", 1.0)),
        )
        self.noise_s = float(_cfg_get(cfg, "noise_s", 0.999))
        self.num_timestep_buckets = int(_cfg_get(cfg, "num_timestep_buckets", 1000))

        # loss
        self.smooth_l1_beta = float(_cfg_get(cfg, "smooth_l1_beta", 0.01))
        self.z_weight = float(_cfg_get(cfg, "z_weight", 2.0))
        self.target_key = str(_cfg_get(cfg, "target_key", "flow_gt"))
        self.valid_key = str(_cfg_get(cfg, "valid_key", "flow_valid"))
        self.dynamic_key = str(_cfg_get(cfg, "dynamic_key", "flow_dynamic"))
        self.dynamic_fallback_to_valid = bool(_cfg_get(cfg, "dynamic_fallback_to_valid", True))

        # deterministic extraction policy (must match train + eval + predict_action)
        self.tap_hidden_index = int(_cfg_get(cfg, "tap_hidden_index", 10))
        n_layers = int(self.model.config.num_layers)
        if not (1 <= self.tap_hidden_index < n_layers):
            raise ValueError(
                f"tap_hidden_index={self.tap_hidden_index} must be in [1, {n_layers}); "
                "index into all_hidden_states=[input]+per-block; -1/num_layers forbidden (final layer)."
            )
        self.extract_t_bucket = int(_cfg_get(cfg, "motion_extract_t_bucket", 0))
        self.extract_noise_mode = str(_cfg_get(cfg, "motion_extract_noise_mode", "zeros"))
        self.detach_conditioning = bool(_cfg_get(cfg, "detach_conditioning", True))

    # ------------------------------------------------------------------ helpers
    def sample_time(self, batch_size, device, dtype):
        s = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return self.noise_s * (1 - s)

    def _encode(self, noised_tokens, t_discretized):
        feats = self.field_encoder(noised_tokens, t_discretized)
        if self.add_pos_embed:
            pos_ids = torch.arange(feats.shape[1], dtype=torch.long, device=feats.device)
            feats = feats + self.position_embedding(pos_ids).unsqueeze(0)
        return feats

    def _pool_target(self, field_gt, valid):
        """Mask-aware avg-pool a dense (B,3,Hgt,Wgt) field + (B,1,Hgt,Wgt) validity to
        the grid_size token resolution (avoids averaging flow over invalid pixels)."""
        g = self.grid_size
        v = valid.float()
        # sanitize non-finite sentinels at invalid pixels (rendered flow/depth carry
        # nan/inf there; nan*0=nan would poison the pooled cell -> NaN loss) — review H4.
        fg = torch.nan_to_num(field_gt.float(), nan=0.0, posinf=0.0, neginf=0.0)
        num = F.adaptive_avg_pool2d(fg * v, (g, g))
        den = F.adaptive_avg_pool2d(v, (g, g)).clamp(min=1e-6)
        pooled = num / den
        pooled_valid = F.adaptive_avg_pool2d(v, (g, g)) > 0.5
        return pooled, pooled_valid  # (B,3,g,g), (B,1,g,g)

    def _supervision_mask(self, valid, dynamic):
        """(B,1,g,g) bool -> supervised cells. dynamic INTERSECTED with valid (dynamic
        cells outside valid have undefined/≈0 pooled GT — review M1); else valid; optional
        fallback to valid on samples with no (valid&dynamic) cells."""
        if dynamic is None:
            return valid
        sup = dynamic & valid
        if self.dynamic_fallback_to_valid:
            no_dyn = sup.flatten(1).any(dim=1).logical_not()  # (B,)
            sup = torch.where(no_dyn.view(-1, 1, 1, 1), valid, sup)
        return sup

    def _encode_past_flow(self, batch, device, training_aug=True):
        """Encode the last-K past flow fields -> (motion_hidden (B,K*N,D), gate (B,1,1)).
        Returns (None, None) when disabled/absent. training_aug=True applies Gaussian noise-aug +
        per-sample dropout (only in the supervised flow forward, for train/infer-gap robustness);
        the action-coupler extraction path passes training_aug=False (deterministic contract).
        past_flow_gt=(B,K,3,H,W); past_flow_valid=(B,K,H,W); has_past_flow=(B,)."""
        if not self.past_flow_enabled:
            return None, None
        if batch is None or self.past_flow_key not in batch:
            # Fail-closed: a supervised batch (has the flow target) that lacks past-flow means the
            # dataloader did not emit it (config not mirrored into datasets.vla_data.scene_flow.
            # past_flow_controlnet). Raise instead of silently training as no-past. Inference/
            # extraction with no FIFO yet (batch None or no target) legitimately returns None.
            if batch is not None and self.target_key in batch:
                raise RuntimeError(
                    "past_flow_controlnet.enabled but the batch lacks "
                    f"'{self.past_flow_key}'. Mirror the config into "
                    "datasets.vla_data.scene_flow.past_flow_controlnet (launcher PAST_FLOW=1 "
                    "sets both the framework and dataloader namespaces)."
                )
            return None, None
        past = batch[self.past_flow_key].to(device).float()  # (B,K,3,H,W)
        B, K = past.shape[0], past.shape[1]
        valid = batch.get(self.past_valid_key, None)
        g = self.grid_size
        pooled = []
        for k in range(K):
            fk = past[:, k]  # (B,3,H,W)
            if valid is not None:
                vk = valid[:, k].to(device).float()
                if vk.dim() == 3:
                    vk = vk.unsqueeze(1)
                pk, _ = self._pool_target(fk, vk)
            else:
                pk = F.adaptive_avg_pool2d(torch.nan_to_num(fk), (g, g))
            pooled.append(pk)  # (B,3,g,g)
        fields = torch.stack(pooled, dim=1)  # (B,K,3,g,g)
        fields = fields.flatten(3).permute(0, 1, 3, 2).contiguous()  # (B,K,N,3)
        if self.training and training_aug and self.past_flow_noise_std > 0:
            fields = fields + torch.randn_like(fields) * self.past_flow_noise_std
        motion_hidden = self.past_flow_encoder(fields)  # (B,K*N,D)
        if self.has_past_key in batch:
            gate = batch[self.has_past_key].to(device).float().view(B, 1, 1)
        else:
            gate = torch.ones(B, 1, 1, device=device)
        if self.training and training_aug and self.past_flow_dropout_p > 0:
            keep = (torch.rand(B, 1, 1, device=device) >= self.past_flow_dropout_p).float()
            gate = gate * keep
        return motion_hidden, gate

    # ------------------------------------------------------------------ forward
    def forward(self, vl_embs, batch, encoder_attention_mask=None):
        """Supervised flow-matching forward. Reads target/valid/dynamic from `batch`
        (dataloader keys). Returns {'flow_loss': ...}. NOTE: this pass is teacher-forced
        (uses the target) and MUST NOT be used to condition the action DiT."""
        device = vl_embs.device
        field_gt = batch[self.target_key]  # (B,3,Hgt,Wgt)
        valid = batch.get(self.valid_key, None)
        dynamic = batch.get(self.dynamic_key, None)
        if valid is None:
            valid = torch.ones(field_gt.shape[0], 1, field_gt.shape[2], field_gt.shape[3], device=device)
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        if dynamic is not None and dynamic.dim() == 3:
            dynamic = dynamic.unsqueeze(1)

        target_grid, valid_grid = self._pool_target(field_gt.to(device), valid.to(device))
        dyn_grid = None
        if dynamic is not None:
            dyn_grid = F.adaptive_avg_pool2d(dynamic.float().to(device), (self.grid_size, self.grid_size)) > 0.5
        sup = self._supervision_mask(valid_grid, dyn_grid)  # (B,1,g,g)
        if "flow_has_gt" in batch:  # dummy no-GT samples carry zero flow_gt -> don't supervise (review H2)
            sup = sup & batch["flow_has_gt"].to(sup.device).view(-1, 1, 1, 1)

        target = target_grid.flatten(2).permute(0, 2, 1).contiguous()  # (B,N,3)
        noise = torch.randn_like(target)
        t = self.sample_time(target.shape[0], target.device, target.dtype)[:, None, None]
        noised = (1 - t) * noise + t * target
        velocity = target - noise
        t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long().clamp_(max=self.num_timestep_buckets - 1)

        feats = self._encode(noised, t_disc)
        _pf_hidden, _pf_gate = self._encode_past_flow(batch, device)
        out = self.model(
            hidden_states=feats,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_disc,
            motion_hidden=_pf_hidden,
            motion_gate=_pf_gate,
        )
        pred_v = self.field_decoder(out)  # (B,N,3)

        # fp32 island for the loss (round-2 code_detail)
        with torch.autocast(device_type="cuda", enabled=False):
            pv = pred_v.float()
            vt = velocity.float()
            diff = F.smooth_l1_loss(pv, vt, reduction="none", beta=self.smooth_l1_beta)
            ch_w = torch.ones(self.field_dim, device=pv.device, dtype=pv.dtype)
            ch_w[-1] = self.z_weight  # z = last channel (camera-frame XYZ)
            diff = diff * ch_w.view(1, 1, -1)
            m = sup.reshape(sup.shape[0], -1, 1).float()  # (B,N,1)
            supervised_cells = sup.sum()  # for the trainer's flow_only zero-supervision guard (H2)
            denom = (m.sum() * float(self.field_dim)).clamp(min=1.0)
            loss = (diff * m).sum() / denom
        return {
            "flow_loss": loss,
            "field_pred_velocity": pred_v.detach(),
            "flow_supervised_cells": supervised_cells.detach(),
        }

    # -------------------------------------------------- conditioning extraction
    def extract_conditioning_hidden(self, vl_embs, encoder_attention_mask=None, past_flow_batch=None):
        """Deterministic, GT-FREE forward -> tapped mid-late hidden for the action
        coupler (round-2 N1/N5). Never sees the target. Detached by default (v1);
        grad flows only in the end-to-end ablation (detach_conditioning=False)."""
        B = vl_embs.shape[0]
        if self.extract_noise_mode == "zeros":
            noised = torch.zeros(B, self.num_tokens, self.field_dim, device=vl_embs.device, dtype=vl_embs.dtype)
        elif self.extract_noise_mode == "fixed":
            gen = torch.Generator(device=vl_embs.device).manual_seed(0)
            noised = torch.randn(
                B, self.num_tokens, self.field_dim, generator=gen, device=vl_embs.device, dtype=vl_embs.dtype
            )
        else:
            raise ValueError(f"motion_extract_noise_mode must be 'zeros'|'fixed', got {self.extract_noise_mode!r}")
        t_disc = torch.full((B,), self.extract_t_bucket, device=vl_embs.device, dtype=torch.long)
        feats = self._encode(noised, t_disc)
        _pf_hidden, _pf_gate = self._encode_past_flow(past_flow_batch, vl_embs.device, training_aug=False)
        with torch.set_grad_enabled(not self.detach_conditioning):
            _, all_hidden = self.model(
                hidden_states=feats,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_disc,
                return_all_hidden_states=True,
                motion_hidden=_pf_hidden,
                motion_gate=_pf_gate,
            )
        hidden = all_hidden[self.tap_hidden_index]  # (B, N, D)
        return hidden.detach() if self.detach_conditioning else hidden

    @torch.no_grad()
    def sample_field(self, vl_embs, encoder_attention_mask=None, past_flow_batch=None, num_steps=None):
        """Sample a flow FIELD by Euler-integrating the flow-matching ODE from noise (rollout /
        offline eval). Reuses _encode + self.model + field_decoder; conditions on vl_embs + the
        past-flow branch. Returns (B, field_dim, grid, grid). Deterministic-ish (past-flow aug off).
        fp32 start noise: the scene DiT runs under an fp32 autocast island; a bf16 start would
        accumulate rounding across the ODE."""
        B = vl_embs.shape[0]
        device = vl_embs.device
        num_steps = int(num_steps or self.sample_num_steps)
        dt = 1.0 / num_steps
        x = torch.randn(B, self.num_tokens, self.field_dim, device=device, dtype=torch.float32)
        pf_hidden, pf_gate = self._encode_past_flow(past_flow_batch, device, training_aug=False)
        for i in range(num_steps):
            t_disc = torch.full(
                (B,), int(i / num_steps * self.num_timestep_buckets), device=device, dtype=torch.long
            )
            feats = self._encode(x, t_disc)
            out = self.model(
                hidden_states=feats,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_disc,
                motion_hidden=pf_hidden,
                motion_gate=pf_gate,
            )
            x = x + dt * self.field_decoder(out)  # (B,N,3)
        return x.permute(0, 2, 1).reshape(B, self.field_dim, self.grid_size, self.grid_size)
