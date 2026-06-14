"""FALCON-style spatial injector for GR00T future tokens.

`PCDInjector_V1` is ported from FALCON's policy head implementation
(`falcon/model/policy_head/base_policy.py`) under the Apache-2.0 license, then
used here with a zero-initialized final projection so step-0 matches baseline.
"""
from __future__ import annotations

import logging
import weakref
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def zero_init(linear_layer: nn.Linear) -> None:
    """Zero-initialize weight and bias of a Linear layer."""
    with torch.no_grad():
        nn.init.zeros_(linear_layer.weight)
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)


class PCDInjector_V1(nn.Module):
    """FALCON PCD injector, vendored under Apache-2.0 with a zero final gate."""

    def __init__(self, point_cloud_dim, hidden_size):
        super().__init__()
        self.proj0 = nn.Sequential(
            nn.Linear(point_cloud_dim, 512, bias=True),
            nn.LayerNorm(512),
        )
        self.adapter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
        )
        self.proj1 = nn.Sequential(
            nn.Linear(1024, hidden_size, bias=True)
        )
        zero_init(self.proj1[0])

    def forward(self, pcd_feat: torch.Tensor) -> torch.Tensor:
        # pcd_feat: (B*L, point_cloud_dim)
        h = self.proj0(pcd_feat)
        h = self.adapter(h)
        h = self.proj1(h)         # -> (B*L, hidden_size)
        return h


class FalconSpatialInjectModule(nn.Module):
    """Owns the FFS net[0] global-pool adapter and action-head bias lifecycle."""

    def __init__(
        self,
        action_model: nn.Module,
        *,
        in_ch: int,
        pooling: str = "amax",
        logging_frequency: int = 20,
        label: str = "[falcon_spatial]",
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.d_dit = int(action_model.future_tokens.weight.shape[-1])
        self.pooling = str(pooling)
        self.logging_frequency = max(int(logging_frequency), 0)
        self.label = str(label)
        self.injector = PCDInjector_V1(self.in_ch, self.d_dit)

        # Keep a weak reference so this wrapper does not register the action head
        # as a child module and duplicate its state_dict keys.
        object.__setattr__(self, "_action_model_ref", weakref.ref(action_model))
        action_model._ffs_spatial_bias = None

        self.forward_step = 0
        self.set_count = 0
        self.clear_count = 0
        self.last_net0_shape: Optional[Tuple[int, ...]] = None
        self.last_pool_shape: Optional[Tuple[int, ...]] = None
        self.last_spatial_vec_shape: Optional[Tuple[int, ...]] = None
        self.last_bias_shape: Optional[Tuple[int, ...]] = None
        self.last_net0_stats: Dict[str, float] = {}

    @property
    def action_model(self) -> nn.Module:
        action_model = self._action_model_ref()
        if action_model is None:
            raise RuntimeError(f"{self.label} action_model reference is no longer alive")
        return action_model

    def _base_param(self) -> nn.Parameter:
        return next(self.injector.parameters())

    def _record_net0_stats(self, net0: torch.Tensor) -> None:
        should_log = self.logging_frequency > 0 and (
            self.forward_step == 1 or self.forward_step % self.logging_frequency == 0
        )
        if not should_log:
            return
        stats_t = net0.detach().float()
        self.last_net0_stats = {
            "mean": float(stats_t.mean().cpu()),
            "std": float(stats_t.std(unbiased=False).cpu()),
            "absmax": float(stats_t.abs().max().cpu()),
        }
        logger.info(
            "%s step=%d net0 mean=%.6g std=%.6g absmax=%.6g pooling=%s",
            self.label,
            self.forward_step,
            self.last_net0_stats["mean"],
            self.last_net0_stats["std"],
            self.last_net0_stats["absmax"],
            self.pooling,
        )

    def pool_net0(self, net0: torch.Tensor) -> torch.Tensor:
        if net0.ndim != 4:
            raise RuntimeError(f"{self.label} net0 must be [B,C,H,W], got {tuple(net0.shape)}")
        if int(net0.shape[1]) != self.in_ch:
            raise RuntimeError(
                f"{self.label} net0 channels {net0.shape[1]} != configured in_ch {self.in_ch}"
            )
        pooling = self.pooling.lower()
        if pooling == "amax":
            spatial_feat = net0.detach().amax(dim=(2, 3))
        elif pooling == "mean":
            spatial_feat = net0.detach().mean(dim=(2, 3))
        else:
            raise ValueError(f"{self.label} unsupported pooling={self.pooling!r}; expected 'amax' or 'mean'")
        self.last_pool_shape = tuple(spatial_feat.shape)
        return spatial_feat

    def compute_spatial_vec(self, net0: torch.Tensor) -> torch.Tensor:
        self.forward_step += 1
        self.last_net0_shape = tuple(net0.shape)
        self._record_net0_stats(net0)
        pooled = self.pool_net0(net0)
        param = self._base_param()
        pooled = pooled.to(device=param.device, dtype=param.dtype)
        spatial_vec = self.injector(pooled)
        head_weight = self.action_model.future_tokens.weight
        spatial_vec = spatial_vec.to(device=head_weight.device, dtype=head_weight.dtype)
        if int(spatial_vec.shape[-1]) != self.d_dit:
            raise RuntimeError(
                f"{self.label} spatial_vec dim {spatial_vec.shape[-1]} != action D_dit {self.d_dit}"
            )
        self.last_spatial_vec_shape = tuple(spatial_vec.shape)
        return spatial_vec

    def forward(self, net0: torch.Tensor) -> torch.Tensor:
        return self.compute_spatial_vec(net0)

    def make_spatial_bias(
        self,
        spatial_vec: torch.Tensor,
        *,
        repeated_diffusion_steps: int = 1,
    ) -> torch.Tensor:
        if spatial_vec.ndim != 2:
            raise RuntimeError(
                f"{self.label} spatial_vec must be [B,D_dit], got {tuple(spatial_vec.shape)}"
            )
        if int(spatial_vec.shape[-1]) != self.d_dit:
            raise RuntimeError(
                f"{self.label} spatial_vec dim {spatial_vec.shape[-1]} != action D_dit {self.d_dit}"
            )
        repeat = int(repeated_diffusion_steps)
        if repeat <= 0:
            raise ValueError(f"{self.label} repeated_diffusion_steps must be positive, got {repeat}")
        if repeat != 1:
            spatial_vec = spatial_vec.repeat(repeat, 1)
        bias = spatial_vec.unsqueeze(1).contiguous()
        self.last_bias_shape = tuple(bias.shape)
        return bias

    def set_head_bias(self, spatial_bias: torch.Tensor) -> None:
        if spatial_bias.ndim != 3 or int(spatial_bias.shape[1]) != 1:
            raise RuntimeError(
                f"{self.label} spatial_bias must be [B,1,D_dit], got {tuple(spatial_bias.shape)}"
            )
        if int(spatial_bias.shape[-1]) != self.d_dit:
            raise RuntimeError(
                f"{self.label} spatial_bias dim {spatial_bias.shape[-1]} != action D_dit {self.d_dit}"
            )
        self.action_model._ffs_spatial_bias = spatial_bias
        self.last_bias_shape = tuple(spatial_bias.shape)
        self.set_count += 1

    def clear_head_bias(self) -> None:
        self.action_model._ffs_spatial_bias = None
        self.clear_count += 1

    def assert_spatial_gate_zero(self, label: Optional[str] = None) -> None:
        prefix = label or self.label
        proj1 = self.injector.proj1[0]
        bad = []
        weight_abs = float(proj1.weight.detach().abs().max().cpu())
        if weight_abs != 0.0:
            bad.append(("proj1[0].weight", weight_abs))
        if proj1.bias is not None:
            bias_abs = float(proj1.bias.detach().abs().max().cpu())
            if bias_abs != 0.0:
                bad.append(("proj1[0].bias", bias_abs))
        if bad:
            raise RuntimeError(f"{prefix} expected zero final spatial gate, got {bad}")


def install_falcon_spatial_inject(
    action_model: nn.Module,
    *,
    in_ch: int,
    pooling: str = "amax",
    logging_frequency: int = 20,
    label: str = "[falcon_spatial]",
) -> FalconSpatialInjectModule:
    module = FalconSpatialInjectModule(
        action_model,
        in_ch=int(in_ch),
        pooling=pooling,
        logging_frequency=logging_frequency,
        label=label,
    )
    module = module.to(dtype=action_model.future_tokens.weight.dtype)
    logger.info(
        "%s installed PCDInjector_V1 in_ch=%d D_dit=%d pooling=%s logging_frequency=%d",
        label,
        int(in_ch),
        int(module.d_dit),
        pooling,
        int(logging_frequency),
    )
    return module
