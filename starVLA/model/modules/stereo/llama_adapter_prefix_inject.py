from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MethodType
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_rope_hook import _PATCHED_FORWARD_DISPATCH, compute_per_token_cam_id

logger = logging.getLogger(__name__)

DEFAULT_PREFIX_LAYERS = [3, 7, 11, 15, 19, 23]


def _language_model_layers(hf_model) -> nn.ModuleList:
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("[llama_prefix] could not locate language_model.layers")
    return lm.layers


def _layer_attention(layer: nn.Module) -> Optional[nn.Module]:
    for child_name in ("self_attn", "attention", "attn"):
        if hasattr(layer, child_name):
            return getattr(layer, child_name)
    return None


def _discover_supported_attention_layers(hf_model) -> List[Tuple[int, nn.Module]]:
    supported = set(_PATCHED_FORWARD_DISPATCH.keys())
    out: List[Tuple[int, nn.Module]] = []
    for layer_idx, layer in enumerate(_language_model_layers(hf_model)):
        attn = _layer_attention(layer)
        if attn is not None and type(attn).__name__ in supported:
            out.append((layer_idx, attn))
    return out


def _text_config(hf_model):
    cfg = hf_model.config
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


def _base_hidden_size(hf_model) -> int:
    cfg = _text_config(hf_model)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise RuntimeError("[llama_prefix] hf_model.config has no hidden_size")


def _module_base_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    raise RuntimeError(f"[llama_prefix] module {type(module).__name__} has no parameters")


def _make_2d_sincos_posenc(
    h: int,
    w: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"[llama_prefix] absorb_dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(quarter, 1)))
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x_freq = xx.reshape(-1, 1) * omega.reshape(1, -1)
    y_freq = yy.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat(
        [torch.sin(x_freq), torch.cos(x_freq), torch.sin(y_freq), torch.cos(y_freq)],
        dim=-1,
    )
    return pos.to(dtype=dtype)


def _linear_with_detached_params(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    weight = linear.weight.detach().to(device=x.device, dtype=x.dtype)
    bias = linear.bias.detach().to(device=x.device, dtype=x.dtype) if linear.bias is not None else None
    return F.linear(x, weight, bias)


def _rms_norm_with_detached_weight(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = getattr(norm, "weight", None)
    if weight is None:
        return norm(x)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    out = (x.float() * scale).to(dtype=x.dtype)
    return out * weight.detach().to(device=x.device, dtype=x.dtype)


def _attention_arg(args, kwargs, name: str, pos: int):
    if name in kwargs:
        return kwargs[name]
    if len(args) > pos:
        return args[pos]
    return None


def _query_valid_mask_from_attention_mask(
    attention_mask: Optional[torch.Tensor],
    fallback_mask: Optional[torch.Tensor],
    *,
    batch: int,
    seqlen: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    mask = attention_mask
    if torch.is_tensor(mask) and mask.ndim == 2 and tuple(mask.shape[:2]) == (batch, seqlen):
        return mask.to(device=device).bool()
    if fallback_mask is not None and tuple(fallback_mask.shape[:2]) == (batch, seqlen):
        return fallback_mask.to(device=device).bool()
    return None


def _masked_mean_l2(x: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is None or mask.numel() == 0:
        return 0.0
    mask = mask.to(device=x.device, dtype=torch.bool)
    if not bool(mask.any().detach().cpu()):
        return 0.0
    vals = x.detach().float().norm(dim=-1)
    return float(vals[mask].mean().cpu())


@dataclass
class LlamaAdapterPrefixState:
    summary: Optional[torch.Tensor] = None
    valid_query_mask: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    forward_step: int = 0
    logging_frequency: int = 20
    enabled: bool = True
    last_logged_step: Dict[int, int] = field(default_factory=dict)
    top_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    top_pre_hook_fn: Optional[Callable] = None
    prepared_net0_shape: Optional[Tuple[int, ...]] = None
    summary_shape: Optional[Tuple[int, ...]] = None
    summary_ready_count: int = 0

    def clear_summary(self) -> None:
        self.summary = None
        self.valid_query_mask = None
        self.per_token_cam_id = None
        self.prepared_net0_shape = None
        self.summary_shape = None


class DepthPromptAbsorber(nn.Module):
    """Absorb FFS net[0] into a small prompt summary with fixed 2D position codes."""

    def __init__(
        self,
        in_ch: int,
        hidden_dim: int,
        *,
        n_prompts: int = 10,
        absorb_dim: int = 256,
        cross_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.hidden_dim = int(hidden_dim)
        self.n_prompts = int(n_prompts)
        self.absorb_dim = int(absorb_dim)
        self.cross_attn_heads = int(cross_attn_heads)
        if self.n_prompts <= 0:
            raise ValueError(f"[llama_prefix] n_prompts must be positive, got {n_prompts}")
        if self.absorb_dim % 4 != 0:
            raise ValueError(f"[llama_prefix] absorb_dim must be divisible by 4, got {absorb_dim}")
        if self.absorb_dim % self.cross_attn_heads != 0:
            raise ValueError(
                f"[llama_prefix] absorb_dim={absorb_dim} must be divisible by "
                f"cross_attn_heads={cross_attn_heads}"
            )

        self.prompts = nn.Parameter(torch.empty(self.n_prompts, self.absorb_dim))
        self.in_proj = nn.Linear(self.in_ch, self.absorb_dim, bias=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.absorb_dim,
            num_heads=self.cross_attn_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.out_proj_absorber = nn.Linear(self.absorb_dim, self.hidden_dim, bias=True)
        nn.init.normal_(self.prompts, mean=0.0, std=0.02)

    def forward(self, net0: torch.Tensor) -> torch.Tensor:
        if net0.ndim != 4:
            raise RuntimeError(f"[llama_prefix] net0 must be [B,C,H,W], got {tuple(net0.shape)}")
        batch, channels, h, w = net0.shape
        if int(channels) != self.in_ch:
            raise RuntimeError(f"[llama_prefix] net0 channels {channels} != configured {self.in_ch}")
        dtype = _module_base_dtype(self)
        x = net0.detach().to(dtype=dtype)
        kv = self.in_proj(x.flatten(2).transpose(1, 2).contiguous())
        pos = _make_2d_sincos_posenc(
            int(h),
            int(w),
            self.absorb_dim,
            device=kv.device,
            dtype=kv.dtype,
        )
        kv = kv + pos.unsqueeze(0)
        prompts = self.prompts.to(device=kv.device, dtype=kv.dtype).unsqueeze(0).expand(batch, -1, -1)
        summary, _ = self.cross_attn(
            query=prompts,
            key=kv,
            value=kv,
            need_weights=False,
        )
        return self.out_proj_absorber(summary)


class PerLayerPrefixAdapter(nn.Module):
    """Per-softmax-layer prefix attention branch with a zero-init tanh gate."""

    def __init__(
        self,
        *,
        layer_idx: int,
        hidden_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        gate_per_head: bool = False,
    ) -> None:
        super().__init__()
        if gate_per_head:
            raise ValueError("[llama_prefix] gate_per_head=True is not part of the finalized spec")
        self.layer_idx = int(layer_idx)
        self.hidden_dim = int(hidden_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"[llama_prefix] layer {layer_idx}: attention heads {num_attention_heads} "
                f"not divisible by kv heads {num_key_value_heads}"
            )
        self.kv_proj = nn.Linear(
            self.hidden_dim,
            2 * self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_debug: Dict[str, object] = {}

    def _recompute_query(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        if position_embeddings is None:
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: missing position_embeddings for Q recompute"
            )
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: expected position_embeddings=(cos,sin), "
                f"got {type(position_embeddings).__name__}"
            )
        batch, seqlen, hidden = hidden_states.shape
        if int(hidden) != self.hidden_dim:
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: hidden dim {hidden} != {self.hidden_dim}"
            )

        q_dtype = attn.q_proj.weight.dtype
        q_raw = _linear_with_detached_params(attn.q_proj, hidden_states.to(dtype=q_dtype))
        expected = self.num_attention_heads * self.head_dim * 2
        if int(q_raw.shape[-1]) != expected:
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: q_proj out {q_raw.shape[-1]} != "
                f"2*num_heads*head_dim {expected}"
            )
        q_chunked = q_raw.view(batch, seqlen, self.num_attention_heads, self.head_dim * 2)
        query_states = torch.chunk(q_chunked, 2, dim=-1)[0]
        query_states = _rms_norm_with_detached_weight(attn.q_norm, query_states).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        return query_states

    def forward_from_attention(
        self,
        *,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        summary: torch.Tensor,
        position_embeddings,
        valid_query_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if summary.ndim != 3:
            raise RuntimeError(f"[llama_prefix] summary must be [B,N,H], got {tuple(summary.shape)}")
        batch, seqlen, _ = hidden_states.shape
        if int(summary.shape[0]) != int(batch):
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: summary batch {summary.shape[0]} != hidden batch {batch}"
            )
        if int(summary.shape[-1]) != self.hidden_dim:
            raise RuntimeError(
                f"[llama_prefix] layer {self.layer_idx}: summary dim {summary.shape[-1]} != {self.hidden_dim}"
            )

        query_states = self._recompute_query(attn, hidden_states, position_embeddings)
        prefix = self.kv_proj(summary.to(dtype=self.kv_proj.weight.dtype))
        prefix = prefix.view(
            batch,
            summary.shape[1],
            2,
            self.num_key_value_heads,
            self.head_dim,
        )
        pk = prefix[:, :, 0].transpose(1, 2).contiguous()
        pv = prefix[:, :, 1].transpose(1, 2).contiguous()
        groups = self.num_attention_heads // self.num_key_value_heads
        pk_rep = pk.repeat_interleave(groups, dim=1)
        pv_rep = pv.repeat_interleave(groups, dim=1)

        prefix_out = F.scaled_dot_product_attention(
            query_states,
            pk_rep.to(dtype=query_states.dtype),
            pv_rep.to(dtype=query_states.dtype),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        merged = prefix_out.transpose(1, 2).reshape(batch, seqlen, self.num_attention_heads * self.head_dim)
        projected = _linear_with_detached_params(attn.o_proj, merged.to(dtype=attn.o_proj.weight.dtype))
        projected = projected.to(dtype=hidden_states.dtype)
        if valid_query_mask is not None:
            projected = projected.masked_fill(~valid_query_mask.to(device=projected.device).unsqueeze(-1), 0)

        self.last_debug = {
            "q_shape": tuple(query_states.shape),
            "pk_shape": tuple(pk.shape),
            "pv_shape": tuple(pv.shape),
            "pk_repeat_shape": tuple(pk_rep.shape),
            "pv_repeat_shape": tuple(pv_rep.shape),
            "merged_shape": tuple(merged.shape),
            "projected_shape": tuple(projected.shape),
            "groups": int(groups),
        }
        return projected


class LlamaAdapterPrefixModule(nn.Module):
    """Single registered parent for all FFS prefix-adapter parameters."""

    def __init__(
        self,
        *,
        in_ch: int,
        hidden_dim: int,
        layer_specs: Sequence[Tuple[int, int, int, int]],
        n_prompts: int = 10,
        absorb_dim: int = 256,
        gate_per_head: bool = False,
        logging_frequency: int = 20,
    ) -> None:
        super().__init__()
        self.state = LlamaAdapterPrefixState(logging_frequency=max(int(logging_frequency), 0))
        self.absorber = DepthPromptAbsorber(
            in_ch=in_ch,
            hidden_dim=hidden_dim,
            n_prompts=n_prompts,
            absorb_dim=absorb_dim,
        )
        self.adapters = nn.ModuleList(
            [
                PerLayerPrefixAdapter(
                    layer_idx=layer_idx,
                    hidden_dim=hidden_dim,
                    num_attention_heads=n_heads,
                    num_key_value_heads=n_kv_heads,
                    head_dim=head_dim,
                    gate_per_head=gate_per_head,
                )
                for layer_idx, n_heads, n_kv_heads, head_dim in layer_specs
            ]
        )

    def clear_summary(self) -> None:
        self.state.clear_summary()

    def prepare_from_net0(self, net0: torch.Tensor) -> torch.Tensor:
        self.clear_summary()
        self.state.forward_step += 1
        self.state.prepared_net0_shape = tuple(net0.shape)
        summary = self.absorber(net0)
        self.state.summary = summary
        self.state.summary_shape = tuple(summary.shape)
        self.state.summary_ready_count += 1
        return summary

    def assert_all_gates_zero(self, label: str = "[llama_prefix]") -> None:
        bad = []
        for adapter in self.adapters:
            if float(adapter.gate.detach().abs().cpu()) != 0.0:
                bad.append((adapter.layer_idx, float(adapter.gate.detach().cpu())))
        if bad:
            raise RuntimeError(f"{label} expected all prefix gates to be zero after warm-start, got {bad}")


def _make_prefix_forward(
    *,
    orig_forward: Callable,
    adapter: PerLayerPrefixAdapter,
    state: LlamaAdapterPrefixState,
    layer_idx: int,
) -> Callable:
    def _forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        output = orig_forward(hidden_states, *args, **kwargs)
        if not state.enabled or state.summary is None:
            return output

        attn_output = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(attn_output) or attn_output.ndim != 3:
            raise RuntimeError(
                f"[llama_prefix] layer {layer_idx} expected attention output [B,S,H], "
                f"got {type(attn_output).__name__}"
            )

        position_embeddings = _attention_arg(args, kwargs, "position_embeddings", 0)
        attention_mask = _attention_arg(args, kwargs, "attention_mask", 1)
        valid_mask = _query_valid_mask_from_attention_mask(
            attention_mask,
            state.valid_query_mask,
            batch=int(hidden_states.shape[0]),
            seqlen=int(hidden_states.shape[1]),
            device=hidden_states.device,
        )

        prefix_out = adapter.forward_from_attention(
            attn=self,
            hidden_states=hidden_states,
            summary=state.summary.to(device=hidden_states.device),
            position_embeddings=position_embeddings,
            valid_query_mask=valid_mask,
        ).to(device=attn_output.device, dtype=attn_output.dtype)
        gate = torch.tanh(adapter.gate.to(device=attn_output.device, dtype=attn_output.dtype))

        if (
            state.logging_frequency > 0
            and state.forward_step % state.logging_frequency == 0
            and state.last_logged_step.get(layer_idx) != state.forward_step
        ):
            state.last_logged_step[layer_idx] = state.forward_step
            image_mask = None
            text_mask = None
            pad_mask = None
            if valid_mask is not None:
                valid = valid_mask.to(device=prefix_out.device, dtype=torch.bool)
                pad_mask = ~valid
            if state.per_token_cam_id is not None and tuple(state.per_token_cam_id.shape[:2]) == tuple(prefix_out.shape[:2]):
                cam = state.per_token_cam_id.to(device=prefix_out.device)
                image_mask = cam >= 0
                text_mask = ~image_mask
                if valid_mask is not None:
                    text_mask = text_mask & valid_mask.to(device=prefix_out.device, dtype=torch.bool)
                    image_mask = image_mask & valid_mask.to(device=prefix_out.device, dtype=torch.bool)

            prefix_l2 = float(prefix_out.detach().float().norm().cpu())
            attn_l2 = float(attn_output.detach().float().norm().cpu())
            ratio = prefix_l2 / max(attn_l2, 1.0e-12)
            logger.info(
                "[llama_prefix] fwd=%d layer=%d gate=%.6g prefix_l2=%.6g attn_l2=%.6g "
                "ratio=%.6g row_mean_l2(image/text/pad)=%.6g/%.6g/%.6g",
                state.forward_step,
                layer_idx,
                float(gate.detach().float().cpu()),
                prefix_l2,
                attn_l2,
                ratio,
                _masked_mean_l2(prefix_out, image_mask),
                _masked_mean_l2(prefix_out, text_mask),
                _masked_mean_l2(prefix_out, pad_mask),
            )

        out = attn_output + gate * prefix_out
        if isinstance(output, tuple):
            return (out, *output[1:])
        return out

    return _forward


def install_llama_adapter_prefix(
    hf_model,
    *,
    in_ch: int,
    n_prompts: int = 10,
    absorb_dim: int = 256,
    gate_per_head: bool = False,
    num_cameras: int = 2,
    spatial_merge_size: int = 2,
    extra_image_cam_id: Optional[int] = None,
    logging_frequency: int = 20,
) -> LlamaAdapterPrefixModule:
    supported = _discover_supported_attention_layers(hf_model)
    target_layers = [idx for idx, _ in supported]
    if target_layers != DEFAULT_PREFIX_LAYERS:
        raise RuntimeError(
            f"[llama_prefix] expected exactly the 6 Qwen softmax layers {DEFAULT_PREFIX_LAYERS}, "
            f"got {target_layers}. Refusing to install on an unreviewed architecture."
        )

    hidden_dim = _base_hidden_size(hf_model)
    layer_specs: List[Tuple[int, int, int, int]] = []
    for layer_idx, attn in supported:
        n_heads = int(getattr(attn, "num_attention_heads", getattr(attn, "num_heads", 0)))
        n_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(attn, "num_key_value_heads", 0)))
        head_dim = int(getattr(attn, "head_dim", 0))
        if n_heads <= 0 or n_kv_heads <= 0 or head_dim <= 0:
            cfg = _text_config(hf_model)
            n_heads = int(getattr(cfg, "num_attention_heads", n_heads))
            n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_kv_heads))
            head_dim = int(getattr(cfg, "head_dim", head_dim))
        if n_heads <= 0 or n_kv_heads <= 0 or head_dim <= 0:
            raise RuntimeError(f"[llama_prefix] could not derive attention shape for layer {layer_idx}")
        layer_specs.append((int(layer_idx), n_heads, n_kv_heads, head_dim))

    module = LlamaAdapterPrefixModule(
        in_ch=in_ch,
        hidden_dim=hidden_dim,
        layer_specs=layer_specs,
        n_prompts=n_prompts,
        absorb_dim=absorb_dim,
        gate_per_head=gate_per_head,
        logging_frequency=logging_frequency,
    )
    module = module.to(dtype=_module_base_dtype(supported[0][1]))
    state = module.state
    image_token_id = int(hf_model.config.image_token_id)

    for adapter, (layer_idx, attn) in zip(module.adapters, supported):
        if getattr(attn, "_ffs_prefix_adapter_installed", False):
            raise RuntimeError(f"[llama_prefix] layer {layer_idx} already has prefix adapter installed")
        orig_forward = attn.forward
        attn._ffs_prefix_adapter_installed = True
        attn._ffs_prefix_adapter_layer_idx = int(layer_idx)
        attn._ffs_prefix_adapter_orig_forward = orig_forward
        attn.forward = MethodType(
            _make_prefix_forward(
                orig_forward=orig_forward,
                adapter=adapter,
                state=state,
                layer_idx=int(layer_idx),
            ),
            attn,
        )

    def _pre_forward_hook(module_, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        attention_mask = kwargs.get("attention_mask", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.valid_query_mask = None
        state.per_token_cam_id = None
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
            state.valid_query_mask = attention_mask.to(dtype=torch.bool)
        if input_ids is None or image_grid_thw is None:
            return None
        state.per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=int(num_cameras),
            spatial_merge_size=int(spatial_merge_size),
            extra_image_cam_id=extra_image_cam_id,
        )
        return None

    state.top_pre_hook_fn = _pre_forward_hook
    state.top_pre_hook_handle = hf_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    logger.info(
        "[llama_prefix] installed on language_model.layers%s "
        "(n_prompts=%d, absorb_dim=%d, num_cameras=%d, spatial_merge=%d, gate_per_head=%s)",
        target_layers,
        int(n_prompts),
        int(absorb_dim),
        int(num_cameras),
        int(spatial_merge_size),
        bool(gate_per_head),
    )
    return module
