# QwenPI depth-token FFS variant with LoRA adapters on the Qwen VLM.
#
# This keeps QwenPIDepthTokenFFS intact for the frozen-VLM Run A path. The
# parent builds FFS, stereo cam-rope, and depth-token hooks first; this subclass
# then wraps the already-hooked HF model with PEFT LoRA for Run B.

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import logging

import torch

from starVLA.model.framework.VLM4A.QwenPI_DepthTokenFFS import (
    QwenPI_DepthTokenFFS,
    QwenPIDepthTokenFFSDefaultConfig,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


def _lora_ffs_depth_token_defaults():
    cfg = dict(QwenPIDepthTokenFFSDefaultConfig().ffs_depth_token)
    cfg.update(
        {
            "vlm_lora": True,
            "vlm_lora_r": 16,
            "vlm_lora_alpha": 32,
            "vlm_lora_dropout": 0.0,
            "vlm_lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
    )
    return cfg


@dataclass
class QwenPIDepthTokenLoRAFFSDefaultConfig(QwenPIDepthTokenFFSDefaultConfig):
    name: str = "QwenPIDepthTokenLoRAFFS"
    ffs_depth_token: dict = field(default_factory=_lora_ffs_depth_token_defaults)


@FRAMEWORK_REGISTRY.register("QwenPIDepthTokenLoRAFFS")
class QwenPI_DepthTokenLoRAFFS(QwenPI_DepthTokenFFS):
    """QwenPI depth-token FFS with trainable LoRA adapters on the VLM."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIDepthTokenLoRAFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_depth_token", {})
        self.vlm_lora_enabled = bool(ffs_cfg.get("vlm_lora", True))
        self.vlm_lora_r = int(ffs_cfg.get("vlm_lora_r", 16))
        self.vlm_lora_alpha = int(ffs_cfg.get("vlm_lora_alpha", 32))
        self.vlm_lora_dropout = float(ffs_cfg.get("vlm_lora_dropout", 0.0))
        self.vlm_lora_target_modules = self._normalize_target_modules(
            ffs_cfg.get("vlm_lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        )
        self.vlm_lora_active = False

        if not self.vlm_lora_enabled:
            logger.warning("[DepthToken-LoRA] vlm_lora=False; running parent depth-token FFS path")
            return

        self._validate_lora_targets_exist()
        self._apply_vlm_lora()
        self._enable_lora_input_grads()
        self.vlm_lora_active = True

        trainable_lora, trainable_vlm = self._count_trainable_lora_params()
        logger.info(
            "[DepthToken-LoRA] active: r=%s alpha=%s dropout=%s targets=%s "
            "trainable_lora_params=%s trainable_vlm_params=%s",
            self.vlm_lora_r,
            self.vlm_lora_alpha,
            self.vlm_lora_dropout,
            self.vlm_lora_target_modules,
            trainable_lora,
            trainable_vlm,
        )
        if trainable_lora <= 0:
            raise RuntimeError("[DepthToken-LoRA] PEFT wrapping produced no trainable LoRA params")

    @staticmethod
    def _normalize_target_modules(raw_targets):
        if isinstance(raw_targets, str):
            return [x.strip() for x in raw_targets.split(",") if x.strip()]
        return [str(x) for x in list(raw_targets)]

    def _validate_lora_targets_exist(self) -> None:
        module_leaf_names = {
            name.rsplit(".", 1)[-1]
            for name, _module in self.qwen_vl_interface.model.named_modules()
            if name
        }
        missing = [name for name in self.vlm_lora_target_modules if name not in module_leaf_names]
        if missing:
            raise RuntimeError(
                "[DepthToken-LoRA] requested LoRA target module(s) not found in Qwen VLM: "
                f"{missing}; available sample={sorted(module_leaf_names)[:40]}"
            )

    def _apply_vlm_lora(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "[DepthToken-LoRA] peft is required for QwenPIDepthTokenLoRAFFS"
            ) from exc

        lora_config = LoraConfig(
            r=self.vlm_lora_r,
            lora_alpha=self.vlm_lora_alpha,
            target_modules=self.vlm_lora_target_modules,
            lora_dropout=self.vlm_lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.qwen_vl_interface.model = get_peft_model(self.qwen_vl_interface.model, lora_config)
        peft_model = self.qwen_vl_interface.model
        self._reinstall_depth_token_outer_hook(peft_model)

    def _reinstall_depth_token_outer_hook(self, top_module) -> None:
        handles = getattr(self, "_depth_token_hook_handles", None)
        if not handles:
            return

        old_outer_handle = handles[0]
        reinstall = getattr(old_outer_handle, "reinstall_outer_hook", None)
        hook_fn = getattr(old_outer_handle, "depth_token_outer_hook_fn", None)
        if reinstall is None and hook_fn is None:
            raise RuntimeError(
                "[DepthToken-LoRA] depth-token outer pre-hook metadata is unavailable; "
                "cannot re-register it on the PEFT wrapper"
            )

        old_outer_handle.remove()
        if reinstall is not None:
            new_outer_handle = reinstall(top_module)
        else:
            new_outer_handle = top_module.register_forward_pre_hook(hook_fn, with_kwargs=True)
            setattr(new_outer_handle, "depth_token_outer_hook_fn", hook_fn)

        self._depth_token_hook_handles = (new_outer_handle, *tuple(handles[1:]))
        logger.info("[DepthToken-LoRA] re-registered depth-token outer pre-hook on PEFT wrapper")

    def _enable_lora_input_grads(self) -> None:
        if hasattr(self.qwen_vl_interface, "enable_input_require_grads"):
            self.qwen_vl_interface.enable_input_require_grads()
            return
        if hasattr(self.qwen_vl_interface.model, "enable_input_require_grads"):
            self.qwen_vl_interface.model.enable_input_require_grads()
            return
        raise RuntimeError(
            "[DepthToken-LoRA] VLM does not expose enable_input_require_grads(); "
            "LoRA with gradient checkpointing may not receive gradients"
        )

    def _count_trainable_lora_params(self):
        trainable_lora = 0
        trainable_vlm = 0
        for name, param in self.qwen_vl_interface.model.named_parameters():
            if param.requires_grad:
                trainable_vlm += param.numel()
                if self._is_lora_param_name(name):
                    trainable_lora += param.numel()
        return trainable_lora, trainable_vlm

    @staticmethod
    def _is_lora_param_name(name: str) -> bool:
        return (
            ".lora_A." in name
            or ".lora_B." in name
            or ".lora_embedding_A" in name
            or ".lora_embedding_B" in name
        )

    def _remap_non_peft_vlm_key(self, key: str, own_keys: set[str]) -> str:
        prefix = "qwen_vl_interface.model."
        peft_prefix = prefix + "base_model.model."
        if not key.startswith(prefix) or key.startswith(peft_prefix):
            return key

        remapped = peft_prefix + key[len(prefix):]
        if remapped in own_keys:
            return remapped

        for target_name in self.vlm_lora_target_modules:
            needle = f".{target_name}."
            if needle not in remapped:
                continue
            if remapped.endswith(".weight") or remapped.endswith(".bias"):
                candidate = remapped.replace(needle, f".{target_name}.base_layer.", 1)
                if candidate in own_keys:
                    return candidate
        return remapped

    def _maybe_remap_baseline_state_dict(self, state_dict, own_keys: set[str]):
        prefix = "qwen_vl_interface.model."
        peft_prefix = prefix + "base_model.model."
        vlm_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
        if not vlm_keys or any(k.startswith(peft_prefix) for k in vlm_keys):
            return state_dict, 0

        remapped = OrderedDict()
        n_remapped = 0
        for key, value in state_dict.items():
            new_key = self._remap_non_peft_vlm_key(key, own_keys)
            if new_key != key:
                n_remapped += 1
            remapped[new_key] = value
        if hasattr(state_dict, "_metadata"):
            remapped._metadata = state_dict._metadata
        return remapped, n_remapped

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        if not getattr(self, "vlm_lora_active", False):
            return super().load_state_dict(
                state_dict,
                strict=strict,
                assign=assign,
                init_from_baseline=init_from_baseline,
            )

        own_keys = set(self.state_dict().keys())
        if init_from_baseline:
            state_dict, n_remapped = self._maybe_remap_baseline_state_dict(state_dict, own_keys)
            if n_remapped:
                logger.info(
                    "[DepthToken-LoRA audit] remapped %s non-PEFT VLM warm-start key(s) "
                    "under qwen_vl_interface.model.base_model.model",
                    n_remapped,
                )

        provided = set(state_dict.keys())

        def _is_branch_key(k: str) -> bool:
            return k.startswith("ffs.") or k.startswith("depth_token_projector.")

        def _is_depth_projector_key(k: str) -> bool:
            return k.startswith("depth_token_projector.")

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    ".language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        def _is_lora_key(k: str) -> bool:
            return k.startswith("qwen_vl_interface.model.") and self._is_lora_param_name(k)

        expected_depth_keys = {k for k in own_keys if _is_depth_projector_key(k)}
        provided_depth_keys = {k for k in provided if _is_depth_projector_key(k)}
        if provided_depth_keys and provided_depth_keys != expected_depth_keys:
            missing = sorted(expected_depth_keys - provided_depth_keys)[:20]
            extra = sorted(provided_depth_keys - expected_depth_keys)[:20]
            raise RuntimeError(
                "[DepthToken-LoRA audit] partial depth-token projector checkpoint detected; "
                f"missing_first={missing}, unexpected_depth_first={extra}. "
                "Provide all depth_token_projector keys or none for baseline warm-start."
            )

        if init_from_baseline:
            allowed_missing = {
                k for k in own_keys - provided
                if _is_branch_key(k) or _is_stereo_cam_rope_key(k) or _is_lora_key(k)
            }
        else:
            allowed_missing = set()

        suspicious_missing = (own_keys - provided) - allowed_missing
        unexpected = provided - own_keys
        lora_missing = {k for k in allowed_missing if _is_lora_key(k)}

        if allowed_missing:
            logger.info(
                "[DepthToken-LoRA audit] ckpt missing %s optional key(s) "
                "(ffs / depth_token_projector / stereo_cam_rope / lora)",
                len(allowed_missing),
            )
        if lora_missing:
            logger.info(
                "[DepthToken-LoRA audit] initializing %s LoRA adapter key(s) fresh",
                len(lora_missing),
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if strict:
                raise RuntimeError(
                    f"[DepthToken-LoRA audit] {len(suspicious_missing)} unexpected missing keys "
                    f"under strict=True; first: {sample}"
                )
            logger.warning(
                "[DepthToken-LoRA audit] %s missing keys (strict=False, accepting); first: %s",
                len(suspicious_missing),
                sample[:5],
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[DepthToken-LoRA audit] {len(unexpected)} UNEXPECTED keys in ckpt not present "
                f"in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " -- refusing silent drop.")
            logger.warning(msg + " -- dropping these keys (strict=False).")
            state_dict = OrderedDict((k, v) for k, v in state_dict.items() if k not in unexpected)

        forwarded_strict = strict and not allowed_missing
        return super(QwenPI_DepthTokenFFS, self).load_state_dict(
            state_dict,
            strict=forwarded_strict,
            assign=assign,
        )
