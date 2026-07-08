# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.ffs_cache_startup import maybe_validate_ffs_cache_startup
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    is_main_process,
    normalize_dotlist_args,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg, key, default=False) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if is_main_process():
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._scene_flow_grad_ratio_failure_count = 0
        self._scene_flow_grad_ratio_low_warned = False
        self._scene_flow_step0_audit_done = False
        self._scene_flow_lambda_ema = None
        self._scene_flow_raw_ratio_ema = None
        self._scene_flow_last_gradnorm_update_step = -1
        self._scene_flow_online_zero_stage_checked = False

    def _scene_flow_cfg(self):
        framework = _cfg_get(self.config, "framework", {})
        action_cfg = _cfg_get(framework, "action_model", {})
        return _cfg_get(action_cfg, "scene_flow", {}) or {}

    def _scene_flow_enabled(self) -> bool:
        return _cfg_bool(self._scene_flow_cfg(), "enabled", False)

    def _scene_predictor_cfg(self):
        framework = _cfg_get(self.config, "framework", {})
        return _cfg_get(framework, "scene_predictor", {}) or {}

    def _scene_predictor_enabled(self) -> bool:
        return _cfg_bool(self._scene_predictor_cfg(), "enabled", False)

    def _scene_predictor_lambda(self) -> float:
        return float(_cfg_get(self._scene_predictor_cfg(), "flow_lambda", 0.1))

    def _loss_mode(self) -> str:
        m = str(_cfg_get(_cfg_get(self.config, "trainer", {}), "loss_mode", "joint")).strip().lower()
        if m not in {"joint", "flow_only"}:
            raise ValueError(f"trainer.loss_mode must be 'joint' or 'flow_only', got {m!r} (review H5)")
        return m

    def _maybe_validate_ffs_cache_startup(self) -> None:
        maybe_validate_ffs_cache_startup(
            accelerator=self.accelerator,
            model=self.model,
            vla_train_dataloader=self.vla_train_dataloader,
            raw_dataset=getattr(self, "_ffs_cache_raw_dataset", None),
        )

    def _scene_flow_lambda(self) -> float:
        return float(_cfg_get(self._scene_flow_cfg(), "flow_lambda", 0.05))

    def _scene_flow_online_grad_norm_enabled(self) -> bool:
        return self._scene_flow_enabled() and _cfg_bool(self._scene_flow_cfg(), "online_grad_norm_enabled", False)

    def _ensure_scene_flow_online_controller_state(self) -> None:
        if self._scene_flow_lambda_ema is None:
            self._scene_flow_lambda_ema = self._scene_flow_lambda()

    def _scene_flow_lambda_warmup_multiplier(self, step=None) -> float:
        cfg = self._scene_flow_cfg()
        warmup_steps = max(0, int(_cfg_get(cfg, "lambda_warmup_steps", 100)))
        if warmup_steps <= 0:
            return 1.0
        step0_warmup_steps = max(0, int(_cfg_get(cfg, "step0_flow_warmup_steps", 1)))
        current_step = self.completed_steps if step is None else int(step)
        elapsed_steps = max(0, current_step - step0_warmup_steps + 1)
        return min(1.0, float(elapsed_steps) / float(warmup_steps))

    def _effective_scene_flow_weight(self) -> float:
        if not self._scene_flow_enabled():
            return 0.0
        warmup_steps = int(_cfg_get(self._scene_flow_cfg(), "step0_flow_warmup_steps", 1))
        if self.completed_steps < warmup_steps:
            return 0.0
        if self._scene_flow_online_grad_norm_enabled():
            self._ensure_scene_flow_online_controller_state()
            return float(self._scene_flow_lambda_ema)
        return self._scene_flow_lambda()

    def _grad_norm(self, grads) -> float:
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return 0.0
        return float(torch.sqrt(total).item())

    def _scene_flow_param_groups_for_calibration(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        action_params = []
        decoder_params = []
        for name, param in unwrapped.named_parameters():
            if not param.requires_grad:
                continue
            if "scene_flow_decoder" in name:
                decoder_params.append(param)
            elif name.startswith("action_model.") or ".action_model." in name:
                action_params.append(param)
        return action_params, decoder_params

    def _scene_flow_probe_batch(self, batch_vla, probe_samples: int = 1):
        if isinstance(batch_vla, list):
            return batch_vla[:probe_samples]
        if isinstance(batch_vla, tuple):
            return batch_vla[:probe_samples]
        if isinstance(batch_vla, dict):
            probe = {}
            for key, value in batch_vla.items():
                if torch.is_tensor(value) and value.shape[:1]:
                    probe[key] = value[:probe_samples]
                elif isinstance(value, np.ndarray) and value.shape[:1]:
                    probe[key] = value[:probe_samples]
                elif isinstance(value, list):
                    probe[key] = value[:probe_samples]
                else:
                    probe[key] = value
            return probe
        return batch_vla

    def _grad_sq_norm_tensor(self, grads, ref_tensor) -> torch.Tensor:
        total = None
        device = ref_tensor.device
        for grad in grads:
            if grad is None:
                continue
            value = grad.detach().to(device=device, dtype=torch.float32).pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return total

    def _distributed_sum_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _distributed_max_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    def _scene_flow_tensor_metric(self, value, device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(device=device, dtype=torch.float32)
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    def _scene_flow_global_raw_grad_ratio_metrics_from_losses(self, action_loss, flow_loss) -> dict:
        local_probe_failed = False
        local_probe_exception = None
        action_sq = torch.zeros((), device=action_loss.device, dtype=torch.float32)
        flow_sq = torch.zeros((), device=action_loss.device, dtype=torch.float32)
        try:
            action_params, _ = self._scene_flow_param_groups_for_calibration()
            if not action_params:
                raise RuntimeError("scene-flow online grad-norm needs a non-empty action param group")
            action_grads = torch.autograd.grad(
                action_loss,
                action_params,
                retain_graph=True,
                allow_unused=True,
            )
            flow_action_grads = torch.autograd.grad(
                flow_loss,
                action_params,
                allow_unused=True,
            )
            action_sq = self._grad_sq_norm_tensor(action_grads, action_loss)
            flow_sq = self._grad_sq_norm_tensor(flow_action_grads, action_loss)
        except Exception as exc:
            local_probe_failed = True
            local_probe_exception = exc

        probe_failed = bool(
            self._distributed_max_tensor(torch.tensor(float(local_probe_failed), device=action_loss.device)).item() > 0.0
        )
        if probe_failed:
            if local_probe_exception is not None:
                raise RuntimeError("scene-flow online grad-norm probe autograd failed on this rank") from local_probe_exception
            raise RuntimeError("scene-flow online grad-norm probe autograd failed on another rank")

        action_sq = self._distributed_sum_tensor(action_sq)
        flow_sq = self._distributed_sum_tensor(flow_sq)
        action_norm = float(torch.sqrt(action_sq.clamp_min(0.0)).item())
        flow_norm = float(torch.sqrt(flow_sq.clamp_min(0.0)).item())
        eps = float(_cfg_get(self._scene_flow_cfg(), "grad_norm_epsilon", 1.0e-12))
        raw_ratio = flow_norm / max(action_norm, eps)
        return {
            "action_norm": action_norm,
            "flow_action_norm": flow_norm,
            "raw_ratio": raw_ratio,
        }

    def _scene_flow_grad_ratio_metrics_from_losses(self, action_loss, flow_loss) -> dict:
        action_params, decoder_params = self._scene_flow_param_groups_for_calibration()
        if not action_params or not decoder_params:
            raise RuntimeError(
                f"scene-flow grad-ratio needs non-empty action/decoder param groups, "
                f"got {len(action_params)}/{len(decoder_params)}"
            )
        flow_calibration_loss = self._scene_flow_lambda() * flow_loss
        action_grads = torch.autograd.grad(
            action_loss,
            action_params,
            retain_graph=True,
            allow_unused=True,
        )
        flow_action_grads = torch.autograd.grad(
            flow_calibration_loss,
            action_params,
            retain_graph=True,
            allow_unused=True,
        )
        flow_decoder_grads = torch.autograd.grad(
            flow_calibration_loss,
            decoder_params,
            retain_graph=True,
            allow_unused=True,
        )
        action_norm = self._grad_norm(action_grads)
        flow_action_norm = self._grad_norm(flow_action_grads)
        flow_decoder_norm = self._grad_norm(flow_decoder_grads)
        ratio = flow_action_norm / max(action_norm, 1e-12)
        return {
            "scene_flow/grad_norm_action_loss_action_path": action_norm,
            "scene_flow/grad_norm_weighted_flow_action_path": flow_action_norm,
            "scene_flow/grad_norm_weighted_flow_decoder": flow_decoder_norm,
            "scene_flow/weighted_flow_to_action_grad_ratio": ratio,
        }

    def _maybe_scene_flow_grad_ratio_metrics(self, batch_vla) -> dict:
        cfg = self._scene_flow_cfg()
        if not self._scene_flow_enabled():
            return {}
        if self._scene_flow_online_grad_norm_enabled():
            return {}
        grad_ratio_steps = int(_cfg_get(cfg, "grad_ratio_steps", 20))
        if self.completed_steps >= grad_ratio_steps:
            return {}

        rng_state = self._snapshot_rng_state()
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            probe_batch = self._scene_flow_probe_batch(batch_vla)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = unwrapped.forward(probe_batch)
                action_loss = output_dict["action_loss"]
                flow_loss = output_dict.get("flow_loss", None)
            if flow_loss is None or not torch.is_tensor(flow_loss) or not flow_loss.requires_grad:
                return {}
            metrics = self._scene_flow_grad_ratio_metrics_from_losses(action_loss, flow_loss)
            min_ratio = float(_cfg_get(cfg, "grad_ratio_min_warn", 0.01))
            ratio = float(metrics["scene_flow/weighted_flow_to_action_grad_ratio"])
            if (
                min_ratio > 0
                and ratio < min_ratio
                and self.completed_steps + 1 >= grad_ratio_steps
                and not self._scene_flow_grad_ratio_low_warned
            ):
                logger.warning(
                    f"scene-flow weighted flow/action grad ratio {ratio:.3e} is below "
                    f"grad_ratio_min_warn={min_ratio:.3e}; flow loss may be underweighted"
                )
                self._scene_flow_grad_ratio_low_warned = True
            return metrics
        except Exception as exc:
            self._scene_flow_grad_ratio_failure_count += 1
            warn_every = max(1, int(_cfg_get(cfg, "grad_ratio_failure_warn_every", 10)))
            if self._scene_flow_grad_ratio_failure_count == 1 or self._scene_flow_grad_ratio_failure_count % warn_every == 0:
                logger.warning(
                    f"scene-flow grad-ratio dry-run failed "
                    f"(count={self._scene_flow_grad_ratio_failure_count}): {exc}"
                )
            if _cfg_bool(cfg, "grad_ratio_raise_on_failure", False):
                raise
            return {}
        finally:
            self._restore_rng_state(rng_state)
            self.optimizer.zero_grad()

    def _scene_flow_online_skip_metrics(self, reason: str, optimizer_step: int, supervised_pixels=None) -> dict:
        reason_codes = {
            "missing_flow_loss": 1.0,
            "missing_supervised_pixels": 2.0,
            "low_supervised_pixels": 3.0,
            "degenerate_norm": 4.0,
            "exception": 5.0,
        }
        metrics = {
            "scene_flow/online_update_skipped": 1.0,
            "scene_flow/online_skip_reason_code": reason_codes.get(reason, 0.0),
            f"scene_flow/online_skip_{reason}": 1.0,
            "scene_flow/online_optimizer_step": float(optimizer_step),
            "scene_flow/lambda_ema": float(self._scene_flow_lambda_ema),
            "scene_flow/online_last_update_step": float(self._scene_flow_last_gradnorm_update_step),
        }
        if supervised_pixels is not None:
            metrics["scene_flow/online_probe_supervised_pixels"] = float(supervised_pixels)
        if self._scene_flow_raw_ratio_ema is not None:
            metrics["scene_flow/online_raw_ratio_ema"] = float(self._scene_flow_raw_ratio_ema)
        return metrics

    def _scene_flow_online_state_metrics(self) -> dict:
        if not self._scene_flow_online_grad_norm_enabled():
            return {}
        self._ensure_scene_flow_online_controller_state()
        metrics = {
            "scene_flow/lambda_ema": float(self._scene_flow_lambda_ema),
            "scene_flow/online_last_update_step": float(self._scene_flow_last_gradnorm_update_step),
            "scene_flow/online_lambda_warmup_multiplier": self._scene_flow_lambda_warmup_multiplier(
                step=self.completed_steps + 1
            ),
        }
        if self._scene_flow_raw_ratio_ema is not None:
            metrics["scene_flow/online_raw_ratio_ema"] = float(self._scene_flow_raw_ratio_ema)
        return metrics

    def _scene_flow_update_lambda_from_raw_ratio(self, raw_ratio: float, optimizer_step: int) -> dict:
        cfg = self._scene_flow_cfg()
        self._ensure_scene_flow_online_controller_state()
        eps = float(_cfg_get(cfg, "grad_norm_epsilon", 1.0e-12))
        target_ratio = float(_cfg_get(cfg, "target_grad_ratio", 0.05))
        lambda_min = float(_cfg_get(cfg, "lambda_min", 1.0e-3))
        lambda_max = float(_cfg_get(cfg, "lambda_max", 1.0e4))
        decay = float(_cfg_get(cfg, "lambda_ema_decay", 0.97))
        jump_cap = min(2.0, max(1.0, float(_cfg_get(cfg, "lambda_jump_cap", 2.0))))

        if self._scene_flow_raw_ratio_ema is None:
            self._scene_flow_raw_ratio_ema = float(raw_ratio)
        else:
            self._scene_flow_raw_ratio_ema = decay * float(self._scene_flow_raw_ratio_ema) + (1.0 - decay) * float(raw_ratio)

        target_lambda = target_ratio / max(float(self._scene_flow_raw_ratio_ema), eps)
        target_lambda = min(max(target_lambda, lambda_min), lambda_max)
        static_lambda = self._scene_flow_lambda()
        warmup_multiplier = self._scene_flow_lambda_warmup_multiplier(step=optimizer_step)
        warmup_target_lambda = static_lambda + warmup_multiplier * (target_lambda - static_lambda)

        old_lambda = float(self._scene_flow_lambda_ema)
        ema_candidate = decay * old_lambda + (1.0 - decay) * warmup_target_lambda
        if jump_cap > 1.0:
            jump_base = max(abs(old_lambda), lambda_min, eps)
            ema_candidate = min(max(ema_candidate, jump_base / jump_cap), jump_base * jump_cap)
        new_lambda = min(max(ema_candidate, lambda_min), lambda_max)

        self._scene_flow_lambda_ema = new_lambda

        return {
            "scene_flow/online_target_lambda": float(target_lambda),
            "scene_flow/online_warmup_target_lambda": float(warmup_target_lambda),
            "scene_flow/online_lambda_before_update": float(old_lambda),
            "scene_flow/lambda_ema": float(new_lambda),
            "scene_flow/online_raw_ratio_ema": float(self._scene_flow_raw_ratio_ema),
        }

    def _validate_scene_flow_online_config(self) -> None:
        if not self._scene_flow_online_grad_norm_enabled():
            return
        cfg = self._scene_flow_cfg()
        lambda_min = float(_cfg_get(cfg, "lambda_min", 1.0e-3))
        lambda_max = float(_cfg_get(cfg, "lambda_max", 1.0e4))
        decay = float(_cfg_get(cfg, "lambda_ema_decay", 0.97))
        every_n = int(_cfg_get(cfg, "online_grad_norm_every_n_steps", 10))
        probe_samples = int(_cfg_get(cfg, "probe_samples", 4))
        jump_cap = float(_cfg_get(cfg, "lambda_jump_cap", 2.0))
        lambda_warmup_steps = int(_cfg_get(cfg, "lambda_warmup_steps", 100))
        min_supervised_pixels = float(_cfg_get(cfg, "min_supervised_pixels", 128.0))
        if float(_cfg_get(cfg, "target_grad_ratio", 0.05)) <= 0:
            raise ValueError("scene-flow online grad-norm target_grad_ratio must be > 0")
        if lambda_min > lambda_max:
            raise ValueError("scene-flow online grad-norm lambda_min must be <= lambda_max")
        if not (0.95 <= decay <= 0.98):
            raise ValueError("scene-flow online grad-norm lambda_ema_decay must be in [0.95, 0.98]")
        if every_n <= 0:
            raise ValueError("scene-flow online grad-norm online_grad_norm_every_n_steps must be > 0")
        if not (2 <= probe_samples <= 4):
            raise ValueError("scene-flow online grad-norm probe_samples must be in [2, 4]")
        if not (1.0 <= jump_cap <= 2.0):
            raise ValueError("scene-flow online grad-norm lambda_jump_cap must be in [1, 2]")
        if lambda_warmup_steps < 0:
            raise ValueError("scene-flow online grad-norm lambda_warmup_steps must be >= 0")
        if min_supervised_pixels < 0:
            raise ValueError("scene-flow online grad-norm min_supervised_pixels must be >= 0")

    def _scene_flow_deepspeed_zero_stage(self):
        state = getattr(self.accelerator, "state", None)
        plugin = getattr(state, "deepspeed_plugin", None)
        candidates = []
        if plugin is not None:
            candidates.extend(
                [
                    getattr(plugin, "deepspeed_config", None),
                    getattr(getattr(plugin, "hf_ds_config", None), "config", None),
                ]
            )
        for cfg in candidates:
            if not isinstance(cfg, dict):
                continue
            zero_cfg = cfg.get("zero_optimization", {})
            if isinstance(zero_cfg, dict) and "stage" in zero_cfg:
                return int(zero_cfg["stage"])
            if "zero_stage" in cfg:
                return int(cfg["zero_stage"])
        return None

    def _assert_scene_flow_online_zero_stage_supported(self) -> None:
        if self._scene_flow_online_zero_stage_checked or not self._scene_flow_online_grad_norm_enabled():
            return
        zero_stage = self._scene_flow_deepspeed_zero_stage()
        if zero_stage is None:
            logger.warning("scene-flow online grad-norm could not determine DeepSpeed ZeRO stage; expected ZeRO 0/1/2")
        elif zero_stage > 2:
            raise RuntimeError(
                f"scene-flow online grad-norm supports DeepSpeed ZeRO 0/1/2 only; got ZeRO stage {zero_stage}"
            )
        self._scene_flow_online_zero_stage_checked = True

    def _maybe_update_scene_flow_online_gradnorm(self, batch_vla, optimizer_step: int) -> dict:
        cfg = self._scene_flow_cfg()
        if not self._scene_flow_online_grad_norm_enabled():
            return {}
        if not self.accelerator.sync_gradients:
            return {}
        self._ensure_scene_flow_online_controller_state()
        self._assert_scene_flow_online_zero_stage_supported()
        every_n = max(1, int(_cfg_get(cfg, "online_grad_norm_every_n_steps", 10)))
        if optimizer_step <= 0 or optimizer_step % every_n != 0:
            return {}
        if self._scene_flow_last_gradnorm_update_step == optimizer_step:
            return {}

        rng_state = self._snapshot_rng_state()
        try:
            device = getattr(self.accelerator, "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)
            output_dict = {}
            action_loss = None
            flow_loss = None
            local_probe_failed = False
            local_probe_exception = None
            probe_samples = int(_cfg_get(cfg, "probe_samples", 4))
            probe_samples = min(4, max(2, probe_samples))
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                probe_batch = self._scene_flow_probe_batch(batch_vla, probe_samples=probe_samples)
                with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = unwrapped.forward(probe_batch)
                    action_loss = output_dict["action_loss"]
                    flow_loss = output_dict.get("flow_loss", None)
                if not torch.is_tensor(action_loss):
                    raise RuntimeError("scene-flow online grad-norm probe action_loss is not a tensor")
                device = action_loss.device
            except Exception as exc:
                local_probe_failed = True
                local_probe_exception = exc

            probe_failed = bool(
                self._distributed_max_tensor(torch.tensor(float(local_probe_failed), device=device)).item() > 0.0
            )
            if probe_failed:
                if local_probe_exception is not None:
                    raise RuntimeError("scene-flow online grad-norm probe forward failed on this rank") from local_probe_exception
                raise RuntimeError("scene-flow online grad-norm probe forward failed on another rank")

            flow_missing = float(flow_loss is None or not torch.is_tensor(flow_loss) or not flow_loss.requires_grad)
            flow_missing = bool(self._distributed_max_tensor(torch.tensor(flow_missing, device=device)).item() > 0.0)
            if flow_missing:
                return self._scene_flow_online_skip_metrics("missing_flow_loss", optimizer_step)

            supervised_value = output_dict.get("flow_supervised_pixel_count", None)
            supervised_missing = float(supervised_value is None)
            supervised_missing = bool(
                self._distributed_max_tensor(torch.tensor(supervised_missing, device=device)).item() > 0.0
            )
            if supervised_missing:
                return self._scene_flow_online_skip_metrics("missing_supervised_pixels", optimizer_step)
            supervised_pixels = float(
                self._distributed_sum_tensor(self._scene_flow_tensor_metric(supervised_value, device)).item()
            )
            min_supervised_pixels = float(_cfg_get(cfg, "min_supervised_pixels", 128.0))
            if supervised_pixels < min_supervised_pixels:
                return self._scene_flow_online_skip_metrics(
                    "low_supervised_pixels",
                    optimizer_step,
                    supervised_pixels=supervised_pixels,
                )

            grad_metrics = self._scene_flow_global_raw_grad_ratio_metrics_from_losses(action_loss, flow_loss)
            action_norm = float(grad_metrics["action_norm"])
            flow_action_norm = float(grad_metrics["flow_action_norm"])
            raw_ratio = float(grad_metrics["raw_ratio"])
            eps = float(_cfg_get(cfg, "grad_norm_epsilon", 1.0e-12))
            if (
                not np.isfinite(action_norm)
                or not np.isfinite(flow_action_norm)
                or not np.isfinite(raw_ratio)
                or action_norm <= eps
                or flow_action_norm <= eps
            ):
                return self._scene_flow_online_skip_metrics(
                    "degenerate_norm",
                    optimizer_step,
                    supervised_pixels=supervised_pixels,
                )

            lambda_metrics = self._scene_flow_update_lambda_from_raw_ratio(raw_ratio, optimizer_step)
            self._scene_flow_last_gradnorm_update_step = int(optimizer_step)
            metrics = {
                "scene_flow/online_update_skipped": 0.0,
                "scene_flow/online_optimizer_step": float(optimizer_step),
                "scene_flow/online_probe_samples_per_rank": float(probe_samples),
                "scene_flow/online_probe_supervised_pixels": supervised_pixels,
                "scene_flow/online_raw_ratio": raw_ratio,
                "scene_flow/online_action_trunk_grad_norm": action_norm,
                "scene_flow/online_flow_trunk_grad_norm": flow_action_norm,
                "scene_flow/online_weighted_flow_to_action_grad_ratio": raw_ratio
                * float(self._scene_flow_lambda_ema),
                "scene_flow/online_last_update_step": float(self._scene_flow_last_gradnorm_update_step),
            }
            metrics.update(lambda_metrics)
            return metrics
        except Exception as exc:
            self._scene_flow_grad_ratio_failure_count += 1
            warn_every = max(1, int(_cfg_get(cfg, "grad_ratio_failure_warn_every", 10)))
            if self._scene_flow_grad_ratio_failure_count == 1 or self._scene_flow_grad_ratio_failure_count % warn_every == 0:
                logger.warning(
                    f"scene-flow online grad-norm update failed "
                    f"(count={self._scene_flow_grad_ratio_failure_count}): {exc}"
                )
            if _cfg_bool(cfg, "grad_ratio_raise_on_failure", False):
                raise
            return self._scene_flow_online_skip_metrics("exception", optimizer_step)
        finally:
            self._restore_rng_state(rng_state)
            self.optimizer.zero_grad()

    def _scene_flow_controller_state_path(self, checkpoint_path) -> Path:
        path = Path(checkpoint_path)
        name = path.name
        for suffix in ("_pytorch_model.pt", "_model.safetensors"):
            if name.endswith(suffix):
                return path.with_name(name[: -len(suffix)] + "_scene_flow_gradnorm.json")
        if path.is_dir():
            state_path = path / "scene_flow_gradnorm.json"
            if state_path.exists():
                return state_path
            sibling_state_path = Path(str(path) + "_scene_flow_gradnorm.json")
            if sibling_state_path.exists():
                return sibling_state_path
            return state_path
        return Path(str(path) + "_scene_flow_gradnorm.json")

    def _save_scene_flow_controller_state(self, checkpoint_path) -> None:
        if not self._scene_flow_online_grad_norm_enabled() or not self.accelerator.is_main_process:
            return
        self._ensure_scene_flow_online_controller_state()
        state = {
            "lambda_ema": float(self._scene_flow_lambda_ema),
            "last_update_step": int(self._scene_flow_last_gradnorm_update_step),
            "raw_ratio_ema": None
            if self._scene_flow_raw_ratio_ema is None
            else float(self._scene_flow_raw_ratio_ema),
            "completed_steps": int(self.completed_steps),
        }
        state_path = self._scene_flow_controller_state_path(checkpoint_path)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def _load_scene_flow_controller_state(self, checkpoint_path) -> None:
        if not self._scene_flow_online_grad_norm_enabled():
            return
        self._ensure_scene_flow_online_controller_state()
        state_path = self._scene_flow_controller_state_path(checkpoint_path)
        payload = None
        if self.accelerator.is_main_process:
            if state_path.exists():
                with open(state_path, "r") as f:
                    payload = json.load(f)
            else:
                logger.warning(
                    f"scene-flow online grad-norm state not found at {state_path}; "
                    "resuming with a fresh controller state"
                )
        if dist.is_available() and dist.is_initialized():
            obj_list = [payload]
            dist.broadcast_object_list(obj_list, src=0)
            payload = obj_list[0]
        if not payload:
            return
        self._scene_flow_lambda_ema = float(payload.get("lambda_ema", self._scene_flow_lambda_ema))
        self._scene_flow_last_gradnorm_update_step = int(
            payload.get("last_update_step", self._scene_flow_last_gradnorm_update_step)
        )
        raw_ratio_ema = payload.get("raw_ratio_ema", None)
        self._scene_flow_raw_ratio_ema = None if raw_ratio_ema is None else float(raw_ratio_ema)

    def _snapshot_rng_state(self):
        state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state) -> None:
        torch.set_rng_state(state["cpu"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _scene_flow_step0_forward_audit(self, batch_vla) -> dict:
        cfg = self._scene_flow_cfg()
        if not self._scene_flow_enabled():
            return {}
        if self._scene_flow_step0_audit_done or self.completed_steps != 0:
            return {}
        if not _cfg_bool(cfg, "step0_action_loss_audit", True):
            return {}

        unwrapped = self.accelerator.unwrap_model(self.model)
        action_model = getattr(unwrapped, "action_model", None)
        if action_model is None or not hasattr(action_model, "scene_flow_enabled"):
            return {}

        rng_state = self._snapshot_rng_state()
        old_enabled = bool(action_model.scene_flow_enabled)
        try:
            with torch.no_grad():
                action_model.scene_flow_enabled = False
                self._restore_rng_state(rng_state)
                baseline_out = unwrapped.forward(batch_vla)
                baseline_action_loss = baseline_out["action_loss"]

                action_model.scene_flow_enabled = old_enabled
                self._restore_rng_state(rng_state)
                flow_out = unwrapped.forward(batch_vla)
                flow_action_loss = flow_out["action_loss"]
        finally:
            action_model.scene_flow_enabled = old_enabled

        diff = torch.abs(flow_action_loss.detach().float() - baseline_action_loss.detach().float()).item()
        self._restore_rng_state(rng_state)
        atol = float(_cfg_get(cfg, "step0_action_loss_audit_atol", 1e-4))
        self._scene_flow_step0_audit_done = True
        if diff > atol:
            raise RuntimeError(
                "scene-flow step-0 forward audit failed: action_loss changed with fixed RNG "
                f"(diff={diff}, atol={atol})"
            )
        return {
            "scene_flow/step0_forward_action_loss_baseline": baseline_action_loss.item(),
            "scene_flow/step0_forward_action_loss_enabled": flow_action_loss.item(),
            "scene_flow/step0_forward_action_loss_absdiff": diff,
        }

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        train_only = (
            self.config.trainer.train_only
            if (self.config and hasattr(self.config.trainer, "train_only"))
            else None
        )
        self.model = self.freeze_backbones(
            self.model,
            freeze_modules=freeze_modules,
            train_only=train_only,
        )
        self.print_trainable_parameters(self.model)

        self._ffs_cache_raw_dataset = getattr(self.vla_train_dataloader, "dataset", None)
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self._maybe_validate_ffs_cache_startup()
        self._validate_scene_flow_online_config()
        self._assert_scene_flow_online_zero_stage_supported()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self._load_scene_flow_controller_state(self.resume_from_checkpoint)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(
                self.model, pretrained_checkpoint, reload_modules=reload_modules, init_from_baseline=True
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self._load_scene_flow_controller_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            self._save_scene_flow_controller_state(checkpoint_path)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and is_main_process():
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        audit_metrics = self._scene_flow_step0_forward_audit(batch_vla)
        grad_ratio_metrics = self._maybe_scene_flow_grad_ratio_metrics(batch_vla)
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                flow_loss = output_dict.get("flow_loss", None)
                flow_weight = self._effective_scene_flow_weight()
                total_loss = action_loss
                if self._scene_flow_enabled() and flow_loss is not None:
                    total_loss = action_loss + flow_weight * flow_loss
                elif self._scene_predictor_enabled():
                    # Scene-flow dual-DiT cascade (independent of the legacy aux path).
                    sup_cells = output_dict.get("flow_supervised_cells", None)
                    if self._loss_mode() == "flow_only":
                        if flow_loss is None:
                            raise RuntimeError(
                                "loss_mode=flow_only but forward returned no flow_loss "
                                "(scene_predictor stage-1 needs flow GT in the batch)."
                            )
                        # Zero-supervision guard (review H2): an all-empty mask makes
                        # flow_loss==0 -> silent no-op training. Refuse. NOTE flow_only
                        # assumes the action head is FROZEN via freeze_modules (else its
                        # params get no grad -> DeepSpeed/DDP unused-param hang; review M3).
                        if sup_cells is not None and float(sup_cells) <= 0:
                            raise RuntimeError(
                                "loss_mode=flow_only but the batch has ZERO supervised scene "
                                "cells (empty masks / flow_has_gt all False). Check GT sidecars / "
                                "gt_only_sampler; refusing a no-op step."
                            )
                        total_loss = flow_loss
                    elif flow_loss is not None:
                        total_loss = action_loss + self._scene_predictor_lambda() * flow_loss
                    else:
                        # joint + scene_predictor enabled but NO flow_loss (no sample had the
                        # scene target key) -> would silently train action-only for the whole
                        # run. Fail loud (review H3).
                        raise RuntimeError(
                            "scene_predictor.enabled + loss_mode=joint but forward returned no "
                            "flow_loss (no sample had the scene target key). Check data_mix / "
                            "scene_predictor.target_key / GT sidecars."
                        )

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        online_gradnorm_metrics = {}
        if self.accelerator.sync_gradients and self._scene_flow_online_grad_norm_enabled():
            online_gradnorm_metrics = self._maybe_update_scene_flow_online_gradnorm(
                batch_vla,
                optimizer_step=self.completed_steps + 1,
            )

        metrics = {
            "action_dit_loss": action_loss.item(),
            "total_loss": total_loss.item(),
        }
        if self._scene_flow_enabled() and flow_loss is not None:
            metrics.update(
                {
                    "scene_flow/loss": flow_loss.item(),
                    "scene_flow/lambda": self._scene_flow_lambda(),
                    "scene_flow/effective_lambda": flow_weight,
                }
            )
            for key in (
                "flow_supervised_samples",
                "flow_gt_samples",
                "dynamic_pixel_count",
                "dynamic_outside_valid_pixel_count",
                "flow_supervised_pixel_count",
                "nonzero_flow_batches",
                "flow_valid_fallback_samples",
            ):
                value = output_dict.get(key, None)
                if torch.is_tensor(value):
                    value = value.item()
                if value is not None:
                    metrics[f"scene_flow/{key}"] = value
            metrics.update(grad_ratio_metrics)
            metrics.update(audit_metrics)
        if self._scene_predictor_enabled() and flow_loss is not None:
            # scene-flow dual-DiT cascade metrics (review H6 — the legacy block above is
            # gated on _scene_flow_enabled, so the new path logged nothing before).
            metrics["scene_predictor/flow_loss"] = flow_loss.item()
            metrics["scene_predictor/lambda"] = self._scene_predictor_lambda()
            _sc = output_dict.get("flow_supervised_cells", None)
            if torch.is_tensor(_sc):
                metrics["scene_predictor/supervised_cells"] = _sc.item()
        if self._scene_flow_online_grad_norm_enabled():
            metrics.update(self._scene_flow_online_state_metrics())
            metrics.update(online_gradnorm_metrics)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    if os.environ.get("METHOD10_DRYRUN_SEED_BEFORE_BUILD") == "1":
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
        seed = cfg.seed + rank if hasattr(cfg, "seed") else rank + 3047
        set_seed(seed)
        logger.info("METHOD10 dry-run seeded before model build for shared baseline/cache init")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and is_main_process():
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
