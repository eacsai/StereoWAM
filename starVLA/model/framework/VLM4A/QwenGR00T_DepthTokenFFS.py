from dataclasses import dataclass, field
from typing import Optional

import logging
import os
import torch
import torch.distributed as dist

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import (
    QwenGR00TFFSBase,
    qwen_vlm_base_dtype,
    qwen_vlm_hidden_size,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.stereo.depth_token_inject import (
    DepthTokenProjector,
    clear_state as clear_depth_state,
    get_state as get_depth_state,
    install_depth_token_hooks,
    set_depth_tokens,
)
from starVLA.model.modules.stereo.ffs_net0_cache import (
    _maybe_cache_dir,
    read_ffs_net0_pooled_cache_batch,
    run_ffs_net0_cache_startup_check,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _optional_str(value) -> Optional[str]:
    return None if value is None else str(value)


def _optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class QwenGR00TDepthTokenFFSDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_DepthTokenFFS"
    ffs_depth_token: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "num_cameras": 2,
            "left_ref_idx": 1,
            "primary_view_idx": 0,
            "inject_cam_id": 1,
            "num_depth_tokens": 16,
            "pool_hw": 4,
            "ffs_cache_dir": None,
            "depth_prompt": "Left image depth features:",
            # Action-head-access ablation. False (default, GR00T behavior) keeps the
            # inserted depth tokens in the sequence the action head cross-attends to;
            # True strips them after the VLM forward so only the original tokens reach
            # the action head (mirrors QwenPI_DepthTokenFFS). Class default stays False
            # + 16/4 above so existing depth-token runs are byte-unchanged; the 64/8 +
            # strip flag are passed by the new-run launcher only.
            "strip_depth_tokens": False,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_DepthTokenFFS")
class QwenGR00T_DepthTokenFFS(QwenGR00TFFSBase):
    """GR00T + FFS #4: insert zero-init net[0] depth tokens into the VLM sequence."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenGR00TDepthTokenFFSDefaultConfig, self.config)
        llm_dim = self._sync_actual_vlm_hidden_dim()

        ffs_cfg = self.config.framework.get("ffs_depth_token", {})
        self._init_frozen_ffs_net0(ffs_cfg, "[GR00T-DepthToken-FFS]")
        self._ffs_depth_token_cfg = ffs_cfg

        self.num_depth_tokens = int(ffs_cfg.get("num_depth_tokens", 16))
        self.depth_pool_hw = int(ffs_cfg.get("pool_hw", 4))
        self.depth_token_prompt = str(ffs_cfg.get("depth_prompt", "Left image depth features:"))
        self.strip_depth_tokens = bool(ffs_cfg.get("strip_depth_tokens", False))
        self._ffs_cache_dir = _maybe_cache_dir(ffs_cfg.get("ffs_cache_dir", None))
        self._ffs_cache_handles = {}
        self._ffs_cache_batches = 0
        self._ffs_cache_hits = 0
        self._ffs_cache_row_misses = 0
        self._ffs_cache_whole_batch_live = 0
        self._ffs_cache_lookup_batches = 0
        self._ffs_cache_lookup_hits = 0
        self._ffs_cache_lookup_misses = 0
        self._ffs_cache_live_warn_count = 0
        self._ffs_cache_miss_warn_count = 0
        self._ffs_cache_dataset_or_none = None
        self._ffs_cache_startup_checked = False
        self._ffs_cache_disabled_runtime = False

        data_cfg = _cfg_get(_cfg_get(self.config, "datasets", None), "vla_data", None)
        trainer_vla_cfg = _cfg_get(_cfg_get(self.config, "trainer", None), "vla_data", None)
        self._ffs_cache_expected_data_root = _optional_str(_cfg_get(data_cfg, "data_root_dir", None))
        self._ffs_cache_expected_data_mix = _optional_str(_cfg_get(data_cfg, "data_mix", None))
        self._ffs_cache_expected_video_backend = _optional_str(_cfg_get(trainer_vla_cfg, "video_backend", None))
        self._ffs_cache_expected_delete_pause_frame = _optional_bool(_cfg_get(data_cfg, "delete_pause_frame", None))

        self.depth_token_projector = DepthTokenProjector(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            num_tokens=self.num_depth_tokens,
            pool_hw=self.depth_pool_hw,
        ).to(dtype=qwen_vlm_base_dtype(self.qwen_vl_interface))

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        hf_model = self.qwen_vl_interface.model
        self._depth_token_hook_handles = install_depth_token_hooks(
            hf_model=hf_model,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.inject_cam_id,
            cam_rope_state=getattr(self, "_stereo_cam_rope_state", None),
            image_token_id=int(hf_model.config.image_token_id),
        )

    def _build_depthtoken_qwenvl_inputs(self, batch_images, instructions):
        """Build [primary image][depth prompt text][left_view image][instruction] messages."""

        assert len(batch_images) == len(instructions), "Images and instructions must have the same length"
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            if len(imgs) <= max(self.primary_view_idx, self.left_ref_idx):
                raise ValueError(
                    "[GR00T-DepthToken-FFS] expected primary,left_view stereo images with "
                    f"at least {max(self.primary_view_idx, self.left_ref_idx) + 1} views, got {len(imgs)}"
                )

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                task_prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                task_prompt = instruction

            content = [
                {"type": "image", "image": imgs[self.primary_view_idx]},
                {"type": "text", "text": self.depth_token_prompt},
                {"type": "image", "image": imgs[self.left_ref_idx]},
                {"type": "text", "text": task_prompt},
            ]
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.qwen_vl_interface.model, "device", None)
        if model_device is None:
            model_device = next(self.qwen_vl_interface.model.parameters()).device
        return batch_inputs.to(model_device)

    def _encode_last_hidden_with_ffs(self, batch_images, instructions, sample_ids=None) -> torch.Tensor:
        qwen_inputs = self._build_depthtoken_qwenvl_inputs(
            batch_images=batch_images,
            instructions=instructions,
        )
        self._prepare_ffs_for_vlm(batch_images, sample_ids=sample_ids)
        try:
            last_hidden = self._run_qwenvl_forward(qwen_inputs)
            if self.strip_depth_tokens:
                last_hidden = self._strip_depth_tokens(last_hidden)
                # Fix#1: depth rows stripped back out -> (B, seq_len) -> plain mask.
                self._stash_pending_mask(qwen_inputs, num_insert=0)
            else:
                # Fix#1: depth tokens kept mid-sequence -> extend by num_depth_tokens.
                self._stash_pending_mask(qwen_inputs, num_insert=self.num_depth_tokens)
            return last_hidden
        finally:
            self._cleanup_ffs_after_vlm()

    def _strip_depth_tokens(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Drop the inserted depth-token rows so the action head sees only the original
        (non-depth) tokens — the 'strip' arm of the action-head-access ablation.

        Mirrors QwenPI_DepthTokenFFS._encode_vl_hidden_states (keep_mask gather). Applied
        inside _encode_last_hidden_with_ffs so BOTH training forward and predict_action
        strip identically (no train-on-N / eval-on-N+depth mismatch). The depth tokens
        still pass through the VLM (so the real tokens + projector are depth-trained); only
        their hidden rows are removed before the action head.
        """
        state = get_depth_state()
        keep_mask = state.keep_mask
        if keep_mask is None:
            raise RuntimeError(
                "[GR00T-DepthToken-FFS] strip_depth_tokens=True but the depth-token hook "
                "did not produce keep_mask; language_model pre-hook may not have run"
            )
        false_counts = (~keep_mask).sum(dim=1)
        expected = torch.full_like(false_counts, self.num_depth_tokens)
        if not torch.equal(false_counts, expected):
            raise RuntimeError(
                f"[GR00T-DepthToken-FFS] expected {self.num_depth_tokens} inserted tokens "
                f"per sample, got {false_counts.detach().cpu().tolist()}"
            )
        batch_size = int(keep_mask.shape[0])
        kept_len = int(keep_mask.shape[1] - self.num_depth_tokens)
        if last_hidden.shape[0] != batch_size or last_hidden.shape[1] != keep_mask.shape[1]:
            raise RuntimeError(
                f"[GR00T-DepthToken-FFS] hidden state shape {tuple(last_hidden.shape)} "
                f"does not match expanded keep_mask {tuple(keep_mask.shape)}"
            )
        mask = keep_mask.to(device=last_hidden.device)
        return last_hidden[mask].view(batch_size, kept_len, last_hidden.shape[-1])

    def _compute_depth_token_net0_pooled_live(self, batch_images) -> torch.Tensor:
        net0 = self._compute_ffs_feature(batch_images)
        return self.depth_token_projector.pool(net0).to(dtype=torch.float32)

    def set_ffs_cache_dataset(self, dataset_or_none) -> None:
        self._ffs_cache_dataset_or_none = dataset_or_none
        self._ffs_cache_startup_checked = False

    attach_ffs_cache_dataset = set_ffs_cache_dataset

    def _ffs_cache_fallback_control_device(self):
        try:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
        except Exception as exc:
            logger.warning("[GR00T-DepthToken-FFS] failed to use CUDA cache broadcast device: %s", exc)
        return torch.device("cpu")

    def _ffs_cache_control_device(self):
        try:
            device = next(self.parameters()).device
        except Exception as exc:
            logger.warning("[GR00T-DepthToken-FFS] failed to infer cache broadcast device: %s", exc)
            device = self._ffs_cache_fallback_control_device()
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl" and device.type != "cuda":
            fallback = self._ffs_cache_fallback_control_device()
            if fallback.type == "cuda":
                device = fallback
            else:
                logger.warning("[GR00T-DepthToken-FFS] NCCL initialized but CUDA is unavailable for cache broadcast")
        return device

    def maybe_validate_ffs_cache_startup(self) -> None:
        cache_dir = _maybe_cache_dir(self._ffs_cache_dir)
        if cache_dir is None or self._ffs_cache_startup_checked:
            return
        if self._ffs_cache_disabled_runtime:
            self._ffs_cache_startup_checked = True
            return

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        broadcast_device = self._ffs_cache_control_device() if distributed else None
        code = 1
        error = None
        if rank == 0:
            try:
                run_ffs_net0_cache_startup_check(
                    cache_dir=cache_dir,
                    dataset_or_none=self._ffs_cache_dataset_or_none,
                    handles=self._ffs_cache_handles,
                    ffs_sha256=getattr(self, "_ffs_actual_sha256", None),
                    ffs_cfg=self._ffs_depth_token_cfg,
                    pool_hw=self.depth_pool_hw,
                    num_depth_tokens=self.num_depth_tokens,
                    device=next(self.depth_token_projector.parameters()).device,
                    compute_pooled_fn=self._compute_depth_token_net0_pooled_live,
                    expected_data_root=self._ffs_cache_expected_data_root,
                    expected_data_mix=self._ffs_cache_expected_data_mix,
                    expected_video_backend=self._ffs_cache_expected_video_backend,
                    expected_delete_pause_frame=self._ffs_cache_expected_delete_pause_frame,
                )
            except Exception as exc:
                error = str(exc)
                if os.environ.get("FFS_CACHE_ON_MISMATCH", "").strip().lower() == "fallback_live":
                    code = 2
                    logger.error(
                        "FFS cache invalid — precompute env != training env; falling back live because "
                        "FFS_CACHE_ON_MISMATCH=fallback_live. detail=%s",
                        error,
                    )
                else:
                    code = 0
                    logger.error("FFS cache invalid — precompute env != training env. detail=%s", error)

        if distributed:
            verdict = torch.tensor([code], dtype=torch.uint8, device=broadcast_device)
            dist.broadcast(verdict, src=0)
            code = int(verdict.item())

        if code == 0:
            if rank == 0 and error:
                raise RuntimeError(f"FFS cache invalid — precompute env != training env: {error}")
            raise RuntimeError("FFS cache invalid — precompute env != training env; rank 0 logged details")
        if code == 2:
            self._ffs_cache_disabled_runtime = True
            self._ffs_cache_startup_checked = True
            logger.warning("[GR00T-DepthToken-FFS] FFS net0 cache disabled at runtime; using live FFS")
            return

        self._ffs_cache_startup_checked = True

    def _info_ffs_cache_live_batch(self, reason: str) -> None:
        count = int(getattr(self, "_ffs_cache_live_warn_count", 0))
        if count < 3 or count % 100 == 0:
            logger.info(
                "[GR00T-DepthToken-FFS] FFS net0 cache configured but serving whole batch live "
                "reason=%s; this is expected for eval/predict_action without dataset ids",
                reason,
            )
        self._ffs_cache_live_warn_count = count + 1

    def _check_ffs_cache_hitrate_floor(self, *, sample_ids, stats) -> None:
        if sample_ids is None:
            return
        self._ffs_cache_lookup_batches += 1
        self._ffs_cache_lookup_hits += int(stats.hits)
        self._ffs_cache_lookup_misses += int(stats.row_misses) + int(stats.whole_batch_live)

        min_batches = int(os.environ.get("FFS_CACHE_MIN_HITRATE_BATCHES", "200"))
        min_hitrate = float(os.environ.get("FFS_CACHE_MIN_HITRATE", "0.01"))
        if min_batches <= 0 or min_hitrate <= 0:
            return
        batches = int(self._ffs_cache_lookup_batches)
        if batches < min_batches:
            return

        hits = int(self._ffs_cache_lookup_hits)
        misses = int(self._ffs_cache_lookup_misses)
        total = hits + misses
        hit_rate = float(hits) / float(total) if total else 0.0
        if hit_rate < min_hitrate:
            raise RuntimeError(
                "FFS cache hit-rate ~0 after "
                f"{batches} batches - cache likely mismatched; aborting. "
                f"hits={hits} misses_or_live={misses} hit_rate={hit_rate:.6f} "
                f"threshold={min_hitrate:.6f}"
            )

    def _warn_ffs_cache_misses(self, miss_details) -> None:
        if not miss_details:
            return
        count = int(getattr(self, "_ffs_cache_miss_warn_count", 0))
        if count < 3 or count % 100 == 0:
            logger.warning(
                "[GR00T-DepthToken-FFS] FFS net0 cache row miss count=%d preview=%s; "
                "falling back to live FFS for those rows",
                len(miss_details),
                miss_details[:3],
            )
        self._ffs_cache_miss_warn_count = count + 1

    def _log_ffs_cache_telemetry(self) -> None:
        batches = int(getattr(self, "_ffs_cache_batches", 0))
        if batches <= 3 or batches % 100 == 0:
            hits = int(getattr(self, "_ffs_cache_hits", 0))
            misses = int(getattr(self, "_ffs_cache_row_misses", 0))
            live = int(getattr(self, "_ffs_cache_whole_batch_live", 0))
            total_lookup = hits + misses
            hit_rate = float(hits) / float(total_lookup) if total_lookup else 0.0
            logger.info(
                "[GR00T-DepthToken-FFS] FFS net0 cache telemetry batches=%d hits=%d "
                "row_misses=%d whole_batch_live_samples=%d hit_rate=%.4f",
                batches,
                hits,
                misses,
                live,
                hit_rate,
            )

    def _get_depth_token_net0_pooled(self, batch_images, sample_ids=None) -> torch.Tensor:
        cache_dir = _maybe_cache_dir(self._ffs_cache_dir)
        if cache_dir is None or self._ffs_cache_disabled_runtime:
            return self._compute_depth_token_net0_pooled_live(batch_images)
        if self._ffs_cache_dataset_or_none is not None and not self._ffs_cache_startup_checked:
            self.maybe_validate_ffs_cache_startup()
            if self._ffs_cache_disabled_runtime:
                return self._compute_depth_token_net0_pooled_live(batch_images)

        self._ffs_cache_batches += 1
        device = next(self.depth_token_projector.parameters()).device
        pooled, stats = read_ffs_net0_pooled_cache_batch(
            cache_dir=cache_dir,
            batch_images=batch_images,
            sample_ids=sample_ids,
            handles=self._ffs_cache_handles,
            ffs_sha256=getattr(self, "_ffs_actual_sha256", None),
            ffs_cfg=self._ffs_depth_token_cfg,
            pool_hw=self.depth_pool_hw,
            num_depth_tokens=self.num_depth_tokens,
            device=device,
            live_fallback_fn=self._compute_depth_token_net0_pooled_live,
            dataset_or_none=self._ffs_cache_dataset_or_none,
            expected_data_root=self._ffs_cache_expected_data_root,
            expected_data_mix=self._ffs_cache_expected_data_mix,
            expected_video_backend=self._ffs_cache_expected_video_backend,
            expected_delete_pause_frame=self._ffs_cache_expected_delete_pause_frame,
        )
        self._ffs_cache_hits += int(stats.hits)
        self._ffs_cache_row_misses += int(stats.row_misses)
        self._ffs_cache_whole_batch_live += int(stats.whole_batch_live)
        if stats.live_reason is not None:
            self._info_ffs_cache_live_batch(stats.live_reason)
        self._warn_ffs_cache_misses(stats.miss_details)
        self._check_ffs_cache_hitrate_floor(sample_ids=sample_ids, stats=stats)
        self._log_ffs_cache_telemetry()
        return pooled

    def _prepare_ffs_for_vlm(self, batch_images, sample_ids=None):
        clear_depth_state()
        if self._ffs_cache_dir is None:
            # Preserve the default-off live path exactly: raw net0 enters the
            # existing projector, which performs the pool and Linear.
            net0 = self._compute_ffs_feature(batch_images)
            depth_tokens = self.depth_token_projector(net0)
        else:
            net0_pooled = self._get_depth_token_net0_pooled(batch_images, sample_ids=sample_ids)
            depth_tokens = self.depth_token_projector.project_pooled(net0_pooled)
        set_depth_tokens(depth_tokens)

    def _cleanup_ffs_after_vlm(self) -> None:
        # Keep state through backward: checkpointed VLM layers may rerun the
        # sequence hook while gradients flow into depth_token_projector.
        return None

    def _ffs_key_prefixes(self):
        return ("ffs.", "depth_token_projector.")

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        # Default False (strict): a standalone / eval / resume load MUST contain the
        # FFS-adapter keys. ONLY the trainer's warm-start path passes
        # init_from_baseline=True (via signature inspection) to allow the adapter
        # keys to be fresh-initialised off a non-FFS baseline (e.g. B).
        own_keys = set(self.state_dict().keys())
        provided_keys = set(state_dict.keys())

        expected_projector = {key for key in own_keys if key.startswith("depth_token_projector.")}
        provided_projector = {key for key in provided_keys if key.startswith("depth_token_projector.")}
        if provided_projector and provided_projector != expected_projector:
            missing = sorted(expected_projector - provided_projector)[:20]
            extra = sorted(provided_projector - expected_projector)[:20]
            raise RuntimeError(
                "[GR00T-DepthToken-FFS audit] partial depth-token projector checkpoint; "
                f"missing_first={missing}, unexpected_depth_first={extra}"
            )

        # Keep the hidden-size sync strict when loading older checkpoints whose
        # config still advertised a stale vl_hidden_dim.
        actual_hidden = qwen_vlm_hidden_size(self.qwen_vl_interface)
        if int(self.config.framework.qwenvl.vl_hidden_dim) != actual_hidden:
            self.config.framework.qwenvl.vl_hidden_dim = actual_hidden

        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
            init_from_baseline=init_from_baseline,
        )
