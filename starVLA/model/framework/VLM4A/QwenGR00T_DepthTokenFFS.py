from dataclasses import dataclass, field
from typing import Optional

import torch

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
from starVLA.model.tools import FRAMEWORK_REGISTRY


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
            "primary_idx": 1,
            "right_view_idx": 0,
            "primary_cam_id": 1,
            "num_depth_tokens": 16,
            "pool_hw": 4,
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

        self.num_depth_tokens = int(ffs_cfg.get("num_depth_tokens", 16))
        self.depth_pool_hw = int(ffs_cfg.get("pool_hw", 4))
        self.depth_token_prompt = str(ffs_cfg.get("depth_prompt", "Left image depth features:"))
        self.strip_depth_tokens = bool(ffs_cfg.get("strip_depth_tokens", False))

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
            primary_cam_id=self.primary_cam_id,
            cam_rope_state=getattr(self, "_stereo_cam_rope_state", None),
            image_token_id=int(hf_model.config.image_token_id),
        )

    def _build_depthtoken_qwenvl_inputs(self, batch_images, instructions):
        """Build [right image][depth prompt text][left image][instruction] messages."""

        assert len(batch_images) == len(instructions), "Images and instructions must have the same length"
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            if len(imgs) <= max(self.primary_idx, self.right_view_idx):
                raise ValueError(
                    "[GR00T-DepthToken-FFS] expected right-first stereo images with "
                    f"at least {max(self.primary_idx, self.right_view_idx) + 1} views, got {len(imgs)}"
                )

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                task_prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                task_prompt = instruction

            content = [
                {"type": "image", "image": imgs[self.right_view_idx]},
                {"type": "text", "text": self.depth_token_prompt},
                {"type": "image", "image": imgs[self.primary_idx]},
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

    def _encode_last_hidden_with_ffs(self, batch_images, instructions) -> torch.Tensor:
        qwen_inputs = self._build_depthtoken_qwenvl_inputs(
            batch_images=batch_images,
            instructions=instructions,
        )
        self._prepare_ffs_for_vlm(batch_images)
        try:
            last_hidden = self._run_qwenvl_forward(qwen_inputs)
            if self.strip_depth_tokens:
                last_hidden = self._strip_depth_tokens(last_hidden)
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

    def _prepare_ffs_for_vlm(self, batch_images):
        clear_depth_state()
        net0 = self._compute_ffs_feature(batch_images)
        depth_tokens = self.depth_token_projector(net0)
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
