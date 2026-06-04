"""QwenPI + LLaMA-Adapter-style multi-layer gated FFS injection.

This framework keeps the Qwen VLM and FoundationStereo frozen. A shared FFS hint
projector produces primary-view image-token hints, and six zero-initialized Linear
gates inject residuals at the six cam_rope softmax layers. There is no copied trunk
branch and no sequence-length change.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import os
import sys

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.stereo.multilayer_gated_inject import (
    assert_gates_zero,
    clear_state as clear_llama_adapter_state,
    install_multilayer_gated_ffs_hooks,
    set_ffs_feature,
)
from starVLA.model.modules.stereo.vlm_controlnet import FFSControlNetHint

logger = logging.getLogger(__name__)


@dataclass
class QwenPILlamaAdapterFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPILlamaAdapterFFS"
    ffs_llama_adapter: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_expected_sha256": None,
            "ffs_feature_source": "gru_hidden",
            "gru_hidden_dim": 16,
            "ffs_image_size": 256,
            "primary_idx": 0,
            "right_view_idx": 1,
            "num_cameras": 2,
            "hint_hidden_dim": 256,
            "reverse_image_order": True,
        }
    )


_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPILlamaAdapterFFS")
class QwenPI_LlamaAdapterFFS(QwenPI):
    """QwenPI with frozen FFS hints gated into the six cam_rope layers."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPILlamaAdapterFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_llama_adapter", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))
        self.num_cameras = int(ffs_cfg.get("num_cameras", 2))
        self.reverse_image_order = bool(ffs_cfg.get("reverse_image_order", True))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "gru_hidden"))
        if self.ffs_feature_source != "gru_hidden":
            raise ValueError(
                f"[LlamaAdapter-FFS] only ffs_feature_source=gru_hidden is supported; "
                f"got {self.ffs_feature_source!r}"
            )
        if self.num_cameras != 2:
            raise ValueError("[LlamaAdapter-FFS] this framework currently expects num_cameras=2")
        if not (0 <= self.primary_idx < self.num_cameras):
            raise ValueError(
                f"[LlamaAdapter-FFS] primary_idx={self.primary_idx} out of range for num_cameras={self.num_cameras}"
            )
        if not (0 <= self.right_view_idx < self.num_cameras):
            raise ValueError(
                f"[LlamaAdapter-FFS] right_view_idx={self.right_view_idx} out of range for num_cameras={self.num_cameras}"
            )
        if self.primary_idx == self.right_view_idx:
            raise ValueError("[LlamaAdapter-FFS] primary_idx and right_view_idx must differ")
        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")

        self._ffs_sha256_verified = False
        self._ffs_expected_sha256 = ffs_cfg.get("ffs_expected_sha256")
        self._ffs_actual_sha256 = None
        if self._ffs_expected_sha256:
            import hashlib

            with open(ffs_model_path, "rb") as fh:
                self._ffs_actual_sha256 = hashlib.sha256(fh.read()).hexdigest()
            if self._ffs_actual_sha256 != self._ffs_expected_sha256:
                raise RuntimeError(
                    f"[LlamaAdapter-FFS] SHA256 mismatch: expected={self._ffs_expected_sha256} "
                    f"got={self._ffs_actual_sha256}"
                )
            self._ffs_sha256_verified = True
            logger.info(f"[LlamaAdapter-FFS] FFS SHA256 verified ({self._ffs_expected_sha256[:12]})")
        else:
            logger.warning("[LlamaAdapter-FFS] no ffs_expected_sha256 set; pickle load uses weights_only=False")

        _ffs_register_fixup()
        logger.info(f"[LlamaAdapter-FFS] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for param in self.ffs.parameters():
            param.requires_grad = False

        self.ffs_feat_dim = int(ffs_cfg.get("gru_hidden_dim", 16))
        try:
            self.ffs.args.mixed_precision = False
        except Exception:
            self.ffs.args["mixed_precision"] = False
        self._ffs_captured_net0 = None

        def _capture_net0_hook(_module, _inputs, output):
            self._ffs_captured_net0 = output[0][0]

        self.ffs.update_block.register_forward_hook(_capture_net0_hook)
        logger.info(
            f"[LlamaAdapter-FFS] feature_source=gru_hidden net0_dim={self.ffs_feat_dim}, "
            f"valid_iters={int(self.ffs.args.valid_iters)}"
        )

        self._llama_adapter_inject_depths = self._discover_cam_rope_depths()
        if len(self._llama_adapter_inject_depths) != 6:
            raise RuntimeError(
                f"[LlamaAdapter-FFS] expected 6 cam_rope softmax layers for Qwen3.5-0.8B, "
                f"got {len(self._llama_adapter_inject_depths)} at {self._llama_adapter_inject_depths}"
            )
        self._assert_softmax_layers([self._language_model_layers()[idx] for idx in self._llama_adapter_inject_depths])

        llm_dim = int(self.config.framework.qwenvl.vl_hidden_dim)
        input_embeddings = self.qwen_vl_interface.model.get_input_embeddings()
        base_dtype = input_embeddings.weight.dtype
        self._llama_adapter_base_dtype = base_dtype

        self.hint = FFSControlNetHint(
            in_ch=self.ffs_feat_dim,
            llm_dim=llm_dim,
            hidden_dim=int(ffs_cfg.get("hint_hidden_dim", 256)),
        ).to(dtype=base_dtype)
        self.gates = nn.ModuleList([nn.Linear(llm_dim, llm_dim) for _ in self._llama_adapter_inject_depths])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.gates.to(dtype=base_dtype)
        assert_gates_zero(self.gates)

        spatial_merge = int(self.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
        self._llama_adapter_state = install_multilayer_gated_ffs_hooks(
            self.qwen_vl_interface.model,
            hint=self.hint,
            gates=self.gates,
            inject_depths=self._llama_adapter_inject_depths,
            num_cameras=self.num_cameras,
            spatial_merge_size=spatial_merge,
            primary_cam_id=self.primary_idx,
            right_view_idx=self.right_view_idx,
            reverse_image_order=self.reverse_image_order,
            cam_rope_state=self._stereo_cam_rope_state,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
        )
        hint_params = sum(p.numel() for p in self.hint.parameters() if p.requires_grad)
        gate_params = sum(p.numel() for p in self.gates.parameters() if p.requires_grad)
        logger.info(
            f"[LlamaAdapter-FFS] installed gated FFS injection; depths={self._llama_adapter_inject_depths}, "
            f"reverse_image_order={self.reverse_image_order}, trainable_hint_params={hint_params}, "
            f"trainable_gate_params={gate_params}, gates all-zero confirmed"
        )

    def train(self, mode: bool = True):
        result = super().train(mode)
        # FFS is a frozen feature extractor; keep eval-time normalization/dropout
        # semantics even when the trainable adapter/action head are in train mode.
        if hasattr(self, "ffs"):
            self.ffs.eval()
        return result

    def _language_model_layers(self):
        inner = getattr(self.qwen_vl_interface.model, "model", None) or self.qwen_vl_interface.model
        lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
        if lm is None or not hasattr(lm, "layers"):
            raise RuntimeError("[LlamaAdapter-FFS] could not locate language_model.layers")
        return lm.layers

    def _discover_cam_rope_depths(self) -> List[int]:
        depths: List[int] = []
        for idx, layer in enumerate(self._language_model_layers()):
            attn = self._get_layer_attn(layer, required=False)
            if attn is not None and hasattr(attn, "stereo_cam_layer"):
                depths.append(idx)
        return depths

    @staticmethod
    def _get_layer_attn(layer: nn.Module, required: bool = True):
        for child_name in ("self_attn", "attention", "attn"):
            if hasattr(layer, child_name):
                return getattr(layer, child_name)
        if required:
            raise RuntimeError(f"[LlamaAdapter-FFS] no attention child on {type(layer).__name__}")
        return None

    def _assert_softmax_layers(self, layers: List[nn.Module]) -> None:
        bad = []
        for depth, layer in zip(self._llama_adapter_inject_depths, layers):
            attn_name = type(self._get_layer_attn(layer)).__name__
            if attn_name != "Qwen3_5Attention":
                bad.append((depth, attn_name))
        if bad:
            raise RuntimeError(f"[LlamaAdapter-FFS] non-softmax cam_rope layers found: {bad}")

    def _imgs_to_ffs_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        from torchvision import transforms

        to_tensor = transforms.ToTensor()
        device = next(self.parameters()).device
        resize = transforms.Resize(
            (self.ffs_image_size, self.ffs_image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        out = []
        for example_imgs in batch_images:
            if view_idx >= len(example_imgs):
                raise IndexError(
                    f"[LlamaAdapter-FFS] sample has {len(example_imgs)} images, cannot read view_idx={view_idx}"
                )
            img = example_imgs[view_idx]
            if not torch.is_tensor(img):
                img = to_tensor(img)
            else:
                if img.ndim != 3 or int(img.shape[0]) not in (1, 3):
                    raise ValueError(
                        f"[LlamaAdapter-FFS] tensor image for view_idx={view_idx} must be CHW with 1 or 3 channels; "
                        f"got shape={tuple(img.shape)}"
                    )
                img = img.float()
                img_min = float(img.detach().min().item())
                img_max = float(img.detach().max().item())
                if img_min < 0.0 or img_max > 255.0:
                    raise ValueError(
                        f"[LlamaAdapter-FFS] tensor image values must be in [0, 1] or [0, 255]; "
                        f"got min={img_min:.3f}, max={img_max:.3f}"
                    )
                if img_max > 2.0:
                    img = img / 255.0
            img = resize(img.unsqueeze(0)).squeeze(0)
            out.append(img * 255.0)
        return torch.stack(out, dim=0).to(device).float()

    def _compute_ffs_feature(self, batch_images: List) -> torch.Tensor:
        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)
        right = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)
        batch_size = primary.shape[0]
        stacked = torch.cat([primary, right], dim=0)
        with torch.no_grad():
            if next(self.ffs.parameters()).dtype != torch.float32:
                self.ffs.float()
            image1 = stacked[:batch_size].float()
            image2 = stacked[batch_size:].float()
            self._ffs_captured_net0 = None
            with torch.amp.autocast("cuda", enabled=False):
                self.ffs(image1, image2, iters=int(self.ffs.args.valid_iters), test_mode=True)
            if self._ffs_captured_net0 is None:
                raise RuntimeError(
                    "[LlamaAdapter-FFS] gru_hidden: update_block hook did not fire; FFS forward changed?"
                )
            ffs_feat = self._ffs_captured_net0
            if ffs_feat.shape[1] != self.ffs_feat_dim:
                raise RuntimeError(
                    f"[LlamaAdapter-FFS] net[0] channels {ffs_feat.shape[1]} != configured "
                    f"gru_hidden_dim {self.ffs_feat_dim}"
                )
        return ffs_feat.detach()

    def _vlm_ordered_images(self, batch_images: List) -> List[List]:
        ordered = []
        for sample_idx, example_imgs in enumerate(batch_images):
            if len(example_imgs) != self.num_cameras:
                raise ValueError(
                    f"[LlamaAdapter-FFS] sample {sample_idx} has {len(example_imgs)} images; "
                    f"expected exactly num_cameras={self.num_cameras} for stereo FFS."
                )
            max_idx = max(self.primary_idx, self.right_view_idx)
            if len(example_imgs) <= max_idx:
                raise IndexError(
                    f"[LlamaAdapter-FFS] sample {sample_idx} has {len(example_imgs)} images, "
                    f"need primary={self.primary_idx} and right={self.right_view_idx}"
                )
            if self.reverse_image_order:
                ordered.append([example_imgs[self.right_view_idx], example_imgs[self.primary_idx]])
            else:
                ordered.append(list(example_imgs))
        return ordered

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        clear_llama_adapter_state(self._llama_adapter_state)
        ffs_feat = self._compute_ffs_feature(batch_images)
        set_ffs_feature(self._llama_adapter_state, ffs_feat)
        vlm_images = self._vlm_ordered_images(batch_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=vlm_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = True):
        own_keys = set(self.state_dict().keys())
        provided_ffs_keys = {k for k in state_dict.keys() if k.startswith("ffs.")}
        if provided_ffs_keys:
            logger.info(
                f"[LlamaAdapter-FFS audit] ignoring {len(provided_ffs_keys)} checkpoint ffs.* keys; "
                "frozen FFS is loaded only from the SHA-verified ffs_model_path"
            )
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ffs.")}
        provided = set(state_dict.keys())

        def _is_adapter_key(k: str) -> bool:
            return k.startswith("ffs.") or k.startswith("hint.") or k.startswith("gates.")

        def _is_trainable_adapter_key(k: str) -> bool:
            return k.startswith("hint.") or k.startswith("gates.")

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    "qwen_vl_interface.model.model.language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        expected_adapter_keys = {k for k in own_keys if _is_trainable_adapter_key(k)}
        provided_adapter_keys = {k for k in provided if _is_trainable_adapter_key(k)}
        adapter_keys_absent = not provided_adapter_keys
        if provided_adapter_keys and provided_adapter_keys != expected_adapter_keys:
            missing = sorted(expected_adapter_keys - provided_adapter_keys)[:20]
            extra = sorted(provided_adapter_keys - expected_adapter_keys)[:20]
            raise RuntimeError(
                "[LlamaAdapter-FFS audit] partial adapter checkpoint detected; "
                f"missing_first={missing}, unexpected_adapter_first={extra}. "
                "Provide all hint/gates keys or none so warm-start is unambiguous."
            )

        if init_from_baseline:
            allowed_missing = {
                k for k in own_keys - provided
                if _is_adapter_key(k) or _is_stereo_cam_rope_key(k)
            }
        else:
            allowed_missing = set()

        suspicious_missing = (own_keys - provided) - allowed_missing
        unexpected = provided - own_keys

        if allowed_missing:
            logger.info(
                f"[LlamaAdapter-FFS audit] ckpt missing {len(allowed_missing)} optional keys "
                "(ffs / hint / gates / stereo_cam_rope)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if init_from_baseline:
                raise RuntimeError(
                    f"[LlamaAdapter-FFS audit] warm-start ckpt architecture mismatch: "
                    f"these non-adapter params would stay randomly initialized; "
                    f"align the launcher config with the checkpoint. "
                    f"missing {len(suspicious_missing)} keys; first: {sample}"
                )
            if strict:
                raise RuntimeError(
                    f"[LlamaAdapter-FFS audit] {len(suspicious_missing)} unexpected missing keys "
                    f"under strict=True; first: {sample}"
                )
            logger.warning(
                f"[LlamaAdapter-FFS audit] {len(suspicious_missing)} missing keys "
                f"(strict=False, accepting); first: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[LlamaAdapter-FFS audit] {len(unexpected)} UNEXPECTED keys in ckpt not present "
                f"in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " -- refusing silent drop.")
            logger.warning(msg + " -- dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        forwarded_strict = strict and not allowed_missing
        result = super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)
        if init_from_baseline and adapter_keys_absent:
            assert_gates_zero(self.gates)
            logger.info("[LlamaAdapter-FFS audit] warm-start adapter keys absent; gates remain all-zero")
        return result
