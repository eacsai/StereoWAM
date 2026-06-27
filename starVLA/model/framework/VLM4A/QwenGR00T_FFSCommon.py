from __future__ import annotations

import hashlib
import inspect
import logging
import os
import sys
import contextlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id
from starVLA.model.modules.stereo.ffs_net0_cache import ffs_tf32_disabled
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = logging.getLogger(__name__)


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


def _cfg_contains(cfg, key: str) -> bool:
    if cfg is None:
        return False
    try:
        return key in cfg
    except TypeError:
        return hasattr(cfg, key)


def _ffs_stereo_convention() -> str:
    value = os.environ.get("FFS_STEREO_CONVENTION", "leftprimary").strip().lower()
    if value in {"", "leftprimary", "clean", "clean_leftprimary"}:
        return "leftprimary"
    raise ValueError(
        f"FFS_STEREO_CONVENTION must be 'leftprimary' (the only supported clean path), got {value!r}"
    )


_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup() -> None:
    import core.foundation_stereo as _fs  # noqa: F401


def qwen_vlm_hidden_size(qwen_vl_interface) -> int:
    model_cfg = qwen_vl_interface.model.config
    if hasattr(model_cfg, "hidden_size"):
        return int(model_cfg.hidden_size)
    if hasattr(model_cfg, "text_config") and hasattr(model_cfg.text_config, "hidden_size"):
        return int(model_cfg.text_config.hidden_size)
    raise RuntimeError("[GR00T-FFS] could not derive VLM hidden_size from model.config")


def qwen_vlm_base_dtype(qwen_vl_interface) -> torch.dtype:
    get_input_embeddings = getattr(qwen_vl_interface.model, "get_input_embeddings", None)
    input_embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
    if input_embeddings is not None and getattr(input_embeddings, "weight", None) is not None:
        return input_embeddings.weight.dtype
    return next(qwen_vl_interface.model.parameters()).dtype


def language_model_layers(hf_model) -> nn.ModuleList:
    inner = getattr(hf_model, "model", None) or hf_model
    lm = getattr(inner, "language_model", None) or getattr(inner, "model", None)
    if lm is None or not hasattr(lm, "layers"):
        raise RuntimeError("[GR00T-FFS] could not locate language_model.layers")
    return lm.layers


def layer_attention(layer: nn.Module, required: bool = True) -> Optional[nn.Module]:
    for child_name in ("self_attn", "attention", "attn"):
        if hasattr(layer, child_name):
            return getattr(layer, child_name)
    if required:
        raise RuntimeError(f"[GR00T-FFS] no attention child on {type(layer).__name__}")
    return None


def _raise_primary_grid_mismatch(
    label: str,
    sample_idx: int,
    h_tok: int,
    w_tok: int,
    n_primary: int,
    primary_cam_id: int,
) -> None:
    expected = h_tok * w_tok
    raise RuntimeError(
        f"[{label}] sample {sample_idx}: primary_cam_id={primary_cam_id} expected "
        f"primary token grid {h_tok}x{w_tok} = {expected}, got {n_primary} tokens. "
        "image_grid_thw / image-token layout mismatch; refusing silent FFS injection."
    )


class FFSPerTokenProjector(nn.Module):
    """FFS net[0] map -> VLM-token residual with a configurable (zero|identity)-init Linear gate."""

    def __init__(self, in_ch: int, llm_dim: int, hidden_dim: int = 256, gate_init: str = "zero"):
        super().__init__()
        self.in_ch = int(in_ch)
        self.llm_dim = int(llm_dim)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(self.in_ch, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.llm_dim, kernel_size=1),
        )
        self.zero_proj = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        gate_init = str(gate_init).lower()
        if gate_init == "zero":
            # zero-init gate: step-0 residual == 0 (warm-start parity with the base VLM).
            nn.init.zeros_(self.zero_proj.weight)
            nn.init.zeros_(self.zero_proj.bias)
        elif gate_init == "identity":
            # identity-init gate: step-0 injects the full spatial_proj(net[0]) residual,
            # still learnable. Forces the FFS signal into primary tokens from step 0 so the
            # model cannot trivially learn a closed gate that ignores the disparity feature.
            nn.init.eye_(self.zero_proj.weight)
            nn.init.zeros_(self.zero_proj.bias)
        else:
            raise ValueError(
                f"FFSPerTokenProjector gate_init must be 'zero' or 'identity', got {gate_init!r}"
            )
        self.gate_init = gate_init

    def forward(self, ffs_feat_b: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
        proj_dtype = next(self.parameters()).dtype
        x = self.spatial_proj(ffs_feat_b.to(dtype=proj_dtype))
        x = F.interpolate(x, size=(h_tok, w_tok), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2).contiguous().squeeze(0)
        return self.zero_proj(x)


@dataclass
class FFSLayerHookState:
    ffs_feat: Optional[torch.Tensor] = None
    per_token_cam_id: Optional[torch.Tensor] = None
    primary_grid: Optional[List[Tuple[int, int]]] = None
    enabled: bool = True


def clear_layer_hook_state(state: FFSLayerHookState) -> None:
    state.ffs_feat = None
    state.per_token_cam_id = None
    state.primary_grid = None


def install_ffs_vlm_layer_residual_hooks(
    hf_model,
    projectors: nn.ModuleList,
    num_cameras: int,
    spatial_merge_size: int,
    primary_cam_id: int,
    image_token_id: Optional[int] = None,
    target_layer_indices: Optional[Sequence[int]] = None,
    label: str = "GR00T-ControlNet-FFS",
) -> FFSLayerHookState:
    """Install #2 hooks: add zero-init FFS residuals at SELECTED VLM layer outputs.

    `target_layer_indices` chooses WHICH decoder layers receive a residual. When
    None, a residual is added at EVERY layer (legacy all-layer behaviour). For #2 we
    pass the 6 cam_rope softmax layer indices [3,7,11,15,19,23] so the FFS disparity
    is injected only where the geometric (cam_rope) attention lives; the
    linear-attention layers (no RoPE/cam_rope) are skipped, matching #3's softmax-only
    coverage. One projector per target layer (len(projectors) == len(targets)).
    """

    image_token_id = int(image_token_id if image_token_id is not None else hf_model.config.image_token_id)
    layers = language_model_layers(hf_model)
    targets = list(range(len(layers))) if target_layer_indices is None else [int(i) for i in target_layer_indices]
    for _i in targets:
        if _i < 0 or _i >= len(layers):
            raise RuntimeError(f"[{label}] target layer index {_i} out of range [0,{len(layers)})")
    if len(projectors) != len(targets):
        raise RuntimeError(
            f"[{label}] projector/target-layer count mismatch: {len(projectors)} vs "
            f"{len(targets)} (targets={targets})"
        )

    state = FFSLayerHookState()

    def _outer_pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        image_grid_thw = kwargs.get("image_grid_thw", None)
        state.per_token_cam_id = None
        state.primary_grid = None
        if input_ids is None or image_grid_thw is None:
            return None

        state.per_token_cam_id = compute_per_token_cam_id(
            input_ids=input_ids,
            image_token_id=image_token_id,
            image_grid_thw=image_grid_thw,
            num_cameras=num_cameras,
            spatial_merge_size=spatial_merge_size,
        )
        s2 = max(int(spatial_merge_size), 1)
        grid: List[Tuple[int, int]] = []
        for b in range(input_ids.shape[0]):
            row = b * num_cameras + primary_cam_id
            if row >= image_grid_thw.shape[0]:
                grid.append((0, 0))
                continue
            h_tok = int(image_grid_thw[row, 1]) // s2
            w_tok = int(image_grid_thw[row, 2]) // s2
            grid.append((h_tok, w_tok))
        state.primary_grid = grid

        if state.ffs_feat is not None:
            image_mask = input_ids == image_token_id
            for b, (h_tok, w_tok) in enumerate(grid):
                if not bool(image_mask[b].any()):
                    continue
                n_primary = int((state.per_token_cam_id[b] == primary_cam_id).sum().item())
                if n_primary == 0 or h_tok * w_tok != n_primary:
                    _raise_primary_grid_mismatch(label, b, h_tok, w_tok, n_primary, primary_cam_id)
        return None

    def _make_layer_hook(proj_idx: int):
        def _layer_hook(module, args, output):
            if not state.enabled:
                return None
            if state.ffs_feat is None or state.per_token_cam_id is None or state.primary_grid is None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            add = torch.zeros_like(hidden)
            B = hidden.shape[0]
            if state.ffs_feat.shape[0] != B:
                raise RuntimeError(
                    f"[{label}] FFS batch {state.ffs_feat.shape[0]} != VLM batch {B}"
                )

            projector = projectors[proj_idx]
            proj_dtype = next(projector.parameters()).dtype
            for b in range(B):
                primary_pos = (state.per_token_cam_id[b] == primary_cam_id).nonzero(as_tuple=True)[0]
                n_primary = int(primary_pos.numel())
                h_tok, w_tok = state.primary_grid[b]
                if h_tok * w_tok != n_primary:
                    _raise_primary_grid_mismatch(label, b, h_tok, w_tok, n_primary, primary_cam_id)
                if n_primary == 0:
                    continue
                residual = projector(state.ffs_feat[b : b + 1].to(dtype=proj_dtype), h_tok, w_tok)
                add[b, primary_pos] = residual.to(dtype=add.dtype, device=add.device)

            out_hidden = hidden + add
            if isinstance(output, tuple):
                return (out_hidden, *output[1:])
            return out_hidden

        return _layer_hook

    hf_model.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True)
    for proj_pos, layer_idx in enumerate(targets):
        layers[layer_idx].register_forward_hook(_make_layer_hook(proj_pos))

    logger.info(
        "[%s] installed VLM residual hooks on %d target layers %s (of %d total), primary_cam_id=%d",
        label,
        len(targets),
        targets,
        len(layers),
        primary_cam_id,
    )
    return state


class QwenGR00TNet0FFSMixin:
    """Shared frozen Fast-FoundationStereo net[0] extraction helpers."""

    def _init_frozen_ffs_net0(self, ffs_cfg, label: str, capture_net0: bool = True) -> None:
        self._ffs_label = label
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self._ffs_model_path = ffs_model_path
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.ffs_feat_dim = int(ffs_cfg.get("gru_hidden_dim", 16))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "gru_hidden"))
        self.num_cameras = int(ffs_cfg.get("num_cameras", 2))
        if os.environ.get("FFS_DISABLE_UNROTATE", "").strip():
            raise ValueError(
                f"{label} no longer accepts FFS_DISABLE_UNROTATE; the FFS path is "
                "always the clean leftprimary convention now."
            )

        self.stereo_convention = _ffs_stereo_convention()
        legacy_keys = [
            key
            for key in ("primary_idx", "right_view_idx", "primary_cam_id")
            if _cfg_contains(ffs_cfg, key)
        ]
        if legacy_keys:
            raise ValueError(
                f"{label} clean leftprimary convention refuses legacy FFS keys {legacy_keys}. "
                "Use left_ref_idx=1, primary_view_idx=0, inject_cam_id=1 for new runs."
            )
        self.left_ref_idx = int(ffs_cfg.get("left_ref_idx", 1))
        self.primary_view_idx = int(ffs_cfg.get("primary_view_idx", 0))
        self.inject_cam_id = int(ffs_cfg.get("inject_cam_id", 1))
        expected = {
            "left_ref_idx": (self.left_ref_idx, 1),
            "primary_view_idx": (self.primary_view_idx, 0),
            "inject_cam_id": (self.inject_cam_id, 1),
        }
        self.ffs_image1_idx = self.left_ref_idx
        self.ffs_image2_idx = self.primary_view_idx
        self.view_order = ("primary", "left_view")
        self.reference_view = "left_view"
        self.net0_frame = "left_view"

        bad = {k: (got, want) for k, (got, want) in expected.items() if got != want}
        if bad:
            raise ValueError(f"{label} invalid {self.stereo_convention} stereo constants: {bad}")
        if self.num_cameras != 2:
            raise ValueError(f"{label} requires num_cameras=2, got {self.num_cameras}")
        # Compatibility aliases for older helper modules. New code should use the
        # explicit names above.
        self.primary_idx = self.primary_view_idx
        self.right_view_idx = self.left_ref_idx
        self.primary_cam_id = self.inject_cam_id
        if self.ffs_feature_source != "gru_hidden":
            raise ValueError(f"{label} only supports ffs_feature_source='gru_hidden'")
        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"{label} FFS model not found: {ffs_model_path}")

        expected_sha256 = ffs_cfg.get("ffs_expected_sha256", None)
        self._ffs_expected_sha256 = str(expected_sha256) if expected_sha256 else None
        def _cache_dir_enabled(value) -> bool:
            return value is not None and str(value).strip().lower() not in {"", "none", "null"}

        utonia_cache_enabled = _cache_dir_enabled(ffs_cfg.get("utonia_cache_dir", None))
        ffs_cache_enabled = _cache_dir_enabled(ffs_cfg.get("ffs_cache_dir", None))
        self._ffs_pin_tf32_off = ffs_cache_enabled or os.environ.get("FFS_PIN_TF32_OFF", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        should_hash = bool(expected_sha256) or utonia_cache_enabled or ffs_cache_enabled
        actual = None
        if should_hash:
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
        self._ffs_actual_sha256 = actual
        if expected_sha256:
            if actual != expected_sha256:
                raise RuntimeError(
                    f"{label} FFS SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info("%s FFS SHA256 verified (%s)", label, expected_sha256[:12])
        else:
            logger.warning("%s no ffs_expected_sha256 set; torch.load uses weights_only=False", label)

        _ffs_register_fixup()
        logger.info("%s loading frozen FFS from %s", label, ffs_model_path)
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for param in self.ffs.parameters():
            param.requires_grad = False
        try:
            self.ffs.args.mixed_precision = False
        except Exception:
            self.ffs.args["mixed_precision"] = False

        self._ffs_captured_net0 = None
        self._ffs_net0_hook_handle = None
        if capture_net0:
            def _capture_net0_hook(_module, _inputs, output):
                self._ffs_captured_net0 = output[0][0]

            self._ffs_net0_hook_handle = self.ffs.update_block.register_forward_hook(_capture_net0_hook)

    def _imgs_to_ffs_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        from torchvision import transforms

        to_tensor = transforms.ToTensor()
        resize = transforms.Resize(
            (self.ffs_image_size, self.ffs_image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        device = next(self.parameters()).device
        out = []
        for example_imgs in batch_images:
            # The dataloader packs camera-major with frames inner: at T>1 the list is
            # [cam0_t0, cam0_t1, cam1_t0, ...] and a bare index would silently pick
            # a same-camera frame as the "other eye" (garbage disparity). FFS stereo
            # is defined for exactly num_cameras single-frame images per sample.
            if len(example_imgs) != int(self.num_cameras):
                raise ValueError(
                    f"{self._ffs_label} expected exactly {self.num_cameras} single-frame "
                    f"images per sample (camera-major), got {len(example_imgs)} — "
                    "multi-frame (T>1) stereo pairing is not supported."
                )
            if view_idx >= len(example_imgs):
                raise IndexError(
                    f"{self._ffs_label} view_idx={view_idx} out of range for {len(example_imgs)} images"
                )
            img = example_imgs[view_idx]
            if not torch.is_tensor(img):
                img = to_tensor(img)
            img = resize(img.unsqueeze(0)).squeeze(0)
            out.append(img * 255.0)
        return torch.stack(out, dim=0).to(device).float()

    def _compute_ffs_feature(self, batch_images: List) -> torch.Tensor:
        image1 = self._imgs_to_ffs_tensor(batch_images, self.ffs_image1_idx)
        image2 = self._imgs_to_ffs_tensor(batch_images, self.ffs_image2_idx)
        # Upright ffs(left_view, primary). FoundationStereo image1 is the
        # geometric-left/reference view, and net[0] stays in that left_view frame.
        B = image1.shape[0]
        with torch.no_grad():
            if next(self.ffs.parameters()).dtype != torch.float32:
                self.ffs.float()
            # The frozen FoundationStereo net contains BatchNorm + Dropout layers.
            # The trainer's model.train() recursively flips self.ffs back into TRAIN
            # mode, which would make the train-time net[0] feature (batch-stat BN +
            # active dropout, plus drifting BN running stats) differ from the
            # eval-time feature (running-stat BN, no dropout) — a silent train/eval
            # mismatch in the injected disparity. Re-assert eval() before EVERY FFS
            # forward so the injected feature is deterministic and identical between
            # training and evaluation. (requires_grad=False alone does NOT do this;
            # BN/Dropout behaviour is governed by module.training, not by autograd.)
            self.ffs.eval()
            self._ffs_captured_net0 = None
            tf32_ctx = ffs_tf32_disabled() if getattr(self, "_ffs_pin_tf32_off", False) else contextlib.nullcontext()
            with tf32_ctx, torch.amp.autocast("cuda", enabled=False):
                self.ffs(
                    image1.float(),
                    image2.float(),
                    iters=int(self.ffs.args.valid_iters),
                    test_mode=True,
                )
            if self._ffs_captured_net0 is None:
                raise RuntimeError(f"{self._ffs_label} update_block net[0] hook did not fire")
            ffs_feat = self._ffs_captured_net0
            if ffs_feat.shape[0] != B:
                raise RuntimeError(f"{self._ffs_label} net[0] batch {ffs_feat.shape[0]} != {B}")
            if ffs_feat.shape[1] != self.ffs_feat_dim:
                raise RuntimeError(
                    f"{self._ffs_label} net[0] channels {ffs_feat.shape[1]} != configured "
                    f"gru_hidden_dim {self.ffs_feat_dim}"
                )
        return ffs_feat.detach()

    def _compute_ffs_disparity(self, batch_images: List) -> torch.Tensor:
        image1 = self._imgs_to_ffs_tensor(batch_images, self.ffs_image1_idx)
        image2 = self._imgs_to_ffs_tensor(batch_images, self.ffs_image2_idx)
        B = image1.shape[0]
        with torch.no_grad():
            if next(self.ffs.parameters()).dtype != torch.float32:
                self.ffs.float()
            self.ffs.eval()
            tf32_ctx = ffs_tf32_disabled() if getattr(self, "_ffs_pin_tf32_off", False) else contextlib.nullcontext()
            with tf32_ctx, torch.amp.autocast("cuda", enabled=False):
                disp_up = self.ffs(
                    image1.float(),
                    image2.float(),
                    iters=int(self.ffs.args.valid_iters),
                    test_mode=True,
                )
        if not torch.is_tensor(disp_up):
            raise RuntimeError(
                f"{self._ffs_label} expected FFS test_mode=True to return a Tensor disparity, "
                f"got {type(disp_up).__name__}"
            )
        if disp_up.shape != (B, 1, self.ffs_image_size, self.ffs_image_size):
            raise RuntimeError(
                f"{self._ffs_label} FFS disparity shape {tuple(disp_up.shape)} != "
                f"{(B, 1, self.ffs_image_size, self.ffs_image_size)}"
            )
        return disp_up.detach().clone()


class QwenGR00TFFSBase(QwenGR00TNet0FFSMixin, Qwen_GR00T):
    """Common GR00T FFS path used by training forward and predict_action."""

    def __init__(self, config):
        super().__init__(config)
        # The parallel PRoPE camera branch is NOT validated together with FFS
        # frameworks: depth-token insertion shifts image rows AFTER cam_branch's
        # pre-hook caches positions (silent geometry corruption), and the two
        # load-audit layers do not compose. Refuse the combination outright.
        if bool(self.config.framework.qwenvl.get("stereo_cam_branch_enabled", False)):
            raise RuntimeError(
                "[GR00T-FFS] stereo_cam_branch_enabled=True is unsupported with FFS "
                "frameworks (token insertion invalidates the branch's cached image "
                "positions). Disable cam_branch or use the plain QwenGR00T framework."
            )

    def _sync_actual_vlm_hidden_dim(self) -> int:
        llm_dim = qwen_vlm_hidden_size(self.qwen_vl_interface)
        self.config.framework.qwenvl.vl_hidden_dim = llm_dim
        return llm_dim

    def _prepare_ffs_for_vlm(self, batch_images: List, sample_ids=None) -> None:
        raise NotImplementedError

    def _cleanup_ffs_after_vlm(self) -> None:
        return None

    def _run_qwenvl_forward(self, qwen_inputs) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return qwenvl_outputs.hidden_states[-1]

    def _encode_last_hidden_with_ffs(
        self,
        batch_images: List,
        instructions: List[str],
        sample_ids=None,
    ) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        prepare_kwargs = {}
        if "sample_ids" in inspect.signature(self._prepare_ffs_for_vlm).parameters:
            prepare_kwargs["sample_ids"] = sample_ids
        self._prepare_ffs_for_vlm(batch_images, **prepare_kwargs)
        try:
            return self._run_qwenvl_forward(qwen_inputs)
        finally:
            self._cleanup_ffs_after_vlm()

    def _sample_ids_from_examples(self, examples: List[dict]):
        ids = []
        for example in examples:
            if all(key in example for key in ("dataset_name", "traj_id", "base_index")):
                ids.append((example["dataset_name"], example["traj_id"], example["base_index"]))
            else:
                ids.append(None)
        return ids if any(sample_id is not None for sample_id in ids) else None

    def _encode_last_hidden_with_optional_sample_ids(
        self,
        batch_images: List,
        instructions: List[str],
        sample_ids=None,
    ) -> torch.Tensor:
        encode_kwargs = {}
        if "sample_ids" in inspect.signature(self._encode_last_hidden_with_ffs).parameters:
            encode_kwargs["sample_ids"] = sample_ids
        return self._encode_last_hidden_with_ffs(batch_images, instructions, **encode_kwargs)

    def _scene_flow_cfg(self):
        action_cfg = self.config.framework.action_model if self.config and hasattr(self.config, "framework") else {}
        return _cfg_get(action_cfg, "scene_flow", {}) or {}

    def _scene_flow_enabled(self) -> bool:
        return _cfg_bool(self._scene_flow_cfg(), "enabled", False)

    def _collect_scene_flow_targets(self, examples: List[dict], device: torch.device):
        required = ("flow_gt", "flow_valid", "flow_dynamic")
        if not examples or not all(all(key in example for key in required) for example in examples):
            return None
        flow_gt = torch.as_tensor(
            np.stack([np.asarray(example["flow_gt"], dtype=np.float32) for example in examples]),
            device=device,
            dtype=torch.float32,
        )
        flow_valid = torch.as_tensor(
            np.stack([np.asarray(example["flow_valid"], dtype=bool) for example in examples]),
            device=device,
            dtype=torch.bool,
        )
        flow_dynamic = torch.as_tensor(
            np.stack([np.asarray(example["flow_dynamic"], dtype=bool) for example in examples]),
            device=device,
            dtype=torch.bool,
        )
        flow_has_gt = torch.as_tensor(
            [bool(example.get("flow_has_gt", False)) for example in examples],
            device=device,
            dtype=torch.bool,
        )
        return flow_gt, flow_valid, flow_dynamic, flow_has_gt

    def _compute_scene_flow_loss(
        self,
        flow_pred: torch.Tensor,
        flow_gt: torch.Tensor,
        flow_valid: torch.Tensor,
        flow_dynamic: torch.Tensor,
        flow_has_gt: torch.Tensor,
    ):
        cfg = self._scene_flow_cfg()
        target_hw = tuple(int(x) for x in flow_gt.shape[1:3])
        pred = F.interpolate(
            flow_pred.float(),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        target = flow_gt.permute(0, 3, 1, 2).contiguous().float()
        valid = flow_valid.bool()
        raw_dynamic = flow_dynamic.bool()
        dynamic_outside_valid = raw_dynamic & ~valid
        dynamic = raw_dynamic & valid
        mask_mode = str(_cfg_get(cfg, "mask_mode", "dynamic")).lower()
        if mask_mode == "valid":
            supervised = valid
            used_valid_fallback = pred.new_tensor(0.0)
        elif mask_mode == "dynamic":
            supervised = dynamic
            used_valid_fallback = pred.new_tensor(0.0)
            if _cfg_bool(cfg, "dynamic_fallback_to_valid", True):
                no_dynamic = dynamic.flatten(1).sum(dim=1) == 0
                has_valid = valid.flatten(1).sum(dim=1) > 0
                fallback = no_dynamic & has_valid
                if bool(fallback.any().item()):
                    supervised = torch.where(fallback[:, None, None], valid, supervised)
                    used_valid_fallback = fallback.float().sum()
        else:
            raise ValueError(f"Unsupported scene_flow mask_mode={mask_mode!r}; expected 'dynamic' or 'valid'")

        beta = float(_cfg_get(cfg, "smooth_l1_beta", 0.01))
        diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
        channel_w = pred.new_tensor((1.0, 1.0, 2.0)).view(1, 3, 1, 1)
        diff = diff * channel_w
        mask = supervised.unsqueeze(1).to(dtype=diff.dtype)
        denom = mask.sum().clamp_min(1.0)
        flow_loss = (diff * mask).sum() / (denom * 3.0)

        supervised_pixels_per_sample = supervised.flatten(1).sum(dim=1)
        dynamic_pixels = dynamic.flatten(1).sum(dim=1)
        metrics = {
            "flow_supervised_samples": (supervised_pixels_per_sample > 0).float().sum(),
            "flow_gt_samples": flow_has_gt.float().sum(),
            "dynamic_pixel_count": dynamic_pixels.float().sum(),
            "dynamic_outside_valid_pixel_count": dynamic_outside_valid.flatten(1).float().sum(),
            "flow_supervised_pixel_count": supervised_pixels_per_sample.float().sum(),
            "nonzero_flow_batches": (dynamic_pixels.sum() > 0).to(dtype=pred.dtype),
            "flow_valid_fallback_samples": used_valid_fallback,
        }
        return flow_loss, metrics

    def forward(self, examples: List[dict] = None, **kwargs):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        sample_ids = self._sample_ids_from_examples(examples)

        last_hidden = self._encode_last_hidden_with_optional_sample_ids(
            batch_images,
            instructions,
            sample_ids=sample_ids,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_output = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        if not isinstance(action_output, dict):
            return {"action_loss": action_output}

        action_loss = action_output["action_loss"]
        output = {"action_loss": action_loss}
        flow_pred = action_output.get("flow_pred", None)
        if self._scene_flow_enabled() and flow_pred is not None:
            original_batch = len(examples)
            if flow_pred.shape[0] != original_batch:
                if flow_pred.shape[0] % original_batch != 0:
                    raise RuntimeError(
                        "scene_flow flow_pred batch is not divisible by the original batch: "
                        f"{flow_pred.shape[0]} vs {original_batch}"
                    )
                flow_pred = flow_pred.view(
                    flow_pred.shape[0] // original_batch,
                    original_batch,
                    *flow_pred.shape[1:],
                )[0]

            targets = self._collect_scene_flow_targets(examples, device=flow_pred.device)
            if targets is None:
                output["flow_loss"] = flow_pred.sum() * 0.0
                output.update(
                    {
                        "flow_supervised_samples": flow_pred.new_tensor(0.0),
                        "flow_gt_samples": flow_pred.new_tensor(0.0),
                        "dynamic_pixel_count": flow_pred.new_tensor(0.0),
                        "dynamic_outside_valid_pixel_count": flow_pred.new_tensor(0.0),
                        "flow_supervised_pixel_count": flow_pred.new_tensor(0.0),
                        "nonzero_flow_batches": flow_pred.new_tensor(0.0),
                        "flow_valid_fallback_samples": flow_pred.new_tensor(0.0),
                    }
                )
            else:
                flow_loss, flow_metrics = self._compute_scene_flow_loss(flow_pred, *targets)
                output["flow_loss"] = flow_loss
                output.update(flow_metrics)

        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: str):
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden = self._encode_last_hidden_with_optional_sample_ids(
            batch_images,
            instructions,
            sample_ids=None,
        )

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _ffs_key_prefixes(self) -> Tuple[str, ...]:
        return ("ffs.",)

    def _frozen_encoder_key_prefixes(self) -> Tuple[str, ...]:
        return ("ffs.",)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        frozen_prefixes = self._frozen_encoder_key_prefixes()
        if not frozen_prefixes:
            return state
        state_prefix = kwargs.get("prefix", "")
        if len(args) >= 2 and isinstance(args[1], str):
            state_prefix = args[1]

        def is_frozen_key(key: str) -> bool:
            if key.startswith(frozen_prefixes):
                return True
            if state_prefix and key.startswith(state_prefix):
                return key[len(state_prefix):].startswith(frozen_prefixes)
            return False

        for key in list(state.keys()):
            if is_frozen_key(key):
                state.pop(key)
        return state

    def _is_ffs_key(self, key: str) -> bool:
        return key.startswith(self._ffs_key_prefixes())

    def load_state_dict(self, state_dict, strict=True, assign=False, init_from_baseline: bool = False):
        # init_from_baseline default is FALSE (strict): a standalone load / eval /
        # resume of an FFS checkpoint MUST contain the FFS-adapter keys, else fail
        # loud. ONLY the explicit warm-start-from-a-non-FFS-baseline path (the
        # trainer's pretrained_checkpoint loader) passes init_from_baseline=True to
        # allow the FFS-adapter keys to be fresh-initialised. This prevents a
        # corrupted/partial FFS checkpoint from silently loading with missing
        # adapter params.
        raw_own_keys = set(super().state_dict().keys())
        frozen_prefixes = self._frozen_encoder_key_prefixes()
        own_keys = {key for key in raw_own_keys if not key.startswith(frozen_prefixes)}
        provided_frozen = {key for key in state_dict.keys() if key.startswith(frozen_prefixes)}
        if provided_frozen:
            logger.info(
                "[GR00T-FFS audit] dropping %d frozen-encoder checkpoint keys; "
                "encoders are loaded from configured paths",
                len(provided_frozen),
            )
            state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith(frozen_prefixes)
            }
        provided_keys = set(state_dict.keys())
        if init_from_baseline:
            # All-or-none per adapter family: a checkpoint carrying SOME keys of an
            # FFS family while missing others is a truncated/corrupted FFS checkpoint,
            # not a baseline — fresh-initialising the gaps would silently corrupt the
            # run. ('ffs.' frozen-net keys are exempt: loaded + sha-verified separately.)
            fresh_init_prefixes = list(self._ffs_key_prefixes())
            if self._scene_flow_enabled():
                fresh_init_prefixes.append("action_model.scene_flow_decoder.")
            for prefix in fresh_init_prefixes:
                if prefix in frozen_prefixes:
                    continue
                own_family = {k for k in own_keys if k.startswith(prefix)}
                provided_family = {k for k in provided_keys if k.startswith(prefix)}
                if provided_family and (own_family - provided_family):
                    sample = sorted(own_family - provided_family)[:10]
                    raise RuntimeError(
                        f"[GR00T-FFS audit] checkpoint has a PARTIAL '{prefix}' family: "
                        f"{len(provided_family)} present, {len(own_family - provided_family)} "
                        f"missing (first={sample}). Refusing fresh-init of a truncated family."
                    )
        allowed_missing = set(raw_own_keys - own_keys)
        if init_from_baseline:
            allowed_missing.update({key for key in own_keys - provided_keys if self._is_ffs_key(key)})
            if self._scene_flow_enabled():
                allowed_missing.update(
                    {
                        key
                        for key in own_keys - provided_keys
                        if key.startswith("action_model.scene_flow_decoder.")
                    }
                )
        legacy_cam_rope_keys = (
            {
                key
                for key in provided_keys
                if key.startswith("stereo_cam_rope_layers.") or ".stereo_cam_layer." in key
            }
            if init_from_baseline and getattr(self, "stereo_cam_rope_layers", None) is None
            else set()
        )
        suspicious_missing = (own_keys - provided_keys) - allowed_missing
        unexpected = (provided_keys - own_keys) - legacy_cam_rope_keys

        if suspicious_missing:
            sample = sorted(suspicious_missing)[:20]
            raise RuntimeError(
                f"[GR00T-FFS audit] warm-start would leave non-FFS params uninitialized; "
                f"missing {len(suspicious_missing)} keys, first={sample}"
            )
        if unexpected:
            sample = sorted(unexpected)[:20]
            raise RuntimeError(
                f"[GR00T-FFS audit] checkpoint has {len(unexpected)} unexpected keys; "
                f"first={sample}. Refusing silent drop."
            )

        if allowed_missing:
            logger.info(
                "[GR00T-FFS audit] allowing %d missing FFS-only keys during warm-start",
                len(allowed_missing),
            )
        forwarded_strict = strict and not allowed_missing
        # Forward init_from_baseline so the next audit layer in the MRO (Qwen_GR00T's
        # cam_branch-aware override) sees the caller's intent instead of its False
        # default — the two allowances must compose, not veto each other.
        return super().load_state_dict(
            state_dict, strict=forwarded_strict, assign=assign, init_from_baseline=init_from_baseline
        )
