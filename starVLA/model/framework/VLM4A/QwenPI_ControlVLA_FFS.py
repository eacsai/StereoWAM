# QwenPI + Fast-FoundationStereo ControlVLA-style branch (V4).
#
# V4 vs V1 (QwenPI_ControlNet_FFS):
#   V1: per-DiT-layer projector adds residual onto vl_embs_list[i]
#       (modifies condition tokens before they enter action cross-attention).
#   V4: per-DiT-layer parallel K/V branch (ControlVLA Sec 3.2 §π_g) inside the
#       action DiT cross-attn block. Same Q, parallel K_z/V_z from FFS, zero-init.
#       Strict step-0 byte-identical because to_k_z = to_v_z = 0.
#
# This file is structurally a sibling of QwenPI_ControlNet_FFS.py: same FFS load,
# same SHA256 guard, same load_state_dict audit. Only the *injection mechanism*
# differs (handled by install_controlvla_branches + module-level state).

from dataclasses import dataclass, field
from typing import List, Optional
import os, sys, logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI as QwenPI, QwenPIDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.stereo.controlvla_branch import (
    install_controlvla_branches,
    set_ffs_tokens,
    clear_ffs_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class QwenPIControlVLAFFSDefaultConfig(QwenPIDefaultConfig):
    name: str = "QwenPIControlVLAFFS"
    ffs_controlvla: dict = field(
        default_factory=lambda: {
            "ffs_model_path": "./playground/Pretrained_models/Fast-FoundationStereo/20-30-48/model_best_bp2_serialize.pth",
            "ffs_scale": 0,
            "ffs_image_size": 256,
            "ffs_pool_size": 8,            # AdaptiveAvgPool2d output (T_ffs = pool_size**2)
            "primary_idx": 0,
            "right_view_idx": 1,
            "ffs_expected_sha256": None,
            # V3=A: when True, also run FFS cost_volume + classifier to get
            # init_disp [B, 1, H/4, W/4], concat as extra channel to FFS features.
            # Gives the branch explicit stereo geometry signal (vs implicit
            # features-only). Adds ~10 ms/step + ~500 MB GPU mem.
            "use_init_disp": False,
            # 2026-05-29: which FFS tensor to inject as the stereo signal.
            #   "backbone"   = concat[left,right] of the FFS Feature pyramid. FoundationStereo's
            #                  Feature module is a MONOCULAR 2D CNN run per-view BEFORE the cost
            #                  volume, so this carries NO left-right matching / disparity signal
            #                  (the model must learn correspondence itself). Legacy default.
            #   "gru_hidden" = FFS GRU hidden state net[0] AFTER the cost volume + refinement
            #                  iterations: a single fused, disparity-aware map (B, hidden, H/4, W/4).
            #                  This is the actual stereo feature. Costs the full FFS forward.
            "ffs_feature_source": "backbone",
        }
    )


# Bootstrap Fast-FoundationStereo import path (same as V1).
_FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "/data/wangqiwei/ICLR2026/Fast-FoundationStereo")
if os.path.isdir(_FFS_REPO_DIR) and _FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, _FFS_REPO_DIR)


def _ffs_register_fixup():
    import core.foundation_stereo as _fs  # noqa: F401


@FRAMEWORK_REGISTRY.register("QwenPIControlVLAFFS")
class QwenPIControlVLAFFS(QwenPI):
    """QwenPI + FFS stereo, ControlVLA-style parallel K/V branch in action DiT.

    The branch is installed once at __init__ time on every cross-attn block of
    self.action_model.model.transformer_blocks. At every forward pass
    `_encode_vl_hidden_states` recomputes FFS stereo tokens (pooled to
    ffs_pool_size x ffs_pool_size) and stashes them on a module-level state
    holder; the patched block.forward consumes them inside SDPA.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.config = merge_framework_config(QwenPIControlVLAFFSDefaultConfig, self.config)

        ffs_cfg = self.config.framework.get("ffs_controlvla", {})
        ffs_model_path = str(ffs_cfg.get("ffs_model_path"))
        self.ffs_scale = int(ffs_cfg.get("ffs_scale", 0))
        self.ffs_image_size = int(ffs_cfg.get("ffs_image_size", 256))
        self.ffs_pool_size = int(ffs_cfg.get("ffs_pool_size", 8))
        self.primary_idx = int(ffs_cfg.get("primary_idx", 0))
        self.right_view_idx = int(ffs_cfg.get("right_view_idx", 1))
        self.use_init_disp = bool(ffs_cfg.get("use_init_disp", False))
        self.ffs_feature_source = str(ffs_cfg.get("ffs_feature_source", "backbone"))
        if self.ffs_feature_source not in ("backbone", "gru_hidden"):
            raise ValueError(
                f"[ControlVLA-FFS] ffs_feature_source must be 'backbone' or 'gru_hidden', "
                f"got {self.ffs_feature_source!r}"
            )
        if self.ffs_feature_source == "gru_hidden" and self.use_init_disp:
            # net[0] is already disparity-aware (refined by the GRU over the cost
            # volume), so the explicit init_disp channel is redundant here and the
            # init_disp code path needs the backbone pyramid. Keep the two mutually
            # exclusive to avoid an extra FFS feature pass.
            raise ValueError(
                "[ControlVLA-FFS] use_init_disp=True is not supported with "
                "ffs_feature_source='gru_hidden' (net[0] already carries disparity)."
            )

        if not os.path.isfile(ffs_model_path):
            raise FileNotFoundError(f"FFS model not found: {ffs_model_path}")

        # SHA256 guard (codex round-2 HIGH#2 — pickle safety).
        expected_sha256 = ffs_cfg.get("ffs_expected_sha256")
        if expected_sha256:
            import hashlib
            with open(ffs_model_path, "rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"[ControlVLA-FFS] SHA256 mismatch: expected={expected_sha256} got={actual}"
                )
            logger.info(f"[ControlVLA-FFS] FFS SHA256 verified ({expected_sha256[:12]}…)")
        else:
            logger.warning(
                f"[ControlVLA-FFS] no ffs_expected_sha256 set — pickle load via weights_only=False"
            )

        _ffs_register_fixup()
        logger.info(f"[ControlVLA-FFS] loading frozen FFS from {ffs_model_path}")
        self.ffs = torch.load(ffs_model_path, map_location="cpu", weights_only=False)
        self.ffs.eval()
        for p in self.ffs.parameters():
            p.requires_grad = False

        if self.ffs_feature_source == "gru_hidden":
            # net[0] = FFS GRU hidden state after cost volume + refinement, a single
            # fused disparity-aware map (B, C_net0, H/4, W/4). No L+R concat — net[0]
            # already fuses both views through the cost volume.
            # C_net0 is NOT hidden_dims[0] (=128): the distilled 20-30-48 FFS GRU hidden
            # is 16 channels (empirically verified net[0]=(B,16,H/4,W/4)). Configurable
            # for other FFS models; a forward-time assert validates the actual shape.
            self.ffs_feat_dim = int(ffs_cfg.get("gru_hidden_dim", 16))
            # FFS.forward() wraps its body in autocast(dtype=fp16) when args.mixed_precision
            # is True. Under bf16 training, DeepSpeed casts the frozen FFS params to bf16, and
            # the fp16 autocast then collides with bf16 weights ("Expected weight to have type
            # Float but got BFloat16" inside the cost-agg distill blocks). Disable FFS's internal
            # autocast so the full forward runs in its native (bf16) param dtype throughout —
            # matching the manual-submodule path that use_init_disp already uses successfully.
            try:
                self.ffs.args.mixed_precision = False
            except Exception:
                self.ffs.args["mixed_precision"] = False
            # Capture net[0] from the last GRU update of each FFS forward via a hook.
            self._ffs_captured_net0 = None
            def _capture_net0_hook(_module, _inputs, output):
                # update_block.forward returns (net_list, mask, delta_disp);
                # net_list[0] is the refined hidden state (B, hidden, H/4, W/4).
                # Fires once per GRU iteration → the last write is the final state.
                self._ffs_captured_net0 = output[0][0]
            self.ffs.update_block.register_forward_hook(_capture_net0_hook)
            logger.info(
                f"[ControlVLA-FFS] feature_source=gru_hidden net0_dim={self.ffs_feat_dim} "
                f"channels, pool={self.ffs_pool_size}x{self.ffs_pool_size}, "
                f"valid_iters={int(self.ffs.args.valid_iters)}"
            )
        else:
            ffs_feat_dim_per_view = int(self.ffs.feature.d_out[self.ffs_scale])
            # V3=A: extra +1 channel for init_disp [B, 1, H/4, W/4] concat into ffs_stereo
            ffs_feat_dim_base = ffs_feat_dim_per_view * 2          # concat L+R
            self.ffs_feat_dim = ffs_feat_dim_base + (1 if self.use_init_disp else 0)
            logger.info(
                f"[ControlVLA-FFS] feature_source=backbone ffs_scale={self.ffs_scale} "
                f"per_view={ffs_feat_dim_per_view} concat={ffs_feat_dim_base} "
                f"+ init_disp={1 if self.use_init_disp else 0} = {self.ffs_feat_dim} channels, "
                f"pool={self.ffs_pool_size}x{self.ffs_pool_size}"
            )

        # Install per-layer parallel K/V branches into action DiT.
        # CODEX FIX (round-2 HIGH-2): expected count is now `num_layers` from
        # the action_model config (architectural truth), not `count(cross_attn !=
        # None)` (tautological with install's predicate). Plus require_all_cross_attn
        # raises if interleave_self_attention is on so "per-layer ControlVLA" actually
        # is per-layer.
        dit_inner = self.action_model.model
        configured_num_layers = int(
            self.config.framework.action_model.diffusion_model_cfg.get(
                "num_layers", len(dit_inner.transformer_blocks)
            )
        )
        self._n_controlvla_blocks = install_controlvla_branches(
            action_dit_model=dit_inner,
            ffs_token_dim=self.ffs_feat_dim,
            expected_n_total_blocks=configured_num_layers,
            require_all_cross_attn=True,
        )

        # CODEX FIX (round-3 MED-3): add learnable positional embedding to FFS
        # tokens. Without this, the branch attention is permutation-invariant
        # over the (pool_size x pool_size) FFS spatial tokens — official
        # ControlVLA adds `control_cond_pos_emb` to control memory for the same
        # reason. Shared across all DiT layers (lives at framework level), small
        # init so step-0 K_z+V_z=0 still gives parity (PE adds noise to
        # to_k_z input which is multiplied by zero weight = zero output).
        n_ffs_tokens = self.ffs_pool_size * self.ffs_pool_size
        self.ffs_pos_emb = nn.Parameter(
            torch.randn(1, n_ffs_tokens, self.ffs_feat_dim) * 0.02
        )
        logger.info(
            f"[ControlVLA-FFS] added learnable ffs_pos_emb shape=(1, {n_ffs_tokens}, {self.ffs_feat_dim})"
        )

    # --- helpers ---------------------------------------------------------- #

    def _imgs_to_ffs_tensor(self, batch_images: List, view_idx: int) -> torch.Tensor:
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        device = next(self.parameters()).device
        size = self.ffs_image_size
        resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)
        out = []
        for example_imgs in batch_images:
            img = example_imgs[view_idx]
            if not torch.is_tensor(img):
                img = to_tensor(img)
            img = resize(img.unsqueeze(0)).squeeze(0)
            out.append(img * 255.0)
        return torch.stack(out, dim=0).to(device).float()

    # --- override parent encode ------------------------------------------- #

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> List[torch.Tensor]:
        # CODEX FIX (MED-2 cleanup hygiene): clear any leftover state from a
        # previous forward that may have errored out before completing. This
        # ensures fresh state per forward, paired with set_ffs_tokens at the end.
        clear_ffs_tokens()

        # 1) Base Qwen forward (cam_rope baked in by parent QwenPI hooks). vl_embs
        #    is NOT mutated here — branch injects into action cross-attn, not condition.
        vl_embs_list = super()._encode_vl_hidden_states(batch_images, instructions)

        # 2) Stereo pair -> FFS features for both views (frozen).
        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)
        right   = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)
        B = primary.shape[0]
        ffs_dtype = next(self.ffs.parameters()).dtype
        stacked = torch.cat([primary, right], dim=0).to(dtype=ffs_dtype)  # (2B, 3, H, W)
        with torch.no_grad():
            mean = stacked.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = stacked.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            stacked_normed = (stacked / 255.0 - mean) / std
            if self.ffs_feature_source == "gru_hidden":
                # Run the FULL FFS forward (feature -> cost volume -> GRU refinement)
                # and capture net[0] via the registered hook. Frozen + no_grad.
                # net[0] is a single fused, disparity-aware map (no L+R concat).
                #
                # ⚠ FFS.forward() normalizes inputs INTERNALLY (foundation_stereo.py
                # normalize_image expects RAW 0-255 RGB and applies the SAME ImageNet
                # mean/std we use manually). So pass the RAW 0-255 `stacked` here — NOT
                # `stacked_normed`, which would double-normalize and corrupt the cost
                # volume / net[0]. (The backbone .feature() path below DOES need the
                # manual normalization because .feature() does not normalize.)
                # Run the frozen FFS entirely in fp32 (its native design dtype). DeepSpeed
                # casts the frozen FFS params to bf16 at prepare(), but FFS internally mixes
                # fp32 intermediates (coords / correlation in the GRU geometry encoder) with
                # its conv weights; under bf16 params that clashes ("Input type (float) and
                # bias type (BFloat16)" at update.py convc1). FFS's own autocast can't fix it
                # because args.mixed_precision=False turns its internal autocast into an
                # explicit *disable* that overrides any outer autocast. Simplest robust fix:
                # keep FFS in fp32 throughout (it is frozen + no_grad, so this is cheap and
                # numerically correct), with autocast off. net[0] is cast to the branch dtype
                # at the token stage below. Cast happens once (subsequent forwards are no-ops).
                if next(self.ffs.parameters()).dtype != torch.float32:
                    self.ffs.float()
                image1 = stacked[:B].float()
                image2 = stacked[B:].float()
                self._ffs_captured_net0 = None
                with torch.amp.autocast("cuda", enabled=False):
                    self.ffs(
                        image1, image2,
                        iters=int(self.ffs.args.valid_iters), test_mode=True,
                    )
                if self._ffs_captured_net0 is None:
                    raise RuntimeError(
                        "[ControlVLA-FFS] gru_hidden: update_block hook did not fire "
                        "— FFS forward path changed?"
                    )
                ffs_stereo = self._ffs_captured_net0  # (B, C_net0, H/4, W/4)
                if ffs_stereo.shape[1] != self.ffs_feat_dim:
                    raise RuntimeError(
                        f"[ControlVLA-FFS] net[0] channel dim {ffs_stereo.shape[1]} != "
                        f"configured ffs_feat_dim {self.ffs_feat_dim} — set "
                        f"ffs_controlvla.gru_hidden_dim={ffs_stereo.shape[1]} to match this FFS model."
                    )
            else:
                ffs_pyramid = self.ffs.feature(stacked_normed)
                features_left  = [o[:B] for o in ffs_pyramid]
                features_right = [o[B:] for o in ffs_pyramid]
                ffs_left  = features_left[self.ffs_scale]   # (B, C, h, w)
                ffs_right = features_right[self.ffs_scale]
                ffs_stereo = torch.cat([ffs_left, ffs_right], dim=1)  # (B, 2C, h, w)

                # V3=A: explicit stereo geometry. Run FFS cost_volume + classifier
                # (skip the iterative GRU updates) → single-channel init disparity
                # at 1/4 res. Concat as extra channel of ffs_stereo. ~+10 ms/step.
                if self.use_init_disp:
                    from core.submodule import (
                        build_gwc_volume_optimized_pytorch1,
                        build_concat_volume_optimized_pytorch1,
                        disparity_regression,
                    )
                    max_disp_div4 = self.ffs.args.max_disp // 4
                    gwc_volume = build_gwc_volume_optimized_pytorch1(
                        features_left[0], features_right[0],
                        max_disp_div4, self.ffs.cv_group,
                        normalize=self.ffs.args.normalize,
                    )
                    left_tmp  = self.ffs.proj_cmb(features_left[0])
                    right_tmp = self.ffs.proj_cmb(features_right[0])
                    concat_volume = build_concat_volume_optimized_pytorch1(
                        left_tmp, right_tmp, maxdisp=max_disp_div4,
                    )
                    comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
                    comb_volume = self.ffs.corr_stem(comb_volume)
                    comb_volume = self.ffs.corr_feature_att(comb_volume, features_left[0])
                    comb_volume = self.ffs.cost_agg(comb_volume, features_left)
                    logits = self.ffs.classifier(comb_volume).squeeze(1)
                    prob = F.softmax(logits, dim=1)
                    init_disp = disparity_regression(prob, max_disp_div4)
                    # init_disp: (B, 1, H/4, W/4). Raw disparity values lie roughly
                    # in [0, max_disp_div4] disparity units. CODEX FIX (round-1 MED):
                    # normalize by max_disp_div4 so the channel sits in [0, 1] and
                    # does not dominate FFS feature magnitudes when concatenated.
                    init_disp_norm = init_disp / float(max_disp_div4)
                    ffs_stereo = torch.cat([ffs_stereo, init_disp_norm.to(ffs_stereo.dtype)], dim=1)
                    # ffs_stereo now (B, 2C+1, H/4, W/4)

            # AdaptiveAvgPool to reduce T_ffs (4096 → 64 by default).
            ffs_pooled = F.adaptive_avg_pool2d(
                ffs_stereo, (self.ffs_pool_size, self.ffs_pool_size)
            )
            # Flatten spatial -> (B, T_ffs, ffs_feat_dim).
            ffs_tokens = ffs_pooled.flatten(2).transpose(1, 2).contiguous().detach()

        # 3) Add learnable positional embedding (round-3 MED-3 fix). Lives in
        # framework, broadcast over batch via PyTorch's implicit (1, T, D) →
        # (B, T, D). Zero-init parity NOT broken because to_k_z/to_v_z weights
        # are zero → branch output stays 0 regardless of PE noise.
        ffs_tokens = ffs_tokens + self.ffs_pos_emb.to(ffs_tokens.dtype)

        # 4) Cast to branch dtype and stash on module state.
        # NOTE: in round-3 redesign, the K_z/V_z linears live on attn1 itself
        # (attn1 is ControlVLAAttention). Find any patched block and use its
        # to_k_z dtype as the branch dtype.
        any_block = next(
            (b for b in self.action_model.model.transformer_blocks
             if hasattr(b.attn1, "to_k_z")),
            None,
        )
        if any_block is not None:
            branch_dtype = any_block.attn1.to_k_z.weight.dtype
            ffs_tokens = ffs_tokens.to(dtype=branch_dtype)
        set_ffs_tokens(ffs_tokens)

        # 4) Return vl_embs UNCHANGED — V4 injects in action cross-attn, not here.
        return vl_embs_list

    # --- init ckpt audit (matches V1's pattern) --------------------------- #

    def load_state_dict(self, state_dict, strict=True, assign=False,
                        init_from_baseline: bool = True):
        """Permissive load gated on init_from_baseline=True. Allows missing
        FFS / ControlVLA / stereo_cam_rope keys when initialising from a
        baseline ckpt that predates this framework; raises on any unexpected
        keys under strict=True.

        CODEX FIX (round-3 MED-2): the round-2 redesign moved trunk Attention
        params under `.attn1.base.*` (was `.attn1.*`). When loading a legacy
        baseline ckpt produced before this framework existed, the old keys
        would be silently dropped as 'unexpected' and the new `.base.*` keys
        would stay random — corrupting the trunk. Remap them here BEFORE the
        audit so the legacy keys land at the right place.
        """
        # Legacy key remap: .attn1.<to_q|to_k|to_v|to_out>.* -> .attn1.base.<...>.*
        # Only remap keys NOT already at .attn1.base.* (avoid double-prefix).
        remapped = {}
        n_legacy = 0
        for k, v in state_dict.items():
            if (
                ".attn1." in k
                and ".attn1.base." not in k
                and any(s in k for s in (".to_q.", ".to_k.", ".to_v.", ".to_out."))
            ):
                new_k = k.replace(".attn1.", ".attn1.base.", 1)
                remapped[new_k] = v
                n_legacy += 1
            else:
                remapped[k] = v
        if n_legacy > 0:
            logger.info(
                f"[ControlVLA-FFS] remapped {n_legacy} legacy attn1.* trunk keys "
                f"to attn1.base.* (round-3 MED-2 fix)"
            )
        state_dict = remapped

        own_keys = set(self.state_dict().keys())
        provided = set(state_dict.keys())

        def _is_branch_key(k: str) -> bool:
            # round-2/3 redesign: branch params at .attn1.to_k_z / .attn1.to_v_z
            # (original Attention moved to .attn1.base.*). ffs_pos_emb is the
            # round-3 MED-3 positional embedding parameter.
            return (
                k.startswith("ffs.")
                or k == "ffs_pos_emb"
                or ".attn1.to_k_z." in k
                or ".attn1.to_v_z." in k
            )

        def _is_stereo_cam_rope_key(k: str) -> bool:
            return (
                k.startswith("stereo_cam_embed.")
                or k.startswith("stereo_cam_rope_layers.")
                or (
                    "qwen_vl_interface.model.model.language_model.layers." in k
                    and ".self_attn.stereo_cam_layer." in k
                )
            )

        if init_from_baseline:
            allowed_missing = {
                k for k in own_keys - provided
                if _is_branch_key(k) or _is_stereo_cam_rope_key(k)
            }
        else:
            allowed_missing = set()

        suspicious_missing = (own_keys - provided) - allowed_missing
        unexpected = provided - own_keys

        if allowed_missing:
            logger.info(
                f"[ControlVLA-FFS audit] ckpt missing {len(allowed_missing)} "
                f"optional-branch keys (ffs / controlvla_branch / stereo_cam_rope)"
            )
        if suspicious_missing:
            sample = sorted(suspicious_missing)[:10]
            if strict:
                raise RuntimeError(
                    f"[ControlVLA-FFS audit] {len(suspicious_missing)} unexpected "
                    f"missing keys under strict=True; first: {sample}"
                )
            logger.warning(
                f"[ControlVLA-FFS audit] {len(suspicious_missing)} missing keys "
                f"(strict=False, accepting); first: {sample[:5]}"
            )
        if unexpected:
            sample = sorted(unexpected)[:10]
            msg = (
                f"[ControlVLA-FFS audit] {len(unexpected)} UNEXPECTED keys in "
                f"ckpt not present in current model; first: {sample}"
            )
            if strict:
                raise RuntimeError(msg + " — refusing silent drop.")
            logger.warning(msg + " — dropping these keys (strict=False).")
            state_dict = {k: v for k, v in state_dict.items() if k not in unexpected}

        forwarded_strict = strict and not allowed_missing
        return super().load_state_dict(state_dict, strict=forwarded_strict, assign=assign)
