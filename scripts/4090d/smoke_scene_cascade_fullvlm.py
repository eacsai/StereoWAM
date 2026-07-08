#!/usr/bin/env python3
"""Full-VLM step-0 smoke for the scene-flow dual-DiT cascade (real Qwen VLM, GPU).

The unit smoke (smoke_scene_cascade_step0.py) proves the coupler is byte-zero on tiny
dummy DiTs. THIS smoke builds the REAL QwenGR00T (Qwen3.5-0.8B VLM + GR00T action DiT +
scene-flow DiT + zero-init coupler) from the actual config path and asserts the claims
that only the full stack can verify — the pre-launch gate before any long run:

  (A) BUILD           : model constructs; scene_predictor present; action DiT coupler on
                        even/cross-attn blocks; cross_attention_dim aligned to VLM hidden.
  (B) FORWARD         : model(examples-with-flow-GT) -> finite action_loss AND finite
                        flow_loss (exercises config-mutation, dim-align, _collect_scene_targets
                        HWC->BCHW, fork_rng, .repeat order, dtype path).
  (C) STEP0_EQ_BASELINE : action_loss with the cascade ENABLED == action_loss with it
                        force-disabled, byte-for-byte (same seed). Proves the zero-init
                        coupler injects nothing at init AND fork_rng leaves the action
                        head's RNG stream untouched. Prints STEP0_EQ_BASELINE=PASS/FAIL
                        for the gpu-smoke-train --injection parser.
  (D) EXTRACT_NO_RNG  : extract_conditioning_hidden is deterministic (same twice) and
                        consumes NO global RNG (zeros-noise, dropout=0) at the real dim.
  (E) WARMSTART_LOAD  : a fresh cascade model load_state_dict(plain-ish state, strict=False,
                        init_from_baseline=True) fresh-inits EXACTLY the scene_predictor.* /
                        motion_coupler.* families and nothing else (stage-1 warm-start path).

READ-ONLY: builds from the base config, feeds synthetic tensors, writes no checkpoints.
Run in the starVLA .venv on a FREE GPU:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/4090d/smoke_scene_cascade_fullvlm.py \
    --base-vlm ./playground/Pretrained_models/Qwen3.5-0.8B
"""
import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse the battle-tested construction helpers from the shared tools namespace.
# The old cluster-specific scripts package has been retired; keep this smoke independent
# of removed training-machine launchers.
from scripts.tools.smoke_groot_ffs import (
    autocast_cuda,
    build_framework_model,
    leftprimary_pair,
    move_model,
)

from starVLA.model.framework.share_tools import apply_config_compat

FAIL = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def _u(cfg, path, value):
    OmegaConf.update(cfg, path, value, force_add=True)


def build_cfg(args):
    cfg = OmegaConf.load(args.config_yaml)
    _compat = apply_config_compat(cfg)  # mutates in place AND returns; capture like the launcher does
    if _compat is not None:
        cfg = _compat
    _u(cfg, "framework.name", "QwenGR00T")
    _u(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    # raw stereo into the VLM (leftprimary): cam_branch optional (matches winning ② / ⑤)
    _u(cfg, "framework.qwenvl.stereo_cam_rope_enabled", False)
    if args.cam_branch:
        _u(cfg, "framework.qwenvl.stereo_cam_branch_enabled", True)
        _u(cfg, "framework.qwenvl.stereo_cam_branch_heads", 4)
        _u(cfg, "framework.qwenvl.stereo_cam_branch_head_dim", 128)
    _u(cfg, "framework.action_model.diffusion_model_cfg.interleave_self_attention", True)
    # === the cascade under test ===
    _u(cfg, "framework.scene_predictor.enabled", True)
    _u(cfg, "framework.scene_predictor.target_key", args.target_key)
    _u(cfg, "framework.scene_predictor.tap_hidden_index", args.tap_hidden_index)
    _u(cfg, "framework.scene_predictor.detach_conditioning", True)
    _u(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    _u(cfg, "trainer.pretrained_checkpoint", "")
    _u(cfg, "trainer.freeze_modules", "")
    return cfg


def make_examples(model, args, *, with_flow_gt):
    """Synthetic leftprimary stereo pairs + (optionally) per-sample scene-flow GT keys
    exactly as the dataloader emits them: flow_gt (H,W,3) fp32, flow_valid/flow_dynamic
    (H,W) bool, flow_has_gt bool. GT resolution is arbitrary (head mask-pools to grid)."""
    fw = model.config.framework
    action_dim = int(fw.action_model.action_dim)
    state_dim = int(fw.action_model.get("state_dim", action_dim))
    horizon = int(model.action_horizon)
    hgt = args.gt_size
    examples = []
    for idx in range(args.batch_size):
        images = leftprimary_pair(2000 + idx, args.image_size, shift=6 + idx)
        action = np.zeros((horizon, action_dim), dtype=np.float32)
        action[:, 0] = np.linspace(-0.2, 0.2, horizon, dtype=np.float32)
        ex = {
            "image": images,
            "lang": "pick up the object",
            "action": action,
            "state": np.zeros((1, state_dim), dtype=np.float32),
        }
        if with_flow_gt:
            rng = np.random.RandomState(4242 + idx)
            ex[args.target_key] = (0.05 * rng.randn(hgt, hgt, 3)).astype(np.float32)
            valid = np.zeros((hgt, hgt), dtype=bool)
            valid[hgt // 4 : 3 * hgt // 4, hgt // 4 : 3 * hgt // 4] = True  # a central valid patch
            dynamic = np.zeros((hgt, hgt), dtype=bool)
            dynamic[hgt // 3 : 2 * hgt // 3, hgt // 3 : 2 * hgt // 3] = True  # smaller dynamic patch
            ex["flow_valid"] = valid
            ex["flow_dynamic"] = dynamic
            ex["flow_has_gt"] = True
        examples.append(ex)
    return examples


def _seed(s=1234):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-vlm", required=True)
    ap.add_argument("--config-yaml", default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--gt-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--tap-hidden-index", type=int, default=10)
    ap.add_argument("--target-key", default="flow_gt")
    ap.add_argument("--cam-branch", action="store_true")
    ap.add_argument("--skip-warmstart", action="store_true", help="skip (E): don't build a 2nd VLM")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA required for the full-VLM smoke (real bf16 path)", flush=True)
        sys.exit(2)
    device = torch.device("cuda")

    print("[build] constructing QwenGR00T + scene_predictor cascade (loads Qwen VLM)...", flush=True)
    cfg = build_cfg(args)
    model = build_framework_model(cfg)
    model = move_model(model, device)
    model.eval()

    # ---------------- (A) BUILD ----------------
    sp = getattr(model, "scene_predictor", None)
    check("(A) scene_predictor built", sp is not None)
    dit = model.action_model.model
    n_coupler = len(dit.motion_coupler) if getattr(dit, "motion_coupler", None) is not None else 0
    check("(A) action DiT motion_coupler present on even blocks", n_coupler > 0, f"{n_coupler} couplers")
    if sp is not None:
        vlm_hidden = int(model.qwen_vl_interface.model.config.hidden_size)
        sp_xdim = int(sp.model.config.cross_attention_dim)
        check("(A) scene DiT cross_attention_dim == VLM hidden", sp_xdim == vlm_hidden, f"{sp_xdim}=={vlm_hidden}")
        check("(A) scene DiT inner_dim == action coupler K/V dim", int(sp.model.inner_dim) == int(dit.inner_dim),
              f"{int(sp.model.inner_dim)}=={int(dit.inner_dim)}")

    # ---------------- (B) FORWARD ----------------
    examples = make_examples(model, args, with_flow_gt=True)
    _seed()
    with torch.no_grad():
        losses = model(examples)
    a_loss = losses.get("action_loss")
    f_loss = losses.get("flow_loss")
    check("(B) forward returns finite action_loss", a_loss is not None and torch.isfinite(a_loss).all(),
          f"action_loss={None if a_loss is None else float(a_loss)}")
    check("(B) forward returns finite flow_loss", f_loss is not None and torch.isfinite(f_loss).all(),
          f"flow_loss={None if f_loss is None else float(f_loss)}")

    # ---------------- (C) STEP0_EQ_BASELINE ----------------
    # action_loss must be identical whether the cascade is enabled (zero coupler + fork_rng)
    # or force-disabled. Reset seed identically before each forward.
    _seed()
    with torch.no_grad():
        loss_on = model(examples)["action_loss"].float().item()
    model.scene_predictor_enabled = False  # force-disable the injection+extraction path
    try:
        _seed()
        with torch.no_grad():
            loss_off = model(examples)["action_loss"].float().item()
    finally:
        model.scene_predictor_enabled = True
    maxabs = abs(loss_on - loss_off)
    step0_ok = maxabs == 0.0
    check("(C) action_loss cascade-on == cascade-off (zero coupler + fork_rng restores RNG)",
          step0_ok, f"|on-off|={maxabs:.3e} on={loss_on:.6f} off={loss_off:.6f}")
    print(f"STEP0_EQ_BASELINE={'PASS' if step0_ok else 'FAIL'} maxabs={maxabs:.3e}", flush=True)

    # ---------------- (D) EXTRACT_NO_RNG ----------------
    if sp is not None:
        _seed()
        with torch.no_grad():
            qi = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[e["image"] for e in examples], instructions=[e["lang"] for e in examples]
            )
            with autocast_cuda():
                vlm_out = model.qwen_vl_interface(**qi, output_hidden_states=True, return_dict=True)
            vl = vlm_out.hidden_states[-1]  # bf16 (from the VLM's bf16 autocast block)
            rng_before = torch.cuda.get_rng_state_all()
            # replicate the REAL calling context: forward/predict_action both wrap extraction
            # in autocast(fp32) (QwenGR00T.py action-expert block) — the head's fp32 params
            # only match bf16 vl_embs under that autocast. Calling it bare would (correctly) fail.
            with torch.autocast("cuda", dtype=torch.float32):
                h1 = sp.extract_conditioning_hidden(vl)
                h2 = sp.extract_conditioning_hidden(vl)
            rng_after = torch.cuda.get_rng_state_all()
        det = torch.equal(h1, h2)
        no_leak = all(torch.equal(a, b) for a, b in zip(rng_before, rng_after))
        check("(D) extract_conditioning_hidden deterministic (same twice)", det)
        check("(D) extraction consumes no global CUDA RNG (zeros-noise, dropout=0)", no_leak)
        check("(D) extracted hidden shape (B, grid*grid, inner_dim) + detached",
              tuple(h1.shape) == (args.batch_size, sp.num_tokens, int(sp.model.inner_dim)) and not h1.requires_grad,
              f"{tuple(h1.shape)}")

    # ---------------- (E) WARMSTART_LOAD ----------------
    if not args.skip_warmstart:
        full_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # a "plain-ish baseline" checkpoint has NO cascade families -> they must fresh-init
        plain_state = {
            k: v for k, v in full_state.items()
            if not (k.startswith("scene_predictor.") or k.startswith("action_model.motion_coupler.")
                    or ".motion_coupler." in k)
        }
        dropped = sorted(set(full_state) - set(plain_state))
        print(f"[warmstart] dropped {len(dropped)} cascade keys from the synthetic baseline", flush=True)
        del model
        torch.cuda.empty_cache()
        model2 = move_model(build_framework_model(build_cfg(args)), device)
        try:
            ret = model2.load_state_dict(plain_state, strict=False, init_from_baseline=True)
            missing = list(getattr(ret, "missing_keys", []))
            unexpected = list(getattr(ret, "unexpected_keys", []))
        except TypeError:
            # load_state_dict signature without init_from_baseline
            ret = model2.load_state_dict(plain_state, strict=False)
            missing = list(getattr(ret, "missing_keys", []))
            unexpected = list(getattr(ret, "unexpected_keys", []))

        def _is_cascade(k):
            return (k.startswith("scene_predictor.") or k.startswith("action_model.motion_coupler.")
                    or ".motion_coupler." in k)

        non_cascade_missing = [k for k in missing if not _is_cascade(k)]
        check("(E) warm-start: only cascade families are missing (fresh-init)", not non_cascade_missing,
              f"{len(missing)} missing, {len(non_cascade_missing)} non-cascade e.g. {non_cascade_missing[:4]}")
        check("(E) warm-start: no unexpected keys", not unexpected, f"{unexpected[:4]}")

    print("\n==== FULL-VLM SMOKE " + ("PASS ====" if not FAIL else f"FAIL: {FAIL} ===="), flush=True)
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
