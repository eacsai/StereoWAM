#!/usr/bin/env python3
"""Smoke checks for GR00T FFS #4 depth-image.

The depth-image arm renders FFS disparity as a third QwenVL image. This test
checks the renderer, the gated cam_rope 3-image contract, and train/eval paths.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T_DepthImageFFS import (  # noqa: E402
    render_disp_tensor_as_turbo_pils,
)
from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402

from scripts.h100b.smoke_groot_ffs import (  # noqa: E402
    DEFAULT_BASE_VLM,
    DEFAULT_DATA_MIX,
    DEFAULT_DATA_ROOT,
    DEFAULT_FFS_MODEL,
    DEFAULT_FFS_REPO_DIR,
    DEFAULT_FFS_SHA256,
    DEFAULT_PRETRAINED_CKPT,
    batch_images,
    build_framework_model,
    clone_cfg,
    install_ffs_repo,
    instructions,
    load_ckpt_config,
    make_examples,
    move_model,
    run_check,
    update_cfg,
)


FRAMEWORK = "QwenGR00T_DepthImageFFS"


def ffs_depth_image_cfg(args: argparse.Namespace) -> dict:
    return {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.ffs_image_size,
        "num_cameras": 2,
        "primary_idx": 1,
        "right_view_idx": 0,
        "primary_cam_id": 1,
        "depth_prompt": "Below is the stereo disparity (depth) map of the left view:",
    }


def build_depthimage_cfg(
    args: argparse.Namespace,
    ckpt_cfg,
    *,
    freeze_modules: str,
    cam_rope_enabled: bool = False,
):
    # cam_rope default OFF: the #4 relaunch trains with CAM_ROPE=0 (inert-cam_rope
    # bypass), so the gate must smoke the configuration that actually gets launched.
    cfg = clone_cfg(ckpt_cfg)
    update_cfg(cfg, "framework.name", FRAMEWORK)
    update_cfg(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_enabled", bool(cam_rope_enabled))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_d_c", 16)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_num_cameras", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_baseline_m", 0.06)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_fovy_degrees", 45.0)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_width", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_height", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_spatial_merge", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_init_mode", "zero")
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_right_first", True)
    update_cfg(cfg, "framework.qwenvl.stereo_extra_image_cam_id", 1)
    update_cfg(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False)
    update_cfg(cfg, "framework.ffs_depth_image", ffs_depth_image_cfg(args))
    update_cfg(cfg, "datasets.vla_data.data_root_dir", args.data_root)
    update_cfg(cfg, "datasets.vla_data.data_mix", args.data_mix)
    update_cfg(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    update_cfg(cfg, "trainer.pretrained_checkpoint", "")
    update_cfg(cfg, "trainer.freeze_modules", freeze_modules)
    return cfg


def build_model(args: argparse.Namespace, ckpt_cfg, device: torch.device) -> nn.Module:
    cfg = build_depthimage_cfg(
        args, ckpt_cfg, freeze_modules="", cam_rope_enabled=bool(args.cam_rope_enabled)
    )
    model = build_framework_model(cfg)
    model = TrainerUtils.freeze_backbones(model, freeze_modules="")
    model = move_model(model, device)
    model.eval()
    return model


def image_run_cam_ids(input_ids: torch.Tensor, image_token_id: int, cam_ids: torch.Tensor) -> list[list[int]]:
    out: list[list[int]] = []
    image_mask = input_ids == image_token_id
    for b in range(input_ids.shape[0]):
        mask_b = image_mask[b]
        diff = torch.diff(mask_b.int(), prepend=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
        diff_end = torch.diff(mask_b.int(), append=torch.zeros(1, dtype=torch.int, device=mask_b.device))
        ends = ((diff_end == -1).nonzero(as_tuple=True)[0] + 1).tolist()
        sample_ids = []
        for start, end in zip(starts, ends):
            unique = torch.unique(cam_ids[b, start:end]).detach().cpu().tolist()
            if len(unique) != 1:
                raise AssertionError(f"sample {b} image run {start}:{end} has mixed cam ids {unique}")
            sample_ids.append(int(unique[0]))
        out.append(sample_ids)
    return out


def check_disp_render(model: nn.Module, examples: list[dict], args: argparse.Namespace) -> str:
    imgs = batch_images(examples)
    primary = model._imgs_to_ffs_tensor(imgs, model.primary_idx)
    right = model._imgs_to_ffs_tensor(imgs, model.right_view_idx)
    with torch.inference_mode():
        if next(model.ffs.parameters()).dtype != torch.float32:
            model.ffs.float()
        model.ffs.eval()
        with torch.amp.autocast("cuda", enabled=False):
            disp_up = model.ffs(
                primary.float(),
                right.float(),
                iters=int(model.ffs.args.valid_iters),
                test_mode=True,
            )
    expected_shape = (len(imgs), 1, args.ffs_image_size, args.ffs_image_size)
    if not torch.is_tensor(disp_up) or tuple(disp_up.shape) != expected_shape:
        raise AssertionError(f"disp_up shape {getattr(disp_up, 'shape', None)} != {expected_shape}")

    depth_pils = model._compute_ffs_disp_image(imgs)
    if len(depth_pils) != len(imgs):
        raise AssertionError("depth PIL count does not match batch size")
    bad_sizes = [pil.size for pil in depth_pils if tuple(pil.size) != tuple(model.depth_image_size)]
    if bad_sizes:
        raise AssertionError(f"depth PIL size mismatch: {bad_sizes}")

    flat = torch.full_like(disp_up, 7.0)
    flat_pils = render_disp_tensor_as_turbo_pils(flat, model.depth_image_size)
    flat_arr = np.asarray(flat_pils[0])
    if flat_arr.shape[:2] != tuple(reversed(model.depth_image_size)):
        raise AssertionError(f"flat render shape {flat_arr.shape} does not match {model.depth_image_size}")
    if len(np.unique(flat_arr.reshape(-1, 3), axis=0)) != 1:
        raise AssertionError("flat disparity map did not render to a single finite color")
    return f"disp_up={tuple(disp_up.shape)}, pil_size={depth_pils[0].size}, flat_eps_ok"


def check_cam_rope_message(model: nn.Module, examples: list[dict]) -> str:
    imgs = batch_images(examples)
    instr = instructions(examples)
    depth_pils = model._compute_ffs_disp_image(imgs)
    qwen_inputs = model._build_depthimage_qwenvl_inputs(imgs, instr, depth_pils)
    input_ids = qwen_inputs["input_ids"]
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    spatial_merge = int(model.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
    image_token_id = int(model.qwen_vl_interface.model.config.image_token_id)
    cam = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=image_token_id,
        image_grid_thw=image_grid_thw,
        num_cameras=2,
        spatial_merge_size=spatial_merge,
        extra_image_cam_id=1,
    )
    run_ids = image_run_cam_ids(input_ids, image_token_id, cam)
    for b, ids in enumerate(run_ids):
        if ids != [0, 1, 1]:
            raise AssertionError(f"sample {b}: 3-image cam ids {ids} != [0, 1, 1]")

    two_image_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=imgs, instructions=instr)
    two_input_ids = two_image_inputs["input_ids"]
    two_grid = two_image_inputs.get("image_grid_thw", None)
    implicit = compute_per_token_cam_id(
        input_ids=two_input_ids,
        image_token_id=image_token_id,
        image_grid_thw=two_grid,
        num_cameras=2,
        spatial_merge_size=spatial_merge,
    )
    explicit_none = compute_per_token_cam_id(
        input_ids=two_input_ids,
        image_token_id=image_token_id,
        image_grid_thw=two_grid,
        num_cameras=2,
        spatial_merge_size=spatial_merge,
        extra_image_cam_id=None,
    )
    if not torch.equal(implicit, explicit_none):
        raise AssertionError("extra_image_cam_id=None changed the 2-image cam_id path")
    two_run_ids = image_run_cam_ids(two_input_ids, image_token_id, explicit_none)
    for b, ids in enumerate(two_run_ids):
        if ids != [0, 1]:
            raise AssertionError(f"sample {b}: 2-image cam ids {ids} != [0, 1]")
    return f"three_image={run_ids}; two_image={two_run_ids}; flag_off_byte_identical"


def check_forward_predict(model: nn.Module, examples: list[dict]) -> str:
    calls = {"n": 0}
    original_compute = model._compute_ffs_disp_image

    def wrapped_compute(batch_images):
        calls["n"] += 1
        return original_compute(batch_images)

    model._compute_ffs_disp_image = wrapped_compute
    try:
        with torch.inference_mode():
            out = model(examples=examples)
            pred = model.predict_action(examples)
    finally:
        model._compute_ffs_disp_image = original_compute

    if calls["n"] < 2:
        raise AssertionError("forward + predict_action did not synthesize depth images on both paths")
    if not torch.isfinite(out["action_loss"].detach()).all():
        raise AssertionError("forward action_loss is not finite")
    actions = pred["normalized_actions"]
    if not np.isfinite(actions).all():
        raise AssertionError("predict_action produced non-finite actions")
    return f"depth_image_calls={calls['n']}, action_loss={float(out['action_loss'].detach().cpu()):.6g}, pred_shape={actions.shape}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--ffs-model-path", default=DEFAULT_FFS_MODEL)
    parser.add_argument("--ffs-repo-dir", default=DEFAULT_FFS_REPO_DIR)
    parser.add_argument("--ffs-expected-sha256", default=DEFAULT_FFS_SHA256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ffs-image-size", type=int, default=256)
    parser.add_argument("--cam-rope-enabled", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_ffs_repo(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_cfg, cfg_path = load_ckpt_config(args.pretrained_ckpt)
    print(f"[setup] config={cfg_path}")
    print(f"[setup] framework={FRAMEWORK} cam_rope_enabled={bool(args.cam_rope_enabled)}")

    ok = True
    try:
        model = build_model(args, ckpt_cfg, device)
        examples = make_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
        ok = run_check(FRAMEWORK, "disp_render", lambda: check_disp_render(model, examples, args)) and ok
        ok = run_check(FRAMEWORK, "cam_rope_message", lambda: check_cam_rope_message(model, examples)) and ok
        ok = run_check(FRAMEWORK, "forward_predict", lambda: check_forward_predict(model, examples)) and ok
    except Exception as exc:  # noqa: BLE001
        print(f"[{FRAMEWORK}] FAIL fatal: {exc}")
        traceback.print_exc()
        ok = False
    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
