#!/usr/bin/env python3
"""Smoke checks for QwenGR00T_UtoniaPromptTokenFFS.

The real training comparison is 30k full finetune against Version A. This
smoke checks the contract that can silently break before such a run:
  - primary image token count is 64,
  - 64 Utonia prompt rows are inserted immediately before the left image,
  - inserted rows stay in the action-head sequence,
  - sample_ids drive the cached Utonia grid HIT path,
  - a short forward/training loop has finite loss.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FRAMEWORK = "QwenGR00T_UtoniaPromptTokenFFS"
SUITE = "smoke_suite"
DEFAULT_PRETRAINED_CKPT = (
    "playground/Checkpoints/"
    "qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/"
    "checkpoints/steps_30000_pytorch_model.pt"
)
DEFAULT_BASE_VLM = "./playground/Pretrained_models/Qwen3.5-0.8B"
DEFAULT_FFS_REPO_DIR = "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo"
DEFAULT_FFS_MODEL = (
    "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo/"
    "weights/20-30-48/model_best_bp2_serialize.pth"
)
DEFAULT_FFS_SHA256 = "98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692"
DEFAULT_DATA_ROOT = "playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW"
DEFAULT_DATA_MIX = "libero_all_sfstereo_leftprimary"
DEFAULT_UTONIA_CKPT = "./playground/Pretrained_models/Utonia/utonia.pth"
DEFAULT_SYNTHETIC_CACHE = "playground/Datasets/utonia_cache_prompttoken_smoke"

np = None
torch = None
nn = None
compute_per_token_cam_id = None
clear_depth_state = None
get_depth_state = None
UTONIA_FEATURE_DIM = None
_all_steps_sha256 = None
_image_sha = None
_sha256_file = None
_utonia_geometry_meta = None
TrainerUtils = None
build_framework_model = None
clone_cfg = None
grad_sum = None
install_ffs_repo = None
load_ckpt_config = None
make_pattern = None
move_model = None
run_check = None
update_cfg = None


def _lazy_import_runtime() -> None:
    global np, torch, nn
    global compute_per_token_cam_id, clear_depth_state, get_depth_state
    global UTONIA_FEATURE_DIM, _all_steps_sha256, _image_sha, _sha256_file, _utonia_geometry_meta
    global TrainerUtils
    global build_framework_model, clone_cfg, grad_sum, install_ffs_repo, load_ckpt_config
    global make_pattern, move_model, run_check, update_cfg

    if torch is not None:
        return

    import numpy as _np
    import torch as _torch
    import torch.nn as _nn
    from starVLA.model.modules.stereo.cam_rope_hook import compute_per_token_cam_id as _compute_per_token_cam_id
    from starVLA.model.modules.stereo.depth_token_inject import (
        clear_state as _clear_depth_state,
        get_state as _get_depth_state,
    )
    from starVLA.model.modules.stereo.utonia_pointcloud import (
        UTONIA_FEATURE_DIM as _UTONIA_FEATURE_DIM,
        _all_steps_sha256 as _cache_all_steps_sha256,
        _image_sha as _cache_image_sha,
        _sha256_file as _cache_sha256_file,
        _utonia_geometry_meta as _cache_utonia_geometry_meta,
    )
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils as _TrainerUtils
    from scripts.tools.smoke_groot_ffs import (
        build_framework_model as _build_framework_model,
        clone_cfg as _clone_cfg,
        grad_sum as _grad_sum,
        install_ffs_repo as _install_ffs_repo,
        load_ckpt_config as _load_ckpt_config,
        make_pattern as _make_pattern,
        move_model as _move_model,
        run_check as _run_check,
        update_cfg as _update_cfg,
    )

    np = _np
    torch = _torch
    nn = _nn
    compute_per_token_cam_id = _compute_per_token_cam_id
    clear_depth_state = _clear_depth_state
    get_depth_state = _get_depth_state
    UTONIA_FEATURE_DIM = _UTONIA_FEATURE_DIM
    _all_steps_sha256 = _cache_all_steps_sha256
    _image_sha = _cache_image_sha
    _sha256_file = _cache_sha256_file
    _utonia_geometry_meta = _cache_utonia_geometry_meta
    TrainerUtils = _TrainerUtils
    build_framework_model = _build_framework_model
    clone_cfg = _clone_cfg
    grad_sum = _grad_sum
    install_ffs_repo = _install_ffs_repo
    load_ckpt_config = _load_ckpt_config
    make_pattern = _make_pattern
    move_model = _move_model
    run_check = _run_check
    update_cfg = _update_cfg


def leftprimary_pair(seed: int, size: int, shift: int = 6):
    # leftprimary order: view[0]=primary (base), view[1]=left_view (shifted partner).
    from PIL import Image

    primary = make_pattern(seed, size)
    left_view = np.roll(primary, shift=shift, axis=1)
    return [
        Image.fromarray(primary, mode="RGB"),
        Image.fromarray(left_view, mode="RGB"),
    ]


def sample_ids(examples: list[dict]) -> list[tuple[str, int, int]]:
    return [(ex["dataset_name"], ex["traj_id"], ex["base_index"]) for ex in examples]


def batch_images(examples: list[dict]) -> list:
    return [example["image"] for example in examples]


def instructions(examples: list[dict]) -> list[str]:
    return [example["lang"] for example in examples]


def make_prompt_examples(model: nn.Module, *, batch_size: int, image_size: int) -> list[dict]:
    action_dim = int(model.config.framework.action_model.action_dim)
    state_dim = int(model.config.framework.action_model.get("state_dim", action_dim))
    horizon = int(model.action_horizon)
    examples = []
    for idx in range(batch_size):
        action = np.zeros((horizon, action_dim), dtype=np.float32)
        state = np.zeros((1, state_dim), dtype=np.float32)
        action[:, 0] = np.linspace(-0.25, 0.25, horizon, dtype=np.float32)
        if action_dim > 1:
            action[:, 1] = 0.03 * (idx + 1)
        examples.append(
            {
                "image": leftprimary_pair(3000 + idx, image_size, shift=6 + idx),
                "lang": "pick up the object",
                "action": action,
                "state": state,
                "dataset_name": SUITE,
                "traj_id": 0,
                "base_index": idx,
            }
        )
    return examples


def utonia_prompt_cfg(args: argparse.Namespace, cache_dir: Path) -> dict:
    return {
        "ffs_model_path": args.ffs_model_path,
        "ffs_expected_sha256": args.ffs_expected_sha256,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": args.ffs_image_size,
        "utonia_ckpt_path": args.utonia_ckpt_path,
        "utonia_scale": args.utonia_scale,
        "utonia_enable_flash": False,
        "fovy_degrees": 45.0,
        "baseline_m": 0.06,
        "image_width": args.ffs_image_size,
        "image_height": args.ffs_image_size,
        "backproject_stride": args.backproject_stride,
        "depth_min": 0.05,
        "depth_max": 3.0,
        "disp_eps": 0.001,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "utonia_cache_dir": str(cache_dir),
        "num_point_tokens": 64,
        "point_prompt": "Left image point-cloud features:",
        "gate_init": "zero",
        "inject_hidden_dim": 256,
    }


def write_synthetic_utonia_cache(cache_root: Path, examples: list[dict], pc_cfg: dict) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    suite_dir = cache_root / SUITE
    suite_dir.mkdir(parents=True, exist_ok=True)

    batch = len(examples)
    rng = np.random.default_rng(20260616)
    grid = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(batch, UTONIA_FEATURE_DIM + 1, 8, 8),
    ).astype(np.float16)
    grid[:, -1, :, :] = np.float16(1.0)
    np.save(suite_dir / "grid.f16.npy", grid)
    np.save(suite_dir / "done.npy", np.ones((batch,), dtype=np.bool_))

    index = {(int(ex["traj_id"]), int(ex["base_index"])): row for row, ex in enumerate(examples)}
    with open(suite_dir / "index.pkl", "wb") as fh:
        pickle.dump(index, fh)

    steps = [(int(ex["traj_id"]), int(ex["base_index"])) for ex in examples]
    geom = _utonia_geometry_meta(pc_cfg, (8, 8))
    meta = {
        "suite": SUITE,
        "count": batch,
        "dtype": "float16",
        "dims": [UTONIA_FEATURE_DIM + 1, 8, 8],
        "ffs_model_sha256": _sha256_file(pc_cfg["ffs_model_path"]),
        "utonia_ckpt_sha256": _sha256_file(pc_cfg["utonia_ckpt_path"]),
        "all_steps_sha256": _all_steps_sha256(steps),
        # leftprimary: validate_utonia_cache reads these 4 keys at meta TOP-LEVEL
        # (mirror precompute_utonia_cache.py), not only inside "geometry".
        "view_order": geom["view_order"],
        "reference_view": geom["reference_view"],
        "net0_frame": geom["net0_frame"],
        "unrotate": geom["unrotate"],
        "geometry": geom,
        "image_sha256_samples": [
            {
                "row": row,
                "traj_id": int(ex["traj_id"]),
                "base_index": int(ex["base_index"]),
                "image_sha256": [_image_sha(img) for img in ex["image"]],
            }
            for row, ex in enumerate(examples)
        ],
    }
    with open(suite_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return cache_root


def build_prompt_cfg(args: argparse.Namespace, ckpt_cfg, cache_dir: Path):
    cfg = clone_cfg(ckpt_cfg)
    update_cfg(cfg, "framework.name", FRAMEWORK)
    update_cfg(cfg, "framework.qwenvl.base_vlm", args.base_vlm)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_enabled", bool(args.cam_rope))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_d_c", 16)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_num_cameras", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_baseline_m", 0.06)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_fovy_degrees", 45.0)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_width", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_image_height", args.ffs_image_size)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_spatial_merge", 2)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_init_mode", "zero")
    update_cfg(cfg, "framework.qwenvl.stereo_cam_rope_right_first", True)
    update_cfg(cfg, "framework.qwenvl.stereo_epipolar_mask_enabled", False)
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_enabled", bool(args.cam_branch))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_heads", int(args.cam_branch_heads))
    update_cfg(cfg, "framework.qwenvl.stereo_cam_branch_head_dim", int(args.cam_branch_head_dim))
    update_cfg(cfg, "framework.action_model.diffusion_model_cfg.interleave_self_attention", True)
    update_cfg(cfg, "framework.utonia_pointcloud", utonia_prompt_cfg(args, cache_dir))
    update_cfg(cfg, "datasets.vla_data.data_root_dir", args.data_root)
    update_cfg(cfg, "datasets.vla_data.data_mix", args.data_mix)
    update_cfg(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size)
    update_cfg(cfg, "trainer.pretrained_checkpoint", "")
    update_cfg(cfg, "trainer.freeze_modules", "")
    update_cfg(cfg, "trainer.train_only", "")
    return cfg


def build_model(args: argparse.Namespace, ckpt_cfg, cache_dir: Path, device: torch.device) -> nn.Module:
    cfg = build_prompt_cfg(args, ckpt_cfg, cache_dir)
    model = build_framework_model(cfg)
    model = TrainerUtils.freeze_backbones(model, freeze_modules="")
    return move_model(model, device)


def find_subsequence(haystack: torch.Tensor, needle: list[int], end: int) -> int:
    values = haystack[:end].detach().cpu().tolist()
    n = len(needle)
    for idx in range(0, max(len(values) - n + 1, 0)):
        if values[idx : idx + n] == needle:
            return idx
    return -1


def position_columns(position_ids: torch.Tensor, batch_idx: int, cols: torch.Tensor) -> torch.Tensor:
    pos_b = position_ids[..., batch_idx, :]
    return pos_b.index_select(-1, cols.to(position_ids.device))


def check_singleton_projector_shape(model: nn.Module, examples: list[dict]) -> str:
    clear_depth_state()
    one = examples[:1]
    model._prepare_ffs_for_vlm(batch_images(one), sample_ids=sample_ids(one))
    state = get_depth_state()
    if state.depth_tokens is None:
        raise AssertionError("state.depth_tokens missing after singleton prepare")
    expected = (1, 64, int(model.config.framework.qwenvl.vl_hidden_dim))
    if tuple(state.depth_tokens.shape) != expected:
        raise AssertionError(f"singleton Utonia tokens shape {tuple(state.depth_tokens.shape)} != {expected}")
    clear_depth_state()
    return f"singleton_tokens={expected}"


def check_sequence_contract(model: nn.Module, examples: list[dict], args: argparse.Namespace) -> str:
    model.eval()
    clear_depth_state()
    qwen_inputs = model._build_utonia_prompttoken_qwenvl_inputs(batch_images(examples), instructions(examples))
    input_ids = qwen_inputs["input_ids"]
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    spatial_merge = int(model.config.framework.qwenvl.get("stereo_cam_rope_spatial_merge", 2))
    image_token_id = int(model.qwen_vl_interface.model.config.image_token_id)
    original_cam = compute_per_token_cam_id(
        input_ids=input_ids,
        image_token_id=image_token_id,
        image_grid_thw=image_grid_thw,
        num_cameras=2,
        spatial_merge_size=spatial_merge,
    )
    prompt_ids = model.qwen_vl_interface.processor.tokenizer(
        model.point_token_prompt,
        add_special_tokens=False,
    )["input_ids"]

    with torch.inference_mode():
        hidden = model._encode_last_hidden_with_ffs(
            batch_images(examples),
            instructions(examples),
            sample_ids=sample_ids(examples),
        )

    state = get_depth_state()
    if state.keep_mask is None or state.per_token_cam_id is None or state.insert_idx is None:
        raise AssertionError("prompt-token hook did not populate keep_mask/per_token_cam_id/insert_idx")
    if state.depth_tokens is None:
        raise AssertionError("state.depth_tokens is empty")
    if tuple(state.depth_tokens.shape[:2]) != (len(examples), 64):
        raise AssertionError(f"state.depth_tokens shape {tuple(state.depth_tokens.shape)} does not start with (B,64)")
    if hidden.shape[1] != input_ids.shape[1] + 64:
        raise AssertionError(f"hidden seq len {hidden.shape[1]} != original {input_ids.shape[1]} + 64")

    details = []
    for b in range(input_ids.shape[0]):
        left_pos = (original_cam[b] == 1).nonzero(as_tuple=True)[0]
        right_pos = (original_cam[b] == 0).nonzero(as_tuple=True)[0]
        if int(left_pos.numel()) != 64:
            raise AssertionError(f"sample {b}: primary token count expected 64, got {int(left_pos.numel())}")
        if int(right_pos.numel()) == 0:
            raise AssertionError(f"sample {b}: missing right image tokens")
        left_start = int(left_pos.min().item())
        right_end = int(right_pos.max().item()) + 1
        insert_idx = int(state.insert_idx[b].item())
        if insert_idx != left_start:
            raise AssertionError(f"sample {b}: insert_idx={insert_idx}, expected left_start={left_start}")
        prompt_at = find_subsequence(input_ids[b], prompt_ids, left_start)
        if prompt_at < right_end or prompt_at >= left_start:
            raise AssertionError(
                f"sample {b}: point prompt tokens not between right and left image "
                f"(right_end={right_end}, left_start={left_start}, found={prompt_at})"
            )
        inserted_slice = slice(insert_idx, insert_idx + 64)
        if not torch.equal(
            state.per_token_cam_id[b, inserted_slice],
            torch.full_like(state.per_token_cam_id[b, inserted_slice], -1),
        ):
            raise AssertionError(f"sample {b}: inserted Utonia token cam_id segment is not all -1")
        if bool(state.keep_mask[b, inserted_slice].any()):
            raise AssertionError(f"sample {b}: keep_mask did not mark Utonia prompt segment as inserted")
        if int((~state.keep_mask[b]).sum().item()) != 64:
            raise AssertionError(f"sample {b}: keep_mask false count is not 64")
        details.append(f"b{b}:orig={input_ids.shape[1]},expanded={hidden.shape[1]},insert={insert_idx},prompt={prompt_at}")

    if state.position_ids_before is None or state.position_ids_after is None:
        raise AssertionError("prompt-token hook did not observe/expand position_ids")
    before = state.position_ids_before
    after = state.position_ids_after
    if int(after.shape[-1]) != int(before.shape[-1]) + 64:
        raise AssertionError("position_ids did not expand by 64")
    for b in range(input_ids.shape[0]):
        insert_idx = int(state.insert_idx[b].item())
        left_pos = (original_cam[b] == 1).nonzero(as_tuple=True)[0]
        expected = position_columns(before, b, left_pos.to(before.device))
        got = after[..., b, insert_idx : insert_idx + 64]
        if not torch.equal(got.cpu(), expected.cpu()):
            raise AssertionError(f"sample {b}: inserted position_ids do not mirror primary grid")

    depth_max = float(state.depth_tokens.detach().abs().max().cpu().item())
    inserted_max = (
        float(state.inserted_depth_embeds.detach().abs().max().cpu().item())
        if state.inserted_depth_embeds is not None
        else depth_max
    )
    if depth_max > args.zero_atol or inserted_max > args.zero_atol:
        raise AssertionError(f"zero-init Utonia tokens are not near zero: {depth_max=} {inserted_max=}")
    return "; ".join(details) + f"; token_zero_max={depth_max:.6g}; inserted_zero_max={inserted_max:.6g}"


def trainable_params(module: nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def check_cam_branch_post_insert(model: nn.Module, args: argparse.Namespace) -> str:
    if not args.cam_branch:
        return "cam_branch=off"
    cam_state = getattr(model, "_stereo_cam_branch_state", None)
    if cam_state is None:
        raise AssertionError("cam_branch state missing")
    depth_state = model.get_utonia_prompt_insert_state()
    if getattr(depth_state, "per_token_cam_id", None) is None:
        raise AssertionError("depth token state has no post-insert per_token_cam_id")
    if cam_state.image_positions is None:
        raise AssertionError("cam_branch image_positions missing after post-insert refresh")
    cam = depth_state.per_token_cam_id
    image_mask = cam >= 0
    flat = image_mask.nonzero(as_tuple=False)
    bsz = int(cam.shape[0])
    if flat.shape[0] % bsz != 0:
        raise AssertionError(f"post-insert image token count {flat.shape[0]} not divisible by batch {bsz}")
    expected_positions = flat[:, 1].view(bsz, flat.shape[0] // bsz).to(cam_state.image_positions.device)
    if tuple(cam_state.image_positions.shape) != tuple(expected_positions.shape):
        raise AssertionError(
            f"cam_branch image_positions shape {tuple(cam_state.image_positions.shape)} "
            f"!= expected {tuple(expected_positions.shape)}"
        )
    if not torch.equal(cam_state.image_positions, expected_positions):
        raise AssertionError("cam_branch image_positions do not exactly match post-insert image-token positions")
    insert_idx = getattr(depth_state, "insert_idx", None)
    if insert_idx is not None:
        for b in range(bsz):
            lo = int(insert_idx[b].item())
            hi = lo + int(model.num_point_tokens)
            inside = ((cam_state.image_positions[b] >= lo) & (cam_state.image_positions[b] < hi)).any()
            if bool(inside):
                raise AssertionError(f"sample {b}: cam_branch position falls inside inserted token span [{lo},{hi})")
    return f"post_insert_positions={tuple(cam_state.image_positions.shape)}"


def run_train_steps(model: nn.Module, examples: list[dict], args: argparse.Namespace) -> str:
    model.train()
    optimizer = torch.optim.AdamW(trainable_params(model), lr=args.train_lr)
    last_loss = None
    projector_grad = head_grad = 0.0
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(args.loss_seed + step)
        out = model(examples=examples)
        loss = out["action_loss"]
        if not torch.isfinite(loss.detach()).all():
            raise AssertionError(f"step {step}: non-finite loss {loss.detach().float().cpu().item()}")
        loss.backward()
        projector_grad = grad_sum(model.utonia_prompt_projector.parameters())
        head_grad = grad_sum(model.action_model.parameters())
        optimizer.step()
        last_loss = float(loss.detach().float().cpu().item())
    if last_loss is None:
        raise AssertionError("train_steps was zero")
    return (
        f"steps={args.train_steps}, last_loss={last_loss:.6g}, "
        f"projector_grad={projector_grad:.6g}, head_grad={head_grad:.6g}"
    )


def check_cache_hit_path(model: nn.Module, min_hits: int) -> str:
    hits = int(getattr(model, "_utonia_cache_hits", 0))
    misses = int(getattr(model, "_utonia_cache_row_misses", 0))
    live = int(getattr(model, "_utonia_cache_whole_batch_live", 0))
    if hits < min_hits:
        raise AssertionError(f"expected at least {min_hits} Utonia cache hits, got {hits}")
    if misses != 0 or live != 0:
        raise AssertionError(f"expected no Utonia cache fallback; hits={hits} row_misses={misses} live={live}")
    return f"cache_hits={hits}, row_misses={misses}, whole_batch_live={live}"


def run_fromscratch(args: argparse.Namespace, ckpt_cfg, device: torch.device) -> bool:
    cache_dir = Path(args.synthetic_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    seed_cfg = utonia_prompt_cfg(args, cache_dir)

    # Build once to obtain action/state dimensions for examples; then write the
    # synthetic cache and rebuild with the real cache files in place.
    model = build_model(args, ckpt_cfg, cache_dir, device)
    examples = make_prompt_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
    write_synthetic_utonia_cache(cache_dir, examples, seed_cfg)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(args, ckpt_cfg, cache_dir, device)
    examples = make_prompt_examples(model, batch_size=args.batch_size, image_size=args.ffs_image_size)
    ok = True
    ok = run_check(FRAMEWORK, "singleton_projector_shape_b1", lambda: check_singleton_projector_shape(model, examples)) and ok
    ok = run_check(FRAMEWORK, "sequence_prompttoken_position_cache", lambda: check_sequence_contract(model, examples, args)) and ok
    ok = run_check(FRAMEWORK, "cam_branch_post_insert_positions", lambda: check_cam_branch_post_insert(model, args)) and ok
    ok = run_check(FRAMEWORK, "fromscratch_finite_steps", lambda: run_train_steps(model, examples, args)) and ok
    min_hits = 1 + args.batch_size * (args.train_steps + 1)
    ok = run_check(FRAMEWORK, "cache_hit_sample_ids_path", lambda: check_cache_hit_path(model, min_hits=min_hits)) and ok
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fromscratch",), default="fromscratch")
    parser.add_argument("--config-ckpt", default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--base-vlm", default=DEFAULT_BASE_VLM)
    parser.add_argument("--ffs-model-path", default=DEFAULT_FFS_MODEL)
    parser.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", DEFAULT_FFS_REPO_DIR))
    parser.add_argument("--ffs-expected-sha256", default=DEFAULT_FFS_SHA256)
    parser.add_argument("--utonia-ckpt-path", default=DEFAULT_UTONIA_CKPT)
    parser.add_argument("--synthetic-cache-dir", default=DEFAULT_SYNTHETIC_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-mix", default=DEFAULT_DATA_MIX)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ffs-image-size", type=int, default=256)
    parser.add_argument("--backproject-stride", type=int, default=4)
    parser.add_argument("--utonia-scale", type=float, default=4.0)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--train-lr", type=float, default=1e-5)
    parser.add_argument("--loss-seed", type=int, default=123)
    parser.add_argument("--zero-atol", type=float, default=1e-7)
    parser.add_argument("--cam-rope", action="store_true")
    parser.add_argument("--cam-branch", action="store_true")
    parser.add_argument("--cam-branch-heads", type=int, default=4)
    parser.add_argument("--cam-branch-head-dim", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _lazy_import_runtime()
    install_ffs_repo(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_cfg, cfg_path = load_ckpt_config(args.config_ckpt)
    print(f"[setup] config={cfg_path}")
    print(f"[setup] base_vlm={args.base_vlm}")
    print(f"[setup] ffs_model={args.ffs_model_path}")
    print(f"[setup] utonia_ckpt={args.utonia_ckpt_path}")
    print(f"[setup] synthetic_cache_dir={args.synthetic_cache_dir}")
    print("[setup] framework=QwenGR00T_UtoniaPromptTokenFFS image_order=leftprimary (VLM [primary,left_view], FFS ffs(left_view,primary)) inject_cam_id=1")

    ok = True
    try:
        ok = run_fromscratch(args, ckpt_cfg, device) and ok
    except Exception as exc:
        print(f"[{FRAMEWORK}] FAIL fatal: {exc}")
        traceback.print_exc()
        ok = False
    print("SMOKE_ALL_PASS" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
