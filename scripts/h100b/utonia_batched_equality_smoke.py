"""GPU equality + speed smoke for the Utonia batched-offset forward (Method #10 R9 gate).

CORRECTNESS (equality): the transform's GridSample mode='train' is RANDOM, so each independent
transform draws different points. To isolate the BATCHING logic we transform each sample ONCE and
feed the SAME pre-transformed points to the single-forward (_utonia_forward_transformed) AND the
batched-forward (_utonia_forward_transformed_batched); the two MUST agree (a wrong offset/inverse
split silently corrupts features). SPEED: timed on the production methods (which transform per call,
as in training). This is the hard gate before enabling `utonia_batched=true` for a real run.

Run on a GPU host, cwd = the starVLA tree (so ./playground resolves):
  cd <starVLA tree> && FFS_REPO_DIR=<...> PYTHONPATH=$(pwd) <python> scripts/h100b/utonia_batched_equality_smoke.py
Env: BATCH_SIZE(8) STRIDE(4) ATOL(1e-3) RTOL(1e-2) STEPS(3) IMAGE_SIZE(256).
"""
import copy
import os
import sys
import time

FFS_REPO_DIR = os.environ.get("FFS_REPO_DIR", "")
if FFS_REPO_DIR and FFS_REPO_DIR not in sys.path:
    sys.path.insert(0, FFS_REPO_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from starVLA.model.modules.stereo.utonia_pointcloud import _BenchModel, _synthetic_pair  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("UTONIA_EQUALITY_FAIL reason=no_cuda")
        return 2
    image_size = int(os.environ.get("IMAGE_SIZE", "256"))
    stride = int(os.environ.get("STRIDE", "4"))
    bs = int(os.environ.get("BATCH_SIZE", "8"))
    atol = float(os.environ.get("ATOL", "1e-3"))
    rtol = float(os.environ.get("RTOL", "1e-2"))
    steps = int(os.environ.get("STEPS", "3"))
    cfg = {
        "ffs_model_path": os.environ.get(
            "FFS_MODEL_PATH", f"{FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth"
        ),
        "ffs_expected_sha256": None,
        "ffs_feature_source": "gru_hidden",
        "gru_hidden_dim": 16,
        "ffs_image_size": image_size,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "utonia_ckpt_path": os.environ.get(
            "UTONIA_CKPT_PATH", "./playground/Pretrained_models/Utonia/utonia.pth"
        ),
        "utonia_expected_sha256": None,
        "utonia_scale": float(os.environ.get("UTONIA_SCALE", "4.0")),
        "utonia_enable_flash": os.environ.get("UTONIA_ENABLE_FLASH", "false").lower() == "true",
        "fovy_degrees": 45.0,
        "baseline_m": 0.06,
        "image_width": image_size,
        "image_height": image_size,
        "depth_min": 0.05,
        "depth_max": 3.0,
        "disp_eps": 1e-3,
        "backproject_stride": stride,
    }
    examples = [_synthetic_pair(2000 + i, image_size) for i in range(bs)]
    model = _BenchModel(cfg).cuda().eval()

    with torch.inference_mode():
        _disp, samples = model._build_pointcloud_samples(examples, cfg)
        # transform each non-empty sample ONCE; both paths share these identical points
        dicts = []
        for s in samples:
            if int(s.coord.shape[0]) == 0:
                continue
            pnp = {
                "coord": s.coord.detach().cpu().numpy().astype(np.float32),
                "color": s.color.detach().cpu().numpy().astype(np.float32),
                "normal": s.normal.detach().cpu().numpy().astype(np.float32),
            }
            dicts.append(model._utonia_transform(pnp))
        single_feats = [model._utonia_forward_transformed(copy.deepcopy(d)) for d in dicts]
        batched_feats = model._utonia_forward_transformed_batched([copy.deepcopy(d) for d in dicts])

    ok = True
    max_diff = 0.0
    for i, (a, b) in enumerate(zip(single_feats, batched_feats)):
        if tuple(a.shape) != tuple(b.shape):
            print(f"UTONIA_EQUALITY_FAIL sample={i} shape {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        if a.numel() == 0:
            continue
        d = (a.float() - b.float()).abs().max().item()
        max_diff = max(max_diff, d)
        if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
            print(f"UTONIA_EQUALITY sample={i} max_abs_diff={d:.3e} NOT_allclose")
            ok = False
    print(
        f"UTONIA_EQUALITY n_dicts={len(dicts)} bs={bs} stride={stride} "
        f"max_abs_diff={max_diff:.3e} atol={atol} rtol={rtol}"
    )

    def _timeit(fn, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(n, 1)):
            with torch.inference_mode():
                fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / max(n, 1)

    loop_sec = _timeit(lambda: [model._extract_utonia_features_one(s) for s in samples], steps)
    batched_sec = _timeit(lambda: model._extract_utonia_features_batched(samples), steps)
    speedup = (loop_sec / batched_sec) if batched_sec > 0 else 0.0
    print(
        f"UTONIA_EQUALITY_SPEED bs={bs} stride={stride} loop_sec={loop_sec:.4f} "
        f"batched_sec={batched_sec:.4f} speedup={speedup:.2f}x"
    )

    if ok:
        print("UTONIA_EQUALITY_OK")
        return 0
    print("UTONIA_EQUALITY_FAIL reason=mismatch")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
