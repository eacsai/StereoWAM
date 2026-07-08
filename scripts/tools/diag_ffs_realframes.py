"""Read-only diagnostic: does FFS net[0]/disparity behave correctly on REAL LIBERO
stereo frames?

Context: smoke_groot_ffs.py's `net0_left_frame_sanity` failed, but on SYNTHETIC
make_pattern input:
  - identical L/R -> raw disp mean_abs ~137px (should be ~0)
  - swap L/R     -> no sign flip (cos(disp_lr,-disp_rl) ~= -0.29, mean 245 vs 8)
Synthetic random patterns are OUT-OF-DISTRIBUTION for FoundationStereo (trained on real
imagery), so these may be OOD garbage rather than a framework bug. This script tests the
ROBUST, geometry-free invariant on REAL frames (the actual training distribution):
  - identical REAL frame as L/R  -> disparity MUST be ~0 (no parallax is physically possible)
  - real right_view+primary pair -> finite, sensible disparity
  - swapped pair                 -> sign flip
If real-identical -> ~0 while synthetic-identical -> 137, the smoke failure is purely an
OOD-synthetic-input artifact (benign). If real-identical is ALSO large, it's a real bug.

Read-only: loads model + runs FFS forward. No training, no writes, no checkpoints.
"""
import argparse
import os
import sys

import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_groot_ffs as S  # noqa: E402


def load_frame(mp4: str, idx: int = 0) -> Image.Image:
    vframes, _, _ = torchvision.io.read_video(mp4, pts_unit="sec", output_format="THWC")
    n = int(vframes.shape[0])
    idx = max(0, min(idx, n - 1))
    frame = vframes[idx].numpy()  # HWC uint8 RGB
    return Image.fromarray(frame).convert("RGB")


def stats(t: torch.Tensor) -> str:
    return (
        f"mean={float(t.mean()):+.4g} mean_abs={float(t.abs().mean()):.4g} "
        f"std={float(t.std()):.4g}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-ckpt", default=S.DEFAULT_PRETRAINED_CKPT)
    p.add_argument("--base-vlm", default=S.DEFAULT_BASE_VLM)
    p.add_argument("--ffs-model-path", default=S.DEFAULT_FFS_MODEL)
    p.add_argument("--ffs-repo-dir", default=os.environ.get("FFS_REPO_DIR", S.DEFAULT_FFS_REPO_DIR))
    p.add_argument("--ffs-expected-sha256", default=S.DEFAULT_FFS_SHA256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--framework", default="QwenGR00T_VLMInputFFS")
    p.add_argument("--right-mp4", required=True)
    p.add_argument("--primary-mp4", required=True)
    p.add_argument("--frame-idx", type=int, default=0)
    args = p.parse_args()
    # FrameworkContext / build_cfg_from_ckpt / make_examples expect these:
    args.batch_size = 1
    args.ffs_image_size = 256
    args.data_root = S.DEFAULT_DATA_ROOT
    args.data_mix = S.DEFAULT_DATA_MIX
    args.train_steps = 10
    args.train_lr = 1e-5
    args.loss_seed = 123
    args.parity_atol = 2e-3
    args.parity_rtol = 0.0
    args.liveness_min_diff = 1e-4
    args.same_disp_abs_mean_max = 1.0
    args.same_disp_std_max = 0.5
    args.swap_sign_cos_min = 0.15

    S.install_ffs_repo(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but cuda unavailable")
    device = torch.device(args.device)

    ckpt_cfg, cfg_path = S.load_ckpt_config(args.pretrained_ckpt)
    ckpt = S.load_ckpt(args.pretrained_ckpt)
    ctx = S.FrameworkContext(args=args, framework_name=args.framework, ckpt_cfg=ckpt_cfg, ckpt=ckpt, device=device)
    model = ctx.get_model()
    print(f"[diag] framework={args.framework} primary_idx={model.primary_idx} right_view_idx={model.right_view_idx}")

    right = load_frame(args.right_mp4, args.frame_idx)
    primary = load_frame(args.primary_mp4, args.frame_idx)
    print(f"[diag] real frames: right={right.size} primary={primary.size} (native res, no resize)")

    def run(tag: str, pair: list) -> torch.Tensor:
        # run_ffs_raw indexes pair via model.primary_idx / model.right_view_idx
        raw, net0 = S.run_ffs_raw(model, [pair])
        disp = S.extract_disparity_like(raw, batch=1)
        print(f"[{tag}] disp: {stats(disp)} | net0: {stats(net0)}")
        return disp

    # build a real stereo pair with correct slot order
    pair_real = [None, None]
    pair_real[model.right_view_idx] = right
    pair_real[model.primary_idx] = primary

    d_iden_p = run("identical_REAL(primary,primary)", [primary, primary])
    d_iden_r = run("identical_REAL(right,right)", [right, right])
    d_lr = run("real_pair[right,primary]", pair_real)
    d_rl = run("real_pair_SWAPPED", [pair_real[1], pair_real[0]])

    cos = S.cosine_with_negative(d_lr, d_rl)
    iden_p_ma = float(d_iden_p.abs().mean())
    iden_r_ma = float(d_iden_r.abs().mean())
    print("=" * 70)
    print(
        f"[VERDICT] identical_REAL_primary_mean_abs={iden_p_ma:.4g}, "
        f"identical_REAL_right_mean_abs={iden_r_ma:.4g}  (should be ~0 for correct stereo)"
    )
    print(
        f"[VERDICT] real_swap_neg_cos={cos:.4g}  (>0.15 = disparity sign-flips on L/R swap = good)"
    )
    benign = (iden_p_ma < 5.0 and iden_r_ma < 5.0)
    print(f"[VERDICT] identical-real collapses to ~0? {'YES (synthetic-137px was OOD garbage)' if benign else 'NO (REAL concern)'}")
    # Exit code must match the verdict: smoke_groot_ffs designates this script the
    # authoritative orientation check, so a gate wiring it in must not pass vacuously.
    return 0 if (benign and cos > 0.15) else 1


if __name__ == "__main__":
    raise SystemExit(main())
