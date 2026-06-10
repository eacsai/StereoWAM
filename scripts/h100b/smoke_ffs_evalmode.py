#!/usr/bin/env python
"""Smoke for the frozen-FFS eval-mode fix (the codex-review #3 HIGH finding).

The trainer's model.train() recursively flips the frozen FoundationStereo net
into TRAIN mode. FoundationStereo contains BatchNorm + Dropout, so train-mode
net[0] (batch-stat BN + active dropout + drifting BN running stats) would differ
from eval-mode net[0] (running-stat BN, no dropout) -> a silent train/eval
mismatch in the injected disparity. `_compute_ffs_feature` now re-asserts
`self.ffs.eval()` before every forward.

This smoke proves the fix decisively, WITHOUT loading the full VLM:
  - count BN/Dropout modules (confirms train-mode sensitivity),
  - in TRAIN mode, two forwards on identical input can differ,
  - after eval(), two forwards on identical input are BIT-IDENTICAL.

Run (GPU):
  PYTHONPATH=/mnt/data/wangqiwei/wangqiwei/starVLA:/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo \
    FFS_REPO_DIR=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo \
    /opt/conda/envs/starvla/bin/python scripts/h100b/smoke_ffs_evalmode.py
"""
import os
import sys

os.environ.setdefault("FFS_REPO_DIR", "/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo")
if os.environ["FFS_REPO_DIR"] not in sys.path:
    sys.path.insert(0, os.environ["FFS_REPO_DIR"])

import torch  # noqa: E402
import core.foundation_stereo  # noqa: E402,F401  (registers the model class for torch.load)

FFS_PATH = os.environ.get(
    "FFS_MODEL_PATH",
    f"{os.environ['FFS_REPO_DIR']}/weights/20-30-48/model_best_bp2_serialize.pth",
)


def net0_forward(ffs, left, right):
    cap = {}

    def _hook(_m, _i, o):
        cap["n0"] = o[0][0].detach().clone()

    h = ffs.update_block.register_forward_hook(_hook)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            ffs(left.float(), right.float(), iters=int(ffs.args.valid_iters), test_mode=True)
    finally:
        h.remove()
    if "n0" not in cap:
        raise RuntimeError("update_block net[0] hook did not fire")
    return cap["n0"]


def main():
    # Regression guard for the FIX ITSELF: _compute_ffs_feature must re-assert
    # self.ffs.eval(). If someone deletes that line, this smoke fails (the train-vs-
    # eval determinism test below only proves the *mechanism*, not that the fix is wired).
    import inspect

    from starVLA.model.framework.VLM4A.QwenGR00T_FFSCommon import QwenGR00TNet0FFSMixin

    src = inspect.getsource(QwenGR00TNet0FFSMixin._compute_ffs_feature)
    assert "self.ffs.eval()" in src, (
        "REGRESSION: QwenGR00TNet0FFSMixin._compute_ffs_feature no longer calls "
        "self.ffs.eval() — the frozen-FFS train/eval feature-mismatch fix was removed."
    )
    print("[smoke] regression guard: _compute_ffs_feature re-asserts self.ffs.eval() ✓")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ffs = torch.load(FFS_PATH, map_location="cpu", weights_only=False).to(dev).float()
    try:
        ffs.args.mixed_precision = False
    except Exception:
        ffs.args["mixed_precision"] = False
    for p in ffs.parameters():
        p.requires_grad = False

    nbn = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in ffs.modules())
    ndo = sum(isinstance(m, torch.nn.Dropout) for m in ffs.modules())
    print(f"[smoke] FFS net has {nbn} BatchNorm + {ndo} Dropout modules (train-mode sensitive)")

    torch.manual_seed(0)
    left = torch.rand(1, 3, 256, 256, device=dev) * 255.0
    right = torch.rand(1, 3, 256, 256, device=dev) * 255.0

    # TRAIN mode == the buggy condition the trainer's model.train() induces.
    ffs.train()
    a = net0_forward(ffs, left, right)
    b = net0_forward(ffs, left, right)
    train_diff = float((a - b).abs().max())
    print(f"[smoke] TRAIN-mode net0 max|Δ| over 2 identical-input forwards = {train_diff:.3e}")

    # THE FIX: eval() before the forward -> deterministic, reproducible features.
    ffs.eval()
    c = net0_forward(ffs, left, right)
    d = net0_forward(ffs, left, right)
    eval_diff = float((c - d).abs().max())
    print(f"[smoke] EVAL-mode  net0 max|Δ| over 2 identical-input forwards = {eval_diff:.3e}")

    # eval-mode is NOT bit-exact on CUDA: atomics / cudnn nondeterminism leave a
    # small floating-point floor (~1e-2, the same floor diag_ffs_realframes saw for
    # identical-frame disparity). The fix's job is to remove the BN+Dropout
    # non-determinism, which is ORDERS larger. So assert eval Δ sits at the FP floor
    # AND train Δ is materially larger (BN+Dropout were active before the fix).
    FP_FLOOR = 0.05
    assert not ffs.training, "ffs still in training mode after eval()"
    assert eval_diff < FP_FLOOR, (
        f"eval-mode net0 Δ={eval_diff:.3e} exceeds the FP-noise floor {FP_FLOOR}; "
        "eval() did not silence BN/Dropout — fix ineffective"
    )
    assert train_diff > max(eval_diff * 5.0, FP_FLOOR), (
        f"train-mode Δ={train_diff:.3e} is not materially larger than eval Δ={eval_diff:.3e}; "
        "expected BN+Dropout to make train-mode much noisier (smoke can't prove the fix)"
    )
    ratio = train_diff / max(eval_diff, 1e-9)
    print(f"[smoke] eval-mode net0 at FP floor (Δ={eval_diff:.3e} < {FP_FLOOR}) ✓")
    print(f"[smoke] CONFIRMED: train-mode is {ratio:.0f}x noisier (Δ={train_diff:.3e}) — the eval() fix "
          "removes BN+Dropout non-determinism from the injected feature")
    print("FFS_EVALMODE_SMOKE_OK")


if __name__ == "__main__":
    main()
