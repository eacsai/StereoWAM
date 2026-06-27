#!/usr/bin/env python3
"""Apply the FFS stereo left/right ORDER fix (un-rotate + flip-back) to
QwenGR00T_FFSCommon.py on 4090d. Deterministic, idempotent-guarded, keeps a .bak.

Bug: training feeds ffs(primary, right_view) = image1=primary as LEFT, but the
stored LIBERO frames are rotated 180 deg so right_view is the GEOMETRIC LEFT eye.
FoundationStereo only searches non-negative disparity -> wrong order gives ~189px
garbage vs correct ~17px. net[0] is injected at primary's token grid so the feature
must stay in primary's frame. Fix = flip both inputs 180 (un-rotate so primary is
geometric-left), feed ffs, flip net[0]/disparity output back to the rotated primary
frame. codex prewrite review verdict: correct-with-modifications (this is the core).
"""
import shutil, sys

F = "/data/wangqiwei/ICLR2026/starVLA/starVLA/model/framework/VLM4A/QwenGR00T_FFSCommon.py"
src = open(F).read()

if "FFS-ORDER FIX" in src:
    print("ALREADY_PATCHED (found 'FFS-ORDER FIX' marker) -> no-op")
    sys.exit(0)

assert "import torch" in src, "torch import not found"

# --- 1) flip both inputs right after they are created (both _compute_ffs_feature
#        and _compute_ffs_disparity have this identical 2-line block) ---
in_block = (
    "        primary = self._imgs_to_ffs_tensor(batch_images, self.primary_idx)\n"
    "        right = self._imgs_to_ffs_tensor(batch_images, self.right_view_idx)"
)
in_block_new = in_block + "\n" + (
    "        # FFS-ORDER FIX (2026-06-25, memory project_ffs_stereo_order_bug): stored LIBERO\n"
    "        # stereo frames are rotated 180 deg (regenerate_libero_stereo.py [::-1,::-1] on all\n"
    "        # views) -> horizontal axis flipped -> on stored pixels right_view is the GEOMETRIC\n"
    "        # LEFT eye, primary the RIGHT. FoundationStereo treats image1 as left/reference and\n"
    "        # searches NON-NEGATIVE disparity only, so feeding (primary, right) gave inverted\n"
    "        # garbage (~189px vs correct ~17px; point cloud 0.11m vs 1.086m). Un-rotate both\n"
    "        # views so primary becomes the geometric left; the net[0]/disparity output is flipped\n"
    "        # back below to re-align with the rotated primary token grid the residual injects onto.\n"
    "        # NOTE: assumes the rot180 LeRobot stereo convention; a non-rotated render path\n"
    "        # (SSF/render_scripts/_stereo_render_utils.py) would NOT need this flip.\n"
    "        primary = torch.flip(primary, dims=(-2, -1)).contiguous()\n"
    "        right = torch.flip(right, dims=(-2, -1)).contiguous()"
)
n_in = src.count(in_block)
assert n_in == 2, f"expected 2 input blocks (feature+disparity), found {n_in}"
src = src.replace(in_block, in_block_new)

# --- 2) flip net[0] back to the rotated primary frame (feature path; appears once) ---
feat_old = "ffs_feat = self._ffs_captured_net0"
n_feat = src.count(feat_old)
assert n_feat == 1, f"expected 1 '{feat_old}', found {n_feat}"
src = src.replace(
    feat_old,
    "ffs_feat = torch.flip(self._ffs_captured_net0, dims=(-2, -1)).contiguous()  # FFS-ORDER FIX: flip net0 back to rotated primary frame",
)

# --- 3) flip disparity back before the shape check (disparity path; anchor appears once) ---
disp_anchor = "        if disp_up.shape != (B, 1, self.ffs_image_size, self.ffs_image_size):"
n_disp = src.count(disp_anchor)
assert n_disp == 1, f"expected 1 disparity shape-check anchor, found {n_disp}"
src = src.replace(
    disp_anchor,
    "        disp_up = torch.flip(disp_up, dims=(-2, -1)).contiguous()  # FFS-ORDER FIX: flip disparity back to rotated primary frame\n"
    + disp_anchor,
)

shutil.copy(F, F + ".bak_ffsorderfix")
open(F, "w").write(src)
print("PATCH_OK: flipped inputs (x2) + net0 back + disparity back; backup at", F + ".bak_ffsorderfix")
