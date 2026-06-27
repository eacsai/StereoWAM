#!/usr/bin/env python3
"""Env-gate (FFS_DISABLE_UNROTATE) for the FFS un-rotate fix; default un-rotate ON.
Set FFS_DISABLE_UNROTATE=1 to use the OLD (pre-fix) wrong order, e.g. to faithfully
eval a ckpt trained before the fix. Also addresses codex concern #9. Atomic + .bak."""
import os, shutil, tempfile
F = "/data/wangqiwei/ICLR2026/starVLA/starVLA/model/framework/VLM4A/QwenGR00T_FFSCommon.py"
src = open(F).read()
if "_maybe_unrotate" in src:
    print("ALREADY_GATED -> no-op"); raise SystemExit(0)
assert "FFS-ORDER FIX" in src, "base un-rotate fix missing"
assert "import os" in src
anchor = '        self.primary_cam_id = int(ffs_cfg.get("primary_cam_id", 1))'
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + '\n'
    '        # FFS-ORDER env-gate (default un-rotate ON). FFS_DISABLE_UNROTATE=1 -> OLD wrong\n'
    '        # order (faithfully eval a pre-fix ckpt). See project_ffs_stereo_order_bug / codex #9.\n'
    '        self._ffs_unrotate = os.environ.get("FFS_DISABLE_UNROTATE", "").strip().lower() not in {"1", "true", "yes", "on"}')
feat_def = '    def _compute_ffs_feature(self, batch_images: List) -> torch.Tensor:'
assert src.count(feat_def) == 1
helper = ('    def _maybe_unrotate(self, t: torch.Tensor) -> torch.Tensor:\n'
    '        """Un-rotate (180-deg flip) when enabled (default); identity when\n'
    '        FFS_DISABLE_UNROTATE is set. Centralises the FFS-ORDER fix flips."""\n'
    '        return torch.flip(t, dims=(-2, -1)).contiguous() if self._ffs_unrotate else t\n\n')
src = src.replace(feat_def, helper + feat_def)
repls = [('torch.flip(primary, dims=(-2, -1)).contiguous()', 'self._maybe_unrotate(primary)', 2),
         ('torch.flip(right, dims=(-2, -1)).contiguous()', 'self._maybe_unrotate(right)', 2),
         ('torch.flip(self._ffs_captured_net0, dims=(-2, -1)).contiguous()', 'self._maybe_unrotate(self._ffs_captured_net0)', 1),
         ('torch.flip(disp_up, dims=(-2, -1)).contiguous()', 'self._maybe_unrotate(disp_up)', 1)]
for old, new, n in repls:
    assert src.count(old) == n, f"expected {n}x '{old}', found {src.count(old)}"
    src = src.replace(old, new)
shutil.copy(F, F + ".bak_envgate")
d = os.path.dirname(F); fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
with os.fdopen(fd, "w") as fh: fh.write(src)
os.replace(tmp, F)
print("ENVGATE_OK")
