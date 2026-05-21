"""Smoke test: can we inject rightview camera into LIBERO sim env at INFERENCE time?

Reuses make_stereo_env / render_step from SSF/render_scripts/_stereo_render_utils.py
(the *already-working* renderer reference). If this script saves valid left/right
PNGs with non-trivial pixel difference, it proves the eval-time stereo path is
identical to the train-time data generation path — i.e. inference can see the
same kind of right_view input the model was trained on.

Usage:
  export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
  export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export PYTHONPATH=$LIBERO_HOME:/data/wangqiwei/ICLR2026/starVLA:/data/wangqiwei/ICLR2026/starVLA/SSF/render_scripts
  CUDA_VISIBLE_DEVICES=6 /data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python \\
    scripts/4090d/smoke_stereo_camera_eval.py
"""
import os
import sys
import numpy as np
from pathlib import Path

# Make the stereo utils importable (same trick regenerate_libero_stereo.py uses).
THIS_DIR = Path(__file__).resolve().parent
RENDER_SCRIPTS_DIR = THIS_DIR.parent.parent / "SSF" / "render_scripts"
sys.path.insert(0, str(RENDER_SCRIPTS_DIR))

from _stereo_render_utils import make_stereo_env, render_step  # type: ignore
from PIL import Image
from libero.libero import benchmark, get_libero_path

OUT_DIR = THIS_DIR / "smoke_stereo_camera_eval_out"
OUT_DIR.mkdir(exist_ok=True)
RESOLUTION = 256
BASELINE = 0.06  # 6 cm — same as render-time

print(f"[smoke] PID={os.getpid()}  out_dir={OUT_DIR}")

# Pull task[0] from libero_goal — same API path as regenerate_libero_stereo.py
task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
bddl_root = get_libero_path("bddl_files")
task0 = task_suite.get_task(0)
bddl = os.path.join(bddl_root, task0.problem_folder, task0.bddl_file)
print(f"[smoke] task[0]={task0.language!r}\\n        bddl={bddl}")

# Build env with rightview injected — this is the ONLY line that differs from
# vanilla OffScreenRenderEnv. If make_stereo_env returns OK, the mujoco model
# already has a "rightview" camera AND it survives reset().
env = make_stereo_env(bddl_file_name=bddl, baseline=BASELINE, resolution=RESOLUTION)
print("[smoke] env built — make_stereo_env returned OK")

# First reset → verify rightview survives the hard reset (this is the whole
# point of set_xml_processor — without that hook, reset() would silently drop
# the camera).
env.reset()
print("[smoke] env.reset() OK")

# Confirm the compiled MjModel has the camera name.
mj_cam_names = list(env.sim.model.camera_names)
print(f"[smoke] mj_model camera_names = {mj_cam_names}")
assert "rightview" in mj_cam_names, "rightview NOT in compiled mj_model — injection failed"
print("[smoke] ✓ rightview camera present in compiled mj_model after reset()")

# Render the three views using the same path the renderer uses (env.sim.render
# direct, [::-1] vertical flip for canonical orientation — same as renderer).
frames = render_step(env, RESOLUTION)
print(f"[smoke] render_step OK — keys={list(frames.keys())}")
for k, arr in frames.items():
    print(f"    {k}: shape={arr.shape}  dtype={arr.dtype}  min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}")

# Save PNGs for visual inspection.
for k, arr in frames.items():
    Image.fromarray(arr).save(OUT_DIR / f"{k}.png")
    print(f"[smoke] saved {OUT_DIR}/{k}.png")

# Sanity: left vs right MUST differ (different camera position).
# If they are identical → either both renders are coming from the SAME camera
# (injection failed silently) or rightview is at the same pos as agentview.
diff = np.abs(frames["left"].astype(np.int16) - frames["right"].astype(np.int16))
print(f"[smoke] |left - right| mean = {diff.mean():.2f}  max = {diff.max()}  >0_frac = {(diff > 0).mean():.3f}")
assert diff.mean() > 1.0, f"left vs right too similar (mean_diff={diff.mean():.3f}) — right_view likely identical to agentview"
print("[smoke] ✓ left vs right pixel diff confirms rightview is at a different camera pose")

# Bonus: difference image to eyeball — overlap region should show horizontal
# parallax (objects shifted horizontally between L and R).
Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).save(OUT_DIR / "diff_x4.png")
print(f"[smoke] saved diff x4 visualization: {OUT_DIR}/diff_x4.png")

# Run reset() one more time and re-render → if rightview survives a SECOND reset,
# the set_xml_processor hook is doing its job. If it would have been dropped, the
# second render_step would crash.
env.reset()
frames2 = render_step(env, RESOLUTION)
print(f"[smoke] ✓ rightview survived 2nd reset() — render_step still works")

print("\\n[smoke] >>> STEREO CAMERA INJECTION AT INFERENCE TIME: WORKS <<<")
print(f"[smoke] Check {OUT_DIR}/{{left,right,wrist,diff_x4}}.png to eyeball.")
