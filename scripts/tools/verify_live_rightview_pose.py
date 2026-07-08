#!/usr/bin/env python3
"""R3 verification: live-injected rightview camera pose vs the renderer's STATIC
per-suite pose (probe.OURRENDER_RIGHTVIEW_POSES).

Decisive de-risk for orthogrid eval: the live ortho render at eval will use the
STATIC per-suite pose (probe.load_right_view_pose) to unproject the point cloud,
but the eval env injects the rightview camera DYNAMICALLY from XML. If the live
world pose != the static pose, the eval point cloud is geometrically wrong
(silent, no crash). This script prints both and their difference for all 4 suites
(2 tasks each = within-suite consistency check).

Run on 4090d in the LIBERO eval venv with MUJOCO_GL=egl.
READ-ONLY: builds envs, queries poses, prints. Modifies nothing.
"""
import os
import sys
import pathlib
import numpy as np

STARVLA = "/data/wangqiwei/ICLR2026/starVLA"
sys.path.insert(0, STARVLA)
sys.path.insert(0, os.path.join(STARVLA, "SSF/render_scripts"))
sys.path.insert(0, os.path.join(STARVLA, "scripts/tools"))

import probe_orthogonal_multiview_render as probe  # static poses + quat helper
from libero.libero import benchmark, get_libero_path
from _stereo_render_utils import make_stereo_env

RES = 256
BASELINE = 0.06
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]


def cam_world_pose(env, cam_name):
    """World camera position + rotation matrix (camera->world) after forward kinematics."""
    sim = env.sim
    model = sim.model
    data = sim.data
    try:
        cid = model.camera_name2id(cam_name)
    except Exception:
        import mujoco
        cid = mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.forward()
    xpos = np.array(data.cam_xpos[cid], dtype=np.float64)
    xmat = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    return xpos, xmat


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


bd = benchmark.get_benchmark_dict()
print("=" * 100)
print("R3 POSE CHECK: live-injected rightview (world) vs static OURRENDER_RIGHTVIEW_POSES")
print("=" * 100)
worst_dpos = 0.0
worst_dang = 0.0
for suite in SUITES:
    ts = bd[suite]()
    ntask = ts.n_tasks
    static_pos, static_rot = probe.load_right_view_pose(suite)  # (pos_world, R_cam->world)
    for task_id in sorted(set([0, min(1, ntask - 1)])):
        task = ts.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = make_stereo_env(bddl_file_name=str(bddl), baseline=BASELINE, resolution=RES)
        env.reset()
        rv_pos, rv_rot = cam_world_pose(env, "rightview")
        av_pos, av_rot = cam_world_pose(env, "agentview")
        dpos = float(np.linalg.norm(rv_pos - static_pos))
        dang = rot_angle_deg(rv_rot, static_rot)
        worst_dpos = max(worst_dpos, dpos)
        worst_dang = max(worst_dang, dang)
        print(f"\n[{suite} task{task_id}]  (n_tasks={ntask})")
        print(f"  LIVE   rightview world pos = {np.round(rv_pos,6)}")
        print(f"  STATIC rightview       pos = {np.round(static_pos,6)}")
        print(f"  LIVE   agentview  world pos = {np.round(av_pos,6)}  (rv should be av + 0.06 along av local +X)")
        print(f"  --> dpos = {dpos*1000:.3f} mm   d_rot = {dang:.4f} deg   {'OK' if (dpos < 2e-3 and dang < 1.0) else 'MISMATCH'}")
        env.close()
        del env

print("\n" + "=" * 100)
print(f"WORST: dpos={worst_dpos*1000:.3f} mm   d_rot={worst_dang:.4f} deg")
print("VERDICT:", "POSE_MATCH_OK (static pose valid for live eval)" if (worst_dpos < 2e-3 and worst_dang < 1.0)
      else "POSE_MISMATCH (renderer must read pose dynamically from env, NOT static constants)")
print("=" * 100)
