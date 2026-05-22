"""
_stereo_render_utils.py
=======================

Pure utility module: stereo camera injection + rendering helpers shared by
``render_stereo_libero_hdf5.py`` (HDF5 state-replay renderer) and
``eval_libero_stereo.py`` (stereo eval driver).

Extracted from the previous action-replay renderer ``render_stereo_libero.py``
(removed 2026-05-19). The action-replay flow was abandoned because the HDF5
state-replay renderer is bit-exact and avoids the ``episode_idx ==
init_state_idx`` assumption baked into action-replay.

Exports
-------
- ``compute_right_camera_pose(pos, quat_wxyz, baseline)`` — geometric helper
- ``make_stereo_env(bddl_file, baseline, resolution)`` — OffScreenRenderEnv
  whose compiled MuJoCo model has a ``rightview`` camera that survives
  ``env.reset()``
- ``_inject_camera_into_xml_string(xml_str, sibling_name, new_name, baseline)``
  — runtime XML camera injection (used internally by ``make_stereo_env``)
- ``render_step(env, resolution) -> dict`` — pull left / right / wrist frames
  in one call via ``env.sim.render`` (consistent orientation)
- ``write_video(frames, out_path, fps=20)`` — h264 mp4 writer matching the
  IPEC LeRobot release codec/pix_fmt
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Match the LeRobot LIBERO release: 256x256 render then center-crop / resize as needed.
# The publicly released `libero_spatial_no_noops_1.0.0_lerobot` videos are 256x256.
# TODO: confirm exact resolution from `meta/info.json` (`features.observation.images.image.shape`).
RENDER_RESOLUTION = 256

LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)

# LeRobot parquet column names (verified from existing dataset layout in the task brief).
ACTION_COL = "action"
TASK_INDEX_COL = "task_index"
EPISODE_INDEX_COL = "episode_index"


# Mapping from our --suite shorthand to the LIBERO benchmark suite name.
SUITE_NAME_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
}


# -----------------------------------------------------------------------------
# Stereo camera injection
# -----------------------------------------------------------------------------

def _quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion is (w, x, y, z)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def compute_right_camera_pose(
    agentview_pos: np.ndarray,
    agentview_quat_wxyz: np.ndarray,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate `agentview` along its LOCAL +X axis (image-right) by `baseline` meters.

    In MuJoCo a camera looks down its local -Z, with +X = image-right, +Y = image-up.
    Therefore the local +X column of the camera's rotation matrix in world frame IS
    the world-space "right" direction we want.
    """
    R = _quat_to_rotmat(agentview_quat_wxyz)
    right_axis_world = R[:, 0]  # local +X expressed in world
    new_pos = agentview_pos + baseline * right_axis_world
    return new_pos, agentview_quat_wxyz.copy()  # same orientation -> parallel stereo rig



def make_stereo_env(bddl_file_name: str, baseline: float, resolution: int):
    """Build an OffScreenRenderEnv whose compiled MuJoCo model contains a
    `rightview` camera (agentview + horizontal baseline) that SURVIVES `reset()`.

    Why a `set_xml_processor` hook (not just `reset_from_xml_string`)
    ----------------------------------------------------------------
    Robosuite's `BaseEnv.reset()` (robosuite/environments/base.py:238) does a
    HARD RESET by default: it calls `_destroy_sim()` -> `_load_model()` ->
    `_initialize_sim()`, which rebuilds MjSim from the original model (the
    untouched BDDL arena). So if we only call `env.reset_from_xml_string(...)`
    in this constructor, the NEXT `env.reset()` invocation in our caller
    silently throws away our `rightview` camera -- which is exactly what the
    smoke test failure shows.

    Robosuite ships an extension point for this: `set_xml_processor(fn)` (see
    robosuite/environments/base.py:186, used inside `_initialize_sim`). The
    processor is called with the raw scene XML on every reload, and its return
    value becomes the XML that MjSim is built from. Registering our injector
    there means rightview is present *every* time the sim is (re)compiled --
    including the implicit `_initialize_sim()` inside `reset()`,
    `reset_from_xml_string()`, and the very first build done by `__init__`.

    Note on observables
    -------------------
    `camera_names` is consumed by `_setup_observables` at __init__ time, so we
    cannot add `rightview` to `obs["rightview_image"]` after the fact without
    rebuilding the env. That's fine for us: `render_step()` uses
    `env.sim.render(camera_name="rightview", ...)` directly, which only needs
    the camera to exist in the compiled MjModel (which our processor guarantees).
    """
    from libero.libero.envs import OffScreenRenderEnv

    # Closure capturing the baseline so the processor signature matches
    # robosuite's expected `fn(xml_string) -> xml_string`.
    def _rightview_processor(xml_str: str) -> str:
        # If rightview is already present (e.g. processor ran on an XML we
        # previously patched), skip to keep idempotency.
        if 'name="rightview"' in xml_str:
            return xml_str
        return _inject_camera_into_xml_string(
            xml_str,
            sibling_cam_name="agentview",
            new_cam_name="rightview",
            baseline=baseline,
        )

    # Build the env with the standard cameras only -- `rightview` isn't an
    # observable, we will render it on demand via `env.sim.render`.
    #
    # IMPORTANT: `OffScreenRenderEnv.__init__` calls `_initialize_sim` BEFORE
    # we get a chance to register the processor, so the FIRST compiled model
    # won't have rightview. We fix that immediately below with one
    # `reset_from_xml_string` (which re-runs `_initialize_sim`, and this time
    # the processor IS registered, so rightview gets injected).
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
    )

    # Register the processor on the underlying robosuite env. ControlEnv stores
    # the actual robosuite env on `.env`.
    env.env.set_xml_processor(_rightview_processor)

    # Force one rebuild so the freshly-registered processor runs and rightview
    # ends up in the compiled MjModel. Pass the current compiled XML through it
    # directly -- using `reset_from_xml_string(env.sim.model.get_xml())` would
    # double-trigger the processor; `_initialize_sim(xml_string=None)` re-runs
    # _load_model which would re-randomize object placement. The cleanest path
    # is reset_from_xml_string with the processor's output (idempotent thanks
    # to the early-return above).
    patched_xml = _rightview_processor(env.sim.model.get_xml())
    env.env.reset_from_xml_string(patched_xml)

    return env


def _inject_camera_into_xml_string(
    xml_str: str, sibling_cam_name: str, new_cam_name: str, baseline: float
) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_str)
    # MuJoCo cameras can live under worldbody/body/worldbody-camera; we scan all.
    sibling = None
    for cam in root.iter("camera"):
        if cam.get("name") == sibling_cam_name:
            sibling = cam
            break
    if sibling is None:
        raise RuntimeError(f"camera '{sibling_cam_name}' not found in compiled XML")

    pos = np.array([float(x) for x in sibling.get("pos").split()])
    quat = np.array([float(x) for x in sibling.get("quat").split()])
    new_pos, new_quat = compute_right_camera_pose(pos, quat, baseline)

    # Insert as sibling under the same parent.
    parent_map = {child: p for p in root.iter() for child in p}
    parent = parent_map[sibling]
    new_cam = ET.SubElement(parent, "camera")
    new_cam.set("mode", sibling.get("mode", "fixed"))
    new_cam.set("name", new_cam_name)
    new_cam.set("pos", " ".join(f"{v:.10f}" for v in new_pos))
    new_cam.set("quat", " ".join(f"{v:.10f}" for v in new_quat))
    # Preserve fovy/focal if set on the sibling.
    for k in ("fovy", "ipd", "focal", "principal", "resolution"):
        if sibling.get(k) is not None:
            new_cam.set(k, sibling.get(k))

    return ET.tostring(root, encoding="unicode")


def render_step(env, resolution: int) -> dict[str, np.ndarray]:
    """Pull three views directly via env.sim.render (bypass obs dict).

    Why not the obs dict: robosuite's obs[X_image] returns the OpenGL raw
    bottom-up frame, which appears 180-degree rotated relative to the
    canonical orientation. LIBERO's own eval pipeline corrects this with
    obs[X_image][::-1, ::-1]. Meanwhile env.sim.render(...)[::-1] (single
    vertical flip) is the standard MuJoCo recipe and yields the SAME
    natural orientation. Using sim.render for all three cameras keeps
    left/right/wrist orientation strictly consistent and matches IPEC's
    mono LeRobot videos pixel-orientation.
    """
    left = env.sim.render(
        camera_name="agentview", width=resolution, height=resolution, depth=False,
    )[::-1]
    right = env.sim.render(
        camera_name="rightview", width=resolution, height=resolution, depth=False,
    )[::-1]
    wrist = env.sim.render(
        camera_name="robot0_eye_in_hand", width=resolution, height=resolution, depth=False,
    )[::-1]

    return {
        "left": np.ascontiguousarray(left, dtype=np.uint8),
        "right": np.ascontiguousarray(right, dtype=np.uint8),
        "wrist": np.ascontiguousarray(wrist, dtype=np.uint8),
    }



def write_video(frames: list[np.ndarray], out_path: Path, fps: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v / libx264 - LeRobot release uses libx264 with yuv420p.
    # TODO: match exact codec/pix_fmt LeRobot uses so HF datasets viewer plays nicely.
    with imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    ) as w:
        for f in frames:
            w.append_data(f)

