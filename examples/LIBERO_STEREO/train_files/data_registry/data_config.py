"""LIBERO_STEREO benchmark — data config + mixtures for SSF / Stereo VLA (Phase 1).

Self-contained: declares **both** the stereo (`libero_franka_stereo`) and the
mono-primary (`libero_franka_mono_primary`) DataConfigs used by Phase 1
experiments, so the upstream `examples/LIBERO/data_config.py` is left untouched.

Mixtures point to LEROBOT_LIBERO_STEREO_DATA written by render_stereo_libero.py.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig: stereo (primary + right_view, no wrist)
# ---------------------------------------------------------------------------
class Libero4in1StereoDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",    # left eye (== original agentview)
        "video.right_view",       # right eye (baseline 6cm)  ★ SSF new
        # NOTE: wrist dropped in Phase 1 (no wrist stereo). Deferred to Phase 1.5.
    ]
    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.pad", "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig: mono-primary (primary only, no wrist, no right_view)
# ---------------------------------------------------------------------------
class Libero4in1MonoPrimaryDataConfig(Libero4in1StereoDataConfig):
    """Mono baseline — only primary view (no wrist, no right_view).

    Inherits state / action / transform from `Libero4in1StereoDataConfig` to
    guarantee identical normalization / chunk schema; the *only* difference
    from the stereo config is `video_keys`. That way the mono baseline sees
    exactly the same primary frames the stereo run sees — the only delta is
    "model has access to right_view or not", which is the fair Y-vs-Z
    comparison Stereo VLA Phase 1 needs.

    Do NOT confuse with the upstream `libero_franka` (primary + wrist) config
    in examples/LIBERO/data_config.py; that one is kept intact for upstream PI
    ckpt eval / upstream LIBERO mono training.
    """

    video_keys = ["video.primary_image"]


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka_stereo":       Libero4in1StereoDataConfig(),
    "libero_franka_mono_primary": Libero4in1MonoPrimaryDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
# NOTE: the registry only .update()s this dict from each example module — it
# does NOT consult DataConfig classvars. If a robot type is missing here,
# make_LeRobotSingleDataset falls back to EmbodimentTag.NEW_EMBODIMENT with a
# warning, which corrupts dataset_statistics keys / normalization keys. So new
# robot types must be registered here explicitly.
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka_stereo":       EmbodimentTag.FRANKA,
    "libero_franka_mono_primary": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures (stereo variants of the LIBERO suite mixes)
# ---------------------------------------------------------------------------
# Note: dataset folder names match what render_stereo_libero.py writes:
#   playground/Datasets/LEROBOT_LIBERO_STEREO_DATA/{suite_name}/
DATASET_NAMED_MIXTURES = {
    # ──────────────────────────────────────────────────────────────
    # Stereo mixes — load left + right (2 videos, no wrist in Phase 1).
    # ──────────────────────────────────────────────────────────────
    "libero_spatial_stereo": [
        ("libero_spatial", 1.0, "libero_franka_stereo"),
    ],
    "libero_object_stereo": [
        ("libero_object", 1.0, "libero_franka_stereo"),
    ],
    "libero_goal_stereo": [
        ("libero_goal", 1.0, "libero_franka_stereo"),
    ],
    "libero_10_stereo": [
        ("libero_10", 1.0, "libero_franka_stereo"),
    ],
    "libero_all_stereo": [
        ("libero_spatial", 1.0, "libero_franka_stereo"),
        ("libero_object", 1.0, "libero_franka_stereo"),
        ("libero_goal", 1.0, "libero_franka_stereo"),
        ("libero_10", 1.0, "libero_franka_stereo"),
    ],

    # ──────────────────────────────────────────────────────────────
    # MONO BASELINE (FAIR-COMPARISON) mixes — point at the SAME stereo
    # dataset but load with the `libero_franka_mono_primary` config, which
    # only declares video.primary_image (drops right_view, never had wrist).
    # This gives the mono baseline pixel-identical primary input as the
    # stereo run, so the ONLY difference between the two trainings is
    # "model has access to right_view or not". Use these mixes for the
    # apples-to-apples mono baseline (Z) in the Stereo VLA ablation.
    # ──────────────────────────────────────────────────────────────
    "libero_spatial_mono_replay": [
        ("libero_spatial", 1.0, "libero_franka_mono_primary"),
    ],
    "libero_object_mono_replay": [
        ("libero_object", 1.0, "libero_franka_mono_primary"),
    ],
    "libero_goal_mono_replay": [
        ("libero_goal", 1.0, "libero_franka_mono_primary"),
    ],
    # NEW 2026-05-20: paper Z1 baseline — use OFFICIAL IPEC LeRobot data
    # (libero_goal_no_noops_1.0.0_lerobot) with mono-primary config (drops wrist).
    # Same suite, no wrist, NO our re-rendering — pure official data.
    # data_root_dir should be playground/Datasets/LEROBOT_LIBERO_DATA.
    "libero_goal_mono_official": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_mono_primary"),
    ],
    "libero_10_mono_replay": [
        ("libero_10", 1.0, "libero_franka_mono_primary"),
    ],
    # Joint mono mixture using the SAME self-rendered LIBERO data, but with
    # the upstream `libero_franka` config (primary + wrist). Kept for
    # reproducing starVLA's paper Qwen3-vl-PI baseline only — the user's
    # own Stereo VLA experiments use mono_replay below, NOT this.
    "libero_all_mono_selfrender": [
        ("libero_object", 1.0, "libero_franka"),
        ("libero_goal", 1.0, "libero_franka"),
        ("libero_spatial", 1.0, "libero_franka"),
        ("libero_10", 1.0, "libero_franka"),
    ],
    "libero_all_mono_replay": [
        ("libero_spatial", 1.0, "libero_franka_mono_primary"),
        ("libero_object", 1.0, "libero_franka_mono_primary"),
        ("libero_goal", 1.0, "libero_franka_mono_primary"),
        ("libero_10", 1.0, "libero_franka_mono_primary"),
    ],
}
