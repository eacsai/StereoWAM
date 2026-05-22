import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
import torch
import contextlib


@contextlib.contextmanager
def _libero_init_state_compat():
    """Temporarily force weights_only=False on torch.load so LIBERO's
    pickled init_states (numpy.core.multiarray._reconstruct legacy format)
    can be deserialised on torch>=2.6 (which defaults to weights_only=True).
    Scoped so we don't weaken pickle safety globally."""
    _orig = torch.load
    def _patched(*a, **kw):
        return _orig(*a, **{**kw, "weights_only": False})
    torch.load = _patched
    try:
        yield
    finally:
        torch.load = _orig


from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# NEW (2026-05-20): stereo eval — inject rightview camera into LIBERO sim env via
# make_stereo_env (reuses the renderer helper used to regenerate the training data,
# so the eval-time right_view geometry is bit-identical to the training pixels).
import sys as _sys
_STARVLA_DIR = pathlib.Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_STARVLA_DIR / "SSF" / "render_scripts"))
from _stereo_render_utils import make_stereo_env  # type: ignore  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    """OpenVLA convention: model output v ∈ [0, 1] (where 1 ≈ open in IPEC LeRobot),
    binarize with threshold 0.5 and INVERT to LIBERO sim convention where
    -1 = open and +1 = close. Used by checkpoints trained on the official IPEC
    LeRobot dataset (gripper col binarized to {0, 1})."""
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _binarize_gripper_libero_raw(open_val: np.ndarray | float) -> np.ndarray:
    """NEW (2026-05-21): LIBERO-raw convention for checkpoints trained on
    self-rendered data (regenerate_libero_stereo.py) which writes the raw
    LIBERO demo gripper column ∈ {-1, +1} directly to parquet WITHOUT the
    OpenVLA→LeRobot polarity transform that the official IPEC release applies.

    Model trained this way emits gripper output near ±1 with LIBERO sim's
    native semantics (-1 = open, +1 = close). So we pass the SIGN directly:
    no inversion, threshold at 0 (midpoint of {-1, +1}).

    Compare with _binarize_gripper_open: that one inverts AND uses 0.5
    threshold because OpenVLA convention treats v as a [0, 1] open-probability."""
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = -1.0 if v < 0.0 else 1.0
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    video_keys: str = "primary,wrist"  # NEW (2026-05-20): comma-separated image keys (primary|wrist|right_view).
                                       # Default backward compat. Mono primary-only ckpts: --args.video-keys primary.
                                       # Stereo ckpts: --args.video-keys primary,right_view.

    stereo_baseline: float = 0.06  # NEW (2026-05-20): rightview camera baseline in meters.
                                   # Only used when video_keys contains "right_view". 0.06 (6cm) matches
                                   # scripts/4090d/regenerate_libero_stereo.py default — keep in sync.

    gripper_convention: str = "openvla"  # NEW (2026-05-21): {"openvla", "libero_raw"}.
                                         # "openvla": model trained on IPEC LeRobot data with gripper col
                                         #            binarized {0,1} (open-prob). Use _binarize_gripper_open
                                         #            (invert + 0.5 threshold). DEFAULT — back-compat.
                                         # "libero_raw": model trained on self-rendered parquet whose
                                         #            gripper col is raw LIBERO demo {-1, +1} (no transform).
                                         #            Use _binarize_gripper_libero_raw (pass sign, threshold 0).
                                         # Mismatch ↔ gripper polarity completely flipped → catastrophic
                                         # success-rate collapse (see notes on Z/Y self-rendered eval).

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # NEW (codex F4c 2026-05-21): gate observation.state injection. Training-time
    # include_state must match eval-time include_state, otherwise an untrained
    # state_encoder MLP receives state input AND the DiT sequence has an extra
    # token vs training → success rate collapses to 0% (Phase 1 lesson, see comment
    # block in observation-build below). All ckpts as of 2026-05-21 train with
    # include_state=True so the default is safe; only flip to False if you trained
    # without state encoder.
    include_state: bool = True

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


_ALLOWED_VIDEO_KEYS = {"primary", "wrist", "right_view"}
_ALLOWED_GRIPPER_CONVENTIONS = {"openvla", "libero_raw"}


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # NEW (2026-05-21): fail-closed validation. Misconfigured gripper_convention or
    # video_keys silently used to fall back to defaults (typo → default openvla → wrong
    # polarity, catastrophic 35% bug 2026-05-21). Now raise immediately on any unknown
    # value so eval CAN'T silently run with wrong setup.
    if args.gripper_convention not in _ALLOWED_GRIPPER_CONVENTIONS:
        raise ValueError(
            f"--args.gripper-convention={args.gripper_convention!r} not in "
            f"{sorted(_ALLOWED_GRIPPER_CONVENTIONS)}. Self-rendered (un-patched) ckpts MUST "
            f"use 'libero_raw'; official IPEC-data ckpts (or self-rendered with patched "
            f"parquets) use 'openvla'. Pick one explicitly."
        )

    requested_video_keys = [k.strip() for k in args.video_keys.split(",") if k.strip()]
    unknown_video_keys = [k for k in requested_video_keys if k not in _ALLOWED_VIDEO_KEYS]
    if unknown_video_keys:
        raise ValueError(
            f"--args.video-keys contains unknown keys {unknown_video_keys}. "
            f"Allowed: {sorted(_ALLOWED_VIDEO_KEYS)}. Mono ckpts: 'primary'. "
            f"Stereo ckpts: 'primary,right_view'. Note: 'rightview' (no underscore) "
            f"is a common typo for 'right_view'."
        )
    if not requested_video_keys:
        raise ValueError("--args.video-keys must list at least one key (e.g. 'primary')")

    logging.info(f"[eval] gripper_convention={args.gripper_convention}  video_keys={requested_video_keys}  include_state={args.include_state}")

    # NEW (codex F4c 2026-05-21): fail-closed ckpt existence check. Previously a typo
    # in --args.pretrained-path produced a buried FileNotFoundError mid-init after
    # the policy server already started; now we exit upfront with a clear message.
    if args.pretrained_path:
        from pathlib import Path as _P
        if not _P(args.pretrained_path).exists():
            raise FileNotFoundError(
                f"--args.pretrained-path={args.pretrained_path!r} does not exist. "
                f"Check the path, or pass empty string '' to skip ckpt-name-derived output dir."
            )

    # NEW (2026-05-20): stereo eval flag — derived from video_keys. Drives both env
    # construction (make_stereo_env vs vanilla OffScreenRenderEnv) and per-step
    # rightview frame extraction. Computed once here so the inner loop is hot-path clean.
    use_stereo = "right_view" in requested_video_keys
    if use_stereo:
        logging.info(f"[stereo] eval with rightview baseline={args.stereo_baseline}m")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        with _libero_init_state_compat():
            initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed,
            use_stereo=use_stereo, stereo_baseline=args.stereo_baseline,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # NEW (2026-05-20): if stereo, render rightview via env.sim.render
                # (rightview is in compiled MjModel but NOT in robosuite obs dict —
                # see _stereo_render_utils.make_stereo_env docstring "Note on observables").
                # Apply the SAME [::-1, ::-1] 180° rotation that primary/wrist use, matching
                # training-time renderer (scripts/4090d/regenerate_libero_stereo.py:150 does
                # `right_view.append(rv_raw[::-1, ::-1])`). DO NOT use the [::-1]-only convention
                # from render_step in _stereo_render_utils.py — that's for the HDF5 replay path,
                # not the action-replay path used for the actual training data.
                if use_stereo:
                    rv_raw = env.sim.render(
                        camera_name="rightview",
                        width=LIBERO_ENV_RESOLUTION,
                        height=LIBERO_ENV_RESOLUTION,
                        depth=False,
                    )
                    right_img = np.ascontiguousarray(rv_raw[::-1, ::-1])
                else:
                    right_img = None

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                # ⚠ Phase 1 lesson (migrated from removed examples/LIBERO_STEREO/eval_files/eval_libero_stereo.py):
                # If the training config did NOT set include_state, the dataloader never put state
                # into training samples → state_encoder MLP was never gradient-updated. Injecting
                # state at eval feeds an untrained MLP AND adds an extra token to the DiT input
                # sequence (mismatched seq length / position-embedding layout vs training) → success
                # rate collapses to 0%. The current ckpts (Mono-Official / Mono-SelfRender /
                # Stereo-SelfRender, 2026-05-21) all train with include_state=True, so passing state
                # below is correct. If you train a model with include_state=False, gate the
                # "observation.state" key here on the training-time config.
                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "instruction": [str(task_description)],
                }
                if args.include_state:
                    observation["observation.state"] = np.expand_dims(state, axis=0)
                if use_stereo:
                    observation["observation.right_view"] = np.expand_dims(right_img, axis=0)  # (H, W, C)

                # align key with model API. Image list built per --args.video-keys to match
                # training-time video_keys (mono ckpts: primary only; stereo ckpts: primary,right_view;
                # default backward-compat: primary,wrist).
                _img_map = {
                    "primary": observation["observation.primary"][0],
                    "wrist":   observation["observation.wrist_image"][0],
                }
                if use_stereo:
                    _img_map["right_view"] = observation["observation.right_view"][0]
                _selected = [k.strip() for k in args.video_keys.split(",") if k.strip()]
                image_list = [_img_map[k] for k in _selected if k in _img_map]
                example_dict = {
                    "image": image_list,
                    "lang": observation["instruction"][0],
                }

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                # NEW (2026-05-21): dispatch on gripper_convention. See Args.gripper_convention docs.
                if args.gripper_convention == "libero_raw":
                    gripper = _binarize_gripper_libero_raw(open_gripper)
                elif args.gripper_convention == "openvla":
                    gripper = _binarize_gripper_open(open_gripper)
                else:
                    raise ValueError(
                        f"Unknown gripper_convention: {args.gripper_convention!r} "
                        "(expected 'openvla' or 'libero_raw')"
                    )

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed, use_stereo: bool = False, stereo_baseline: float = 0.06):
    """Initializes and returns the LIBERO environment, along with the task description.

    When ``use_stereo=True``, builds the env via ``make_stereo_env`` instead of
    vanilla OffScreenRenderEnv. That injects a sibling ``rightview`` camera into
    the compiled MuJoCo XML via robosuite's ``set_xml_processor`` hook, so the
    camera survives every ``env.reset()`` (verified by
    ``scripts/4090d/smoke_stereo_camera_eval.py``). The baseline MUST match
    training-time renderer (default 0.06m, see
    ``scripts/4090d/regenerate_libero_stereo.py``).
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if use_stereo:
        env = make_stereo_env(
            bddl_file_name=str(task_bddl_file),
            baseline=stereo_baseline,
            resolution=resolution,
        )
    else:
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
        }
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
