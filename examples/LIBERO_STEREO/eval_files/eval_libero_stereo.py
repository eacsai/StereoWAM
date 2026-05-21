import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro

import contextlib


@contextlib.contextmanager
def _libero_init_state_compat():
    """Temporarily force weights_only=False on torch.load so that
    LIBERO's pickled init_states.pruned (numpy.core.multiarray._reconstruct
    legacy format) can be deserialized on torch>=2.6 (which defaults to
    weights_only=True and refuses such pickles).

    Scoped on purpose: a global monkey patch would weaken pickle safety for
    every transitive torch.load in the process (ckpt loads, etc.). Wrap
    only the LIBERO call site that needs it, then restore.
    """
    _orig = torch.load
    def _patched(*a, **kw):
        return _orig(*a, **{**kw, "weights_only": False})
    torch.load = _patched
    try:
        yield
    finally:
        torch.load = _orig

from libero.libero import benchmark, get_libero_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# Reuse the exact stereo camera plumbing used to build the training dataset so
# the geometry (agentview <-> rightview baseline, intrinsics, frame orientation)
# is bit-for-bit identical at eval time.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "SSF" / "render_scripts"))
from _stereo_render_utils import make_stereo_env, render_step  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val):
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    bin_val = 1.0 - 2.0 * (float(arr[0]) > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = -1

    # Must match the baseline used to render the training stereo dataset
    # (SSF/render_scripts/render_stereo_libero_hdf5.py default: 0.06 m).
    stereo_baseline: float = 0.06

    video_out_path: str = "experiments/libero_stereo/logs"
    seed: int = 7
    pretrained_path: str = ""
    unnorm_key: str | None = None
    post_process_action: bool = True
    job_name: str = "test"


def eval_libero_stereo(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        task = task_suite.get_task(task_id)
        with _libero_init_state_compat():
            initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_stereo_env(
            task, LIBERO_ENV_RESOLUTION, args.stereo_baseline, args.seed
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            full_actions = []
            done = False
            step = 0
            logging.info(f"Starting episode {task_episodes + 1}...")

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Render the three views via env.sim.render (same path used by
                # the training stereo renderer) so orientation and intrinsics
                # match the training distribution exactly.
                frames = render_step(env, LIBERO_ENV_RESOLUTION)
                primary = np.ascontiguousarray(frames["left"])
                right_view = np.ascontiguousarray(frames["right"])
                # Phase 1: no wrist input (deferred to Phase 1.5).
                replay_images.append(primary)

                # Plain QwenPI accepts a list of images directly via build_qwenvl_inputs;
                # VLM self-attention fuses [primary, right_view] internally.
                # NOTE: do NOT pass "state" here. Training config did not set
                # include_state, so the dataloader never put state into the
                # training samples → state_encoder was never gradient-updated.
                # Injecting state at eval would feed an untrained MLP and add
                # an extra token to the DiT sequence (mismatched seq length /
                # position-embedding layout) → drives success rate to 0%.
                example_dict = {
                    "image": [primary, right_view],
                    "lang": str(task_description),
                }

                response = client_model.step(example=example_dict, step=step)
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                full_actions.append(delta_action)

                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            if full_actions:
                full_actions = np.stack(full_actions)

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_stereo_env(task, resolution, baseline, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = make_stereo_env(str(task_bddl_file), baseline=baseline, resolution=resolution)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_libero_stereo)
