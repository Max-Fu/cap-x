from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
from typing import Any

import imageio
import numpy as np
import tqdm
import tyro
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path

from capx.utils.openpi import OpenPIWebsocketClient, build_openpi_libero_input


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    """Run upstream OpenPI websocket inference on LIBERO or LIBERO-PRO suites."""

    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: int | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int | None = None
    seed: int = 7
    max_steps_override: int | None = None
    video_out_path: str = "outputs/openpi_libero/videos"
    record_video: bool = False


def _get_max_steps(task_suite_name: str) -> int:
    suite = task_suite_name.lower()
    if suite.startswith("libero_spatial"):
        return 220
    if suite.startswith("libero_object"):
        return 280
    if suite.startswith("libero_goal"):
        return 300
    if suite.startswith("libero_10"):
        return 520
    if suite.startswith("libero_90"):
        return 400
    raise ValueError(f"Unknown LIBERO suite prefix for {task_suite_name}")


def _get_libero_env(task: Any, resolution: int, seed: int) -> tuple[OffScreenRenderEnv, str]:
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _trial_indices(num_trials_per_task: int | None, available_trials: int) -> range:
    if num_trials_per_task is None:
        return range(available_trials)
    return range(min(num_trials_per_task, available_trials))


def run(args: Args) -> None:
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise KeyError(f"Suite {args.task_suite_name!r} not found. Available suites: {sorted(benchmark_dict)}")

    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = args.max_steps_override or _get_max_steps(args.task_suite_name)
    task_ids = [args.task_id] if args.task_id is not None else list(range(task_suite.n_tasks))
    video_root = pathlib.Path(args.video_out_path)
    if args.record_video:
        video_root.mkdir(parents=True, exist_ok=True)

    client = OpenPIWebsocketClient(host=args.host, port=args.port)
    total_episodes = 0
    total_successes = 0

    def _env_success(env_obj: Any) -> bool:
        checker = getattr(env_obj.env, "check_success", None)
        if checker is None:
            checker = getattr(env_obj.env, "_check_success", None)
        if checker is None:
            raise AttributeError("LIBERO env has neither check_success nor _check_success")
        return bool(checker())

    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0
        for episode_idx in tqdm.tqdm(
            _trial_indices(args.num_trials_per_task, len(initial_states)),
            desc=f"task_{task_id}",
            leave=False,
        ):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan: collections.deque[np.ndarray] = collections.deque()
            replay_images: list[np.ndarray] = []
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if args.record_video:
                    replay_images.append(np.asarray(obs["agentview_image"][::-1, ::-1]))

                if not action_plan:
                    openpi_input = build_openpi_libero_input(
                        {
                            "agentview": {"images": {"rgb": obs["agentview_image"][::-1]}},
                            "robot0_eye_in_hand": {
                                "images": {"rgb": obs["robot0_eye_in_hand_image"][::-1]}
                            },
                            "robot_cartesian_pos": np.zeros(8, dtype=np.float64),
                        },
                        prompt=task_description,
                        raw_libero_obs=obs,
                        resize_size=args.resize_size,
                    )
                    action_chunk = np.asarray(client.infer(openpi_input)["actions"], dtype=np.float64)
                    if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
                        raise ValueError(f"Unexpected OpenPI action chunk shape: {action_chunk.shape}")
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float64)[:7]
                try:
                    obs, _, done, _ = env.step(action.tolist())
                except ValueError as exc:
                    if "terminated episode" not in str(exc):
                        raise
                    done = _env_success(env)
                    if done:
                        task_successes += 1
                        total_successes += 1
                    break
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            if args.record_video and replay_images:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    video_root / f"{args.task_suite_name}_task{task_id}_{task_segment}_{suffix}.mp4",
                    [np.asarray(frame) for frame in replay_images],
                    fps=10,
                )

            task_episodes += 1
            total_episodes += 1

        task_rate = task_successes / max(task_episodes, 1)
        total_rate = total_successes / max(total_episodes, 1)
        logging.info(
            "suite=%s task=%d success_rate=%.4f total_rate=%.4f",
            args.task_suite_name,
            task_id,
            task_rate,
            total_rate,
        )

    final_rate = total_successes / max(total_episodes, 1)
    logging.info(
        "completed suite=%s episodes=%d successes=%d success_rate=%.4f",
        args.task_suite_name,
        total_episodes,
        total_successes,
        final_rate,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(tyro.cli(Args))
