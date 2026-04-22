from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any, Literal

import imageio
import numpy as np
import tyro

from capx.envs.simulators.libero import FrankaLiberoEnv
from capx.integrations.franka.libero_reduced import FrankaLiberoVLAMinimalApiReduced


OBJECT_SWAP_TARGET_PROMPTS: dict[int, list[str]] = {
    2: ["salad dressing", "bottle of salad dressing", "salad dressing bottle"],
    6: ["stick of butter", "butter package", "package of butter", "butter"],
    7: ["milk", "carton of milk", "milk carton"],
    9: ["orange juice carton", "carton of orange juice", "orange juice"],
}


@dataclasses.dataclass
class Args:
    mode: Literal["helper", "rollout"] = "helper"
    suite_name: str = "libero_object_swap"
    task_id: int = 9
    goal_prompt: str | None = None
    target_prompts: list[str] = dataclasses.field(default_factory=list)
    basket_prompts: list[str] = dataclasses.field(
        default_factory=lambda: ["basket", "woven basket", "square woven basket"]
    )
    host: str = "127.0.0.1"
    port: int = 8004
    num_trials: int = 20
    seed_start: int = 1
    max_steps: int = 4000
    hover_dz: float = 0.10
    grasp_dz: float = 0.015
    lift_dz: float = 0.16
    place_dz: float = 0.12
    stage_before_openpi: bool = True
    replan_steps: int = 3
    execute_actions: int = 3
    openpi_rounds: int = 1
    max_openpi_steps: int = 150
    execute_actions_per_plan: int = 1
    record_video: bool = False
    wrist_video: bool = False
    output_dir: str = "outputs/openpi_helper_eval"


def _to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        return value

    return {key: convert(value) for key, value in result.items()}


def _save_video(path: pathlib.Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    imageio.mimwrite(path, frames, fps=10)


def run(args: Args) -> None:
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    env = FrankaLiberoEnv(
        suite_name=args.suite_name,
        task_id=args.task_id,
        privileged=False,
        max_steps=args.max_steps,
        seed=args.seed_start,
    )
    api = FrankaLiberoVLAMinimalApiReduced(env)
    goal_prompt = args.goal_prompt or env.handle.task_language
    target_prompts = args.target_prompts
    if not target_prompts and args.suite_name == "libero_object_swap":
        target_prompts = OBJECT_SWAP_TARGET_PROMPTS.get(args.task_id, [])

    successes = 0
    summaries: list[dict[str, Any]] = []

    for offset in range(args.num_trials):
        trial_seed = args.seed_start + offset
        trial_name = f"trial_{offset + 1:02d}"
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        env.reset(seed=trial_seed)
        if args.record_video:
            env.enable_video_capture(True, clear=True, wrist_camera=args.wrist_video)

        before_reward = float(env.compute_reward())
        before_done = bool(env.task_completed())
        before_obs = env.get_observation()

        if args.mode == "rollout":
            result = api.execute_openpi_native_rollout(
                max_steps=args.max_openpi_steps,
                replan_steps=args.replan_steps,
                execute_actions_per_plan=args.execute_actions_per_plan,
                prompt=goal_prompt,
                host=args.host,
                port=args.port,
            )
        else:
            result = api.execute_openpi_local_pick_and_place(
                goal_prompt=goal_prompt,
                target_prompts=target_prompts,
                basket_prompts=args.basket_prompts,
                host=args.host,
                port=args.port,
                replan_steps=args.replan_steps,
                execute_actions=args.execute_actions,
                openpi_rounds=args.openpi_rounds,
                hover_dz=args.hover_dz,
                grasp_dz=args.grasp_dz,
                lift_dz=args.lift_dz,
                place_dz=args.place_dz,
                stage_before_openpi=args.stage_before_openpi,
            )

        after_reward = float(env.compute_reward())
        after_done = bool(env.task_completed())
        after_obs = env.get_observation()
        if after_done:
            successes += 1

        summary = {
            "trial": offset + 1,
            "seed": trial_seed,
            "goal_prompt": goal_prompt,
            "target_prompts": target_prompts,
            "basket_prompts": args.basket_prompts,
            "before_reward": before_reward,
            "after_reward": after_reward,
            "before_done": before_done,
            "after_done": after_done,
            "eef_before": np.asarray(before_obs["robot_cartesian_pos"][:3]).tolist(),
            "eef_after": np.asarray(after_obs["robot_cartesian_pos"][:3]).tolist(),
            "helper_result": _to_jsonable(result),
        }
        summaries.append(summary)

        with (trial_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        if args.record_video:
            _save_video(trial_dir / "video_agentview.mp4", env.get_video_frames(clear=True))
            if args.wrist_video:
                _save_video(trial_dir / "video_wrist.mp4", env.get_wrist_video_frames(clear=True))

        logging.info(
            "trial=%d/%d seed=%d success=%s reward=%.3f openpi_success=%.3f cumulative=%.3f",
            offset + 1,
            args.num_trials,
            trial_seed,
            after_done,
            after_reward,
            float(result.get("openpi_success", 0.0)),
            successes / (offset + 1),
        )

    with (output_root / "aggregate.json").open("w") as f:
        json.dump(
            {
                "suite_name": args.suite_name,
                "task_id": args.task_id,
                "goal_prompt": goal_prompt,
                "mode": args.mode,
                "target_prompts": target_prompts,
                "basket_prompts": args.basket_prompts,
                "episodes": args.num_trials,
                "successes": successes,
                "success_rate": successes / max(args.num_trials, 1),
                "trials": summaries,
            },
            f,
            indent=2,
        )

    logging.info(
        "completed suite=%s task=%d episodes=%d successes=%d success_rate=%.4f",
        args.suite_name,
        args.task_id,
        args.num_trials,
        successes,
        successes / max(args.num_trials, 1),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(tyro.cli(Args))
