from __future__ import annotations

import json
from typing import Any

import numpy as np

from capx.utils.openpi import (
    DEFAULT_OPENPI_HOST,
    DEFAULT_OPENPI_PORT,
    OpenPIWebsocketClient,
    apply_libero_delta_action,
    build_openpi_libero_input,
)


def infer_openpi_gripper_command(gripper_action: float, *, deadband: float = 0.05) -> str:
    """Convert an OpenPI gripper scalar into an open / close / hold command."""
    if gripper_action > deadband:
        return "close"
    if gripper_action < -deadband:
        return "open"
    return "hold"


def build_openpi_subgoals(
    robot_cartesian_pos: np.ndarray,
    action_chunk: np.ndarray,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    gripper_deadband: float = 0.05,
) -> list[dict[str, Any]]:
    """Interpret an OpenPI LIBERO action chunk as a sequence of Cartesian subgoals."""
    current_position = np.asarray(robot_cartesian_pos[:3], dtype=np.float64).reshape(3)
    current_quaternion = np.asarray(robot_cartesian_pos[3:7], dtype=np.float64).reshape(4)

    subgoals: list[dict[str, Any]] = []
    for index, action in enumerate(np.asarray(action_chunk, dtype=np.float64)):
        position, quaternion_wxyz, gripper_action = apply_libero_delta_action(
            current_position,
            current_quaternion,
            action,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )
        subgoals.append(
            {
                "index": index,
                "position": position,
                "quaternion_wxyz": quaternion_wxyz,
                "gripper_action": float(gripper_action),
                "gripper_command": infer_openpi_gripper_command(
                    float(gripper_action),
                    deadband=gripper_deadband,
                ),
                "raw_action": np.asarray(action[:7], dtype=np.float64),
                "translation_delta": np.asarray(action[:3], dtype=np.float64) * translation_scale,
                "rotation_delta_axis_angle": np.asarray(action[3:6], dtype=np.float64)
                * rotation_scale,
            }
        )
        current_position = position
        current_quaternion = quaternion_wxyz
    return subgoals


class FrankaLiberoOpenPIToolMixin:
    """Shared OpenPI-backed VLA helpers for LIBERO Franka APIs."""

    _openpi_client: OpenPIWebsocketClient | None
    _openpi_client_endpoint: tuple[str, int] | None

    def _get_openpi_client(
        self,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
    ) -> OpenPIWebsocketClient:
        endpoint = (host, port)
        if self._openpi_client is None or self._openpi_client_endpoint != endpoint:
            if self._openpi_client is not None:
                self._openpi_client.close()
            self._openpi_client = OpenPIWebsocketClient(host=host, port=port)
            self._openpi_client_endpoint = endpoint
        return self._openpi_client

    def _resolve_openpi_prompt(self, prompt: str | None = None) -> str:
        if prompt is not None and prompt.strip():
            return prompt
        handle = getattr(self._env, "handle", None)
        task_language = getattr(handle, "task_language", "")
        return str(task_language)

    def _emit_openpi_trace(self, event: str, **payload: Any) -> None:
        """Emit a compact stdout trace so CLI runs capture OpenPI tool usage."""
        try:
            serialized = json.dumps(payload, default=str, sort_keys=True)
        except TypeError:
            serialized = str(payload)
        print(f"[openpi-tool] {event} {serialized}")

    def get_openpi_server_info(
        self,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
    ) -> dict[str, Any]:
        """Return OpenPI server metadata to verify the VLA endpoint CaP-X is using.

        Args:
            host: Host of the OpenPI websocket server.
            port: Port of the OpenPI websocket server.

        Returns:
            dict with:
                - "host": resolved host string.
                - "port": resolved port.
                - "metadata": handshake metadata reported by the OpenPI server.
        """
        client = self._get_openpi_client(host=host, port=port)
        result = {
            "host": host,
            "port": port,
            "metadata": dict(client.metadata),
        }
        self._emit_openpi_trace(
            "server_info",
            host=host,
            port=port,
            metadata=result["metadata"],
        )
        return result

    def get_openpi_action_chunk(
        self,
        replan_steps: int = 5,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
    ) -> np.ndarray:
        """Query an upstream OpenPI LIBERO policy server for the next raw action chunk.

        This returns the native OpenPI/LIBERO actions: XYZ delta, axis-angle delta,
        and gripper command. Use this when you want OpenPI's direct policy suggestion
        from the current camera observations and robot state.

        Args:
            replan_steps: Number of actions to keep from the predicted chunk.
            resize_size: Square image size used for OpenPI preprocessing. Default is 224.
            prompt: Optional task instruction override. Defaults to the current LIBERO task language.
            host: Host of the OpenPI websocket server.
            port: Port of the OpenPI websocket server.

        Returns:
            action_chunk: Array of shape (replan_steps, 7) containing raw OpenPI actions.
        """
        obs = self.get_observation()
        raw_obs = getattr(self._env, "_current_obs", None)
        openpi_obs = obs
        if raw_obs is not None and all(
            key in raw_obs for key in ("agentview_image", "robot0_eye_in_hand_image")
        ):
            # Match the upstream eval path exactly when the raw LIBERO observation is available.
            openpi_obs = {
                "agentview": {
                    "images": {"rgb": np.asarray(raw_obs["agentview_image"][::-1])},
                },
                "robot0_eye_in_hand": {
                    "images": {"rgb": np.asarray(raw_obs["robot0_eye_in_hand_image"][::-1])},
                },
                "robot_cartesian_pos": np.asarray(obs["robot_cartesian_pos"], dtype=np.float64),
            }
        payload = build_openpi_libero_input(
            openpi_obs,
            prompt=self._resolve_openpi_prompt(prompt),
            raw_libero_obs=raw_obs,
            resize_size=resize_size,
        )
        response = self._get_openpi_client(host=host, port=port).infer(payload)
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"Unexpected OpenPI action chunk shape: {actions.shape}")
        trimmed = actions[:replan_steps, :7]
        self._emit_openpi_trace(
            "action_chunk",
            host=host,
            port=port,
            prompt=self._resolve_openpi_prompt(prompt),
            replan_steps=int(replan_steps),
            returned_steps=int(trimmed.shape[0]),
        )
        return trimmed

    def get_openpi_native_action_chunk(
        self,
        replan_steps: int = 5,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        sync_from_primary: bool = False,
    ) -> np.ndarray:
        """Query OpenPI using the native OSC_POSE LIBERO env observation contract."""
        get_native_raw_obs = getattr(self._env, "get_openpi_native_raw_obs", None)
        if get_native_raw_obs is None:
            raise RuntimeError("Current environment does not support native OpenPI observations.")

        raw_obs = get_native_raw_obs(sync_from_primary=sync_from_primary)
        openpi_obs = {
            "agentview": {
                "images": {"rgb": np.asarray(raw_obs["agentview_image"][::-1])},
            },
            "robot0_eye_in_hand": {
                "images": {"rgb": np.asarray(raw_obs["robot0_eye_in_hand_image"][::-1])},
            },
        }
        payload = build_openpi_libero_input(
            openpi_obs,
            prompt=self._resolve_openpi_prompt(prompt),
            raw_libero_obs=raw_obs,
            resize_size=resize_size,
        )
        response = self._get_openpi_client(host=host, port=port).infer(payload)
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"Unexpected OpenPI action chunk shape: {actions.shape}")
        trimmed = actions[:replan_steps, :7]
        self._emit_openpi_trace(
            "native_action_chunk",
            host=host,
            port=port,
            prompt=self._resolve_openpi_prompt(prompt),
            replan_steps=int(replan_steps),
            returned_steps=int(trimmed.shape[0]),
            sync_from_primary=bool(sync_from_primary),
        )
        return trimmed

    def get_openpi_subgoal(
        self,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Convert OpenPI's next raw action into an approximate Cartesian subgoal.

        This is an approximation layer for CaP-X's joint-position-controlled LIBERO wrapper.
        It uses the first OpenPI delta action and maps it onto the current end-effector pose.

        Args:
            resize_size: Square image size used for OpenPI preprocessing. Default is 224.
            prompt: Optional task instruction override. Defaults to the current LIBERO task language.
            host: Host of the OpenPI websocket server.
            port: Port of the OpenPI websocket server.
            translation_scale: Multiplier applied to the XYZ delta before composing the subgoal.
            rotation_scale: Multiplier applied to the axis-angle delta before composing the subgoal.

        Returns:
            dict with:
                - "position": (3,) target XYZ in meters.
                - "quaternion_wxyz": (4,) target orientation.
                - "gripper_action": scalar OpenPI gripper command.
                - "gripper_command": discretized open / close / hold interpretation.
                - "raw_action": (7,) raw OpenPI action used to form the subgoal.
        """
        plan = self.plan_with_openpi(
            replan_steps=1,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )
        return plan["subgoals"][0]

    def plan_with_openpi(
        self,
        replan_steps: int = 5,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_deadband: float = 0.05,
    ) -> dict[str, Any]:
        """Ask OpenPI for a short VLA plan and lift it into Cartesian subgoals CaP-X can inspect.

        This is the preferred high-level entrypoint when you want to use the OpenPI policy as a
        tool: it returns the raw action chunk plus an interpreted sequence of Cartesian waypoints
        and discrete gripper commands.

        Args:
            replan_steps: Number of OpenPI actions to keep in the returned plan.
            resize_size: Square image size used for OpenPI preprocessing. Default is 224.
            prompt: Optional task instruction override. Defaults to the current LIBERO task language.
            host: Host of the OpenPI websocket server.
            port: Port of the OpenPI websocket server.
            translation_scale: Multiplier applied to each XYZ delta before composing subgoals.
            rotation_scale: Multiplier applied to each axis-angle delta before composing subgoals.
            gripper_deadband: Values in [-deadband, deadband] are treated as hold.

        Returns:
            dict with:
                - "prompt": task instruction sent to OpenPI.
                - "server_info": metadata about the OpenPI endpoint.
                - "current_robot_cartesian_pos": current robot pose before execution.
                - "current_gripper_open_fraction": current gripper opening in [0, 1].
                - "action_chunk": raw OpenPI action chunk of shape (replan_steps, 7).
                - "subgoals": interpreted Cartesian subgoals derived from the action chunk.
        """
        obs = self.get_observation()
        robot_cartesian_pos = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(8)
        action_chunk = self.get_openpi_action_chunk(
            replan_steps=replan_steps,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
        )
        subgoals = build_openpi_subgoals(
            robot_cartesian_pos,
            action_chunk,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            gripper_deadband=gripper_deadband,
        )
        result = {
            "prompt": self._resolve_openpi_prompt(prompt),
            "server_info": self.get_openpi_server_info(host=host, port=port),
            "current_robot_cartesian_pos": robot_cartesian_pos,
            "current_gripper_open_fraction": float(robot_cartesian_pos[7]),
            "action_chunk": action_chunk,
            "subgoals": subgoals,
        }

        if subgoals:
            first = subgoals[0]
            self._emit_openpi_trace(
                "plan_with_openpi",
                prompt=result["prompt"],
                subgoal_count=len(subgoals),
                first_position=np.round(first["position"], 4).tolist(),
                first_gripper_command=first["gripper_command"],
            )
            self._log_step(
                "plan_with_openpi",
                (
                    "OpenPI proposed "
                    f"{len(subgoals)} subgoal(s); first target={np.round(first['position'], 4).tolist()} "
                    f"gripper={first['gripper_command']}."
                ),
            )
        return result

    def execute_openpi_step(
        self,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        z_approach: float = 0.0,
        apply_gripper: bool = True,
        gripper_deadband: float = 0.05,
    ) -> dict[str, Any]:
        """Query OpenPI and execute the first interpreted subgoal as a single CaP-X tool step."""
        return self.execute_openpi_plan(
            replan_steps=1,
            execute_subgoals=1,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            z_approach=z_approach,
            apply_gripper=apply_gripper,
            gripper_deadband=gripper_deadband,
        )

    def execute_openpi_native_step(
        self,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        sync_from_primary: bool = False,
    ) -> dict[str, Any]:
        """Execute one raw OpenPI delta action through the native OSC_POSE LIBERO env."""
        return self.execute_openpi_native_plan(
            replan_steps=1,
            execute_actions=1,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            sync_from_primary=sync_from_primary,
        )

    def execute_openpi_raw_action(
        self,
        action: np.ndarray,
        *,
        sync_from_primary: bool = False,
    ) -> dict[str, Any]:
        """Execute a provided raw OpenPI action through the native OSC_POSE env."""
        env_execute = getattr(self._env, "execute_openpi_native_action", None)
        if env_execute is None:
            raise RuntimeError("Current environment does not support native OpenPI execution.")
        obs, reward, done, info = env_execute(action, sync_from_primary=sync_from_primary)
        final_robot_cartesian_pos = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(8)
        result = {
            "executed_raw_action": np.asarray(action, dtype=np.float64).reshape(7),
            "native_reward": float(reward),
            "native_done": bool(done),
            "native_info": dict(info),
            "final_robot_cartesian_pos": final_robot_cartesian_pos,
        }
        self._emit_openpi_trace(
            "execute_openpi_raw_action",
            reward=float(reward),
            done=bool(done),
            final_position=np.round(final_robot_cartesian_pos[:3], 4).tolist(),
        )
        return result

    def plan_with_openpi_native(
        self,
        replan_steps: int = 5,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        sync_from_primary: bool = False,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_deadband: float = 0.05,
    ) -> dict[str, Any]:
        """Plan with OpenPI using the native OSC_POSE LIBERO observation path."""
        obs = self.get_observation()
        robot_cartesian_pos = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(8)
        action_chunk = self.get_openpi_native_action_chunk(
            replan_steps=replan_steps,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            sync_from_primary=sync_from_primary,
        )
        subgoals = build_openpi_subgoals(
            robot_cartesian_pos,
            action_chunk,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            gripper_deadband=gripper_deadband,
        )
        return {
            "prompt": self._resolve_openpi_prompt(prompt),
            "server_info": self.get_openpi_server_info(host=host, port=port),
            "current_robot_cartesian_pos": robot_cartesian_pos,
            "current_gripper_open_fraction": float(robot_cartesian_pos[7]),
            "action_chunk": action_chunk,
            "subgoals": subgoals,
            "sync_from_primary": bool(sync_from_primary),
        }

    def execute_openpi_native_plan(
        self,
        replan_steps: int = 5,
        execute_actions: int = 1,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        sync_from_primary: bool = False,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_deadband: float = 0.05,
    ) -> dict[str, Any]:
        """Plan and execute raw OpenPI actions through the native OSC_POSE LIBERO env."""
        env_execute_many = getattr(self._env, "execute_openpi_native_actions", None)
        if env_execute_many is None:
            raise RuntimeError("Current environment does not support native OpenPI execution.")
        if execute_actions < 1:
            raise ValueError("execute_actions must be at least 1")

        plan = self.plan_with_openpi_native(
            replan_steps=replan_steps,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            sync_from_primary=sync_from_primary,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            gripper_deadband=gripper_deadband,
        )
        raw_actions = np.asarray(plan["action_chunk"][:execute_actions], dtype=np.float64)
        obs, reward, done, info, executed_steps = env_execute_many(
            raw_actions, sync_from_primary=False
        )
        final_robot_cartesian_pos = np.asarray(
            obs["robot_cartesian_pos"],
            dtype=np.float64,
        ).reshape(8)
        result = dict(plan)
        result["executed_raw_actions"] = raw_actions
        result["executed_action_count"] = int(executed_steps)
        result["native_reward"] = float(reward)
        result["native_done"] = bool(done)
        result["native_info"] = dict(info)
        result["final_robot_cartesian_pos"] = final_robot_cartesian_pos
        self._emit_openpi_trace(
            "execute_openpi_native_plan",
            prompt=result["prompt"],
            reward=float(reward),
            done=bool(done),
            executed_action_count=int(executed_steps),
            final_position=np.round(final_robot_cartesian_pos[:3], 4).tolist(),
        )
        self._log_step(
            "execute_openpi_native_plan",
            (
                f"Executed {int(executed_steps)} native OpenPI delta action(s); "
                f"final pose={np.round(final_robot_cartesian_pos[:3], 4).tolist()} reward={float(reward):.3f}."
            ),
        )
        return result

    def execute_openpi_native_rollout(
        self,
        max_steps: int = 150,
        replan_steps: int = 5,
        execute_actions_per_plan: int = 1,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        sync_from_primary: bool = False,
    ) -> dict[str, Any]:
        """Keep replanning with OpenPI until done or a step budget is exhausted."""
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if replan_steps < 1:
            raise ValueError("replan_steps must be at least 1")
        if execute_actions_per_plan < 1:
            raise ValueError("execute_actions_per_plan must be at least 1")

        attempts: list[dict[str, Any]] = []
        total_executed_steps = 0
        obs = self.get_observation()
        reward = 0.0
        done = False
        info: dict[str, Any] = {}
        while total_executed_steps < max_steps and not done:
            remaining = max_steps - total_executed_steps
            execute_actions = min(execute_actions_per_plan, remaining)
            attempt = self.execute_openpi_native_plan(
                replan_steps=replan_steps,
                execute_actions=execute_actions,
                resize_size=resize_size,
                prompt=prompt,
                host=host,
                port=port,
                sync_from_primary=sync_from_primary if total_executed_steps == 0 else False,
            )
            attempts.append(attempt)
            total_executed_steps += int(attempt["executed_action_count"])
            reward = float(attempt["native_reward"])
            done = bool(attempt["native_done"])
            info = dict(attempt["native_info"])
            obs = self.get_observation()
            if int(attempt["executed_action_count"]) < execute_actions:
                break

        final_robot_cartesian_pos = np.asarray(
            obs["robot_cartesian_pos"],
            dtype=np.float64,
        ).reshape(8)
        result = {
            "prompt": self._resolve_openpi_prompt(prompt),
            "server_info": self.get_openpi_server_info(host=host, port=port),
            "attempts": attempts,
            "attempt_count": len(attempts),
            "executed_action_count": int(total_executed_steps),
            "execute_actions_per_plan": int(execute_actions_per_plan),
            "native_reward": float(reward),
            "native_done": bool(done),
            "native_info": info,
            "final_robot_cartesian_pos": final_robot_cartesian_pos,
            "max_steps": int(max_steps),
            "replan_steps": int(replan_steps),
        }
        self._emit_openpi_trace(
            "execute_openpi_native_rollout",
            prompt=result["prompt"],
            max_steps=int(max_steps),
            replan_steps=int(replan_steps),
            attempt_count=len(attempts),
            executed_action_count=int(total_executed_steps),
            reward=float(reward),
            done=bool(done),
            final_position=np.round(final_robot_cartesian_pos[:3], 4).tolist(),
        )
        return result

    def execute_openpi_plan(
        self,
        replan_steps: int = 5,
        execute_subgoals: int = 1,
        resize_size: int = 224,
        prompt: str | None = None,
        host: str = DEFAULT_OPENPI_HOST,
        port: int = DEFAULT_OPENPI_PORT,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        z_approach: float = 0.0,
        apply_gripper: bool = True,
        gripper_deadband: float = 0.05,
    ) -> dict[str, Any]:
        """Use OpenPI as a first-class CaP-X tool by planning and optionally executing subgoals.

        This helper asks OpenPI for a short action chunk, interprets it as Cartesian subgoals,
        and executes the first `execute_subgoals` waypoints through CaP-X's own IK + motion tools.

        Args:
            replan_steps: Number of OpenPI actions to request and interpret.
            execute_subgoals: How many interpreted subgoals to execute from the planned chunk.
            resize_size: Square image size used for OpenPI preprocessing. Default is 224.
            prompt: Optional task instruction override. Defaults to the current LIBERO task language.
            host: Host of the OpenPI websocket server.
            port: Port of the OpenPI websocket server.
            translation_scale: Multiplier applied to each XYZ delta before composing subgoals.
            rotation_scale: Multiplier applied to each axis-angle delta before composing subgoals.
            z_approach: Optional approach distance applied to each executed `goto_pose` call.
            apply_gripper: Whether to apply OpenPI's discrete open / close suggestion after each move.
            gripper_deadband: Values in [-deadband, deadband] are treated as hold.

        Returns:
            dict containing the plan plus:
                - "executed_subgoals": executed subgoal records.
                - "executed_subgoal_count": number of executed subgoals.
                - "final_robot_cartesian_pos": robot pose after execution.
        """
        if execute_subgoals < 1:
            raise ValueError("execute_subgoals must be at least 1")

        result = self.plan_with_openpi(
            replan_steps=replan_steps,
            resize_size=resize_size,
            prompt=prompt,
            host=host,
            port=port,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            gripper_deadband=gripper_deadband,
        )
        planned_subgoals = result["subgoals"][:execute_subgoals]
        current_gripper_open_fraction = float(result["current_gripper_open_fraction"])
        executed_subgoals: list[dict[str, Any]] = []

        for subgoal in planned_subgoals:
            self.goto_pose(
                subgoal["position"],
                subgoal["quaternion_wxyz"],
                z_approach=z_approach,
            )

            executed_gripper_command = "hold"
            if apply_gripper:
                executed_gripper_command, current_gripper_open_fraction = self._apply_openpi_gripper_command(
                    float(subgoal["gripper_action"]),
                    current_gripper_open_fraction,
                    deadband=gripper_deadband,
                )

            executed_subgoal = dict(subgoal)
            executed_subgoal["executed_gripper_command"] = executed_gripper_command
            executed_subgoals.append(executed_subgoal)

        final_robot_cartesian_pos = np.asarray(
            self._env.get_observation()["robot_cartesian_pos"],
            dtype=np.float64,
        ).reshape(8)
        result["executed_subgoals"] = executed_subgoals
        result["executed_subgoal_count"] = len(executed_subgoals)
        result["final_robot_cartesian_pos"] = final_robot_cartesian_pos
        self._emit_openpi_trace(
            "execute_openpi_plan",
            prompt=result["prompt"],
            requested_subgoals=int(execute_subgoals),
            executed_subgoals=len(executed_subgoals),
            final_position=np.round(final_robot_cartesian_pos[:3], 4).tolist(),
        )

        self._log_step(
            "execute_openpi_plan",
            (
                "Executed "
                f"{len(executed_subgoals)} OpenPI-guided subgoal(s); "
                f"final pose={np.round(final_robot_cartesian_pos[:3], 4).tolist()}."
            ),
        )
        return result

    def _apply_openpi_gripper_command(
        self,
        gripper_action: float,
        current_gripper_open_fraction: float,
        *,
        deadband: float = 0.05,
    ) -> tuple[str, float]:
        command = infer_openpi_gripper_command(gripper_action, deadband=deadband)
        if command == "open":
            if current_gripper_open_fraction < 0.95:
                self.open_gripper()
            return command, 1.0
        if command == "close":
            if current_gripper_open_fraction > 0.05:
                self.close_gripper()
            return command, 0.0
        return command, float(current_gripper_open_fraction)
