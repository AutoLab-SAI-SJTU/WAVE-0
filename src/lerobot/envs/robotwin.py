# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import contextlib
import importlib
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import spaces

from lerobot.processor.core import RobotObservation

logger = logging.getLogger(__name__)

ROBOTWIN_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
ACTION_DIM = 14
DEFAULT_CAMERA_H = 240
DEFAULT_CAMERA_W = 320
DEFAULT_EPISODE_LENGTH = 1200
OFFICIAL_INSTRUCTION_ENV = "LEROBOT_ROBOTWIN_OFFICIAL_INSTRUCTION"
OFFICIAL_INSTRUCTION_TYPE_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_TYPE"
OFFICIAL_INSTRUCTION_MAX_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_MAX"
EXPERT_CHECK_MAX_ATTEMPTS_ENV = "LEROBOT_ROBOTWIN_EXPERT_CHECK_MAX_ATTEMPTS"
FAIL_FAST_ENV = "ROBOTWIN_FAIL_FAST"
TASK_CONFIG_ENV = "LEROBOT_ROBOTWIN_TASK_CONFIG"
SKIP_OBS_WITHIN_CHUNK_ENV = "LEROBOT_ROBOTWIN_SKIP_OBS_WITHIN_CHUNK"
RENDER_ALL_CAMERAS_ENV = "LEROBOT_ROBOTWIN_RENDER_ALL_CAMERAS"
CLEAR_CACHE_FREQ_ENV = "LEROBOT_ROBOTWIN_CLEAR_CACHE_FREQ"
SUPPORTED_TASK_CONFIGS = ("demo_clean", "demo_randomized")

ROBOTWIN_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

_SETUP_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_non_retryable_setup_error(error: Exception) -> bool:
    """Changing scene seeds cannot fix a CUDA runtime or missing dependency."""
    markers = (
        "cuda error:",
        "cuda out of memory",
        "no kernel image is available for execution on the device",
        "cuda driver version is insufficient",
        "cuda-capable device is detected",
    )
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImportError) or any(marker in str(current).lower() for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def _arm_for_block(block: Any) -> str:
    return "left" if float(block.get_pose().p[0]) < 0 else "right"


def _episode_instruction_info(task_name: str, env: Any) -> dict[str, str] | None:
    if task_name == "beat_block_hammer":
        block_x = float(env.block.get_functional_point(0, "pose").p[0])
        return {
            "{A}": "020_hammer/base0",
            "{a}": "left" if block_x < 0 else "right",
        }
    if task_name == "blocks_ranking_rgb":
        return {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    if task_name == "blocks_ranking_size":
        return {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    return None


def _generate_official_instruction(
    task_name: str, env: Any, episode_info: dict[str, Any] | None = None
) -> str:
    fallback = task_name.replace("_", " ")
    instruction_info = None
    if isinstance(episode_info, dict):
        candidate = episode_info.get("info")
        if isinstance(candidate, dict):
            instruction_info = candidate
    if instruction_info is None:
        instruction_info = _episode_instruction_info(task_name, env)
    if instruction_info is None:
        return fallback
    try:
        from description.utils.generate_episode_instructions import generate_episode_descriptions
    except Exception:
        logger.warning(
            "Could not load RoboTwin instruction generator; using %r.", fallback, exc_info=True
        )
        return fallback

    instruction_type = os.environ.get(OFFICIAL_INSTRUCTION_TYPE_ENV, "seen")
    try:
        maximum = int(os.environ.get(OFFICIAL_INSTRUCTION_MAX_ENV, "100"))
    except ValueError:
        maximum = 100
    results = generate_episode_descriptions(task_name, [instruction_info], max_descriptions=maximum)
    if not results:
        return fallback
    options = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen")
    return str(np.random.choice(options)) if options else fallback


def _load_setup_kwargs(task_name: str) -> dict[str, Any]:
    task_config = os.environ.get(TASK_CONFIG_ENV, "demo_clean")
    if task_config not in SUPPORTED_TASK_CONFIGS:
        raise ValueError(
            f"Unsupported RoboTwin task config {task_config!r}; "
            f"expected one of {SUPPORTED_TASK_CONFIGS}."
        )
    cache_key = (task_name, task_config)
    if cache_key in _SETUP_CACHE:
        return dict(_SETUP_CACHE[cache_key])

    from envs import CONFIGS_PATH

    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as file:
        args = yaml.safe_load(file)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as file:
        embodiment_types = yaml.safe_load(file)

    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) == 1:
        robot_file = embodiment_types[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"Unexpected RoboTwin embodiment: {embodiment}")

    with open(os.path.join(args["left_robot_file"], "config.yml"), encoding="utf-8") as file:
        args["left_embodiment_config"] = yaml.safe_load(file)
    with open(os.path.join(args["right_robot_file"], "config.yml"), encoding="utf-8") as file:
        args["right_embodiment_config"] = yaml.safe_load(file)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as file:
        camera_config = yaml.safe_load(file)

    head_camera = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera]["h"]
    args["head_camera_w"] = camera_config[head_camera]["w"]
    args["render_freq"] = 0
    args["eval_mode"] = True
    args["task_name"] = task_name
    args["task_config"] = task_config
    _SETUP_CACHE[cache_key] = args
    return dict(args)


def _load_task(task_name: str) -> type:
    try:
        module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Could not import RoboTwin task {task_name!r}. Run evaluation from the RoboTwin root "
            "and include that directory in PYTHONPATH."
        ) from error
    task_class = getattr(module, task_name, None)
    if task_class is None:
        raise AttributeError(f"Task class {task_name!r} not found in envs/{task_name}.py")
    return task_class


class RoboTwinEnv(gym.Env):
    """Gymnasium wrapper for the official RoboTwin 2.0 task API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        observation_height: int = DEFAULT_CAMERA_H,
        observation_width: int = DEFAULT_CAMERA_W,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
    ):
        super().__init__()
        self.task_name = task_name
        self.task = task_name
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self._reset_stride = n_envs
        self.camera_names = list(camera_names)
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.episode_length = episode_length
        self._max_episode_steps = episode_length
        self._env: Any | None = None
        self._step_count = 0
        self._next_seed = episode_index
        self._episode_count = 0
        self._policy_scene_active = False
        self._skip_next_observation = False
        self._last_observation: RobotObservation | None = None
        self._black_frame = np.zeros((observation_height, observation_width, 3), dtype=np.uint8)

        image_spaces = {
            camera: spaces.Box(
                low=0,
                high=255,
                shape=(observation_height, observation_width, 3),
                dtype=np.uint8,
            )
            for camera in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_spaces),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ACTION_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = _load_task(self.task_name)()

    def _set_skip_next_observation(self, skip: bool) -> None:
        """Ask the next step() call to return the cached observation.

        This is used during chunked-policy evaluation. pi0/pi0.5 only consumes
        RGB observations when it predicts a new action chunk; actions popped
        from the internal queue do not depend on the observation passed to
        select_action(). Skipping RoboTwin get_obs() within a chunk avoids the
        expensive SAPIEN RGB render, matching RoboTwin/FastWAM's
        skip_get_obs_within_replan behavior.
        """
        self._skip_next_observation = bool(skip) and _env_flag(SKIP_OBS_WITHIN_CHUNK_ENV)

    def _get_obs(self) -> RobotObservation:
        assert self._env is not None
        raw = self._env.get_obs()
        camera_data = raw.get("observation", {})
        images = {}
        for camera in self.camera_names:
            value = camera_data.get(camera)
            image = value.get("rgb") if value else None
            images[camera] = (
                self._black_frame if image is None else np.asarray(image, dtype=np.uint8)[..., :3]
            )

        vector = (raw.get("joint_action") or {}).get("vector")
        if vector is None:
            state = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            array = np.asarray(vector, dtype=np.float32).ravel()
            state = (
                array[:ACTION_DIM]
                if array.size >= ACTION_DIM
                else np.zeros(ACTION_DIM, dtype=np.float32)
            )
        observation = {"pixels": images, "agent_pos": state}
        self._last_observation = observation
        return observation

    def _close_task_env(self, *, clear_cache: bool = False) -> None:
        assert self._env is not None
        if hasattr(self._env, "close_env"):
            with contextlib.suppress(Exception):
                self._env.close_env(clear_cache=clear_cache)

    def _finish_policy_episode(self) -> None:
        if not self._policy_scene_active:
            return
        frequency = int(
            os.environ.get(CLEAR_CACHE_FREQ_ENV, _load_setup_kwargs(self.task_name).get("clear_cache_freq", 0))
        )
        if frequency < 0:
            raise ValueError(f"{CLEAR_CACHE_FREQ_ENV} must be nonnegative.")
        clear_cache = frequency > 0 and self._episode_count > 0 and self._episode_count % frequency == 0
        self._close_task_env(clear_cache=clear_cache)
        self._policy_scene_active = False

    def _setup_official_episode(self, requested_seed: int) -> tuple[int, dict[str, Any]]:
        """Select and recreate a seed using RoboTwin's official expert-check protocol."""
        assert self._env is not None
        try:
            max_attempts = int(os.environ.get(EXPERT_CHECK_MAX_ATTEMPTS_ENV, "1000"))
        except ValueError:
            max_attempts = 1000
        if max_attempts <= 0:
            raise ValueError(f"{EXPERT_CHECK_MAX_ATTEMPTS_ENV} must be positive.")

        candidate_seed = max(requested_seed, self._next_seed)
        last_error: Exception | None = None
        for _ in range(max_attempts):
            setup_kwargs = _load_setup_kwargs(self.task_name)
            setup_kwargs.update(
                seed=candidate_seed,
                is_test=True,
                eval_mode=True,
                now_ep_num=self._episode_count,
                render_freq=0,
            )
            try:
                # This expert rollout is part of RoboTwin's evaluation protocol. Besides
                # rejecting invalid scenes, some tasks initialize success-check fields
                # (for example arm_tag and origin_z) inside play_once().
                with torch.enable_grad():
                    self._env.setup_demo(**setup_kwargs)
                    episode_info = self._env.play_once()
                plan_success = bool(getattr(self._env, "plan_success", False))
                expert_success = plan_success and bool(self._env.check_success())
            except Exception as error:
                last_error = error
                self._close_task_env()
                if _env_flag(FAIL_FAST_ENV) or _is_non_retryable_setup_error(error):
                    raise
                logger.warning(
                    "Skipping invalid RoboTwin expert seed %d for task %s: %s",
                    candidate_seed,
                    self.task_name,
                    error,
                )
                # A failed setup may leave a partially initialized robot behind.
                self._env = _load_task(self.task_name)()
                candidate_seed += self._reset_stride
                continue

            self._close_task_env()
            if not expert_success:
                logger.warning(
                    "Skipping RoboTwin seed %d for task %s because the expert did not succeed.",
                    candidate_seed,
                    self.task_name,
                )
                candidate_seed += self._reset_stride
                continue

            # Recreate exactly the same validated scene for policy evaluation, as done
            # by RoboTwin's script/eval_policy.py.
            with torch.enable_grad():
                self._env.setup_demo(**setup_kwargs)
            self._policy_scene_active = True
            return candidate_seed, episode_info

        message = (
            f"Could not find a valid expert seed for RoboTwin task {self.task_name!r} "
            f"after {max_attempts} attempts starting at seed {requested_seed}."
        )
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def reset(self, seed: int | None = None, **kwargs) -> tuple[RobotObservation, dict]:
        self._ensure_env()
        self._finish_policy_episode()
        super().reset(seed=seed)
        assert self._env is not None
        requested_seed = self.episode_index if seed is None else int(seed)
        actual_seed, episode_info = self._setup_official_episode(requested_seed)
        self.episode_index = actual_seed + self._reset_stride
        self._next_seed = self.episode_index
        self._episode_count += 1
        self._step_count = 0
        self._skip_next_observation = False

        if _env_flag(OFFICIAL_INSTRUCTION_ENV, default=True):
            self.task_description = _generate_official_instruction(
                self.task_name, self._env, episode_info
            )
            if hasattr(self._env, "set_instruction"):
                self._env.set_instruction(instruction=self.task_description)
            logger.info(
                "RoboTwin instruction | task=%s | %s", self.task_name, self.task_description
            )
        else:
            self.task_description = self.task_name.replace("_", " ")
        observation = self._get_obs()
        return observation, {
            "is_success": False,
            "task": self.task_name,
            "seed": actual_seed,
        }

    def step(self, action: np.ndarray):
        assert self._env is not None
        action = np.asarray(action)
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")
        with torch.enable_grad():
            self._env.take_action(action)
        self._step_count += 1

        success = bool(getattr(self._env, "eval_success", False))
        if not success and hasattr(self._env, "check_success"):
            success = bool(self._env.check_success())
        terminated = success
        truncated = self._step_count >= self.episode_length
        info = {"task": self.task_name, "is_success": success, "step": self._step_count}
        if self._skip_next_observation and not (terminated or truncated) and self._last_observation is not None:
            observation = self._last_observation
        else:
            observation = self._get_obs()
        self._skip_next_observation = False
        if terminated or truncated:
            info["final_info"] = {"task": self.task_name, "is_success": success}
            self._finish_policy_episode()
        return observation, float(success), terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        # The terminal scene is closed after caching its final observation.
        if not self._policy_scene_active and self._last_observation is not None:
            observation = self._last_observation
        else:
            observation = self._get_obs()
        images = observation["pixels"]
        if _env_flag(RENDER_ALL_CAMERAS_ENV):
            camera_frames = [images[name] for name in ROBOTWIN_CAMERA_NAMES if name in images]
            if camera_frames:
                return np.concatenate(camera_frames, axis=1)
        return images.get("head_camera", next(iter(images.values())))

    def close(self) -> None:
        if self._env is not None:
            self._finish_policy_episode()
            self._close_task_env(clear_cache=True)
            self._env = None


def _make_env_fns(
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    observation_height: int,
    observation_width: int,
    episode_length: int,
) -> list[Callable[[], RoboTwinEnv]]:
    def make_one(episode_index: int) -> RoboTwinEnv:
        return RoboTwinEnv(
            task_name=task_name,
            episode_index=episode_index,
            n_envs=n_envs,
            camera_names=camera_names,
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
        )

    return [partial(make_one, index) for index in range(n_envs)]


def create_robotwin_envs(
    task: str,
    n_envs: int,
    env_cls: Callable = gym.vector.SyncVectorEnv,
    camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
    observation_height: int = DEFAULT_CAMERA_H,
    observation_width: int = DEFAULT_CAMERA_W,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    task_names = [name.strip() for name in task.split(",") if name.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboTwin task name.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"`n_envs` must be a positive integer, got {n_envs!r}.")
    if not callable(env_cls):
        raise ValueError("`env_cls` must be a callable vector environment constructor.")
    unknown = [name for name in task_names if name not in ROBOTWIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown RoboTwin tasks: {unknown}")

    result = defaultdict(dict)
    for task_name in task_names:
        functions = _make_env_fns(
            task_name,
            n_envs,
            list(camera_names),
            observation_height,
            observation_width,
            episode_length,
        )
        result[task_name][0] = env_cls(functions)
        logger.info("Created RoboTwin task=%s, n_envs=%d", task_name, n_envs)
    return {key: dict(value) for key, value in result.items()}
