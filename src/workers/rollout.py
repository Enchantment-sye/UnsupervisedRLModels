
from collections import defaultdict
import logging
import time
import numpy as np
import functools
import torch
from utils import utils
from data_structs.trajectory_batch import TrajectoryBatch
from envs.isaaclab.viewer_runtime import reapply_active_viewer_preset, warmup_render_capture
from safety import build_safety_controller

_LOG = logging.getLogger(__name__)

class SkillRolloutWorker:
    def __init__(
            self,
            seed,
            time_limit,
            cur_extra_keys,
            pixeled = False,
            config = None,
    ):
        self._observations = []
        self._last_observations = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._lengths = []
        self._agent_infos = defaultdict(list)
        self._env_infos = defaultdict(list)
        self._prev_obs = None
        self._video_frame = None
        self._last_valid_video_frame = None
        self._pending_render_bootstrap = False
        self._path_length = 0
        self._time_limit_override = None
        self._timing_totals = {
            "TimeSamplingEnv": 0.0,
            "TimeImagePostprocess": 0.0,
        }
        self._cur_extra_keys = cur_extra_keys
        self._render = False
        self._deterministic_policy = None
        self._seed = seed
        self._time_limit = time_limit
        self.pixeled = pixeled
        self._config = config
        self._safety_controller = None
        self._prev_safe_action = None
        self._current_timestep_info = {}
        self.worker_init()

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            utils.set_seed_everywhere(self._seed)

    def get_attrs(self, keys):
        attr_dict = {}
        for key in keys:
            attr_dict[key] = functools.reduce(getattr, [self] + key.split('.'))
        return attr_dict

    def start_rollout(self, env, policy, deterministic_policy=False):
        """Begin a new rollout."""
        self._path_length = 0
        self._last_valid_video_frame = None
        self._pending_render_bootstrap = False
        reset_started = time.perf_counter()
        timestep = env.reset()
        self._timing_totals["TimeSamplingEnv"] += (time.perf_counter() - reset_started)
        # gymnasium: (obs, info)
        if isinstance(timestep, tuple) and len(timestep) >= 1:
            timestep = timestep[0]

        if not isinstance(timestep, dict):
            raise TypeError(f"env.reset() must return dict-like obs, got: {type(timestep)}")

        train_image_tensor = getattr(env, "get_train_image_tensor", lambda: None)()
        if self.pixeled and train_image_tensor is not None:
            self._prev_obs = train_image_tensor
        elif self.pixeled and "image" in timestep:
            self._prev_obs = timestep["image"]
        elif not self.pixeled and "state" in timestep:
            self._prev_obs = timestep["state"]
        else:
            raise KeyError("reset obs must contain key 'image' or 'state'")
        self._pixeled_obs = timestep["image"]
        self._video_frame = self._pixeled_obs
        self._current_timestep_info = timestep.get("info", {}) or {}
        self._prev_safe_action = None
        self._accumulate_env_timing_metrics(env)

        self._prev_extra = None

        policy.reset()
        policy._force_use_mode_actions = deterministic_policy

    def _clone_frame(self, frame):
        if frame is None:
            return None
        if torch.is_tensor(frame):
            return frame.detach().cpu().numpy().copy()
        return np.asarray(frame).copy()

    def _default_record_frame(self, *, state_record_pixeled=False):
        if state_record_pixeled:
            return self._clone_frame(self._pixeled_obs)
        return self._clone_frame(self._prev_obs)

    def _capture_record_frame(self, env, *, state_record_pixeled=False, video_frame_source=None):
        if not state_record_pixeled:
            return self._clone_frame(self._prev_obs)
        if video_frame_source is not None and hasattr(env, "capture_video_frame"):
            return env.capture_video_frame(source=video_frame_source)
        return self._clone_frame(self._pixeled_obs)

    def _resolve_record_frame(self, env, *, state_record_pixeled=False, video_frame_source=None):
        frame = self._capture_record_frame(
            env,
            state_record_pixeled=state_record_pixeled,
            video_frame_source=video_frame_source,
        )
        if frame is not None:
            frame = self._clone_frame(frame)
            self._last_valid_video_frame = self._clone_frame(frame)
            return frame
        if self._last_valid_video_frame is not None:
            return self._clone_frame(self._last_valid_video_frame)
        return self._default_record_frame(state_record_pixeled=state_record_pixeled)

    def _bootstrap_video_frame(self, env, *, state_record_pixeled=False, video_frame_source=None):
        if (
            state_record_pixeled
            and isinstance(video_frame_source, str)
            and video_frame_source.lower() == "render"
            and hasattr(env, "capture_video_frame")
        ):
            reapply_active_viewer_preset(env)
            frame, stabilized = warmup_render_capture(
                lambda: self._capture_record_frame(
                    env,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                ),
            )
            if frame is not None:
                if stabilized:
                    frame = self._clone_frame(frame)
                    self._last_valid_video_frame = self._clone_frame(frame)
                    return frame
                self._pending_render_bootstrap = True
                _LOG.warning(
                    "IsaacLab render video did not stabilize within %d attempts after reset; "
                    "deferring video bootstrap until after the first environment step.",
                    10,
                )
                return None
            self._pending_render_bootstrap = True
            return None
        else:
            frame = self._capture_record_frame(
                env,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            )
            if frame is not None:
                frame = self._clone_frame(frame)
                self._last_valid_video_frame = self._clone_frame(frame)
                return frame

        if self._last_valid_video_frame is not None:
            return self._clone_frame(self._last_valid_video_frame)
        return self._default_record_frame(state_record_pixeled=state_record_pixeled)

    def _complete_pending_render_bootstrap(self, env, *, state_record_pixeled=False, video_frame_source=None):
        if not self._pending_render_bootstrap:
            return self._video_frame
        frame, stabilized = warmup_render_capture(
            lambda: self._capture_record_frame(
                env,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            ),
        )
        self._pending_render_bootstrap = False
        if frame is not None:
            frame = self._clone_frame(frame)
            self._last_valid_video_frame = self._clone_frame(frame)
            if not stabilized:
                _LOG.warning(
                    "IsaacLab render video bootstrap was still unstable after the first step; "
                    "using the last non-black frame from the post-step warmup.",
                )
            return frame
        return self._resolve_record_frame(
            env,
            state_record_pixeled=state_record_pixeled,
            video_frame_source=video_frame_source,
        )

    def step_rollout(self, env, policy, extra=None, state_record_pixeled = False, video_frame_source=None):
        """Take a single time-step in the current rollout."""
        cur_time_limit = self._time_limit if self._time_limit_override is None else self._time_limit_override

        if extra is None:
            extra = {}
        if self._path_length < cur_time_limit:
            cur_extra_key = 'skill' if 'skill' in self._cur_extra_keys else None

            if cur_extra_key is None:
                agent_input = self._prev_obs
            else:
                cur_extra = extra[cur_extra_key]
                if torch.is_tensor(self._prev_obs):
                    agent_input = utils.get_torch_concat_obs(self._prev_obs, cur_extra, dim=1)
                else:
                    agent_input = utils.get_np_concat_obs(self._prev_obs, cur_extra)
                self._prev_extra = cur_extra

            raw_action, agent_info = policy.get_action(agent_input)
            safe_action, safety_report = self._filter_action_if_needed(
                env,
                raw_action,
                agent_input=agent_input,
            )

            # 有的 env 期望 dict action，有的直接吃 ndarray
            step_started = time.perf_counter()
            try:
                timestep = env.step({"action": safe_action})
            except Exception:
                timestep = env.step(safe_action)
            self._timing_totals["TimeSamplingEnv"] += (time.perf_counter() - step_started)
            # gymnasium: (obs, reward, terminated, truncated, info)
            timestep["obs"] = timestep["state"] if not self.pixeled else timestep["image"]
            if isinstance(timestep, tuple) and len(timestep) == 5:
                obs, reward, terminated, truncated, info = timestep
                is_terminal = bool(terminated or truncated)

                if isinstance(obs, dict):
                    next_obs = obs.get("image") if self.pixeled else obs.get("state", obs)
                else:
                    next_obs = obs

                timestep = {
                    "reward": reward,
                    "is_terminal": is_terminal,
                    "info": info if info is not None else {},
                    "obs": next_obs,
                }
                if state_record_pixeled:
                    timestep.update({
                        "image": obs.get("image")
                    })
            train_image_tensor = getattr(env, "get_train_image_tensor", lambda: None)()
            if self.pixeled and train_image_tensor is not None:
                self._prev_obs = train_image_tensor
            else:
                self._prev_obs = timestep.get("obs", self._prev_obs)
            self._pixeled_obs = timestep.get("image", self._pixeled_obs)
            self._accumulate_env_timing_metrics(env)
            if self._video_frame is None:
                self._video_frame = self._complete_pending_render_bootstrap(
                    env,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                )

            # --- record ---
            self._observations.append(self._video_frame)
            self._rewards.append(timestep.get('reward', 0.0))
            self._actions.append(safe_action)

            agent_info = dict(agent_info)
            if safety_report is not None:
                agent_info["raw_action"] = np.asarray(raw_action, dtype=np.float32).copy()
                agent_info["safe_action"] = np.asarray(safe_action, dtype=np.float32).copy()
                agent_info["safety_correction_norm"] = np.float32(
                    np.linalg.norm(np.asarray(safe_action, dtype=np.float32) - np.asarray(raw_action, dtype=np.float32))
                )

            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k in self._cur_extra_keys:
                self._agent_infos[k].append(extra[k])
            info_dict = timestep.get('info', {}) or {}
            self._current_timestep_info = info_dict
            if safety_report is not None:
                info_dict = {
                    **info_dict,
                    **safety_report.to_env_info(),
                }
            for k, v in info_dict.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(bool(timestep.get('is_terminal', False)))

            self._video_frame = self._resolve_record_frame(
                env,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            )
            if not self._terminals[-1]:
                return False
        self._terminals[-1] = True
        self._lengths.append(self._path_length)
        self._last_observations.append(self._video_frame)
        return True

    def _filter_action_if_needed(self, env, raw_action, *, agent_input):
        controller = self._get_safety_controller(env)
        if controller is None:
            return raw_action, None
        safety_state = self._current_safety_state(env)
        denorm = getattr(env, "safety_denormalize_action", None)
        norm = getattr(env, "safety_normalize_action", None)
        safe_action, report = controller.filter_action(
            raw_action=raw_action,
            safety_state=safety_state,
            policy_obs=agent_input,
            prev_action=self._prev_safe_action,
            action_to_physical=denorm if callable(denorm) else None,
            action_from_physical=norm if callable(norm) else None,
        )
        self._prev_safe_action = np.asarray(safe_action, dtype=np.float32).copy()
        return safe_action, report

    def _get_safety_controller(self, env):
        if self._safety_controller is not None:
            return self._safety_controller
        if self._config is None:
            return None
        self._safety_controller = build_safety_controller(self._config, env=env, logger=_LOG)
        return self._safety_controller

    def _current_safety_state(self, env):
        info = self._current_timestep_info or {}
        if isinstance(info, dict) and "safety_state" in info:
            return info.get("safety_state")
        getter = getattr(env, "get_safety_state", None)
        if callable(getter):
            return getter()
        return None

    def consume_timing_metrics(self):
        metrics = dict(self._timing_totals)
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
        return metrics

    def _accumulate_env_timing_metrics(self, env):
        try:
            consume = getattr(env, "consume_timing_metrics", None)
        except (AttributeError, ValueError):
            return
        if not callable(consume):
            return
        metrics = consume() or {}
        for key, value in metrics.items():
            self._timing_totals[key] = self._timing_totals.get(key, 0.0) + float(value)

    def collect_rollout(self, env):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(env.spec, np.asarray(observations),
                               np.asarray(last_observations),
                               np.asarray(actions), np.asarray(rewards),
                               np.asarray(terminals), dict(env_infos),
                               dict(agent_infos), np.asarray(lengths,
                                                             dtype='i'))

    def rollout(
        self,
        env,
        policy,
        extra=None,
        deterministic_policy=False,
        state_record_pixeled=False,
        video_frame_source=None,
    ):
        """Sample a single rollout of the agent in the environment.
        Params:
            extra: {'skill': skill}
        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout(env, policy, deterministic_policy=deterministic_policy)
        self._video_frame = self._bootstrap_video_frame(
            env,
            state_record_pixeled=state_record_pixeled,
            video_frame_source=video_frame_source,
        )
        while not self.step_rollout(
            env,
            policy,
            extra,
            state_record_pixeled=state_record_pixeled,
            video_frame_source=video_frame_source,
        ):
            pass
        return self.collect_rollout(env)
