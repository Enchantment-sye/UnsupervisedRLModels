from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from core.stage_contract import uses_skill_inputs
from utils import utils

from .adapters.single_agent_box import _GALAXEA_STATE_KEY_ORDER
from .adapters.vision import ImageAcquisitionError, IsaacLabImageProvider, is_image_array


class IsaacLabParallelTrajectoryCollector:
    def __init__(self, cfg, env_handles):
        self.cfg = cfg
        self._raw_env = env_handles["raw_env"]
        self._task_spec = env_handles["task_spec"]
        self._request = env_handles["request"]
        self._action_low = np.asarray(env_handles["action_low"], dtype=np.float32)
        self._action_high = np.asarray(env_handles["action_high"], dtype=np.float32)
        self._action_mask = np.isfinite(self._action_low) & np.isfinite(self._action_high)
        self._num_envs = int(self._request.num_envs)
        self._image_provider = IsaacLabImageProvider(self._raw_env, self._request, self._task_spec)
        self._timing_totals = {
            "TimeSamplingEnv": 0.0,
            "TimeImagePostprocess": 0.0,
        }

    def collect(self, policy, *, target_num_trajectories, sample_extra_fn):
        paths = []
        extras = [sample_extra_fn() for _ in range(self._num_envs)]
        buffers = [self._new_slot_buffer() for _ in range(self._num_envs)]
        path_lengths = np.zeros(self._num_envs, dtype=np.int32)

        reset_started = time.perf_counter()
        reset_output = self._raw_env.reset(seed=int(self.cfg.seed))
        self._timing_totals["TimeSamplingEnv"] += (time.perf_counter() - reset_started)
        obs_dict, extras_dict = self._normalize_reset_output(reset_output)
        current_policy_obs, current_record_obs = self._extract_batches(obs_dict, extras_dict)

        while len(paths) < target_num_trajectories:
            agent_input = current_policy_obs
            if self._uses_skill_inputs():
                stacked_skills = np.stack([np.asarray(extra["skill"], dtype=np.float32) for extra in extras], axis=0)
                if torch is not None and torch.is_tensor(agent_input):
                    agent_input = utils.get_torch_concat_obs(agent_input, stacked_skills, dim=1)
                else:
                    agent_input = utils.get_np_concat_obs(agent_input, stacked_skills)

            actions, agent_infos = policy.get_actions(agent_input)
            env_action = self._denormalize_actions(actions)

            step_started = time.perf_counter()
            step_output = self._raw_env.step(self._prepare_action_tensor(env_action))
            self._timing_totals["TimeSamplingEnv"] += (time.perf_counter() - step_started)
            if not isinstance(step_output, tuple) or len(step_output) != 5:
                raise TypeError(
                    "Isaac Lab vector step must return (obs_dict, reward, terminated, truncated, extras)."
                )
            next_obs_dict, reward, terminated, truncated, step_extras = step_output

            reward = self._to_numpy(reward).reshape(-1)
            terminated = self._to_numpy(terminated).reshape(-1).astype(bool)
            truncated = self._to_numpy(truncated).reshape(-1).astype(bool)
            done_mask = np.logical_or(terminated, truncated)

            path_lengths += 1
            time_limit_done = path_lengths >= int(self.cfg.time_limit)
            manual_reset_ids = np.where(np.logical_and(time_limit_done, ~done_mask))[0]
            done_mask = np.logical_or(done_mask, time_limit_done)

            if manual_reset_ids.size > 0:
                self._reset_slots(manual_reset_ids)
                next_obs_dict = self._raw_env.unwrapped._get_observations()
                step_extras = {}

            next_policy_obs, next_record_obs = self._extract_batches(next_obs_dict, step_extras)

            for slot in range(self._num_envs):
                self._append_slot_step(
                    buffers[slot],
                    current_record_obs[slot],
                    actions[slot],
                    reward[slot],
                    done_mask[slot],
                    agent_infos,
                    slot,
                    extra=extras[slot],
                )
                if done_mask[slot]:
                    paths.append(self._finalize_slot_buffer(buffers[slot], next_record_obs[slot]))
                    buffers[slot] = self._new_slot_buffer()
                    path_lengths[slot] = 0
                    extras[slot] = sample_extra_fn()
                    if len(paths) >= target_num_trajectories:
                        break

            current_policy_obs = next_policy_obs
            current_record_obs = next_record_obs

        return paths[:target_num_trajectories]

    def consume_timing_metrics(self):
        metrics = dict(self._timing_totals)
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
        return metrics

    def _extract_batches(self, obs_dict, extras):
        state_batch = self._extract_state_batch(obs_dict)
        if not self.cfg.encoder:
            return state_batch, state_batch

        started = time.perf_counter()
        try:
            image_tensor = self._image_provider.capture_tensor(
                obs_dict,
                extras,
                allow_placeholder=not self.cfg.encoder,
                batched=True,
            )
        except ImageAcquisitionError:
            image_tensor = self._image_provider.placeholder_image_tensor()

        if image_tensor is None:
            record_obs = np.repeat(
                self._image_provider.placeholder_image()[None, ...],
                self._num_envs,
                axis=0,
            )
            self._timing_totals["TimeImagePostprocess"] += (time.perf_counter() - started)
            return record_obs, record_obs

        if torch is not None and torch.is_tensor(image_tensor):
            record_obs = image_tensor.detach().cpu().numpy()
        else:
            record_obs = np.asarray(image_tensor)
        self._timing_totals["TimeImagePostprocess"] += (time.perf_counter() - started)
        return image_tensor, record_obs

    def _extract_state_batch(self, obs_dict):
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
        else:
            policy_obs = obs_dict

        chunks = []
        self._collect_state_chunks(policy_obs, chunks)
        if not chunks:
            array = self._to_numpy(policy_obs)
            return array.astype(np.float32).reshape(array.shape[0], -1)
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def _collect_state_chunks(self, value, chunks):
        if isinstance(value, dict):
            for key in self._ordered_state_keys(value):
                if self._should_skip_state_key(key):
                    continue
                self._collect_state_chunks(value[key], chunks)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                self._collect_state_chunks(nested, chunks)
            return
        array = self._to_numpy(value)
        if is_image_array(array):
            return
        array = array.astype(np.float32)
        if array.ndim == 1:
            array = array[:, None]
        else:
            array = array.reshape(array.shape[0], -1)
        chunks.append(array)

    def _ordered_state_keys(self, mapping):
        keys = list(mapping.keys())
        preferred = [key for key in _GALAXEA_STATE_KEY_ORDER if key in mapping]
        remaining = sorted(key for key in keys if key not in preferred)
        return [*preferred, *remaining]

    @staticmethod
    def _looks_visual_key(key: str) -> bool:
        lowered = key.lower()
        return any(token in lowered for token in ("rgb", "image", "pixel", "camera", "depth", "segmentation"))

    def _should_skip_state_key(self, key: str) -> bool:
        lowered = key.lower()
        if lowered in ("last_joints", "last_joint", "previous_joints"):
            return True
        return self._looks_visual_key(key)

    def _append_slot_step(self, buffer, obs, action, reward, done, agent_infos, slot, *, extra):
        buffer["observations"].append(self._copy_obs(obs))
        buffer["actions"].append(np.asarray(action, dtype=np.float32).copy())
        buffer["rewards"].append(float(reward))
        buffer["dones"].append(bool(done))
        for key, value in agent_infos.items():
            buffer["agent_infos"][key].append(np.asarray(value[slot]).copy())
        if extra is not None and "skill" in extra:
            buffer["agent_infos"]["skill"].append(np.asarray(extra["skill"], dtype=np.float32).copy())

    def _finalize_slot_buffer(self, buffer, last_observation):
        observations = np.asarray(buffer["observations"])
        actions = np.asarray(buffer["actions"], dtype=np.float32)
        rewards = np.asarray(buffer["rewards"], dtype=np.float32)
        dones = np.asarray(buffer["dones"], dtype=bool)
        agent_infos = {key: np.asarray(value) for key, value in buffer["agent_infos"].items()}
        return {
            "observations": observations,
            "next_observations": np.concatenate([observations[1:], [self._copy_obs(last_observation)]], axis=0),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "agent_infos": agent_infos,
            "env_infos": {},
        }

    @staticmethod
    def _new_slot_buffer():
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "agent_infos": defaultdict(list),
        }

    @staticmethod
    def _copy_obs(obs):
        if torch is not None and torch.is_tensor(obs):
            return obs.detach().cpu().numpy().copy()
        return np.asarray(obs).copy()

    def _denormalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        orig = (actions + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low
        return np.where(self._action_mask, orig, actions)

    def _prepare_action_tensor(self, action):
        if torch is not None:
            device = getattr(getattr(self._raw_env, "unwrapped", self._raw_env), "device", None)
            return torch.as_tensor(action, dtype=torch.float32, device=device)
        return action

    def _reset_slots(self, slot_ids):
        if slot_ids.size == 0:
            return
        env = getattr(self._raw_env, "unwrapped", self._raw_env)
        env_ids = torch.as_tensor(slot_ids, dtype=torch.long, device=getattr(env, "device", None))
        env._reset_idx(env_ids)

    @staticmethod
    def _normalize_reset_output(reset_output):
        if isinstance(reset_output, tuple) and len(reset_output) >= 2:
            return reset_output[0], reset_output[1] or {}
        return reset_output, {}

    @staticmethod
    def _to_numpy(value):
        if torch is not None and torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _uses_skill_inputs(self):
        return uses_skill_inputs(self.cfg)
