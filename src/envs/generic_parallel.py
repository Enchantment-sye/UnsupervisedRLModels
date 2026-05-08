from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from core.stage_contract import uses_skill_inputs
from envs.wrappers import Async
from utils import utils


class GenericProcessTrajectoryCollector:
    """Process-parallel collector for standard URL-style single-agent envs."""

    def __init__(self, cfg, *, num_workers: int | None = None):
        self.cfg = cfg
        requested_workers = int(num_workers if num_workers is not None else getattr(cfg, "n_parallel", 1))
        self._num_workers = max(1, requested_workers)
        self._workers = [
            Async(self._make_constructor(worker_id), strategy="process")
            for worker_id in range(self._num_workers)
        ]
        self._timing_totals = self._new_timing_totals()

    def collect(self, policy, *, target_num_trajectories, sample_extra_fn):
        start = time.perf_counter()
        old_force_mode = getattr(policy, "_force_use_mode_actions", None)
        if old_force_mode is not None:
            policy._force_use_mode_actions = False
        if hasattr(policy, "reset"):
            policy.reset()
        try:
            paths = self._collect_streaming(policy, int(target_num_trajectories), sample_extra_fn)
        finally:
            if old_force_mode is not None:
                policy._force_use_mode_actions = old_force_mode
            self._timing_totals["TimeParallelSampler"] += time.perf_counter() - start
        return paths

    def collect_fixed(
            self,
            policy,
            *,
            extras,
            deterministic_policy: bool,
            state_record_pixeled: bool = False,
            video_frame_source=None,
            reset_perturbations=None):
        extras = list(extras)
        if not extras:
            return []
        reset_perturbations = self._normalize_reset_perturbations(reset_perturbations, len(extras))

        start = time.perf_counter()
        had_force_mode = hasattr(policy, "_force_use_mode_actions")
        old_force_mode = getattr(policy, "_force_use_mode_actions", None)
        if hasattr(policy, "reset"):
            policy.reset()
        policy._force_use_mode_actions = bool(deterministic_policy)
        try:
            paths = self._collect_fixed_impl(
                policy,
                extras=extras,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
                reset_perturbations=reset_perturbations,
            )
        finally:
            if had_force_mode:
                policy._force_use_mode_actions = old_force_mode
            else:
                delattr(policy, "_force_use_mode_actions")
            self._timing_totals["TimeParallelSampler"] += time.perf_counter() - start
        return paths

    def consume_timing_metrics(self):
        metrics = dict(self._timing_totals)
        metrics["ParallelSamplerNumWorkers"] = float(self._num_workers)
        self._timing_totals = self._new_timing_totals()
        return metrics

    def close(self):
        for worker in self._workers:
            worker.close()

    def _collect_streaming(self, policy, target_num_trajectories, sample_extra_fn):
        paths = []
        active_slots = list(range(min(self._num_workers, max(1, int(target_num_trajectories)))))
        extras = [None for _ in range(self._num_workers)]
        for slot in active_slots:
            extras[slot] = sample_extra_fn()
        buffers = [self._new_slot_buffer() for _ in range(self._num_workers)]
        path_lengths = np.zeros(self._num_workers, dtype=np.int32)

        timesteps = self._reset_slots(active_slots)
        current_policy_obs = [None for _ in range(self._num_workers)]
        current_record_obs = [None for _ in range(self._num_workers)]
        initial_policy_obs = self._extract_policy_obs_batch(timesteps, slots=active_slots)
        for row, slot in enumerate(active_slots):
            current_policy_obs[slot] = initial_policy_obs[row]
            current_record_obs[slot] = initial_policy_obs[row].copy()

        while len(paths) < target_num_trajectories:
            obs_batch = np.stack([np.asarray(current_policy_obs[slot]) for slot in active_slots], axis=0)
            active_extras = [extras[slot] for slot in active_slots]
            actions, agent_infos = self._policy_actions(policy, obs_batch, active_extras)
            timesteps = self._step_slots(active_slots, actions)
            next_policy_obs_batch = self._extract_policy_obs_batch(timesteps, slots=active_slots)
            next_policy_obs = [None for _ in range(self._num_workers)]
            next_record_obs = [None for _ in range(self._num_workers)]
            for row, slot in enumerate(active_slots):
                next_policy_obs[slot] = next_policy_obs_batch[row]
                next_record_obs[slot] = next_policy_obs_batch[row].copy()

            for row, slot in enumerate(active_slots):
                timestep = timesteps[row]
                path_lengths[slot] += 1
                done = self._is_done(timestep) or path_lengths[slot] >= int(getattr(self.cfg, "time_limit", 0) or 0)
                self._append_slot_step(
                    buffers[slot],
                    current_record_obs[slot],
                    actions[row],
                    self._reward(timestep),
                    done,
                    agent_infos,
                    row,
                    extra=extras[slot],
                    env_info=self._env_info(timestep),
                )
                if done:
                    paths.append(self._finalize_slot_buffer(buffers[slot], next_record_obs[slot]))
                    buffers[slot] = self._new_slot_buffer()
                    path_lengths[slot] = 0
                    if len(paths) >= target_num_trajectories:
                        break
                    extras[slot] = sample_extra_fn()
                    reset_timestep = self._reset_slot(slot)
                    next_policy_obs[slot] = self._extract_policy_obs(slot, reset_timestep)
                    next_record_obs[slot] = next_policy_obs[slot].copy()

            for slot in active_slots:
                current_policy_obs[slot] = next_policy_obs[slot]
                current_record_obs[slot] = next_record_obs[slot]

        return paths[:target_num_trajectories]

    def _collect_fixed_impl(self, policy, *, extras, state_record_pixeled: bool, video_frame_source, reset_perturbations):
        total = len(extras)
        paths_by_index = [None for _ in range(total)]
        buffers = [self._new_slot_buffer() for _ in range(self._num_workers)]
        active_extra_indices = [None for _ in range(self._num_workers)]
        current_policy_obs = [None for _ in range(self._num_workers)]
        current_record_obs = [None for _ in range(self._num_workers)]
        path_lengths = np.zeros(self._num_workers, dtype=np.int32)
        next_extra_idx = 0

        for slot in range(self._num_workers):
            if next_extra_idx >= total:
                break
            active_extra_indices[slot] = next_extra_idx
            next_extra_idx += 1
            self._set_next_reset_perturbation(slot, reset_perturbations[active_extra_indices[slot]])
            timestep = self._reset_slot(slot)
            policy_obs = self._extract_policy_obs(slot, timestep)
            current_policy_obs[slot] = policy_obs
            current_record_obs[slot] = self._extract_record_obs(
                slot,
                timestep,
                policy_obs,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            )

        while any(path is None for path in paths_by_index):
            active_slots = [slot for slot, idx in enumerate(active_extra_indices) if idx is not None]
            if not active_slots:
                break

            obs_batch = np.stack([np.asarray(current_policy_obs[slot]) for slot in active_slots], axis=0)
            active_extras = [extras[active_extra_indices[slot]] for slot in active_slots]
            actions, agent_infos = self._policy_actions(policy, obs_batch, active_extras)
            timestep_by_slot = self._step_active_slots(active_slots, actions)

            for row, slot in enumerate(active_slots):
                timestep = timestep_by_slot[slot]
                extra_idx = active_extra_indices[slot]
                next_policy_obs = self._extract_policy_obs(slot, timestep)
                next_record_obs = self._extract_record_obs(
                    slot,
                    timestep,
                    next_policy_obs,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                )
                path_lengths[slot] += 1
                done = self._is_done(timestep) or path_lengths[slot] >= int(getattr(self.cfg, "time_limit", 0) or 0)
                self._append_slot_step(
                    buffers[slot],
                    current_record_obs[slot],
                    actions[row],
                    self._reward(timestep),
                    done,
                    agent_infos,
                    row,
                    extra=extras[extra_idx],
                    env_info=self._env_info(timestep),
                )

                if done:
                    paths_by_index[extra_idx] = self._finalize_slot_buffer(buffers[slot], next_record_obs)
                    buffers[slot] = self._new_slot_buffer()
                    path_lengths[slot] = 0
                    if next_extra_idx < total:
                        active_extra_indices[slot] = next_extra_idx
                        next_extra_idx += 1
                        self._set_next_reset_perturbation(slot, reset_perturbations[active_extra_indices[slot]])
                        reset_timestep = self._reset_slot(slot)
                        policy_obs = self._extract_policy_obs(slot, reset_timestep)
                        current_policy_obs[slot] = policy_obs
                        current_record_obs[slot] = self._extract_record_obs(
                            slot,
                            reset_timestep,
                            policy_obs,
                            state_record_pixeled=state_record_pixeled,
                            video_frame_source=video_frame_source,
                        )
                    else:
                        active_extra_indices[slot] = None
                        current_policy_obs[slot] = None
                        current_record_obs[slot] = None
                    continue

                current_policy_obs[slot] = next_policy_obs
                current_record_obs[slot] = next_record_obs

        return [path for path in paths_by_index if path is not None]

    def _make_constructor(self, worker_id: int):
        cfg = copy.deepcopy(self.cfg)
        cfg.seed = int(getattr(self.cfg, "seed", 0)) + int(worker_id)

        def _construct():
            from envs import make_env

            return make_env(mode="train", config=cfg)

        return _construct

    def _policy_actions(self, policy, obs_batch, extras):
        agent_input = obs_batch
        if uses_skill_inputs(self.cfg):
            stacked_skills = np.stack(
                [np.asarray(extra["skill"], dtype=np.float32) for extra in extras],
                axis=0,
            )
            if torch is not None and torch.is_tensor(agent_input):
                agent_input = utils.get_torch_concat_obs(agent_input, stacked_skills, dim=1)
            else:
                agent_input = utils.get_np_concat_obs(agent_input, stacked_skills)
        actions, agent_infos = policy.get_actions(agent_input)
        return np.asarray(actions, dtype=np.float32), dict(agent_infos)

    def _reset_slots(self, slots):
        started = time.perf_counter()
        promises = [self._workers[slot].reset() for slot in slots]
        timesteps = [promise() for promise in promises]
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - started
        return timesteps

    def _reset_slot(self, slot):
        started = time.perf_counter()
        timestep = self._workers[slot].reset(blocking=True)
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - started
        return timestep

    def _set_next_reset_perturbation(self, slot, perturbation):
        if perturbation is None:
            return
        seed, scale = perturbation
        if float(scale) <= 0.0:
            return
        started = time.perf_counter()
        self._workers[slot].call(
            "set_next_reset_perturbation",
            int(seed),
            float(scale),
        )()
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - started

    def _step_slots(self, slots, actions):
        started = time.perf_counter()
        promises = [
            self._workers[slot].step({"action": np.asarray(actions[row], dtype=np.float32)})
            for row, slot in enumerate(slots)
        ]
        timesteps = [promise() for promise in promises]
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - started
        return timesteps

    def _step_active_slots(self, active_slots, actions):
        started = time.perf_counter()
        promises = {
            slot: self._workers[slot].step({"action": np.asarray(actions[row], dtype=np.float32)})
            for row, slot in enumerate(active_slots)
        }
        timesteps = {slot: promise() for slot, promise in promises.items()}
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - started
        return timesteps

    def _extract_policy_obs_batch(self, timesteps, *, slots=None):
        slots = list(range(len(timesteps))) if slots is None else list(slots)
        started = time.perf_counter()
        batch = np.stack(
            [self._extract_policy_obs(slot, timestep) for slot, timestep in zip(slots, timesteps)],
            axis=0,
        )
        self._timing_totals["TimeImagePostprocess"] += time.perf_counter() - started
        return batch

    def _extract_policy_obs(self, _slot, timestep):
        timestep = self._normalize_timestep(timestep)
        key = "image" if bool(getattr(self.cfg, "encoder", 0)) else "state"
        if key not in timestep:
            key = "obs"
        return np.asarray(timestep[key], dtype=np.uint8 if key == "image" else np.float32).reshape(-1)

    def _extract_record_obs(self, slot, timestep, policy_obs, *, state_record_pixeled: bool, video_frame_source):
        if not state_record_pixeled:
            return np.asarray(policy_obs).copy()
        if video_frame_source is not None:
            try:
                frame = self._workers[slot].call("capture_video_frame", video_frame_source)()
            except Exception:
                frame = None
            if frame is not None:
                return np.asarray(frame).copy()
        timestep = self._normalize_timestep(timestep)
        if "image" in timestep:
            return np.asarray(timestep["image"]).copy()
        return np.asarray(policy_obs).copy()

    def _append_slot_step(self, buffer, obs, action, reward, done, agent_infos, row, *, extra, env_info):
        buffer["observations"].append(np.asarray(obs).copy())
        buffer["actions"].append(np.asarray(action, dtype=np.float32).copy())
        buffer["rewards"].append(float(reward))
        buffer["dones"].append(bool(done))
        for key, value in agent_infos.items():
            buffer["agent_infos"][key].append(np.asarray(value[row]).copy())
        if extra is not None and "skill" in extra:
            buffer["agent_infos"]["skill"].append(np.asarray(extra["skill"], dtype=np.float32).copy())
        for key, value in (env_info or {}).items():
            if key == "state":
                continue
            try:
                buffer["env_infos"][key].append(np.asarray(value).copy())
            except ValueError:
                pass

    def _finalize_slot_buffer(self, buffer, last_observation):
        observations = np.asarray(buffer["observations"])
        actions = np.asarray(buffer["actions"], dtype=np.float32)
        rewards = np.asarray(buffer["rewards"], dtype=np.float32)
        dones = np.asarray(buffer["dones"], dtype=bool)
        agent_infos = {key: np.asarray(value) for key, value in buffer["agent_infos"].items()}
        env_infos = {key: np.asarray(value) for key, value in buffer["env_infos"].items()}
        return {
            "observations": observations,
            "next_observations": np.concatenate(
                [observations[1:], [np.asarray(last_observation).copy()]],
                axis=0,
            ),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "agent_infos": agent_infos,
            "env_infos": env_infos,
        }

    @staticmethod
    def _new_slot_buffer():
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "agent_infos": defaultdict(list),
            "env_infos": defaultdict(list),
        }

    @staticmethod
    def _new_timing_totals():
        return {
            "TimeSamplingEnv": 0.0,
            "TimeImagePostprocess": 0.0,
            "TimeParallelSampler": 0.0,
        }

    @staticmethod
    def _normalize_reset_perturbations(reset_perturbations, expected_length):
        if reset_perturbations is None:
            return [None] * int(expected_length)
        reset_perturbations = list(reset_perturbations)
        if len(reset_perturbations) != int(expected_length):
            raise ValueError(
                f"Expected {expected_length} reset perturbations, got {len(reset_perturbations)}."
            )
        return reset_perturbations

    @staticmethod
    def _normalize_timestep(timestep):
        if isinstance(timestep, tuple) and len(timestep) == 5:
            obs, reward, terminated, truncated, info = timestep
            obs_dict = dict(obs) if isinstance(obs, dict) else {"obs": obs}
            obs_dict.update({
                "reward": reward,
                "is_first": False,
                "is_last": bool(terminated or truncated),
                "is_terminal": bool(terminated),
                "info": info or {},
            })
            return obs_dict
        if isinstance(timestep, tuple) and len(timestep) >= 1:
            timestep = timestep[0]
        if not isinstance(timestep, dict):
            raise TypeError(f"Expected dict-like env timestep, got {type(timestep)!r}")
        return timestep

    @classmethod
    def _reward(cls, timestep):
        return float(cls._normalize_timestep(timestep).get("reward", 0.0))

    @classmethod
    def _is_done(cls, timestep):
        timestep = cls._normalize_timestep(timestep)
        return bool(timestep.get("is_last", False) or timestep.get("is_terminal", False))

    @classmethod
    def _env_info(cls, timestep):
        info = cls._normalize_timestep(timestep).get("info", {}) or {}
        return dict(info) if isinstance(info, dict) else {}
