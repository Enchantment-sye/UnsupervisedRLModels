from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from envs.wrappers import Async
from core.stage_contract import uses_skill_inputs
from utils import utils
from safety import build_safety_controller


class GalaxeaSimProcessTrajectoryCollector:
    """Process-parallel trajectory collector for GalaxeaManipSim single-agent envs."""

    def __init__(self, cfg, *, num_envs: int):
        self.cfg = cfg
        self._num_envs = max(1, int(num_envs))
        self._workers = [
            Async(self._make_constructor(worker_id), strategy="process")
            for worker_id in range(self._num_envs)
        ]
        self._timing_totals = {
            "TimeSamplingEnv": 0.0,
            "TimeImagePostprocess": 0.0,
        }
        self._safety_controllers = [
            build_safety_controller(cfg, env=None)
            for _ in range(self._num_envs)
        ]
        self._prev_safe_actions = [None for _ in range(self._num_envs)]

    def collect(self, policy, *, target_num_trajectories, sample_extra_fn):
        paths = []
        extras = [sample_extra_fn() for _ in range(self._num_envs)]
        buffers = [self._new_slot_buffer() for _ in range(self._num_envs)]

        reset_started = time.perf_counter()
        timesteps = [promise() for promise in [worker.reset() for worker in self._workers]]
        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - reset_started
        current_policy_obs = self._extract_policy_obs_batch(timesteps)
        current_record_obs = current_policy_obs.copy()
        current_safety_states = [self._extract_safety_state(timestep) for timestep in timesteps]

        while len(paths) < target_num_trajectories:
            agent_input = current_policy_obs
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
            safe_actions, safety_reports = self._filter_actions(actions, current_safety_states, agent_input)
            agent_infos = dict(agent_infos)
            if any(report is not None for report in safety_reports):
                agent_infos["raw_action"] = np.asarray(actions, dtype=np.float32).copy()
                agent_infos["safe_action"] = np.asarray(safe_actions, dtype=np.float32).copy()
                agent_infos["safety_correction_norm"] = np.linalg.norm(
                    np.asarray(safe_actions, dtype=np.float32) - np.asarray(actions, dtype=np.float32),
                    axis=1,
                ).astype(np.float32)

            step_started = time.perf_counter()
            promises = [
                worker.step({"action": np.asarray(safe_actions[slot], dtype=np.float32)})
                for slot, worker in enumerate(self._workers)
            ]
            timesteps = [promise() for promise in promises]
            self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - step_started
            next_policy_obs = self._extract_policy_obs_batch(timesteps)
            next_record_obs = next_policy_obs.copy()
            next_safety_states = [self._extract_safety_state(timestep) for timestep in timesteps]

            for slot, timestep in enumerate(timesteps):
                done = bool(timestep.get("is_last", False) or timestep.get("is_terminal", False))
                self._append_slot_step(
                    buffers[slot],
                    current_record_obs[slot],
                    safe_actions[slot],
                    timestep.get("reward", 0.0),
                    done,
                    agent_infos,
                    slot,
                    extra=extras[slot],
                    env_info=self._merge_safety_env_info(timestep.get("info", {}) or {}, safety_reports[slot]),
                )
                if done:
                    paths.append(self._finalize_slot_buffer(buffers[slot], next_record_obs[slot]))
                    buffers[slot] = self._new_slot_buffer()
                    self._prev_safe_actions[slot] = None
                    if len(paths) >= target_num_trajectories:
                        break

                    extras[slot] = sample_extra_fn()
                    reset_started = time.perf_counter()
                    reset_timestep = self._workers[slot].reset(blocking=True)
                    self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - reset_started
                    next_policy_obs[slot] = self._extract_policy_obs(slot, reset_timestep)
                    next_record_obs[slot] = next_policy_obs[slot]
                    next_safety_states[slot] = self._extract_safety_state(reset_timestep)

            current_policy_obs = next_policy_obs
            current_record_obs = next_record_obs
            current_safety_states = next_safety_states

        return paths[:target_num_trajectories]

    def collect_fixed(
            self,
            policy,
            *,
            extras,
            deterministic_policy: bool,
            state_record_pixeled: bool = False,
            video_frame_source=None,
    ):
        extras = list(extras)
        if not extras:
            return []

        old_force_mode = getattr(policy, "_force_use_mode_actions", None)
        if hasattr(policy, "reset"):
            policy.reset()
        policy._force_use_mode_actions = bool(deterministic_policy)

        try:
            return self._collect_fixed_impl(
                policy,
                extras=extras,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            )
        finally:
            if old_force_mode is not None:
                policy._force_use_mode_actions = old_force_mode

    def _collect_fixed_impl(self, policy, *, extras, state_record_pixeled: bool, video_frame_source):
        total = len(extras)
        paths_by_index = [None for _ in range(total)]
        buffers = [self._new_slot_buffer() for _ in range(self._num_envs)]
        active_extra_indices = [None for _ in range(self._num_envs)]
        current_policy_obs = [None for _ in range(self._num_envs)]
        current_record_obs = [None for _ in range(self._num_envs)]
        current_safety_states = [None for _ in range(self._num_envs)]
        next_extra_idx = 0

        for slot in range(self._num_envs):
            if next_extra_idx >= total:
                break
            active_extra_indices[slot] = next_extra_idx
            next_extra_idx += 1
            reset_started = time.perf_counter()
            timestep = self._workers[slot].reset(blocking=True)
            self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - reset_started
            policy_obs = self._extract_policy_obs(slot, timestep)
            current_policy_obs[slot] = policy_obs
            current_record_obs[slot] = self._extract_record_obs(
                slot,
                timestep,
                policy_obs,
                state_record_pixeled=state_record_pixeled,
                video_frame_source=video_frame_source,
            )
            current_safety_states[slot] = self._extract_safety_state(timestep)
            self._prev_safe_actions[slot] = None

        while any(path is None for path in paths_by_index):
            active_slots = [slot for slot, idx in enumerate(active_extra_indices) if idx is not None]
            if not active_slots:
                break

            agent_input = np.stack([np.asarray(current_policy_obs[slot]) for slot in active_slots], axis=0)
            active_extras = [extras[active_extra_indices[slot]] for slot in active_slots]
            if uses_skill_inputs(self.cfg):
                stacked_skills = np.stack(
                    [np.asarray(extra["skill"], dtype=np.float32) for extra in active_extras],
                    axis=0,
                )
                if torch is not None and torch.is_tensor(agent_input):
                    agent_input = utils.get_torch_concat_obs(agent_input, stacked_skills, dim=1)
                else:
                    agent_input = utils.get_np_concat_obs(agent_input, stacked_skills)

            actions, agent_infos = policy.get_actions(agent_input)
            active_safety_states = [current_safety_states[slot] for slot in active_slots]
            safe_actions, safety_reports = self._filter_actions(
                actions,
                active_safety_states,
                agent_input,
                slots=active_slots,
            )
            agent_infos = dict(agent_infos)
            if any(report is not None for report in safety_reports):
                agent_infos["raw_action"] = np.asarray(actions, dtype=np.float32).copy()
                agent_infos["safe_action"] = np.asarray(safe_actions, dtype=np.float32).copy()
                agent_infos["safety_correction_norm"] = np.linalg.norm(
                    np.asarray(safe_actions, dtype=np.float32) - np.asarray(actions, dtype=np.float32),
                    axis=1,
                ).astype(np.float32)

            step_started = time.perf_counter()
            promises = {
                slot: self._workers[slot].step({"action": np.asarray(safe_actions[row], dtype=np.float32)})
                for row, slot in enumerate(active_slots)
            }
            timesteps = {slot: promise() for slot, promise in promises.items()}
            self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - step_started

            for row, slot in enumerate(active_slots):
                timestep = timesteps[slot]
                extra_idx = active_extra_indices[slot]
                next_policy_obs = self._extract_policy_obs(slot, timestep)
                next_record_obs = self._extract_record_obs(
                    slot,
                    timestep,
                    next_policy_obs,
                    state_record_pixeled=state_record_pixeled,
                    video_frame_source=video_frame_source,
                )
                done = bool(timestep.get("is_last", False) or timestep.get("is_terminal", False))
                self._append_slot_step(
                    buffers[slot],
                    current_record_obs[slot],
                    safe_actions[row],
                    timestep.get("reward", 0.0),
                    done,
                    agent_infos,
                    row,
                    extra=extras[extra_idx],
                    env_info=self._merge_safety_env_info(timestep.get("info", {}) or {}, safety_reports[row]),
                )

                if done:
                    paths_by_index[extra_idx] = self._finalize_slot_buffer(buffers[slot], next_record_obs)
                    buffers[slot] = self._new_slot_buffer()
                    self._prev_safe_actions[slot] = None
                    if next_extra_idx < total:
                        active_extra_indices[slot] = next_extra_idx
                        next_extra_idx += 1
                        reset_started = time.perf_counter()
                        reset_timestep = self._workers[slot].reset(blocking=True)
                        self._timing_totals["TimeSamplingEnv"] += time.perf_counter() - reset_started
                        policy_obs = self._extract_policy_obs(slot, reset_timestep)
                        current_policy_obs[slot] = policy_obs
                        current_record_obs[slot] = self._extract_record_obs(
                            slot,
                            reset_timestep,
                            policy_obs,
                            state_record_pixeled=state_record_pixeled,
                            video_frame_source=video_frame_source,
                        )
                        current_safety_states[slot] = self._extract_safety_state(reset_timestep)
                    else:
                        active_extra_indices[slot] = None
                        current_policy_obs[slot] = None
                        current_record_obs[slot] = None
                        current_safety_states[slot] = None
                    continue

                current_policy_obs[slot] = next_policy_obs
                current_record_obs[slot] = next_record_obs
                current_safety_states[slot] = self._extract_safety_state(timestep)

        return [path for path in paths_by_index if path is not None]

    def consume_timing_metrics(self):
        metrics = dict(self._timing_totals)
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
        return metrics

    def close(self):
        for worker in self._workers:
            worker.close()

    def _make_constructor(self, worker_id: int):
        cfg = copy.deepcopy(self.cfg)
        cfg.seed = int(getattr(self.cfg, "seed", 0)) + worker_id

        def _construct():
            from envs import make_env

            return make_env(mode="train", config=cfg)

        return _construct

    def _extract_policy_obs_batch(self, timesteps):
        started = time.perf_counter()
        batch = np.stack(
            [self._extract_policy_obs(slot, timestep) for slot, timestep in enumerate(timesteps)],
            axis=0,
        )
        self._timing_totals["TimeImagePostprocess"] += time.perf_counter() - started
        return batch

    def _extract_policy_obs(self, slot, timestep):
        if self.cfg.encoder:
            image_getter = self._workers[slot].call("get_train_image_tensor")
            return np.asarray(image_getter(), dtype=np.uint8).reshape(-1)
        return np.asarray(timestep["state"], dtype=np.float32).reshape(-1)

    def _extract_record_obs(self, slot, timestep, policy_obs, *, state_record_pixeled: bool, video_frame_source):
        if not state_record_pixeled:
            return np.asarray(policy_obs).copy()
        if video_frame_source is not None:
            frame_getter = self._workers[slot].call("capture_video_frame", video_frame_source)
            frame = frame_getter()
            if frame is not None:
                return np.asarray(frame).copy()
        if isinstance(timestep, dict) and "image" in timestep:
            return np.asarray(timestep["image"]).copy()
        return np.asarray(policy_obs).copy()

    def _append_slot_step(self, buffer, obs, action, reward, done, agent_infos, slot, *, extra, env_info):
        buffer["observations"].append(np.asarray(obs).copy())
        buffer["actions"].append(np.asarray(action, dtype=np.float32).copy())
        buffer["rewards"].append(float(reward))
        buffer["dones"].append(bool(done))
        for key, value in agent_infos.items():
            buffer["agent_infos"][key].append(np.asarray(value[slot]).copy())
        if extra is not None and "skill" in extra:
            buffer["agent_infos"]["skill"].append(np.asarray(extra["skill"], dtype=np.float32).copy())
        for key, value in env_info.items():
            if key == "state":
                continue
            try:
                buffer["env_infos"][key].append(np.asarray(value).copy())
            except ValueError:
                pass

    def _filter_actions(self, actions, safety_states, agent_input, *, slots=None):
        safe_actions = np.asarray(actions, dtype=np.float32).copy()
        reports = [None for _ in range(self._num_envs)]
        active_slots = list(range(self._num_envs)) if slots is None else list(slots)
        reports = [None for _ in range(len(active_slots))]
        for row, slot in enumerate(active_slots):
            controller = self._safety_controllers[slot]
            if controller is None:
                continue
            safe, report = controller.filter_action(
                raw_action=safe_actions[row],
                safety_state=safety_states[row],
                policy_obs=agent_input[row] if hasattr(agent_input, "__getitem__") else None,
                prev_action=self._prev_safe_actions[slot],
                action_to_physical=lambda action, slot=slot: self._worker_call(slot, "safety_denormalize_action", action),
                action_from_physical=lambda action, slot=slot: self._worker_call(slot, "safety_normalize_action", action),
            )
            safe_actions[row] = np.asarray(safe, dtype=np.float32)
            self._prev_safe_actions[slot] = safe_actions[row].copy()
            reports[row] = report
        return safe_actions, reports

    def _worker_call(self, slot, name, *args):
        return self._workers[slot].call(name, *args)()

    @staticmethod
    def _extract_safety_state(timestep):
        info = timestep.get("info", {}) if isinstance(timestep, dict) else {}
        return (info or {}).get("safety_state")

    @staticmethod
    def _merge_safety_env_info(env_info, report):
        if report is None:
            return env_info
        return {**(env_info or {}), **report.to_env_info()}

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
