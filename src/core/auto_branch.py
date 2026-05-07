"""Automatic cascade branching driven by M-policy saturation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from matplotlib.figure import Figure

from core.auto_branch_metrics import compute_m_policy
from core.stage_contract import is_pretraining_stage
from utils import utils


@dataclass
class BranchState:
    branch_id: int
    parent_branch_id: Optional[int]
    layer_id: int
    birth_epoch: int
    m_policy_best: float = 0.0
    split_patience_counter: int = 0
    checks_performed: int = 0
    last_m_policy: Optional[float] = None
    last_ratio: Optional[float] = None
    last_should_split: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            'branch_id': int(self.branch_id),
            'parent_branch_id': None if self.parent_branch_id is None else int(self.parent_branch_id),
            'layer_id': int(self.layer_id),
            'birth_epoch': int(self.birth_epoch),
            'm_policy_best': float(self.m_policy_best),
            'split_patience_counter': int(self.split_patience_counter),
            'checks_performed': int(self.checks_performed),
            'last_m_policy': None if self.last_m_policy is None else float(self.last_m_policy),
            'last_ratio': None if self.last_ratio is None else float(self.last_ratio),
            'last_should_split': bool(self.last_should_split),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, object]) -> "BranchState":
        return cls(
            branch_id=int(state['branch_id']),
            parent_branch_id=state.get('parent_branch_id'),
            layer_id=int(state['layer_id']),
            birth_epoch=int(state['birth_epoch']),
            m_policy_best=float(state.get('m_policy_best', 0.0)),
            split_patience_counter=int(state.get('split_patience_counter', 0)),
            checks_performed=int(state.get('checks_performed', 0)),
            last_m_policy=state.get('last_m_policy'),
            last_ratio=state.get('last_ratio'),
            last_should_split=bool(state.get('last_should_split', False)),
        )


class AutoBranchController:
    """Manage fresh/recent buffers and automatic cascade stage growth."""

    def __init__(self, config, agent, task_adapter, work_dir, logger, writer):
        self.cfg = config
        self.agent = agent
        self.task_adapter = task_adapter
        self.work_dir = work_dir
        self.logger = logger
        self.writer = writer
        self.enabled = bool(getattr(config.auto_branch, 'enabled', False))
        self.recent_buffers: Dict[int, deque] = {}
        self.branch_states: Dict[int, BranchState] = {}
        self.active_branch_id = 0
        self.split_event_id = 0

        if not self.enabled:
            return

        self._validate_runtime_requirements()
        self._bootstrap_from_policy_state()
        self.logger.info(
            "[AutoBranch] enabled interval=%d recent_epochs=%d k=%d points_per_traj=%d distance_mode=%s",
            self.cfg.auto_branch.check_interval_epochs,
            self.cfg.auto_branch.recent_buffer_epochs,
            self.cfg.auto_branch.knn_k,
            self.cfg.auto_branch.representative_points_per_traj,
            self.cfg.auto_branch.distance_mode,
        )

    def _validate_runtime_requirements(self):
        if not bool(self.cfg.cascade.use_cascade):
            raise ValueError("auto_branch requires use_cascade=True")
        if not is_pretraining_stage(self.cfg):
            raise ValueError("auto_branch currently requires stage=pre_training so phi(s)=traj_encoder(obs).mean is available")
        if self.agent.traj_encoder is None:
            raise ValueError("auto_branch requires traj_encoder; raw observation or pixel-space fallbacks are intentionally disabled")

    def _bootstrap_from_policy_state(self):
        current_stage_count = int(self.agent._get_cascade_stage_count())
        for branch_id in range(current_stage_count):
            parent_branch_id = None if branch_id == 0 else branch_id - 1
            self.branch_states[branch_id] = BranchState(
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                layer_id=branch_id,
                birth_epoch=0,
            )
            self.recent_buffers[branch_id] = deque(maxlen=int(self.cfg.auto_branch.recent_buffer_epochs))
        self.active_branch_id = max(self.branch_states.keys())

    def state_dict(self) -> Dict[str, object]:
        if not self.enabled:
            return {}
        return {
            'enabled': True,
            'active_branch_id': int(self.active_branch_id),
            'split_event_id': int(self.split_event_id),
            'branch_states': [self.branch_states[idx].to_dict() for idx in sorted(self.branch_states.keys())],
            'recent_buffers': {
                str(branch_id): [self._serialize_entry(entry) for entry in entries]
                for branch_id, entries in self.recent_buffers.items()
            },
        }

    def load_state_dict(self, state: Dict[str, object]):
        if not self.enabled or not state:
            return
        self.branch_states = {
            int(item['branch_id']): BranchState.from_dict(item)
            for item in state.get('branch_states', [])
        }
        self.recent_buffers = {
            int(branch_id): deque(
                [self._deserialize_entry(entry) for entry in entries],
                maxlen=int(self.cfg.auto_branch.recent_buffer_epochs),
            )
            for branch_id, entries in state.get('recent_buffers', {}).items()
        }
        if not self.branch_states:
            self._bootstrap_from_policy_state()
        else:
            current_stage_count = int(self.agent._get_cascade_stage_count())
            while len(self.branch_states) < current_stage_count:
                branch_id = len(self.branch_states)
                parent_branch_id = None if branch_id == 0 else branch_id - 1
                self.branch_states[branch_id] = BranchState(
                    branch_id=branch_id,
                    parent_branch_id=parent_branch_id,
                    layer_id=branch_id,
                    birth_epoch=0,
                )
                self.recent_buffers[branch_id] = deque(maxlen=int(self.cfg.auto_branch.recent_buffer_epochs))
        for branch_id in self.branch_states:
            self.recent_buffers.setdefault(branch_id, deque(maxlen=int(self.cfg.auto_branch.recent_buffer_epochs)))
        self.active_branch_id = int(state.get('active_branch_id', max(self.branch_states.keys())))
        self.split_event_id = int(state.get('split_event_id', 0))

    def maybe_handle_epoch_end(self, epoch: int, step_paths, step_itr: int):
        """Update recent buffer every epoch and run split checks on schedule."""
        if not self.enabled:
            return None

        current_branch = self.branch_states[self.active_branch_id]
        recent_entry = self._build_point_entry(
            step_paths,
            branch_state=current_branch,
            epoch=epoch,
            step_itr=step_itr,
            source_tag='recent',
        )
        if recent_entry is not None:
            self.recent_buffers[current_branch.branch_id].append(recent_entry)

        if not self._should_check_epoch(epoch):
            self._log_branch_status(
                epoch=epoch,
                branch_state=current_branch,
                checked=False,
                did_split=False,
                reason='check_interval_not_reached',
            )
            return None

        if self.agent._get_cascade_stage_count() >= int(self.cfg.cascade.num_policy_levels):
            self._log_branch_status(
                epoch=epoch,
                branch_state=current_branch,
                checked=True,
                did_split=False,
                reason='max_policy_levels_reached',
            )
            return None

        check_result = self._evaluate_branch(epoch=epoch, branch_state=current_branch, step_itr=step_itr)
        if not check_result['valid']:
            self._log_branch_status(
                epoch=epoch,
                branch_state=current_branch,
                checked=True,
                did_split=False,
                reason=check_result['reason'],
                result=check_result,
            )
            return check_result

        if check_result['should_split']:
            split_result = self._perform_split(epoch=epoch, branch_state=current_branch, check_result=check_result, step_itr=step_itr)
            self._log_branch_status(
                epoch=epoch,
                branch_state=current_branch,
                checked=True,
                did_split=True,
                reason='split_triggered',
                result=check_result,
                split_result=split_result,
            )
            return split_result

        self._log_branch_status(
            epoch=epoch,
            branch_state=current_branch,
            checked=True,
            did_split=False,
            reason='split_not_triggered',
            result=check_result,
        )
        return check_result

    def _should_check_epoch(self, epoch: int) -> bool:
        return epoch > 0 and epoch % int(self.cfg.auto_branch.check_interval_epochs) == 0

    def _evaluate_branch(self, epoch: int, branch_state: BranchState, step_itr: int) -> Dict[str, object]:
        fresh_entry = self._collect_fresh_entry(epoch=epoch, branch_state=branch_state, step_itr=step_itr)
        if fresh_entry is None:
            return {'valid': False, 'reason': 'fresh_rollout_empty'}

        recent_points = self._collect_recent_points(branch_state.branch_id)
        if recent_points is None or recent_points.shape[0] == 0:
            return {
                'valid': False,
                'reason': 'recent_buffer_empty',
                'fresh_entry': fresh_entry,
            }

        metric_result = compute_m_policy(
            fresh_entry['points'],
            recent_points,
            k=int(self.cfg.auto_branch.knn_k),
            mode=str(self.cfg.auto_branch.distance_mode),
        )
        if not metric_result['valid']:
            metric_result['fresh_entry'] = fresh_entry
            metric_result['recent_points'] = recent_points
            return metric_result

        previous_best = float(branch_state.m_policy_best)
        m_policy = float(metric_result['m_policy'])
        if branch_state.checks_performed == 0 or previous_best <= 0.0:
            ratio_reference = max(m_policy, 1e-8)
        else:
            ratio_reference = max(previous_best, 1e-8)
        ratio = float(m_policy / ratio_reference)

        branch_state.checks_performed += 1
        branch_state.last_m_policy = m_policy
        branch_state.last_ratio = ratio
        branch_state.m_policy_best = max(previous_best, m_policy)

        branch_age = int(epoch - branch_state.birth_epoch)
        age_ok = branch_age >= int(self.cfg.auto_branch.min_branch_age)
        ratio_ok = ratio < float(self.cfg.auto_branch.m_policy_ratio_threshold)
        if age_ok and ratio_ok:
            branch_state.split_patience_counter += 1
        else:
            branch_state.split_patience_counter = 0

        should_split = branch_state.split_patience_counter >= int(self.cfg.auto_branch.split_patience)
        branch_state.last_should_split = bool(should_split)

        result = {
            'valid': True,
            'reason': None,
            'm_policy': m_policy,
            'best': float(branch_state.m_policy_best),
            'ratio': ratio,
            'recent_threshold': float(metric_result['recent_threshold']),
            'effective_k': int(metric_result['effective_k']),
            'fresh_entry': fresh_entry,
            'recent_points': recent_points,
            'frontier_points': metric_result['frontier_points'],
            'fresh_knn_distances': metric_result['fresh_knn_distances'],
            'branch_age': branch_age,
            'should_split': bool(should_split),
            'patience_counter': int(branch_state.split_patience_counter),
        }
        self._write_check_scalars(branch_state, epoch, result)
        return result

    def _collect_fresh_entry(self, epoch: int, branch_state: BranchState, step_itr: int):
        extras = self.task_adapter.build_auto_branch_probe_extras(
            int(self.cfg.auto_branch.fresh_rollout_episodes),
            branch_state.branch_id,
        )
        if not extras:
            return None

        rollout_seed = int(self.cfg.seed + 50000 + branch_state.branch_id * 9973)
        if not self.cfg.auto_branch.seeded_probe_skills:
            rollout_seed += int(epoch)

        try:
            with utils.eval_mode(self.agent.sac_trainer.skill_policy):
                trajectories = self.task_adapter.collect_policy_trajectories(
                    extras,
                    deterministic_policy=True,
                    rollout_seed=rollout_seed,
                    state_record_pixeled=False,
                )
        except Exception as exc:  # pragma: no cover - safety path
            self.logger.exception("[AutoBranch] fresh rollout failed: %s", exc)
            return None

        return self._build_point_entry(
            trajectories,
            branch_state=branch_state,
            epoch=epoch,
            step_itr=step_itr,
            source_tag='fresh',
        )

    def _build_point_entry(self, paths, *, branch_state: BranchState, epoch: int, step_itr: int, source_tag: str):
        sampled_obs = []
        rollout_ids = []
        sample_indices = []
        sampled_skills = []
        has_skill = False

        for rollout_id, path in enumerate(paths):
            observations = np.asarray(path['observations'])
            if observations.shape[0] == 0:
                continue
            rep_indices = self._representative_indices(observations.shape[0])
            if rep_indices.size == 0:
                continue
            sampled_obs.append(observations[rep_indices])
            rollout_ids.append(np.full(rep_indices.shape[0], rollout_id, dtype=np.int64))
            sample_indices.append(rep_indices.astype(np.int64))

            skill_array = path.get('agent_infos', {}).get('skill')
            if skill_array is not None:
                skill_array = np.asarray(skill_array)
                if skill_array.shape[0] >= rep_indices[-1] + 1:
                    skill_samples = skill_array[rep_indices]
                else:
                    skill_samples = np.repeat(skill_array[-1: ], rep_indices.shape[0], axis=0)
                sampled_skills.append(skill_samples.reshape(skill_samples.shape[0], -1).astype(np.float32))
                has_skill = True

        if not sampled_obs:
            return None

        stacked_obs = np.concatenate(sampled_obs, axis=0)
        phi_points = self._encode_phi_points(stacked_obs)
        num_points = phi_points.shape[0]
        parent_branch_id = -1 if branch_state.parent_branch_id is None else int(branch_state.parent_branch_id)
        entry = {
            'source': source_tag,
            'points': phi_points,
            'epoch': np.full(num_points, int(epoch), dtype=np.int64),
            'step_itr': np.full(num_points, int(step_itr), dtype=np.int64),
            'rollout_id': np.concatenate(rollout_ids, axis=0),
            'sample_index': np.concatenate(sample_indices, axis=0),
            'branch_id': np.full(num_points, int(branch_state.branch_id), dtype=np.int64),
            'layer_id': np.full(num_points, int(branch_state.layer_id), dtype=np.int64),
            'parent_branch_id': np.full(num_points, parent_branch_id, dtype=np.int64),
            'skill': np.concatenate(sampled_skills, axis=0) if has_skill else None,
        }
        return entry

    def _representative_indices(self, traj_length: int) -> np.ndarray:
        num_points = min(int(self.cfg.auto_branch.representative_points_per_traj), int(traj_length))
        if num_points <= 0:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.linspace(0, traj_length - 1, num=num_points, dtype=int))

    def _encode_phi_points(self, observations) -> np.ndarray:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.agent.device)
        traj_encoder = self.agent.traj_encoder
        was_training = bool(traj_encoder.training)
        try:
            traj_encoder.eval()
            with torch.no_grad():
                phi = self.agent.encode_phi(obs_tensor, use_target=False)
        finally:
            if was_training:
                traj_encoder.train()
        phi = phi.detach().cpu().float().numpy()
        if phi.ndim == 1:
            phi = phi.reshape(-1, 1)
        return phi.astype(np.float32)

    def _collect_recent_points(self, branch_id: int) -> Optional[np.ndarray]:
        entries = []
        if self.cfg.auto_branch.use_global_recent_buffer:
            for buffer_entries in self.recent_buffers.values():
                entries.extend(list(buffer_entries))
        else:
            entries.extend(list(self.recent_buffers.get(branch_id, [])))
        if not entries:
            return None
        point_sets = [entry['points'] for entry in entries if entry['points'].shape[0] > 0]
        if not point_sets:
            return None
        return np.concatenate(point_sets, axis=0)

    def _perform_split(self, *, epoch: int, branch_state: BranchState, check_result: Dict[str, object], step_itr: int):
        new_branch_id = len(self.branch_states)
        new_layer_id = int(branch_state.layer_id) + 1
        split_event_id = self.split_event_id
        viz_path = None
        if self.cfg.auto_branch.visualize_on_split:
            viz_path = self._save_split_visualization(
                epoch=epoch,
                step_itr=step_itr,
                split_event_id=split_event_id,
                branch_state=branch_state,
                new_branch_id=new_branch_id,
                new_layer_id=new_layer_id,
                fresh_entry=check_result['fresh_entry'],
                recent_points=check_result['recent_points'],
                frontier_points=check_result['frontier_points'],
            )

        self.agent.add_policy_stage()
        self.branch_states[new_branch_id] = BranchState(
            branch_id=new_branch_id,
            parent_branch_id=branch_state.branch_id,
            layer_id=new_layer_id,
            birth_epoch=epoch,
        )
        self.recent_buffers[new_branch_id] = deque(maxlen=int(self.cfg.auto_branch.recent_buffer_epochs))
        self.active_branch_id = new_branch_id
        self.split_event_id += 1
        branch_state.split_patience_counter = 0
        branch_state.last_should_split = True

        split_result = {
            'valid': True,
            'did_split': True,
            'new_branch_id': new_branch_id,
            'new_layer_id': new_layer_id,
            'parent_branch_id': branch_state.branch_id,
            'split_event_id': split_event_id,
            'visualization_path': viz_path,
        }
        if viz_path:
            self.logger.info("[AutoBranch] split visualization saved to %s", viz_path)
        return split_result

    def _save_split_visualization(
            self,
            *,
            epoch: int,
            step_itr: int,
            split_event_id: int,
            branch_state: BranchState,
            new_branch_id: int,
            new_layer_id: int,
            fresh_entry,
            recent_points: np.ndarray,
            frontier_points: np.ndarray,
    ) -> str:
        save_dir = os.path.join(self.work_dir, self.cfg.auto_branch.visualize_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            (
                f"split_epoch{epoch:06d}_branch{branch_state.branch_id}_layer{branch_state.layer_id}_"
                f"newbranch{new_branch_id}_newlayer{new_layer_id}_event{split_event_id}.png"
            ),
        )

        projected_recent, projected_fresh, projected_frontier = self._project_point_sets(
            [recent_points, fresh_entry['points'], frontier_points]
        )
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        parent_color = self._color_for_layer(branch_state.layer_id)
        new_color = self._color_for_layer(new_layer_id)

        if projected_recent.shape[0] > 0:
            ax.scatter(
                projected_recent[:, 0],
                projected_recent[:, 1],
                s=12,
                alpha=0.22,
                color=parent_color,
                label=f"branch {branch_state.branch_id} recent L{branch_state.layer_id}",
            )
        if projected_fresh.shape[0] > 0:
            ax.scatter(
                projected_fresh[:, 0],
                projected_fresh[:, 1],
                s=24,
                alpha=0.75,
                color=parent_color,
                label=f"branch {branch_state.branch_id} fresh L{branch_state.layer_id}",
            )
        if projected_frontier.shape[0] > 0:
            ax.scatter(
                projected_frontier[:, 0],
                projected_frontier[:, 1],
                s=36,
                alpha=0.95,
                marker='x',
                color=new_color,
                label=f"new branch {new_branch_id} frontier L{new_layer_id}",
            )
        ax.set_title(
            f"Split event {split_event_id} | branch {branch_state.branch_id} -> {new_branch_id}"
        )
        ax.set_xlabel("phi-PC1")
        ax.set_ylabel("phi-PC2")
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        self.writer.add_figure(f'auto_branch/split_event_{split_event_id}', fig, step_itr)
        return save_path

    def _project_point_sets(self, point_sets: List[np.ndarray]):
        non_empty = [pts for pts in point_sets if pts is not None and pts.size > 0]
        if not non_empty:
            return [np.zeros((0, 2), dtype=np.float32) for _ in point_sets]

        dim = non_empty[0].shape[1]
        if dim == 1:
            return [
                np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1) if pts is not None and pts.size > 0
                else np.zeros((0, 2), dtype=np.float32)
                for pts in point_sets
            ]
        if dim == 2:
            return [
                pts[:, :2] if pts is not None and pts.size > 0 else np.zeros((0, 2), dtype=np.float32)
                for pts in point_sets
            ]

        stacked = np.concatenate(non_empty, axis=0)
        if stacked.shape[0] < 2 or utils.decomposition is None:
            return [
                pts[:, :2] if pts is not None and pts.size > 0 else np.zeros((0, 2), dtype=np.float32)
                for pts in point_sets
            ]

        try:
            pca = utils.decomposition.PCA(n_components=2)
            projected = pca.fit_transform(stacked)
        except Exception:  # pragma: no cover - visualization fallback
            return [
                pts[:, :2] if pts is not None and pts.size > 0 else np.zeros((0, 2), dtype=np.float32)
                for pts in point_sets
            ]

        split_sets = []
        start = 0
        for pts in point_sets:
            if pts is None or pts.size == 0:
                split_sets.append(np.zeros((0, 2), dtype=np.float32))
                continue
            stop = start + pts.shape[0]
            split_sets.append(projected[start:stop].astype(np.float32))
            start = stop
        return split_sets

    def _color_for_layer(self, layer_id: int):
        from matplotlib import cm

        return cm.get_cmap('tab20')(int(layer_id) % 20)

    def _log_branch_status(
            self,
            *,
            epoch: int,
            branch_state: BranchState,
            checked: bool,
            did_split: bool,
            reason: str,
            result: Optional[Dict[str, object]] = None,
            split_result: Optional[Dict[str, object]] = None):
        result = result or {}
        split_result = split_result or {}
        m_policy = result.get('m_policy', branch_state.last_m_policy)
        ratio = result.get('ratio', branch_state.last_ratio)
        best = result.get('best', branch_state.m_policy_best)
        patience = result.get('patience_counter', branch_state.split_patience_counter)
        branch_age = result.get('branch_age', epoch - branch_state.birth_epoch)
        active_branch_id = split_result.get('new_branch_id', branch_state.branch_id)
        active_layer_id = split_result.get('new_layer_id', branch_state.layer_id)
        self.logger.info(
            "[AutoBranch] epoch=%d branch_id=%d layer_id=%d parent_branch_id=%s checked=%s split=%s "
            "m_policy=%s best=%s ratio=%s patience=%d branch_age=%d reason=%s new_branch_id=%s new_layer_id=%s",
            epoch,
            branch_state.branch_id,
            branch_state.layer_id,
            "None" if branch_state.parent_branch_id is None else branch_state.parent_branch_id,
            checked,
            did_split,
            self._fmt_optional(m_policy),
            self._fmt_optional(best),
            self._fmt_optional(ratio),
            int(patience),
            int(branch_age),
            reason,
            split_result.get('new_branch_id'),
            split_result.get('new_layer_id'),
        )
        self.writer.add_scalar('auto_branch/active_branch_id', active_branch_id, epoch)
        self.writer.add_scalar('auto_branch/active_layer_id', active_layer_id, epoch)
        self.writer.add_scalar('auto_branch/branch_age', branch_age, epoch)
        self.writer.add_scalar('auto_branch/checked', int(bool(checked)), epoch)
        self.writer.add_scalar('auto_branch/split', int(bool(did_split)), epoch)
        if m_policy is not None:
            self.writer.add_scalar('auto_branch/m_policy', float(m_policy), epoch)
        if best is not None:
            self.writer.add_scalar('auto_branch/m_policy_best', float(best), epoch)
        if ratio is not None:
            self.writer.add_scalar('auto_branch/m_policy_ratio', float(ratio), epoch)
        self.writer.add_scalar('auto_branch/patience_counter', int(patience), epoch)

    def _write_check_scalars(self, branch_state: BranchState, epoch: int, result: Dict[str, object]):
        self.writer.add_scalar(f'auto_branch/branch_{branch_state.branch_id}/m_policy', result['m_policy'], epoch)
        self.writer.add_scalar(f'auto_branch/branch_{branch_state.branch_id}/m_policy_best', result['best'], epoch)
        self.writer.add_scalar(f'auto_branch/branch_{branch_state.branch_id}/ratio', result['ratio'], epoch)
        self.writer.add_scalar(f'auto_branch/branch_{branch_state.branch_id}/recent_threshold', result['recent_threshold'], epoch)
        self.writer.add_scalar(f'auto_branch/branch_{branch_state.branch_id}/effective_k', result['effective_k'], epoch)

    def _serialize_entry(self, entry: Dict[str, object]) -> Dict[str, object]:
        return {
            key: value
            for key, value in entry.items()
        }

    def _deserialize_entry(self, entry: Dict[str, object]) -> Dict[str, object]:
        return {
            'source': entry['source'],
            'points': np.asarray(entry['points'], dtype=np.float32),
            'epoch': np.asarray(entry['epoch'], dtype=np.int64),
            'step_itr': np.asarray(entry['step_itr'], dtype=np.int64),
            'rollout_id': np.asarray(entry['rollout_id'], dtype=np.int64),
            'sample_index': np.asarray(entry['sample_index'], dtype=np.int64),
            'branch_id': np.asarray(entry['branch_id'], dtype=np.int64),
            'layer_id': np.asarray(entry['layer_id'], dtype=np.int64),
            'parent_branch_id': np.asarray(entry['parent_branch_id'], dtype=np.int64),
            'skill': None if entry.get('skill') is None else np.asarray(entry['skill'], dtype=np.float32),
        }

    @staticmethod
    def _fmt_optional(value):
        if value is None:
            return "None"
        return f"{float(value):.6f}"
