from __future__ import annotations

import os
import numpy as np
import torch

from utils import utils
from core.isolation_kernel import SoftIsolationKernel


class _RandomActionPolicy:
    def __init__(self, env):
        self._env = env
        self._force_use_mode_actions = False

    def reset(self):
        return None

    def get_action(self, _):
        action_space = None
        for attr_name in ('action_space', 'act_space'):
            try:
                action_space = getattr(self._env, attr_name)
                break
            except Exception:
                continue
        if action_space is None:
            raise AttributeError('RandomActionPolicy could not find action_space or act_space on the environment')
        return _sample_action_from_space(action_space), {}




def _sample_action_from_space(action_space):
    if hasattr(action_space, 'sample'):
        sample = action_space.sample()
    elif isinstance(action_space, dict):
        sample = {key: _sample_action_from_space(value) for key, value in action_space.items()}
    else:
        raise TypeError(f'Unsupported action space type for random warmup: {type(action_space)}')

    if isinstance(sample, dict) and set(sample.keys()) == {'action'}:
        return sample['action']
    return sample


class _PrecomputedDistanceFunction:
    def __init__(self, lookup: torch.Tensor):
        self.lookup = lookup

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        max_index = self.lookup.shape[0] - 1
        x_idx = torch.clamp(torch.round(x[..., 0]).long(), 0, max_index)
        y_idx = torch.clamp(torch.round(y[..., 0]).long(), 0, max_index)
        return self.lookup[x_idx.unsqueeze(-1), y_idx.unsqueeze(-2)]


def plot_trajectories(
        agent,
        task_adapter=None,
        *,
        snapshot_dir=None,
        writer=None,
        logger=None,
        step_itr=None,
        rollout_seed=None,
        n_eval_skills=None,
        n_trajs_per_skill=None) -> None:
    """Trajectory analysis with switchable IKSE or temporal/Soft-DTW metrics."""
    try:
        from sklearn.manifold import Isomap

        step_itr = int(step_itr if step_itr is not None else getattr(agent, 'step_itr', 0))
        writer = writer if writer is not None else getattr(agent, 'writer', None)
        logger = logger if logger is not None else getattr(agent, 'logger', None)
        snapshot_dir = snapshot_dir or getattr(agent, 'snapshot_dir', None)
        if snapshot_dir is None and task_adapter is not None:
            snapshot_dir = task_adapter.work_dir
        if snapshot_dir is None:
            raise ValueError('snapshot_dir is required for trajectory plotting')

        dim_skill = int(_agent_attr(agent, 'dim_skill', 0))
        discrete = bool(_agent_attr(agent, 'discrete', False))
        unit_length = bool(_agent_attr(agent, 'unit_length', True))
        if dim_skill <= 0:
            raise ValueError('Trajectory plotting requires dim_skill > 0')
        if getattr(agent, 'traj_encoder', None) is None:
            raise ValueError('Trajectory plotting requires a loaded trajectory encoder')

        ikse = bool(_agent_attr(agent, 'ikse', False))
        metric_mode = 'IKSE' if ikse else 'Temporal/Soft-DTW'

        metric_num_sampled_points = int(_agent_attr(agent, 'metric_num_sampled_points', 10))
        if metric_num_sampled_points < 1:
            raise ValueError('metric_num_sampled_points must be >= 1')

        default_dbi_rollouts = int(_agent_attr(agent, 'dbi_num_rollouts_per_skill', 3))
        if default_dbi_rollouts < 3:
            raise ValueError('dbi_num_rollouts_per_skill must be >= 3')

        if n_trajs_per_skill is None:
            n_trajs_per_skill = default_dbi_rollouts
        n_trajs_per_skill = int(n_trajs_per_skill)
        if n_trajs_per_skill < 3:
            raise ValueError('n_trajs_per_skill must be >= 3 for DBI evaluation')

        if n_eval_skills is None:
            n_eval_skills = _default_num_eval_skills(agent, discrete, dim_skill)

        base_skills = _build_eval_skills(
            discrete=discrete,
            dim_skill=dim_skill,
            unit_length=unit_length,
            n_eval_skills=n_eval_skills,
        )
        base_rollout_seed = int(rollout_seed if rollout_seed is not None else _agent_attr(agent, 'seed', 0) + 100)
        record_pixeled_for_metrics = bool(_agent_attr(agent, 'encoder', 0))
        record_pixeled_for_grid = True

        entropy_trajectories, entropy_records = _collect_metric_records(
            agent,
            task_adapter,
            base_skills,
            rollout_seed=base_rollout_seed,
            num_rollouts_per_skill=1,
            num_points_per_traj=metric_num_sampled_points,
            record_pixeled_obs=record_pixeled_for_metrics,
        )
        dbi_trajectories, dbi_records = _collect_metric_records(
            agent,
            task_adapter,
            base_skills,
            rollout_seed=base_rollout_seed + 1,
            num_rollouts_per_skill=n_trajs_per_skill,
            num_points_per_traj=metric_num_sampled_points,
            record_pixeled_obs=record_pixeled_for_metrics,
        )

        graph_context = None
        if not ikse and (entropy_records or dbi_records):
            graph_context = _build_temporal_metric_context(
                agent,
                task_adapter,
                base_skills,
                entropy_trajectories=entropy_trajectories,
                entropy_records=entropy_records,
                dbi_trajectories=dbi_trajectories,
                dbi_records=dbi_records,
                rollout_seed=base_rollout_seed,
                record_pixeled_obs=record_pixeled_for_metrics,
                logger=logger,
            )
            metric_mode = f"Temporal/Soft-DTW[{graph_context['soft_dtw_device']}]"

        grid_trajectories = dbi_trajectories if dbi_trajectories else entropy_trajectories
        grid_trajs_per_skill = n_trajs_per_skill if dbi_trajectories else 1
        if grid_trajectories and not record_pixeled_for_metrics and record_pixeled_for_grid:
            grid_skills = np.repeat(base_skills, grid_trajs_per_skill, axis=0)
            grid_trajectories = _collect_trajectories(
                agent,
                task_adapter,
                grid_skills,
                rollout_seed=base_rollout_seed + 2,
                record_pixeled_obs=True,
            )
        if grid_trajectories:
            save_image_grid(
                agent,
                grid_trajectories,
                len(base_skills),
                grid_trajs_per_skill,
                snapshot_dir=snapshot_dir,
                writer=writer,
                step_itr=step_itr,
            )

        metric_backend = 'ikse' if ikse else 'temporal'
        if writer is not None:
            _log_metric_backend_flags(writer, step_itr, metric_backend)

        if not entropy_records and not dbi_records:
            _maybe_log(logger, f'Trajectory analysis skipped [{metric_mode}]: no valid sampled state trajectories were collected.')
            return

        ensemble_size = 100
        subsample_size = 1024

        entropy_raw = None
        entropy_enc = None
        if entropy_records:
            if ikse:
                entropy_raw, entropy_enc = _compute_ikse_entropy_metrics(
                    agent,
                    entropy_records,
                    ensemble_size=ensemble_size,
                    subsample_size=subsample_size,
                )
            else:
                entropy_raw, entropy_enc = _compute_temporal_entropy_metrics(agent, graph_context)

        dbi_raw = None
        dbi_enc = None
        encoded_concat = None
        traj_obs_counts = []
        skill_labels = []
        if dbi_records:
            if ikse:
                dbi_raw, dbi_enc, encoded_concat, traj_obs_counts, skill_labels = _compute_ikse_dbi_metrics(
                    agent,
                    dbi_records,
                    ensemble_size=ensemble_size,
                    subsample_size=subsample_size,
                )
            else:
                dbi_raw, dbi_enc, encoded_concat, traj_obs_counts, skill_labels = _compute_temporal_dbi_metrics(
                    agent,
                    dbi_records,
                    graph_context,
                )

        message_parts = []
        if entropy_raw is not None and entropy_enc is not None:
            message_parts.append(f'Entropy (Raw) = {entropy_raw:.4f} | Entropy (Enc) = {entropy_enc:.4f}')
        if dbi_raw is not None and dbi_enc is not None:
            message_parts.append(f'DBI (Raw) = {dbi_raw:.4f} | DBI (Enc) = {dbi_enc:.4f}')
        if message_parts:
            message = f"Step {step_itr} [{metric_mode}]: " + ' | '.join(message_parts)
            _maybe_log(logger, message)
            print(message)

        if writer is not None:
            if entropy_raw is not None and entropy_enc is not None:
                writer.add_scalar('eval/Entropy_Raw', entropy_raw, step_itr)
                writer.add_scalar('eval/Entropy_Enc', entropy_enc, step_itr)
            if dbi_raw is not None and dbi_enc is not None:
                writer.add_scalar('eval/DBI_Raw', dbi_raw, step_itr)
                writer.add_scalar('eval/DBI_Enc', dbi_enc, step_itr)
            _log_metric_backend_value_tags(
                writer,
                step_itr,
                metric_backend,
                entropy_raw=entropy_raw,
                dbi_raw=dbi_raw,
            )

        if encoded_concat is None or encoded_concat.shape[0] <= 5:
            return

        isomap = Isomap(n_components=2, n_neighbors=min(20, encoded_concat.shape[0] - 1))
        phi_2d = encoded_concat if encoded_concat.shape[1] == 2 else isomap.fit_transform(encoded_concat)
        phi_2d = phi_2d - phi_2d.mean(axis=0)

        with utils.FigManager(snapshot_dir, step_itr, 'PhiIsomap_Traj', writer=writer, global_step=step_itr) as fm:
            ax = fm.ax
            from matplotlib import cm
            cmap = cm.get_cmap('tab20')

            cursor = 0
            for count, skill_idx in zip(traj_obs_counts, skill_labels):
                traj_2d = phi_2d[cursor: cursor + count]
                cursor += count
                if traj_2d.shape[0] == 0:
                    continue

                color = cmap(skill_idx % 20)
                ax.plot(traj_2d[:, 0], traj_2d[:, 1], color=color, alpha=0.7, linewidth=1.0)
                ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=color, marker='.', s=20, alpha=0.5)
                ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=color, marker='x', s=20, alpha=0.8)

            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')

    except Exception as exc:
        print(f'Error in Trajectory Analysis: {exc}')
        import traceback
        traceback.print_exc()


def _log_metric_backend_flags(writer, step_itr, metric_backend: str):
    writer.add_scalar('eval/MetricBackend_IsIKSE', int(metric_backend == 'ikse'), step_itr)
    writer.add_scalar('eval/MetricBackend_UsesTemporalDistance', int(metric_backend == 'temporal'), step_itr)
    writer.add_scalar('eval/MetricBackend_UsesIsolationKernel', int(metric_backend == 'ikse'), step_itr)
    writer.add_scalar('eval/MetricBackend_UsesEuclideanDistance', int(metric_backend == 'euclidean'), step_itr)


def _log_metric_backend_value_tags(writer, step_itr, metric_backend: str, *, entropy_raw, dbi_raw):
    if metric_backend == 'temporal':
        if entropy_raw is not None:
            writer.add_scalar('eval/Entropy_TemporalParticle', entropy_raw, step_itr)
        if dbi_raw is not None:
            writer.add_scalar('eval/DBI_TemporalMedoid', dbi_raw, step_itr)
        if entropy_raw is not None and dbi_raw is not None:
            writer.add_scalar('eval/Score_TemporalParticle_DBI', _metric_score(entropy_raw, dbi_raw), step_itr)
    elif metric_backend == 'ikse':
        if entropy_raw is not None:
            writer.add_scalar('eval/Entropy_IKDE_XY', entropy_raw, step_itr)
        if dbi_raw is not None:
            writer.add_scalar('eval/DBI_IKMeanRatio_Legacy', dbi_raw, step_itr)
            # TODO: add eval/DBI_IKCanonical after canonical IK DBI is implemented.
        if entropy_raw is not None and dbi_raw is not None:
            writer.add_scalar('eval/IKSE_LegacyDBI', _metric_score(entropy_raw, dbi_raw), step_itr)
    elif metric_backend == 'euclidean':
        if entropy_raw is not None:
            writer.add_scalar('eval/Entropy_EuclideanParticle', entropy_raw, step_itr)
        if dbi_raw is not None:
            writer.add_scalar('eval/DBI_EuclideanCanonical', dbi_raw, step_itr)


def _metric_score(entropy_value, dbi_value, *, eps: float = 1e-8):
    return float(entropy_value) / (float(dbi_value) + eps)


def extract_xy_trajectory(trajectory):
    env_infos = trajectory.get('env_infos', {}) or {}
    coordinates = env_infos.get('coordinates')
    next_coordinates = env_infos.get('next_coordinates')
    if coordinates is None or next_coordinates is None:
        return None

    coordinates = np.asarray(coordinates, dtype=np.float32)
    next_coordinates = np.asarray(next_coordinates, dtype=np.float32)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, -1)
    if next_coordinates.ndim == 1:
        next_coordinates = next_coordinates.reshape(1, -1)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] < 2:
        return None
    if next_coordinates.ndim != 2 or next_coordinates.shape[0] == 0 or next_coordinates.shape[1] < 2:
        return None
    return np.concatenate([coordinates[:, :2], next_coordinates[-1:, :2]], axis=0)


def plot_skill_xy_trajectories(
        trajectories,
        *,
        n_trajs_per_skill: int,
        snapshot_dir,
        writer=None,
        step_itr=None,
        plot_axis=None,
        logger=None,
        label: str = 'SkillXYTrajPlot') -> int:
    n_trajs_per_skill = max(1, int(n_trajs_per_skill))
    step_itr = int(step_itr or 0)
    if snapshot_dir is None:
        raise ValueError('snapshot_dir is required for skill XY trajectory plotting')

    records = []
    for idx, trajectory in enumerate(trajectories):
        xy = extract_xy_trajectory(trajectory)
        if xy is None:
            continue
        records.append((idx // n_trajs_per_skill, idx % n_trajs_per_skill, xy))

    if not records:
        _maybe_warn(logger, 'Skill XY trajectory plot skipped: no valid coordinates were collected.')
        return 0

    with utils.FigManager(snapshot_dir, step_itr, label, writer=writer, global_step=step_itr) as fm:
        ax = fm.ax
        try:
            from matplotlib import colormaps
            cmap = colormaps.get_cmap('tab20')
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap('tab20')
        for skill_idx, _traj_idx, xy in records:
            color = cmap(skill_idx % 20)
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                alpha=0.72,
                linewidth=1.1,
            )
            ax.scatter(xy[0, 0], xy[0, 1], color=color, marker='.', s=20, alpha=0.7)
            ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker='x', s=24, alpha=0.9)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Skill XY trajectories ({n_trajs_per_skill} rollouts/skill)')
        _set_xy_plot_axis(ax, [xy for _, _, xy in records], plot_axis)

    return len(records)


def _set_xy_plot_axis(ax, trajectories, plot_axis):
    if isinstance(plot_axis, str) and plot_axis == 'free':
        ax.axis('scaled')
        return
    if plot_axis is not None:
        plot_axis = list(plot_axis)
        if len(plot_axis) == 4:
            ax.axis(plot_axis)
            ax.set_aspect('equal', adjustable='box')
            return

    coords = np.concatenate(trajectories, axis=0)
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    pad = 0.05 * span
    ax.set_xlim(float(x_min - pad), float(x_max + pad))
    ax.set_ylim(float(y_min - pad), float(y_max + pad))
    ax.set_aspect('equal', adjustable='box')


def save_image_grid(agent, trajectories, n_eval_skills: int, n_trajs_per_skill: int, *, snapshot_dir=None, writer=None, step_itr=None) -> None:
    """Save a grid of sampled frames for qualitative inspection when observations are image-like."""
    import cv2

    snapshot_dir = snapshot_dir or getattr(agent, 'snapshot_dir', None)
    writer = writer if writer is not None else getattr(agent, 'writer', None)
    step_itr = int(step_itr if step_itr is not None else getattr(agent, 'step_itr', 0))
    if snapshot_dir is None:
        return

    all_rows = []
    n_frames = 64
    target_h = int(_agent_attr(agent, 'render_size', 128))
    target_w = target_h

    for i in range(n_eval_skills):
        for j in range(n_trajs_per_skill):
            idx = i * n_trajs_per_skill + j
            if idx >= len(trajectories):
                continue

            traj = trajectories[idx]
            images = np.asarray(traj['observations'])
            if images.ndim < 2:
                continue

            total_steps = len(images)
            indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)

            processed_frames = []
            for step_idx in indices:
                img = _prepare_image_frame(images[step_idx], target_h, target_w)
                if img is None:
                    processed_frames = []
                    break
                processed_frames.append(img)

            if not processed_frames:
                continue

            row = np.concatenate(processed_frames, axis=1)
            all_rows.append(row)

    if not all_rows:
        return

    final_grid = np.concatenate(all_rows, axis=0)
    save_path = os.path.join(snapshot_dir, 'plots', f'skill_grid_{step_itr}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if final_grid.shape[-1] == 1:
        grid_bgr = final_grid[..., 0]
    else:
        grid_bgr = cv2.cvtColor(final_grid.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, grid_bgr)

    if writer is not None:
        if final_grid.ndim == 2:
            tb_grid = final_grid[None, ...]
        else:
            tb_grid = final_grid.transpose(2, 0, 1)
        writer.add_image('eval/skill_grid', tb_grid, step_itr)


def _build_eval_skills(*, discrete: bool, dim_skill: int, unit_length: bool, n_eval_skills: int):
    if discrete:
        return np.eye(dim_skill, dtype=np.float32)[:n_eval_skills]

    if dim_skill == 2:
        radius = 1.0 if unit_length else 1.5
        angles = np.linspace(0.0, 2.0 * np.pi, num=max(n_eval_skills, 1), endpoint=False)
        return np.stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
        ], axis=1).astype(np.float32)

    skills = np.random.randn(n_eval_skills, dim_skill).astype(np.float32)
    if unit_length:
        skills = skills / (np.linalg.norm(skills, axis=1, keepdims=True) + 1e-8)
    return skills


def _collect_trajectories(agent, task_adapter, skills, *, rollout_seed=None, record_pixeled_obs: bool):
    if task_adapter is not None:
        seed = int(rollout_seed if rollout_seed is not None else _agent_attr(agent, 'seed', 0) + 100)
        extras = task_adapter.build_skill_extras(skills)
        return task_adapter.collect_policy_trajectories(
            extras,
            deterministic_policy=False,
            rollout_seed=seed,
            state_record_pixeled=record_pixeled_obs,
        )

    if hasattr(agent, '_get_trajectories') and hasattr(agent, '_generate_skill_extras'):
        return agent._get_trajectories(
            batch_size=len(skills),
            extras=agent._generate_skill_extras(skills),
            deterministic_policy=False,
            state_record_pixeled=record_pixeled_obs,
        )

    raise TypeError('plot_trajectories requires either task_adapter or legacy agent trajectory helpers')


def _collect_metric_records(
        agent,
        task_adapter,
        base_skills,
        *,
        rollout_seed,
        num_rollouts_per_skill: int,
        num_points_per_traj: int,
        record_pixeled_obs: bool):
    repeated_skills = np.repeat(base_skills, num_rollouts_per_skill, axis=0)
    trajectories = _collect_trajectories(
        agent,
        task_adapter,
        repeated_skills,
        rollout_seed=rollout_seed,
        record_pixeled_obs=record_pixeled_obs,
    )

    records = []
    for idx, traj in enumerate(trajectories):
        skill_idx = idx // num_rollouts_per_skill
        if skill_idx >= len(base_skills):
            break

        sampled_record = _sample_metric_points_from_trajectory(agent, traj, num_points_per_traj)
        if sampled_record is None:
            continue

        sampled_record['skill_idx'] = int(skill_idx)
        sampled_record['trajectory_idx'] = int(idx % num_rollouts_per_skill)
        sampled_record['trajectory_list_idx'] = int(idx)
        records.append(sampled_record)

    return trajectories, records


def _collect_random_action_trajectories(agent, task_adapter, *, num_rollouts: int, rollout_seed: int, record_pixeled_obs: bool):
    if num_rollouts <= 0:
        return []

    from workers.rollout import SkillRolloutWorker

    env = task_adapter.env if task_adapter is not None else getattr(agent, 'env', None)
    if env is None:
        return []

    random_policy = _RandomActionPolicy(env)
    rollout_worker = SkillRolloutWorker(
        rollout_seed,
        int(_agent_attr(agent, 'time_limit', 0)),
        cur_extra_keys=[],
        pixeled=record_pixeled_obs,
    )

    trajectories = []
    for _ in range(int(num_rollouts)):
        batch = rollout_worker.rollout(
            env,
            random_policy,
            extra=None,
            deterministic_policy=False,
            state_record_pixeled=record_pixeled_obs,
        )
        trajectories.extend(batch.to_trajectory_list())
    return trajectories


def _collect_graph_trajectories(
        agent,
        task_adapter,
        base_skills,
        *,
        rollout_seed: int,
        num_rollouts_per_skill: int,
        record_pixeled_obs: bool):
    if num_rollouts_per_skill <= 0:
        return []
    repeated_skills = np.repeat(base_skills, int(num_rollouts_per_skill), axis=0)
    return _collect_trajectories(
        agent,
        task_adapter,
        repeated_skills,
        rollout_seed=rollout_seed,
        record_pixeled_obs=record_pixeled_obs,
    )


def _sample_uniform_indices(traj_length: int, num_points: int) -> np.ndarray:
    if traj_length <= 0 or num_points <= 0:
        return np.zeros((0,), dtype=np.int64)
    sample_count = min(int(traj_length), int(num_points))
    if sample_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.linspace(0, traj_length - 1, num=sample_count, dtype=int)).astype(np.int64)


def _sample_metric_points_from_trajectory(agent, traj, num_points: int):
    obs = np.asarray(traj['observations'])
    if obs.ndim != 2 or obs.shape[0] == 0:
        return None

    sampled_indices = _sample_uniform_indices(obs.shape[0], num_points)
    if sampled_indices.size == 0:
        return None

    sampled_obs = obs[sampled_indices]
    if not bool(_agent_attr(agent, 'encoder', 0)) and _looks_like_flat_pixel_observation(sampled_obs):
        raise ValueError(
            'Trajectory metrics received flattened pixel observations while encoder=0. '
            'Expected state observations for metric computation.'
        )
    raw_points = sampled_obs[:, 0:2].astype(np.float32)
    if raw_points.ndim != 2 or raw_points.shape[0] == 0 or raw_points.shape[1] == 0:
        return None

    encoded_points = _encode_sampled_points(agent, sampled_obs)
    if encoded_points.ndim != 2 or encoded_points.shape[0] == 0:
        return None

    return {
        'raw_points': raw_points,
        'encoded_points': encoded_points.astype(np.float32),
        'sampled_indices': sampled_indices,
    }


def _encode_sampled_points(agent, observations) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.from_numpy(np.asarray(observations)).float().to(agent.device)
        phi = agent.encode_phi(obs_tensor, use_target=False)
        phi = phi.detach().cpu().float().numpy()
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)
    return phi.astype(np.float32)


def _compute_entropy_metrics(agent, records, *, ensemble_size: int, subsample_size: int, ikse: bool):
    if ikse:
        return _compute_ikse_entropy_metrics(agent, records, ensemble_size=ensemble_size, subsample_size=subsample_size)
    return _compute_particle_entropy_metrics(agent, records)


def _compute_dbi_metrics(agent, records, *, ensemble_size: int, subsample_size: int, ikse: bool):
    if ikse:
        return _compute_ikse_dbi_metrics(agent, records, ensemble_size=ensemble_size, subsample_size=subsample_size)
    return _compute_euclidean_dbi_metrics(agent, records)


def _compute_ikse_entropy_metrics(agent, records, *, ensemble_size: int, subsample_size: int):
    raw_concat = _concat_record_points(records, 'raw_points')
    encoded_concat = _concat_record_points(records, 'encoded_points')
    raw_maps = _fit_and_map_points(raw_concat, ensemble_size, subsample_size, agent.device)
    enc_maps = _fit_and_map_points(encoded_concat, ensemble_size, subsample_size, agent.device)
    entropy_raw = _calculate_ik_entropy(raw_maps, ensemble_size, agent.device)
    entropy_enc = _calculate_ik_entropy(enc_maps, ensemble_size, agent.device)
    return entropy_raw, entropy_enc


def _compute_particle_entropy_metrics(agent, records):
    raw_concat = _concat_record_points(records, 'raw_points')
    encoded_concat = _concat_record_points(records, 'encoded_points')
    entropy_raw = _calculate_particle_entropy(raw_concat, agent.device)
    entropy_enc = _calculate_particle_entropy(encoded_concat, agent.device)
    return entropy_raw, entropy_enc


def _compute_ikse_dbi_metrics(agent, records, *, ensemble_size: int, subsample_size: int):
    raw_concat = _concat_record_points(records, 'raw_points')
    encoded_concat = _concat_record_points(records, 'encoded_points')
    raw_maps = _fit_and_map_points(raw_concat, ensemble_size, subsample_size, agent.device)
    enc_maps = _fit_and_map_points(encoded_concat, ensemble_size, subsample_size, agent.device)

    traj_raw_mappings = []
    traj_enc_mappings = []
    traj_obs_counts = []
    skill_labels = []
    cursor = 0
    for record in records:
        count = int(record['raw_points'].shape[0])
        if count <= 0:
            continue
        traj_raw_mappings.append(raw_maps[cursor:cursor + count].mean(dim=0))
        traj_enc_mappings.append(enc_maps[cursor:cursor + count].mean(dim=0))
        traj_obs_counts.append(count)
        skill_labels.append(int(record['skill_idx']))
        cursor += count

    if not traj_raw_mappings:
        return 0.0, 0.0, encoded_concat, [], []

    labels_tensor = torch.as_tensor(skill_labels, device=agent.device, dtype=torch.long)
    traj_raw_mappings = torch.stack(traj_raw_mappings)
    traj_enc_mappings = torch.stack(traj_enc_mappings)
    dbi_raw = _calculate_dbi(traj_raw_mappings, labels_tensor, ensemble_size, agent.device)
    dbi_enc = _calculate_dbi(traj_enc_mappings, labels_tensor, ensemble_size, agent.device)
    return dbi_raw, dbi_enc, encoded_concat, traj_obs_counts, skill_labels


def _compute_euclidean_dbi_metrics(agent, records):
    encoded_concat, traj_obs_counts, skill_labels = _build_plot_payload(records)
    raw_clusters = _group_points_by_skill(records, 'raw_points')
    enc_clusters = _group_points_by_skill(records, 'encoded_points')
    dbi_raw = _calculate_euclidean_dbi(raw_clusters, agent.device)
    dbi_enc = _calculate_euclidean_dbi(enc_clusters, agent.device)
    return dbi_raw, dbi_enc, encoded_concat, traj_obs_counts, skill_labels


def _build_temporal_metric_context(
        agent,
        task_adapter,
        base_skills,
        *,
        entropy_trajectories,
        entropy_records,
        dbi_trajectories,
        dbi_records,
        rollout_seed: int,
        record_pixeled_obs: bool,
        logger=None):
    temporal_cfg = _resolve_temporal_metric_config(agent)

    warmup_trajectories = _collect_random_action_trajectories(
        agent,
        task_adapter,
        num_rollouts=temporal_cfg['num_warmup_rollouts'],
        rollout_seed=int(rollout_seed + 10),
        record_pixeled_obs=record_pixeled_obs,
    )
    graph_skill_trajectories = _collect_graph_trajectories(
        agent,
        task_adapter,
        base_skills,
        rollout_seed=int(rollout_seed + 11),
        num_rollouts_per_skill=temporal_cfg['graph_rollouts_per_skill'],
        record_pixeled_obs=record_pixeled_obs,
    )

    payloads, payload_lookup = _build_temporal_graph_payloads(
        agent,
        {
            'warmup': warmup_trajectories,
            'graph': graph_skill_trajectories,
            'entropy': entropy_trajectories,
            'dbi': dbi_trajectories,
        }
    )
    if not payloads:
        raise ValueError('Temporal metric evaluation could not collect any valid graph trajectories')

    entropy_records = _attach_graph_node_ids(entropy_records, payload_lookup.get('entropy', {}))
    dbi_records = _attach_graph_node_ids(dbi_records, payload_lookup.get('dbi', {}))
    all_metric_records = entropy_records + dbi_records
    query_node_ids, query_map = _build_query_node_index(all_metric_records)
    if query_node_ids.size < 2:
        raise ValueError('Temporal metric evaluation requires at least two sampled graph nodes')

    for record in entropy_records:
        record['query_positions'] = np.asarray([query_map[int(node_id)] for node_id in record['graph_node_ids']], dtype=np.int64)
    for record in dbi_records:
        record['query_positions'] = np.asarray([query_map[int(node_id)] for node_id in record['graph_node_ids']], dtype=np.int64)

    raw_graph = _build_temporal_graph(
        payloads,
        feature_key='raw_points',
        knn_k=temporal_cfg['knn_k'],
        bridge_cost=temporal_cfg['bridge_cost'],
    )
    enc_graph = _build_temporal_graph(
        payloads,
        feature_key='encoded_points',
        knn_k=temporal_cfg['knn_k'],
        bridge_cost=temporal_cfg['bridge_cost'],
    )
    fallback_cost = max(float(len(payloads) + 1), temporal_cfg['bridge_cost'] * 10.0)
    raw_query_distances = _compute_query_shortest_paths(raw_graph, query_node_ids, fallback_cost=fallback_cost)
    enc_query_distances = _compute_query_shortest_paths(enc_graph, query_node_ids, fallback_cost=fallback_cost)

    if record_pixeled_obs:
        _maybe_log(
            logger,
            'Non-IKSE temporal metrics are using recorded observations for raw graph construction because encoder>0 does not expose raw state in the current eval path.',
        )

    soft_dtw_device = 'cuda' if str(agent.device).startswith('cuda') and torch.cuda.is_available() else 'cpu'
    return {
        'entropy_records': entropy_records,
        'dbi_records': dbi_records,
        'raw_query_distances': raw_query_distances,
        'enc_query_distances': enc_query_distances,
        'soft_dtw_gamma': temporal_cfg['soft_dtw_gamma'],
        'soft_dtw_device': soft_dtw_device,
    }


def _resolve_temporal_metric_config(agent):
    num_warmup_rollouts = int(_agent_attr(agent, 'temporal_graph_num_warmup_rollouts', 32))
    graph_rollouts_per_skill = int(_agent_attr(agent, 'temporal_graph_rollouts_per_skill', 5))
    knn_k = int(_agent_attr(agent, 'temporal_graph_knn_k', 8))
    bridge_cost = float(_agent_attr(agent, 'temporal_bridge_cost', 5.0))
    soft_dtw_gamma = float(_agent_attr(agent, 'soft_dtw_gamma', 1.0))

    if num_warmup_rollouts < 0:
        raise ValueError('temporal_graph_num_warmup_rollouts must be >= 0')
    if graph_rollouts_per_skill < 0:
        raise ValueError('temporal_graph_rollouts_per_skill must be >= 0')
    if knn_k < 1:
        raise ValueError('temporal_graph_knn_k must be >= 1')
    if bridge_cost <= 0:
        raise ValueError('temporal_bridge_cost must be > 0')
    if soft_dtw_gamma <= 0:
        raise ValueError('soft_dtw_gamma must be > 0')

    return {
        'num_warmup_rollouts': num_warmup_rollouts,
        'graph_rollouts_per_skill': graph_rollouts_per_skill,
        'knn_k': knn_k,
        'bridge_cost': bridge_cost,
        'soft_dtw_gamma': soft_dtw_gamma,
    }


def _build_temporal_graph_payloads(agent, trajectory_sets):
    payloads = []
    payload_lookup = {}
    node_offset = 0
    global_traj_idx = 0

    for set_name, trajectories in trajectory_sets.items():
        set_lookup = {}
        for trajectory_list_idx, traj in enumerate(trajectories):
            payload = _extract_graph_trajectory_payload(agent, traj)
            if payload is None:
                continue
            payload['set_name'] = set_name
            payload['trajectory_list_idx'] = int(trajectory_list_idx)
            payload['global_traj_idx'] = int(global_traj_idx)
            payload['node_offset'] = int(node_offset)
            payloads.append(payload)
            set_lookup[int(trajectory_list_idx)] = payload
            node_offset += int(payload['num_nodes'])
            global_traj_idx += 1
        payload_lookup[set_name] = set_lookup

    return payloads, payload_lookup


def _extract_graph_trajectory_payload(agent, traj):
    obs = np.asarray(traj['observations'])
    if obs.ndim != 2 or obs.shape[0] == 0:
        return None

    if not bool(_agent_attr(agent, 'encoder', 0)) and _looks_like_flat_pixel_observation(obs):
        raise ValueError(
            'Temporal graph construction received flattened pixel observations while encoder=0. '
            'Expected state observations for temporal metrics.'
        )

    raw_points = obs[:, 0:2].astype(np.float32)
    if raw_points.ndim != 2 or raw_points.shape[0] == 0 or raw_points.shape[1] == 0:
        return None
    encoded_points = _encode_sampled_points(agent, obs).astype(np.float32)
    if encoded_points.ndim != 2 or encoded_points.shape[0] != raw_points.shape[0]:
        return None

    return {
        'num_nodes': int(raw_points.shape[0]),
        'raw_points': raw_points,
        'encoded_points': encoded_points,
    }


def _attach_graph_node_ids(records, payload_lookup):
    attached = []
    for record in records:
        payload = payload_lookup.get(int(record['trajectory_list_idx']))
        if payload is None:
            continue
        sampled_indices = np.asarray(record['sampled_indices'], dtype=np.int64)
        sampled_indices = sampled_indices[sampled_indices < int(payload['num_nodes'])]
        if sampled_indices.size == 0:
            continue
        new_record = dict(record)
        new_record['graph_node_ids'] = int(payload['node_offset']) + sampled_indices
        attached.append(new_record)
    return attached


def _build_query_node_index(records):
    ordered_node_ids = []
    node_to_query = {}
    for record in records:
        for node_id in np.asarray(record['graph_node_ids'], dtype=np.int64):
            node_id = int(node_id)
            if node_id not in node_to_query:
                node_to_query[node_id] = len(ordered_node_ids)
                ordered_node_ids.append(node_id)
    return np.asarray(ordered_node_ids, dtype=np.int64), node_to_query


def _build_temporal_graph(payloads, *, feature_key: str, knn_k: int, bridge_cost: float):
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    num_nodes = sum(int(payload['num_nodes']) for payload in payloads)
    edge_weights = {}

    node_features = []
    node_traj_ids = []
    max_traj_len = 1
    for payload in payloads:
        start = int(payload['node_offset'])
        count = int(payload['num_nodes'])
        max_traj_len = max(max_traj_len, count)
        node_features.append(np.asarray(payload[feature_key], dtype=np.float32))
        node_traj_ids.append(np.full(count, int(payload['global_traj_idx']), dtype=np.int64))
        for local_idx in range(count - 1):
            src = start + local_idx
            dst = start + local_idx + 1
            _register_edge(edge_weights, src, dst, 1.0)
            _register_edge(edge_weights, dst, src, 1.0)

    if num_nodes == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)

    features = np.concatenate(node_features, axis=0)
    traj_ids = np.concatenate(node_traj_ids, axis=0)
    candidate_neighbors = min(num_nodes, max(max_traj_len + knn_k + 1, knn_k * 32 + 1))
    if candidate_neighbors > 1:
        nn = NearestNeighbors(n_neighbors=candidate_neighbors)
        nn.fit(features)
        _, neighbor_indices = nn.kneighbors(features, return_distance=True)
        for node_idx in range(num_nodes):
            added = 0
            for neighbor_idx in neighbor_indices[node_idx, 1:]:
                neighbor_idx = int(neighbor_idx)
                if traj_ids[node_idx] == traj_ids[neighbor_idx]:
                    continue
                _register_edge(edge_weights, node_idx, neighbor_idx, bridge_cost)
                _register_edge(edge_weights, neighbor_idx, node_idx, bridge_cost)
                added += 1
                if added >= knn_k:
                    break

    if not edge_weights:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    rows = np.fromiter((src for src, _ in edge_weights.keys()), dtype=np.int64)
    cols = np.fromiter((dst for _, dst in edge_weights.keys()), dtype=np.int64)
    data = np.fromiter(edge_weights.values(), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)


def _register_edge(edge_weights, src: int, dst: int, weight: float):
    key = (int(src), int(dst))
    prev = edge_weights.get(key)
    if prev is None or weight < prev:
        edge_weights[key] = float(weight)


def _compute_query_shortest_paths(graph, query_node_ids: np.ndarray, *, fallback_cost: float):
    from scipy.sparse.csgraph import dijkstra

    if query_node_ids.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    distances = dijkstra(graph, directed=False, indices=query_node_ids, return_predecessors=False)
    pairwise = distances[:, query_node_ids]
    finite = pairwise[np.isfinite(pairwise)]
    if finite.size > 0:
        replacement = max(float(fallback_cost), float(finite.max()) * 2.0)
    else:
        replacement = float(fallback_cost)
    pairwise = np.where(np.isfinite(pairwise), pairwise, replacement)
    return pairwise.astype(np.float32)


def _compute_temporal_entropy_metrics(agent, graph_context):
    entropy_records = graph_context['entropy_records']
    if not entropy_records:
        return 0.0, 0.0

    raw_positions = _flatten_record_query_positions(entropy_records)
    enc_positions = raw_positions
    raw_distance_matrix = graph_context['raw_query_distances'][np.ix_(raw_positions, raw_positions)]
    enc_distance_matrix = graph_context['enc_query_distances'][np.ix_(enc_positions, enc_positions)]
    entropy_raw = _calculate_temporal_particle_entropy_from_distances(raw_distance_matrix, agent.device)
    entropy_enc = _calculate_temporal_particle_entropy_from_distances(enc_distance_matrix, agent.device)
    return entropy_raw, entropy_enc


def _compute_temporal_dbi_metrics(agent, records, graph_context):
    encoded_concat, traj_obs_counts, skill_labels = _build_plot_payload(records)
    dbi_records = graph_context['dbi_records']
    if not dbi_records:
        return 0.0, 0.0, encoded_concat, traj_obs_counts, skill_labels

    labels = [int(record['skill_idx']) for record in dbi_records]
    raw_distance_matrix, soft_dtw_device = _calculate_soft_dtw_trajectory_distance_matrix(
        graph_context['raw_query_distances'],
        dbi_records,
        agent.device,
        gamma=graph_context['soft_dtw_gamma'],
    )
    enc_distance_matrix, _ = _calculate_soft_dtw_trajectory_distance_matrix(
        graph_context['enc_query_distances'],
        dbi_records,
        agent.device,
        gamma=graph_context['soft_dtw_gamma'],
    )
    graph_context['soft_dtw_device'] = soft_dtw_device
    dbi_raw = _calculate_medoid_dbi(raw_distance_matrix, labels)
    dbi_enc = _calculate_medoid_dbi(enc_distance_matrix, labels)
    return dbi_raw, dbi_enc, encoded_concat, traj_obs_counts, skill_labels


def _flatten_record_query_positions(records):
    flattened = []
    for record in records:
        flattened.extend(np.asarray(record['query_positions'], dtype=np.int64).tolist())
    return np.asarray(flattened, dtype=np.int64)


def _calculate_soft_dtw_trajectory_distance_matrix(query_distance_lookup, records, device, *, gamma: float):
    use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
    soft_dtw, soft_dtw_device = _build_soft_dtw(query_distance_lookup, gamma=gamma, use_cuda=use_cuda)

    num_records = len(records)
    trajectory_distances = np.zeros((num_records, num_records), dtype=np.float32)
    for i in range(num_records):
        seq_i = np.asarray(records[i]['query_positions'], dtype=np.int64)
        for j in range(i + 1, num_records):
            seq_j = np.asarray(records[j]['query_positions'], dtype=np.int64)
            trajectory_distances[i, j] = _soft_dtw_distance(soft_dtw, seq_i, seq_j, soft_dtw_device)
            trajectory_distances[j, i] = trajectory_distances[i, j]
    return trajectory_distances, soft_dtw_device


def _build_soft_dtw(query_distance_lookup, *, gamma: float, use_cuda: bool):
    try:
        from pysdtw import SoftDTW
    except ImportError as exc:
        raise ImportError(
            'Temporal/Soft-DTW metrics require pysdtw. Install it in the active environment before running non-IKSE trajectory evaluation.'
        ) from exc

    target_device = 'cuda' if use_cuda else 'cpu'
    lookup = torch.as_tensor(query_distance_lookup, device=target_device, dtype=torch.float32)
    distance_function = _PrecomputedDistanceFunction(lookup)
    return SoftDTW(gamma=gamma, dist_func=distance_function, use_cuda=use_cuda), target_device


def _soft_dtw_distance(soft_dtw, seq_a: np.ndarray, seq_b: np.ndarray, device_name: str):
    seq_a_tensor = torch.as_tensor(seq_a, device=device_name, dtype=torch.float32).view(1, -1, 1)
    seq_b_tensor = torch.as_tensor(seq_b, device=device_name, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        distance = soft_dtw(seq_a_tensor, seq_b_tensor)
    return float(distance.reshape(-1)[0].detach().cpu().item())


def _calculate_temporal_particle_entropy_from_distances(distance_matrix, device, *, k: int = 3, eps: float = 1e-9):
    distances = torch.as_tensor(np.asarray(distance_matrix), device=device).float()
    if distances.ndim != 2 or distances.shape[0] < 2:
        return 0.0

    num_points = int(distances.shape[0])
    effective_k = min(int(k), num_points - 1)
    if effective_k < 1:
        return 0.0

    distances = distances.clone()
    distances.fill_diagonal_(float('inf'))
    kth_distances = torch.topk(distances, k=effective_k, largest=False).values[:, -1]
    dtype = distances.dtype
    n_tensor = torch.tensor(float(num_points), dtype=dtype, device=device)
    k_tensor = torch.tensor(float(effective_k), dtype=dtype, device=device)
    entropy = torch.digamma(n_tensor) - torch.digamma(k_tensor) + torch.mean(torch.log(kth_distances + eps))
    return float(entropy.item())


def _calculate_medoid_dbi(distance_matrix, labels, *, eps: float = 1e-8):
    distance_matrix = np.asarray(distance_matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return 0.0

    medoid_indices = []
    spreads = []
    valid_labels = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        if cluster_indices.size == 0:
            continue
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_local = int(np.argmin(cluster_distances.mean(axis=1)))
        medoid_idx = int(cluster_indices[medoid_local])
        medoid_indices.append(medoid_idx)
        spreads.append(float(distance_matrix[cluster_indices, medoid_idx].mean()))
        valid_labels.append(int(label))

    if len(valid_labels) < 2:
        return 0.0

    spreads = np.asarray(spreads, dtype=np.float32)
    medoid_distances = distance_matrix[np.ix_(medoid_indices, medoid_indices)]
    ratios = np.zeros_like(medoid_distances, dtype=np.float32)
    for i in range(len(valid_labels)):
        for j in range(len(valid_labels)):
            if i == j:
                continue
            ratios[i, j] = (spreads[i] + spreads[j]) / (medoid_distances[i, j] + eps)
    return float(ratios.max(axis=1).mean())


def _fit_and_map_points(points, ensemble_size: int, subsample_size: int, device):
    kernel = SoftIsolationKernel(points.shape[1], ensemble_size, subsample_size, device=device).to(device)
    tensor_points = torch.from_numpy(points).to(device).float()
    kernel.fit(tensor_points)
    with torch.no_grad():
        return kernel(tensor_points)


def _concat_record_points(records, key: str) -> np.ndarray:
    point_sets = [np.asarray(record[key], dtype=np.float32) for record in records if np.asarray(record[key]).shape[0] > 0]
    if not point_sets:
        raise ValueError(f'No valid point sets available for {key}')
    return np.concatenate(point_sets, axis=0)


def _build_plot_payload(records):
    if not records:
        return None, [], []
    encoded_concat = _concat_record_points(records, 'encoded_points')
    traj_obs_counts = [int(record['encoded_points'].shape[0]) for record in records if int(record['encoded_points'].shape[0]) > 0]
    skill_labels = [int(record['skill_idx']) for record in records if int(record['encoded_points'].shape[0]) > 0]
    return encoded_concat, traj_obs_counts, skill_labels


def _group_points_by_skill(records, key: str):
    grouped = {}
    for record in records:
        points = np.asarray(record[key], dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        grouped.setdefault(int(record['skill_idx']), []).append(points)
    return [(skill_idx, np.concatenate(grouped[skill_idx], axis=0)) for skill_idx in sorted(grouped)]


def _default_num_eval_skills(agent, discrete: bool, dim_skill: int) -> int:
    requested = int(max(1, _agent_attr(agent, 'num_random_trajectories', 16)))
    return min(dim_skill if discrete else 16, requested)


def _prepare_image_frame(img, target_h, target_w):
    img = np.asarray(img)
    if img.ndim == 1:
        if img.size == target_h * target_w * 3:
            img = img.reshape(target_h, target_w, 3)
        else:
            pixels = img.size // 3
            side = int(np.sqrt(max(pixels, 1)))
            if side * side * 3 != img.size:
                return None
            img = img.reshape(side, side, 3)

    if img.ndim == 2:
        return None

    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[1] > 4:
        img = img.transpose(1, 2, 0)

    if img.ndim != 3 or img.shape[-1] not in (1, 3, 4):
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _looks_like_flat_pixel_observation(obs: np.ndarray) -> bool:
    obs = np.asarray(obs)
    if obs.ndim != 2 or obs.shape[1] < 3:
        return False
    flat_dim = int(obs.shape[1])
    channels, rem = divmod(flat_dim, 3)
    if rem != 0:
        return False
    side = int(np.sqrt(max(channels, 1)))
    return side * side * 3 == flat_dim and side >= 16


def _calculate_dbi(mappings, labels, ensemble_size, device):
    if not isinstance(mappings, torch.Tensor):
        mappings = torch.from_numpy(mappings).to(device).float()

    unique_labels = torch.unique(labels)
    num_clusters = len(unique_labels)
    if num_clusters < 2:
        return 0.0

    centroids = torch.stack([mappings[labels == label].mean(0) for label in unique_labels])
    distances = 1.0 - torch.matmul(centroids, centroids.T) / ensemble_size

    spreads = torch.diag(distances)
    ratios = (spreads.view(-1, 1) + spreads.view(1, -1)) / torch.clamp(distances, min=1e-8)
    ratios.fill_diagonal_(0)
    mean_ratios = torch.sum(ratios, dim=1) / (num_clusters - 1)
    return torch.mean(mean_ratios).item()


def _calculate_ik_entropy(maps, ensemble_size, device):
    if not isinstance(maps, torch.Tensor):
        maps = torch.from_numpy(maps).to(device).float()
    phi_hat = maps.mean(dim=0)
    p_s = torch.sum(maps * phi_hat, dim=1) / ensemble_size
    entropy = -torch.mean(torch.log(p_s + 1e-9))
    return entropy.item()


def _calculate_particle_entropy(points, device, *, k: int = 3, eps: float = 1e-9):
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(np.asarray(points)).to(device).float()
    else:
        points = points.to(device).float()

    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        return 0.0

    num_points = int(points.shape[0])
    effective_k = min(int(k), num_points - 1)
    if effective_k < 1:
        return 0.0

    distances = torch.cdist(points, points, p=2)
    distances.fill_diagonal_(float('inf'))
    kth_distances = torch.topk(distances, k=effective_k, largest=False).values[:, -1]

    dtype = points.dtype
    pi = torch.tensor(np.pi, dtype=dtype, device=device)
    dim = torch.tensor(float(points.shape[1]), dtype=dtype, device=device)
    n_tensor = torch.tensor(float(num_points), dtype=dtype, device=device)
    k_tensor = torch.tensor(float(effective_k), dtype=dtype, device=device)
    log_unit_ball_volume = 0.5 * dim * torch.log(pi) - torch.lgamma(0.5 * dim + 1.0)
    entropy = (
        torch.digamma(n_tensor)
        - torch.digamma(k_tensor)
        + log_unit_ball_volume
        + dim * torch.mean(torch.log(2.0 * kth_distances + eps))
    )
    return entropy.item()


def _calculate_euclidean_dbi(cluster_points, device, *, eps: float = 1e-8):
    valid_clusters = []
    for _, points in cluster_points:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] > 0:
            valid_clusters.append(torch.from_numpy(points).to(device).float())

    if len(valid_clusters) < 2:
        return 0.0

    centroids = torch.stack([cluster.mean(dim=0) for cluster in valid_clusters])
    spreads = torch.stack([
        torch.norm(cluster - centroid, dim=1).mean() if cluster.shape[0] > 1 else torch.zeros((), device=device)
        for cluster, centroid in zip(valid_clusters, centroids)
    ])
    separations = torch.cdist(centroids, centroids, p=2)
    ratios = (spreads.view(-1, 1) + spreads.view(1, -1)) / (separations + eps)
    ratios.fill_diagonal_(0)
    return ratios.max(dim=1).values.mean().item()


def _agent_attr(agent, name, default=None):
    if hasattr(agent, name):
        return getattr(agent, name)
    cfg = getattr(agent, 'cfg', None)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def _maybe_log(logger, message):
    if logger is not None:
        logger.info(message)


def _maybe_warn(logger, message):
    if logger is None:
        return
    warn = getattr(logger, 'warning', None)
    if callable(warn):
        warn(message)
    else:
        logger.info(message)
