from __future__ import annotations

import numpy as np
import torch

from utils import utils
from utils import agent_utils
from data_structs.trajectory_batch import TrajectoryBatch
from envs.kitchen.metrics import calc_kitchen_eval_metrics


def is_kitchen_env(env) -> bool:
    """Heuristic detection for Kitchen envs (unchanged logic, just moved out)."""
    try:
        name = getattr(env, "name", None)
        if isinstance(name, str) and ("kitchen" in name.lower()):
            return True
    except Exception:
        pass
    try:
        clsname = env.__class__.__name__.lower()
        if "kitchen" in clsname:
            return True
    except Exception:
        pass
    try:
        spaces = getattr(env, "obs_space", None)
        if isinstance(spaces, dict) and ("success" in spaces):
            return True
    except Exception:
        pass
    return False


def evaluate_policy(agent) -> None:
    if "pre_training" == agent.stage:
        evaluate_pretrain_policy(agent)
    else:
        evaluate_finetune_policy(agent)


def evaluate_finetune_policy(agent) -> None:
    avg_ret = 0
    kitchen_successes = {}
    completed_tasks_counts = []

    num_episodes = 10
    extras = None
    if agent.stage == "finetune":
        skill = agent.best_skill
        # Run evaluation episodes using rollout_worker; reuse _get_trajectories
        extras = agent._generate_skill_extras(np.repeat(skill[None, :], num_episodes, axis=0))

    trajectories = agent._get_trajectories(
        batch_size=num_episodes,
        extras=extras,
        deterministic_policy=True,
    )
    official_kitchen_metrics = {}
    if is_kitchen_env(agent._env):
        official_kitchen_metrics = calc_kitchen_eval_metrics(trajectories)
    tracker = getattr(agent, "coverage_tracker", None)
    if tracker is not None:
        official_kitchen_metrics.update(tracker.compute_policy_metrics(trajectories))
        official_kitchen_metrics.update(tracker.compute_queue_metrics())
        official_kitchen_metrics.update(tracker.compute_total_metrics())

    # Discounted return
    with utils.GlobalContext({"phase": "eval", "policy": "skill"}):
        perf = utils.log_performance_ex(
            agent.step_itr,
            batch=TrajectoryBatch.from_trajectory_list(agent._env_spec, trajectories),
            discount=agent.discount,
            additional_records=official_kitchen_metrics,
        )
    avg_ret = np.mean(perf["undiscounted_returns"])
    agent.writer.add_scalar("eval/undiscounted_return", avg_ret, agent.step_itr)

    # For kitchen success tracking
    if is_kitchen_env(agent._env):
        for path in trajectories:
            ep_completed_count = 0
            if "env_infos" in path:
                for k, v in path["env_infos"].items():
                    if "success" in k and "distance" not in k:
                        if k not in kitchen_successes:
                            kitchen_successes[k] = []
                        is_completed = v[-1] > 0.5
                        kitchen_successes[k].append(float(is_completed))
                        if is_completed:
                            ep_completed_count += 1
            completed_tasks_counts.append(ep_completed_count)

    # Log Eval Success Rates
    if is_kitchen_env(agent._env):
        official_metrics = official_kitchen_metrics
        for key, value in official_metrics.items():
            agent.writer.add_scalar(f"eval/{key}", value, agent.step_itr)
        agent.writer.add_scalar(
            "eval/kitchen/overall_6task_coverage",
            official_metrics["KitchenOverall"],
            agent.step_itr,
        )
        agent.writer.add_scalar(
            "eval/kitchen/policy_task_coverage",
            official_metrics["KitchenPolicyTaskCoverage"],
            agent.step_itr,
        )
        agent.writer.add_scalar(
            "eval/avg_completed_tasks",
            official_metrics["KitchenAvgCompletedTasksPerTraj"],
            agent.step_itr,
        )

        for k, v in kitchen_successes.items():
            avg_success = np.mean(v)
            agent.writer.add_scalar(f"eval/{k}_rate", avg_success, agent.step_itr)
            agent.logger.info(f"Step {agent.step_itr}: Eval {k} Rate = {avg_success:.2f}")

        # Log Avg Completed Tasks
        if completed_tasks_counts:
            avg_completed = np.mean(completed_tasks_counts)
            agent.writer.add_scalar("eval_legacy/avg_completed_tasks", avg_completed, agent.step_itr)
            agent.logger.info(f"Step {agent.step_itr}: Eval Avg Completed Tasks = {avg_completed:.10f}")
            print(f"Step {agent.step_itr}: Eval Avg Completed Tasks = {avg_completed:.10f}")

    agent.logger.info(f"Step {agent.step_itr}: Eval Discounted Return = {avg_ret}")
    print(f"Step {agent.step_itr}: Eval Discounted Return = {avg_ret}")

    # Record Video if requested
    if agent.eval_record_video:
        agent.logger.info("Recording video...")
        video_extras = {}
        if "finetune" == agent.stage:
            video_extras = agent._generate_skill_extras(np.repeat(skill[None, :], 1, axis=0))
        video_trajectories = agent._get_trajectories(
            batch_size=1,
            extras=video_extras,
            deterministic_policy=True,
            state_record_pixeled=not agent.use_encoder,
        )

        utils.record_video(
            agent.snapshot_dir,
            agent.step_itr,
            "video_finetune",
            video_trajectories,
            skip_frames=agent.video_skip_frames,
            shape=(128, 128),
        )


def evaluate_pretrain_policy(agent) -> None:
    # NOTE: logic is identical to original; only moved out.
    if agent.discrete:
        eye_skills = np.eye(agent.dim_skill)
        random_skills = []
        colors = []
        for i in range(agent.dim_skill):
            num_trajs_per_skill = agent.num_random_trajectories // agent.dim_skill + (
                    i < agent.num_random_trajectories % agent.dim_skill
            )
            for _ in range(num_trajs_per_skill):
                random_skills.append(eye_skills[i])
                colors.append(i)
        random_skills = np.array(random_skills)
        colors = np.array(colors)
        num_evals = len(random_skills)
        from matplotlib import cm
        cmap = "tab10" if agent.dim_skill <= 10 else "tab20"
        random_skill_colors = []
        for i in range(num_evals):
            random_skill_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
        random_skill_colors = np.array(random_skill_colors)
    else:
        random_skills = np.random.randn(agent.num_random_trajectories, agent.dim_skill)
        if agent.unit_length:
            random_skills = random_skills / np.linalg.norm(random_skills, axis=1, keepdims=True)
        random_skill_colors = utils.get_skill_colors(random_skills * 4)

    # Rollout random trajectories
    random_trajectories = agent._get_trajectories(
        batch_size=agent.num_random_trajectories,
        extras=agent._generate_skill_extras(random_skills),
        deterministic_policy=True,
    )

    if False:  # TODO: keep as-is
        with utils.FigManager(
                agent.snapshot_dir, agent.step_itr, "TrajPlot_RandomZ", writer=agent.writer, global_step=agent.step_itr
        ) as fm:
            agent._env.render_trajectories(random_trajectories, random_skill_colors, agent.eval_plot_axis, fm.ax)

    data = agent_utils.process_samples(random_trajectories, agent.cfg.sac_discount)
    last_obs = torch.stack([torch.from_numpy(ob[-1]).float().to(agent.device) for ob in data["obs"]])

    skill_dists = agent.traj_encoder(last_obs)
    skill_means_tensor = agent.encode_phi(last_obs, use_target=False)
    skill_means = skill_means_tensor.detach().cpu().numpy()
    if agent.inner:
        skill_stddevs = torch.ones_like(skill_means_tensor.detach().cpu()).numpy()
    elif agent.traj_latent_normalizer is not None:
        skill_stddevs = torch.ones_like(skill_means_tensor.detach().cpu()).numpy()
    else:
        skill_stddevs = skill_dists.stddev.detach().cpu().numpy()
    skill_samples = skill_means

    skill_colors = random_skill_colors

    with utils.FigManager(
            agent.snapshot_dir, agent.step_itr, "PhiPlot", writer=agent.writer, global_step=agent.step_itr
    ) as fm:
        utils.draw_2d_gaussians(skill_means, skill_stddevs, skill_colors, fm.ax)
        utils.draw_2d_gaussians(
            skill_samples,
            [[0.03, 0.03]] * len(skill_samples),
            skill_colors,
            fm.ax,
            fill=True,
            use_adaptive_axis=True,
            )

    eval_skill_metrics = {}

    # Videos
    if agent.eval_record_video:
        print("Recording video.\n\n\n\n\n")
        if agent.discrete:
            video_skills = np.eye(agent.dim_skill)
            video_skills = video_skills.repeat(agent.num_video_repeats, axis=0)
        else:
            if agent.dim_skill == 2:
                radius = 1.0 if agent.unit_length else 1.5
                video_skills = []
                for angle in [3, 2, 1, 4]:
                    video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_skills.append([0, 0])
                for angle in [0, 5, 6, 7]:
                    video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_skills = np.array(video_skills)
            else:
                video_skills = np.random.randn(16, agent.dim_skill)
                if agent.unit_length:
                    video_skills = video_skills / np.linalg.norm(video_skills, axis=1, keepdims=True)
            video_skills = video_skills.repeat(agent.num_video_repeats, axis=0)

        video_trajectories = agent._get_trajectories(
            batch_size=len(video_skills),
            deterministic_policy=True,
            extras=agent._generate_skill_extras(video_skills),
            state_record_pixeled=not agent.use_encoder,
        )
        utils.record_video(
            agent.snapshot_dir,
            agent.step_itr,
            "Video_RandomZ",
            video_trajectories,
            skip_frames=agent.video_skip_frames,
            shape=(128, 128),
        )

    eval_skill_metrics.update(calc_eval_metrics(agent, random_trajectories, is_skill_trajectories=True))
    tracker = getattr(agent, "coverage_tracker", None)
    if tracker is not None:
        eval_skill_metrics.update(tracker.compute_policy_metrics(random_trajectories))
        eval_skill_metrics.update(tracker.compute_queue_metrics())
        eval_skill_metrics.update(tracker.compute_total_metrics())
    with utils.GlobalContext({"phase": "eval", "policy": "skill"}):
        performance = utils.log_performance_ex(
            agent.step_itr,
            TrajectoryBatch.from_trajectory_list(agent._env_spec, random_trajectories),
            discount=agent.discount,
            additional_records=eval_skill_metrics,
        )
        # Log performance metrics with 'eval/' prefix
        for k, v in performance["scalars"].items():
            agent.writer.add_scalar("eval/" + k, v, agent.step_itr)
        for k, v in performance["histograms"].items():
            agent.writer.add_histogram("eval/" + k, v, agent.step_itr)

    agent._log_eval_metrics()


def calc_eval_metrics(agent, trajectories, is_skill_trajectories: bool = True) -> dict:
    _ = is_skill_trajectories
    eval_metrics = {}
    sum_returns = 0
    for traj in trajectories:
        sum_returns += traj["rewards"].sum()
    eval_metrics["ReturnOverall"] = sum_returns
    if is_kitchen_env(agent._env):
        calc_env_metrics = getattr(agent._env, "calc_eval_metrics", None)
        if callable(calc_env_metrics):
            eval_metrics.update(calc_env_metrics(trajectories, is_option_trajectories=True))
        else:
            eval_metrics.update(calc_kitchen_eval_metrics(trajectories))
    else:
        calc_env_metrics = getattr(agent._env, "calc_eval_metrics", None)
        if callable(calc_env_metrics):
            eval_metrics.update(calc_env_metrics(trajectories, is_option_trajectories=True))
    return eval_metrics
