import argparse
from .base import MetraConfig
from core.stage_contract import (
    SUPPORTED_ALGO_NAMES,
    get_base_algo_name,
    is_cascade_algo,
    validate_stage_config,
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    hidden = argparse.SUPPRESS
    
    # Helper to add arguments from dataclass
    # Mapping for compatibility with existing command lines
    
    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--workspace_root', type=str, default='/share/shangyy')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=1)
    parser.add_argument(
        '--encoder_type',
        type=str,
        default='original',
        choices=['original', 'resnet-101', 'dinov3', 'galaxea-r1lite-triview'],
    )
    parser.add_argument('--finetune_encoder', action='store_true', default=False)
    parser.add_argument('--env-backend', type=str, default='url', choices=['url', 'isaaclab'], help=hidden)
    parser.add_argument('--task', type=str, default='dmc_walker_walk')
    parser.add_argument('--framestack', type=int, default=1)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--flatten_obs', type=int, default=1, choices=[0, 1])
    parser.add_argument('--camera', type=str, default='corner')
    parser.add_argument('--dmc_camera', type=int, default=-1)
    parser.add_argument('--isaaclab-task', type=str, default='', help=hidden)
    parser.add_argument('--isaaclab-num-envs', type=int, default=1, help=hidden)
    parser.add_argument('--isaaclab-headless', type=int, default=1, choices=[0, 1], help=hidden)
    parser.add_argument('--isaaclab-enable-cameras', type=int, default=None, choices=[0, 1], help=hidden)
    parser.add_argument('--isaaclab-render-mode', type=str, default='rgb_array', help=hidden)
    parser.add_argument('--isaaclab-image-source', type=str, default=None, choices=['auto', 'render', 'camera'], help=hidden)
    parser.add_argument('--isaaclab-camera-key', type=str, default=None, help=hidden)
    parser.add_argument(
        '--isaaclab-video-source',
        type=str,
        default='observation',
        choices=['observation', 'render'],
        help=hidden,
    )
    parser.add_argument(
        '--isaaclab-video-viewer-preset',
        type=str,
        default='inherit',
        choices=['inherit', 'panorama_fixed'],
        help=hidden,
    )
    parser.add_argument('--galaxea-sim-headless', dest='galaxea_sim_headless', type=int, default=1, choices=[0, 1])
    parser.add_argument('--galaxea-sim-obs-mode', dest='galaxea_sim_obs_mode', type=str, default=None, choices=['state', 'image'])
    parser.add_argument('--galaxea-sim-image-key', dest='galaxea_sim_image_key', type=str, default='rgb_head')
    parser.add_argument(
        '--galaxea-sim-video-source',
        dest='galaxea_sim_video_source',
        type=str,
        default='observation',
        choices=['observation', 'third_person', 'render'],
    )
    parser.add_argument(
        '--galaxea-sim-video-view-preset',
        dest='galaxea_sim_video_view_preset',
        type=str,
        default='default',
        choices=['default', 'robot_full_body'],
    )
    parser.add_argument(
        '--galaxea-sim-controller-type',
        dest='galaxea_sim_controller_type',
        type=str,
        default='bimanual_joint_position',
        choices=['bimanual_joint_position', 'bimanual_ee_pose', 'bimanual_relaxed_ik'],
    )
    parser.add_argument('--safety-enabled', dest='safety_enabled', type=int, default=0, choices=[0, 1])
    parser.add_argument('--safety-mode', dest='safety_mode', type=str, default='sim', choices=['off', 'sim', 'shadow', 'real'])
    parser.add_argument('--safety-lbsgd-enabled', dest='safety_lbsgd_enabled', type=int, default=1, choices=[0, 1])
    parser.add_argument('--safety-qp-enabled', dest='safety_qp_enabled', type=int, default=1, choices=[0, 1])
    parser.add_argument('--safety-supervisor-enabled', dest='safety_supervisor_enabled', type=int, default=1, choices=[0, 1])
    parser.add_argument(
        '--safety-action-semantics',
        dest='safety_action_semantics',
        type=str,
        default='auto',
        choices=['auto', 'absolute_joint_position', 'delta_joint_position', 'joint_velocity'],
    )
    parser.add_argument('--safety-horizon', dest='safety_horizon', type=int, default=1)
    parser.add_argument('--safety-barrier-eta', dest='safety_barrier_eta', type=float, default=1e-2)
    parser.add_argument('--safety-lbsgd-lr', dest='safety_lbsgd_lr', type=float, default=1e-2)
    parser.add_argument('--safety-lbsgd-steps', dest='safety_lbsgd_steps', type=int, default=8)
    parser.add_argument('--safety-min-barrier-margin', dest='safety_min_barrier_margin', type=float, default=1e-4)
    parser.add_argument('--safety-deviation-weight', dest='safety_deviation_weight', type=float, default=1.0)
    parser.add_argument('--safety-critic-weight', dest='safety_critic_weight', type=float, default=0.0)
    parser.add_argument('--safety-qp-warmup-steps', dest='safety_qp_warmup_steps', type=int, default=0)
    parser.add_argument('--safety-lbsgd-warmup-steps', dest='safety_lbsgd_warmup_steps', type=int, default=0)
    parser.add_argument('--safety-lbsgd-ramp-steps', dest='safety_lbsgd_ramp_steps', type=int, default=0)
    parser.add_argument('--safety-accel-warmup-steps', dest='safety_accel_warmup_steps', type=int, default=0)
    parser.add_argument('--safety-shadow-until-steps', dest='safety_shadow_until_steps', type=int, default=0)
    parser.add_argument('--safety-qpos-margin', dest='safety_qpos_margin', type=float, default=0.05)
    parser.add_argument('--safety-dq-limit-scale', dest='safety_dq_limit_scale', type=float, default=0.25)
    parser.add_argument('--safety-ddq-limit-scale', dest='safety_ddq_limit_scale', type=float, default=0.25)
    parser.add_argument('--safety-tau-limit-scale', dest='safety_tau_limit_scale', type=float, default=0.25)
    parser.add_argument('--safety-lock-torso', dest='safety_lock_torso', type=int, default=1, choices=[0, 1])
    parser.add_argument('--safety-lock-chassis', dest='safety_lock_chassis', type=int, default=1, choices=[0, 1])
    parser.add_argument('--safety-yaml', dest='safety_yaml', type=str, default='')
    parser.add_argument('--safety-distill-safe-action-weight', dest='safety_distill_safe_action_weight', type=float, default=0.0)

    parser.add_argument('--time_limit', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_minibatch_size', type=int, default=256)
    parser.add_argument('--trans_optimization_epochs', type=int, default=200)
    parser.add_argument('--parallel_sampler_enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--parallel_sampler_num_workers', type=int, default=0)
    parser.add_argument('--parallel_sampler_fail_open', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_parallel_sampler_enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--eval_video_parallel_sampler_enabled', type=int, default=1, choices=[0, 1])
    parser.add_argument('--async_video_encoding', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--replay_staging_enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--replay_staging_pin_memory', type=int, default=1, choices=[0, 1])

    parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    parser.add_argument('--n_epochs_per_log', type=int, default=25)
    parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_random_trajectories', type=int, default=48)
    parser.add_argument('--ikse', action='store_true', default=False)
    parser.add_argument('--metric_num_sampled_points', type=int, default=10)
    parser.add_argument('--dbi_num_rollouts_per_skill', type=int, default=3)
    parser.add_argument('--temporal_graph_num_warmup_rollouts', type=int, default=32)
    parser.add_argument('--temporal_graph_rollouts_per_skill', type=int, default=5)
    parser.add_argument('--temporal_graph_knn_k', type=int, default=8)
    parser.add_argument('--temporal_bridge_cost', type=float, default=5.0)
    parser.add_argument('--soft_dtw_gamma', type=float, default=1.0)
    parser.add_argument('--eval_structure_metrics', type=int, default=0, choices=[0, 1])
    parser.add_argument('--eval_structure_metrics_backends', type=str, default='temporal,ikse')
    parser.add_argument('--eval_structure_metrics_interval', type=int, default=1)
    parser.add_argument('--eval_structure_metrics_rollouts_per_skill', type=int, default=3)
    parser.add_argument('--eval_structure_metrics_num_skills', type=int, default=-1)
    parser.add_argument('--eval_structure_metrics_max_trajs', type=int, default=96)
    parser.add_argument('--eval_structure_metrics_max_points', type=int, default=1000)
    parser.add_argument('--eval_structure_metrics_states_per_traj', type=int, default=10)
    parser.add_argument('--eval_structure_metrics_anchor_seed', type=int, default=0)
    parser.add_argument('--eval_structure_metrics_use_video_trajectories', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_structure_metrics_policy_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--eval_structure_metrics_reset_perturb_scale', type=float, default=0.0)
    parser.add_argument('--eval_structure_metrics_degenerate_policy', type=str, default='skip', choices=['skip', 'warn_compute'])
    parser.add_argument('--eval_structure_metrics_fail_open', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_structure_metrics_write_legacy_tags', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)
    parser.add_argument('--motion-analysis-enabled', dest='motion_analysis_enabled', type=int, default=0, choices=[0, 1])
    parser.add_argument('--motion-analysis-video-path', dest='motion_analysis_video_path', type=str, default='')
    parser.add_argument('--motion-analysis-resize-h', dest='motion_analysis_resize_h', type=int, default=128)
    parser.add_argument('--motion-analysis-resize-w', dest='motion_analysis_resize_w', type=int, default=128)
    parser.add_argument('--motion-analysis-blur-kernel', dest='motion_analysis_blur_kernel', type=int, default=3)
    parser.add_argument('--motion-analysis-frame-gap', dest='motion_analysis_frame_gap', type=int, default=1)
    parser.add_argument('--motion-analysis-pixel-threshold-mode', dest='motion_analysis_pixel_threshold_mode',
                        type=str, default='adaptive', choices=['adaptive', 'fixed'])
    parser.add_argument('--motion-analysis-fixed-tau-p', dest='motion_analysis_fixed_tau_p', type=float, default=0.04)
    parser.add_argument('--motion-analysis-smooth-window', dest='motion_analysis_smooth_window', type=int, default=5)
    parser.add_argument('--motion-analysis-large-motion-threshold',
                        dest='motion_analysis_large_motion_threshold', type=float, default=2.0)
    parser.add_argument('--motion-analysis-eps', dest='motion_analysis_eps', type=float, default=1e-8)

    parser.add_argument('--dim_skill', type=int, default=2)

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--algo', type=str, default='metra', choices=SUPPORTED_ALGO_NAMES)

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=300000)
    parser.add_argument('--policy_delay', type=int, default=1, help="Delay policy updates by this factor")
    parser.add_argument('--actor_start_steps', type=int, default=0, help="Steps to warm up critic before updating actor")

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    parser.add_argument('--num_alt_samples', type=int, default=100)
    parser.add_argument('--split_group', type=int, default=65536)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])  # Only for continuous skills
    parser.add_argument('--traj_latent_norm', type=str, default='off', choices=['off', 'rmsnorm', 'layernorm'])
    parser.add_argument('--traj_latent_norm_eps', type=float, default=1e-5)

    parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one',
                                                                         'skill_kme', 'kernel_mmd', 'kernel_sim_dist', 'kernel_sim'])
    parser.add_argument('--dual_lr', type=float, default=None)
    parser.add_argument('--use_kme', action="store_true", default=False,
                        help='whether use kernel mean embedding')
    parser.add_argument('--update_idk', type=int, default=1000,
                        help='rebuild IDK from replay every N updates (0 to disable)')
    parser.add_argument('--idk_subsample_size', type=int, default=256,
                        help='number of phi(s) to build IDK anchors')
    parser.add_argument('--idk_init', type=str, default='gaussian',
                        choices=['gaussian','uniform','replay'],
                        help='initialization of IDK basis in latent space')
    parser.add_argument('--idk_from', type=str, default='traj',
                        choices=['traj','enc'],
                        help='which latent to use for phi(s): traj encoder mean or pixel encoder feat')
    parser.add_argument('--idk_groups', type=int, default=1,
                        help='compute kernel mean with high samples by split to n groups')
    parser.add_argument('--kernel_map', action="store_true", default=False,
                        help='use kernel to map state encoder')
    parser.add_argument('--use_novelty_reward', action="store_true", default=False,
                        help='whether use kernel mean embedding')
    parser.add_argument('--use_target_traj_encoder', action="store_true", default=False,
                        help='whether use target trajectory encoder')
    parser.add_argument('--use_hierarchical_policy', action='store_true', default=False)
    parser.add_argument('--use_hierarchical_skill', action='store_true', default=False)
    parser.add_argument('--num_skill_levels', type=int, default=1)
    parser.add_argument('--epochs_per_skill_stage', type=int, default=50)
    parser.add_argument('--use_hierarchical_phi', action='store_true', default=False)
    parser.add_argument('--hierarchical_phi_depth', type=int, default=0)
    parser.add_argument('--beta_mode', type=str, default='uniform', choices=['uniform', 'exp_unnorm', 'exp_norm'])
    parser.add_argument('--beta_rho', type=float, default=0.5)
    parser.add_argument('--log_beta_values', type=int, default=1, choices=[0, 1])
    parser.add_argument('--stage', type=str, default='pre_training', choices=['pre_training','finetune', 'zero_training'])
    parser.add_argument('--skill_policy_path', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=50.0)
    parser.add_argument('--actor_init_std', type=float, default=1.0)
    parser.add_argument('--actor_max_log_std', type=float, default=2.0)
    parser.add_argument('--ac_backbone', type=str, default='mlp', choices=['mlp', 'simba'])

    parser.add_argument('--simba_actor_hidden_dim', type=int, default=128)
    parser.add_argument('--simba_actor_num_blocks', type=int, default=1)
    parser.add_argument('--simba_critic_hidden_dim', type=int, default=512)
    parser.add_argument('--simba_critic_num_blocks', type=int, default=2)

    parser.add_argument('--simba_mlp_ratio', type=int, default=4)
    parser.add_argument('--simba_rsnorm_momentum', type=float, default=0.999)
    parser.add_argument('--simba_rsnorm_eps', type=float, default=1e-5)
    parser.add_argument('--simba_ln_eps', type=float, default=1e-5)
    
    # Cascade
    parser.add_argument('--use_cascade', action='store_true', default=False)
    parser.add_argument('--num_policy_levels', type=int, default=2)
    parser.add_argument('--num_cascade_stages', dest='num_policy_levels', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--epochs_per_policy_stage', type=int, default=50)
    parser.add_argument('--cascade_init_from_prev', action='store_true', default=False)
    parser.add_argument('--cascade_gate_type', type=str, default='scalar')
    parser.add_argument('--cascade_min_lambda', type=float, default=0.01)
    parser.add_argument('--cascade_max_lambda', type=float, default=0.99)

    # Auto branch
    parser.add_argument('--auto_branch', '--auto_branch_enabled', dest='enabled', action='store_true', default=False)
    parser.add_argument('--auto_branch_check_interval_epochs', dest='check_interval_epochs', type=int, default=20)
    parser.add_argument('--auto_branch_recent_buffer_epochs', dest='recent_buffer_epochs', type=int, default=100)
    parser.add_argument('--auto_branch_knn_k', dest='knn_k', type=int, default=5)
    parser.add_argument('--auto_branch_representative_points_per_traj', dest='representative_points_per_traj', type=int, default=10)
    parser.add_argument('--auto_branch_distance_mode', dest='distance_mode', type=str, default='knn_mean',
                        choices=['knn_mean', 'knn_kth'])
    parser.add_argument('--auto_branch_m_policy_ratio_threshold', dest='m_policy_ratio_threshold', type=float, default=0.1)
    parser.add_argument('--auto_branch_split_patience', dest='split_patience', type=int, default=3)
    parser.add_argument('--auto_branch_min_branch_age', dest='min_branch_age', type=int, default=150)
    parser.add_argument('--auto_branch_fresh_rollout_episodes', dest='fresh_rollout_episodes', type=int, default=48)
    parser.add_argument('--auto_branch_visualize_on_split', dest='visualize_on_split', type=int, default=1, choices=[0, 1])
    parser.add_argument('--auto_branch_visualize_dir', dest='visualize_dir', type=str, default='split_viz')
    parser.add_argument('--auto_branch_use_global_recent_buffer', dest='use_global_recent_buffer', type=int, default=0, choices=[0, 1])
    parser.add_argument('--auto_branch_seeded_probe_skills', dest='seeded_probe_skills', type=int, default=1, choices=[0, 1])

    return parser

def make_config_from_args(args, cls=MetraConfig) -> MetraConfig:
    from envs import normalize_env_backend_for_task

    # Post-process algo settings to set implied flags
    base_algo = get_base_algo_name(args.algo)
    if is_cascade_algo(args.algo):
        args.use_cascade = True

    if base_algo == 'diayn':
        args.inner = 0
        args.dual_reg = 0
    elif base_algo == 'lsd':
        args.dual_dist = 'l2'
    elif base_algo == 'iksd':
        args.dual_dist = 'kernel_sim_dist'
        args.use_kme = True
    elif base_algo == 'csd':
        args.dual_dist = 's2_from_s'
    elif base_algo == 'metra':
        args.dual_dist = 'one'

    args.env_backend = normalize_env_backend_for_task(
        getattr(args, "task", ""),
        getattr(args, "env_backend", "url"),
    )

    cfg = cls()
    arg_dict = vars(args)
    
    # Root level
    for k in ('seed', 'use_gpu', 'sample_cpu'):
        if k in arg_dict:
            setattr(cfg, k, arg_dict[k])
    if arg_dict.get('device') is not None:
        cfg.device = arg_dict['device']
    elif not cfg.use_gpu:
        cfg.device = 'cpu'
    if arg_dict.get('workspace_root') is not None:
        cfg.log.workspace_root = arg_dict['workspace_root']
    
    # Special handling for Contrastive Params (if present)
    if hasattr(cfg, 'contrastive_n_epochs'):
        for k in ['contrastive_n_epochs', 'contrastive_m_epochs', 'contrastive_warmup_epochs', 'contrastive_temperature']:
            if k in arg_dict:
                setattr(cfg, k, arg_dict[k])
        
    # Sub levels
    for sub_cfg in (cfg.env, cfg.log, cfg.net, cfg.algo, cfg.train, cfg.cascade, cfg.auto_branch):
        for k, v in arg_dict.items():
            if hasattr(sub_cfg, k):
                setattr(sub_cfg, k, v)

    safety_arg_map = {
        'safety_enabled': 'enabled',
        'safety_mode': 'mode',
        'safety_lbsgd_enabled': 'lbsgd_enabled',
        'safety_qp_enabled': 'qp_enabled',
        'safety_supervisor_enabled': 'supervisor_enabled',
        'safety_action_semantics': 'action_semantics',
        'safety_horizon': 'horizon',
        'safety_barrier_eta': 'barrier_eta',
        'safety_lbsgd_lr': 'lbsgd_lr',
        'safety_lbsgd_steps': 'lbsgd_steps',
        'safety_min_barrier_margin': 'min_barrier_margin',
        'safety_deviation_weight': 'deviation_weight',
        'safety_critic_weight': 'critic_weight',
        'safety_qp_warmup_steps': 'qp_warmup_steps',
        'safety_lbsgd_warmup_steps': 'lbsgd_warmup_steps',
        'safety_lbsgd_ramp_steps': 'lbsgd_ramp_steps',
        'safety_accel_warmup_steps': 'accel_warmup_steps',
        'safety_shadow_until_steps': 'shadow_until_steps',
        'safety_qpos_margin': 'qpos_margin',
        'safety_dq_limit_scale': 'dq_limit_scale',
        'safety_ddq_limit_scale': 'ddq_limit_scale',
        'safety_tau_limit_scale': 'tau_limit_scale',
        'safety_lock_torso': 'lock_torso',
        'safety_lock_chassis': 'lock_chassis',
        'safety_yaml': 'safety_yaml',
        'safety_distill_safe_action_weight': 'distill_safe_action_weight',
    }
    for arg_name, cfg_name in safety_arg_map.items():
        if arg_name in arg_dict:
            setattr(cfg.safety, cfg_name, arg_dict[arg_name])

    if (
        bool(getattr(cfg.net, 'encoder', 0))
        and str(getattr(cfg.env, 'task', '')).startswith('galaxea_')
        and getattr(cfg.env, 'galaxea_sim_image_key', '') == 'rgb_left_right_head'
    ):
        if cfg.net.encoder_type in ('resnet-101', 'dinov3'):
            raise ValueError(
                "galaxea_sim_image_key='rgb_left_right_head' requires "
                "encoder_type='galaxea-r1lite-triview' (or leave --encoder_type at "
                "'original' for automatic selection); HuggingFace single-image "
                f"encoder_type={cfg.net.encoder_type!r} is not supported for this v1 path."
            )
        cfg.net.encoder_type = 'galaxea-r1lite-triview'

    motion_arg_map = {
        'motion_analysis_enabled': 'enabled',
        'motion_analysis_video_path': 'video_path',
        'motion_analysis_resize_h': 'resize_h',
        'motion_analysis_resize_w': 'resize_w',
        'motion_analysis_blur_kernel': 'blur_kernel',
        'motion_analysis_frame_gap': 'frame_gap',
        'motion_analysis_pixel_threshold_mode': 'pixel_threshold_mode',
        'motion_analysis_fixed_tau_p': 'fixed_tau_p',
        'motion_analysis_smooth_window': 'smooth_window',
        'motion_analysis_large_motion_threshold': 'large_motion_threshold',
        'motion_analysis_eps': 'eps',
    }
    for arg_name, cfg_name in motion_arg_map.items():
        if arg_name in arg_dict:
            setattr(cfg.motion_analysis, cfg_name, arg_dict[arg_name])

    if cfg.algo.hierarchical_phi_depth == 0:
        cfg.algo.hierarchical_phi_depth = cfg.algo.num_skill_levels
    if cfg.algo.num_skill_levels < 1:
        raise ValueError("num_skill_levels must be >= 1")
    if cfg.algo.hierarchical_phi_depth < 1:
        raise ValueError("hierarchical_phi_depth must be >= 1")
    if cfg.cascade.num_policy_levels < 1:
        raise ValueError("num_policy_levels must be >= 1")
    if cfg.auto_branch.check_interval_epochs < 1:
        raise ValueError("auto_branch.check_interval_epochs must be >= 1")
    if cfg.auto_branch.recent_buffer_epochs < 1:
        raise ValueError("auto_branch.recent_buffer_epochs must be >= 1")
    if cfg.auto_branch.knn_k < 1:
        raise ValueError("auto_branch.knn_k must be >= 1")
    if cfg.auto_branch.representative_points_per_traj < 1:
        raise ValueError("auto_branch.representative_points_per_traj must be >= 1")
    if cfg.auto_branch.split_patience < 1:
        raise ValueError("auto_branch.split_patience must be >= 1")
    if cfg.auto_branch.min_branch_age < 0:
        raise ValueError("auto_branch.min_branch_age must be >= 0")
    if cfg.auto_branch.fresh_rollout_episodes < 1:
        raise ValueError("auto_branch.fresh_rollout_episodes must be >= 1")
    if cfg.train.n_parallel < 1:
        raise ValueError("n_parallel must be >= 1")
    if cfg.train.parallel_sampler_num_workers < 0:
        raise ValueError("parallel_sampler_num_workers must be >= 0")
    if cfg.train.traj_batch_size < 1:
        raise ValueError("traj_batch_size must be >= 1")
    if cfg.train.trans_minibatch_size < 1:
        raise ValueError("trans_minibatch_size must be >= 1")
    if cfg.train.trans_optimization_epochs < 1:
        raise ValueError("trans_optimization_epochs must be >= 1")
    if cfg.log.metric_num_sampled_points < 1:
        raise ValueError("metric_num_sampled_points must be >= 1")
    if not isinstance(cfg.log.ikse, bool):
        raise ValueError("ikse must be a boolean flag")
    if cfg.log.dbi_num_rollouts_per_skill < 3:
        raise ValueError("dbi_num_rollouts_per_skill must be >= 3")
    if cfg.log.temporal_graph_num_warmup_rollouts < 0:
        raise ValueError("temporal_graph_num_warmup_rollouts must be >= 0")
    if cfg.log.temporal_graph_rollouts_per_skill < 0:
        raise ValueError("temporal_graph_rollouts_per_skill must be >= 0")
    if cfg.log.temporal_graph_knn_k < 1:
        raise ValueError("temporal_graph_knn_k must be >= 1")
    if cfg.log.temporal_bridge_cost <= 0:
        raise ValueError("temporal_bridge_cost must be > 0")
    if cfg.log.soft_dtw_gamma <= 0:
        raise ValueError("soft_dtw_gamma must be > 0")
    structure_backends = [
        backend.strip()
        for backend in str(cfg.log.eval_structure_metrics_backends).split(',')
        if backend.strip()
    ]
    if not structure_backends:
        raise ValueError("eval_structure_metrics_backends must include at least one backend")
    invalid_structure_backends = sorted(set(structure_backends) - {'temporal', 'ikse'})
    if invalid_structure_backends:
        raise ValueError(
            "eval_structure_metrics_backends only supports temporal and ikse, "
            f"got {invalid_structure_backends}"
        )
    if cfg.log.eval_structure_metrics_interval < 1:
        raise ValueError("eval_structure_metrics_interval must be >= 1")
    if cfg.log.eval_structure_metrics_rollouts_per_skill < 1:
        raise ValueError("eval_structure_metrics_rollouts_per_skill must be >= 1")
    if cfg.log.eval_structure_metrics_num_skills == 0 or cfg.log.eval_structure_metrics_num_skills < -1:
        raise ValueError("eval_structure_metrics_num_skills must be -1 or >= 1")
    if cfg.log.eval_structure_metrics_max_trajs < 1:
        raise ValueError("eval_structure_metrics_max_trajs must be >= 1")
    if cfg.log.eval_structure_metrics_max_points < 1:
        raise ValueError("eval_structure_metrics_max_points must be >= 1")
    if cfg.log.eval_structure_metrics_states_per_traj < 1:
        raise ValueError("eval_structure_metrics_states_per_traj must be >= 1")
    if cfg.log.eval_structure_metrics_policy_mode not in ('deterministic', 'stochastic'):
        raise ValueError("eval_structure_metrics_policy_mode must be deterministic or stochastic")
    if cfg.log.eval_structure_metrics_reset_perturb_scale < 0:
        raise ValueError("eval_structure_metrics_reset_perturb_scale must be >= 0")
    if cfg.log.eval_structure_metrics_degenerate_policy not in ('skip', 'warn_compute'):
        raise ValueError("eval_structure_metrics_degenerate_policy must be skip or warn_compute")
    if cfg.motion_analysis.resize_h < 1 or cfg.motion_analysis.resize_w < 1:
        raise ValueError("motion_analysis resize_h/resize_w must be >= 1")
    if cfg.motion_analysis.blur_kernel < 1:
        raise ValueError("motion_analysis blur_kernel must be >= 1")
    if cfg.motion_analysis.frame_gap < 1:
        raise ValueError("motion_analysis frame_gap must be >= 1")
    if cfg.motion_analysis.smooth_window < 1:
        raise ValueError("motion_analysis smooth_window must be >= 1")
    if cfg.motion_analysis.pixel_threshold_mode not in ('adaptive', 'fixed'):
        raise ValueError("motion_analysis pixel_threshold_mode must be one of {'adaptive', 'fixed'}")
    if cfg.motion_analysis.eps <= 0:
        raise ValueError("motion_analysis eps must be > 0")
    if cfg.safety.mode == 'off':
        cfg.safety.enabled = 0
    if cfg.safety.mode not in ('off', 'sim', 'shadow', 'real'):
        raise ValueError("safety.mode must be one of {'off', 'sim', 'shadow', 'real'}")
    if cfg.safety.horizon < 1:
        raise ValueError("safety.horizon must be >= 1")
    if cfg.safety.lbsgd_steps < 0:
        raise ValueError("safety.lbsgd_steps must be >= 0")
    if cfg.safety.min_barrier_margin <= 0:
        raise ValueError("safety.min_barrier_margin must be > 0")
    if cfg.safety.qp_warmup_steps < 0:
        raise ValueError("safety.qp_warmup_steps must be >= 0")
    if cfg.safety.lbsgd_warmup_steps < 0:
        raise ValueError("safety.lbsgd_warmup_steps must be >= 0")
    if cfg.safety.lbsgd_ramp_steps < 0:
        raise ValueError("safety.lbsgd_ramp_steps must be >= 0")
    if cfg.safety.accel_warmup_steps < 0:
        raise ValueError("safety.accel_warmup_steps must be >= 0")
    if cfg.safety.shadow_until_steps < 0:
        raise ValueError("safety.shadow_until_steps must be >= 0")
    if cfg.safety.qpos_margin < 0:
        raise ValueError("safety.qpos_margin must be >= 0")
    if cfg.safety.dq_limit_scale <= 0:
        raise ValueError("safety.dq_limit_scale must be > 0")
    if cfg.safety.ddq_limit_scale <= 0:
        raise ValueError("safety.ddq_limit_scale must be > 0")
    if cfg.safety.tau_limit_scale <= 0:
        raise ValueError("safety.tau_limit_scale must be > 0")
    if cfg.safety.distill_safe_action_weight < 0:
        raise ValueError("safety.distill_safe_action_weight must be >= 0")
    if (cfg.algo.use_hierarchical_policy or cfg.algo.use_hierarchical_skill) and not cfg.cascade.use_cascade:
        raise ValueError("use_hierarchical_policy/use_hierarchical_skill require use_cascade=True")
    if cfg.algo.use_hierarchical_phi:
        if not cfg.cascade.use_cascade:
            raise ValueError("use_hierarchical_phi requires use_cascade=True")
        if not cfg.algo.use_hierarchical_skill:
            raise ValueError("use_hierarchical_phi requires use_hierarchical_skill=True")
        if cfg.algo.hierarchical_phi_depth != cfg.algo.num_skill_levels:
            raise ValueError("hierarchical_phi_depth must match num_skill_levels in the first implementation")
        if not cfg.algo.inner:
            raise ValueError("use_hierarchical_phi currently requires inner=1")
        if get_base_algo_name(cfg) == 'dads':
            raise ValueError("use_hierarchical_phi is currently implemented for METRA-style reward, not dads")
    validate_stage_config(cfg)
    return cfg
