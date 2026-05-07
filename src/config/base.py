from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnvConfig:
    env_backend: str = 'url'
    task: str = 'dmc_walker_walk'
    time_limit: int = 200
    framestack: int = 1
    action_repeat: int = 1
    render_size: int = 64
    flatten_obs: int = 1 # 0 or 1
    camera: str = 'corner'
    dmc_camera: int = -1
    isaaclab_task: str = ''
    isaaclab_num_envs: int = 1
    isaaclab_headless: int = 1
    isaaclab_enable_cameras: Optional[int] = None
    isaaclab_render_mode: str = 'rgb_array'
    isaaclab_image_source: Optional[str] = None
    isaaclab_camera_key: Optional[str] = None
    isaaclab_video_source: str = 'observation'
    isaaclab_video_viewer_preset: str = 'inherit'
    galaxea_sim_headless: int = 1
    galaxea_sim_obs_mode: Optional[str] = None
    galaxea_sim_image_key: str = 'rgb_head'
    galaxea_sim_video_source: str = 'observation'
    galaxea_sim_video_view_preset: str = 'default'
    galaxea_sim_controller_type: str = 'bimanual_joint_position'

@dataclass
class LogConfig:
    workspace_root: str = '/share/shangyy'
    stage: str = 'pre_training' # pre_training, finetune, zero_training
    run_group: str = 'Debug'
    n_epochs: int = 1000000
    n_epochs_per_eval: int = 125
    n_epochs_per_log: int = 25
    n_epochs_per_tb: int = 25
    n_epochs_per_save: int = 1000
    n_epochs_per_pt_save: int = 1000
    n_epochs_per_pkl_update: Optional[int] = None
    num_video_repeats: int = 2
    eval_record_video: int = 1
    video_skip_frames: int = 1
    eval_plot_axis: Optional[List[float]] = None
    num_random_trajectories: int = 48
    ikse: bool = False
    metric_num_sampled_points: int = 10
    dbi_num_rollouts_per_skill: int = 3
    temporal_graph_num_warmup_rollouts: int = 32
    temporal_graph_rollouts_per_skill: int = 5
    temporal_graph_knn_k: int = 8
    temporal_bridge_cost: float = 5.0
    soft_dtw_gamma: float = 1.0
    eval_structure_metrics: bool = False
    eval_structure_metrics_backends: str = 'temporal,ikse'
    eval_structure_metrics_interval: int = 1
    eval_structure_metrics_rollouts_per_skill: int = 3
    eval_structure_metrics_num_skills: int = -1
    eval_structure_metrics_max_trajs: int = 96
    eval_structure_metrics_max_points: int = 1000
    eval_structure_metrics_states_per_traj: int = 10
    eval_structure_metrics_anchor_seed: int = 0
    eval_structure_metrics_use_video_trajectories: bool = True
    eval_structure_metrics_policy_mode: str = 'deterministic'
    eval_structure_metrics_fail_open: bool = True
    eval_structure_metrics_write_legacy_tags: bool = False

@dataclass
class NetworkConfig:
    encoder: int = 1
    encoder_type: str = 'original' # original, resnet-101, dinov3, galaxea-r1lite-triview
    finetune_encoder: bool = False
    spectral_normalization: int = 0
    model_master_dim: int = 1024
    model_master_num_layers: int = 2
    model_master_nonlinearity: Optional[str] = None # relu, tanh
    sd_batch_norm: int = 1 # 0 or 1
    sd_const_std: int = 1 # 0 or 1
    ac_backbone: str = 'mlp' # mlp, simba
    simba_actor_hidden_dim: int = 128
    simba_actor_num_blocks: int = 1
    simba_critic_hidden_dim: int = 512
    simba_critic_num_blocks: int = 2
    simba_mlp_ratio: int = 4
    simba_rsnorm_momentum: float = 0.999
    simba_rsnorm_eps: float = 1e-5
    simba_ln_eps: float = 1e-5
    actor_init_std: float = 1.0
    actor_max_log_std: float = 2.0

@dataclass
class CascadeConfig:
    use_cascade: bool = False
    num_policy_levels: int = 2
    epochs_per_policy_stage: int = 50
    cascade_init_from_prev: bool = True
    cascade_gate_type: str = 'scalar'  # 'scalar' or 'vector'
    cascade_min_lambda: float = 0.01
    cascade_max_lambda: float = 0.99

@dataclass
class AutoBranchConfig:
    enabled: bool = False
    check_interval_epochs: int = 20
    recent_buffer_epochs: int = 100
    knn_k: int = 5
    representative_points_per_traj: int = 10
    distance_mode: str = 'knn_mean'
    m_policy_ratio_threshold: float = 0.1
    split_patience: int = 3
    min_branch_age: int = 150
    fresh_rollout_episodes: int = 48
    visualize_on_split: bool = True
    visualize_dir: str = 'split_viz'
    use_global_recent_buffer: bool = False
    seeded_probe_skills: bool = True

@dataclass
class MotionAnalysisConfig:
    enabled: int = 0
    video_path: str = ''
    resize_h: int = 128
    resize_w: int = 128
    blur_kernel: int = 3
    frame_gap: int = 1
    pixel_threshold_mode: str = 'adaptive'
    fixed_tau_p: float = 0.04
    smooth_window: int = 5
    large_motion_threshold: float = 2.0
    eps: float = 1e-8

@dataclass
class SafetyConfig:
    enabled: int = 0
    mode: str = "sim"   # off, sim, shadow, real
    lbsgd_enabled: int = 1
    qp_enabled: int = 1
    supervisor_enabled: int = 1
    actor_critic_pixel_only: int = 1

    action_semantics: str = "auto"  # auto, absolute_joint_position, delta_joint_position, joint_velocity
    dt: float = 0.05
    horizon: int = 1

    barrier_eta: float = 1e-2
    lbsgd_lr: float = 1e-2
    lbsgd_steps: int = 8
    lbsgd_backtrack_steps: int = 10
    min_barrier_margin: float = 1e-4
    deviation_weight: float = 1.0
    critic_weight: float = 0.0
    qp_warmup_steps: int = 0
    lbsgd_warmup_steps: int = 0
    lbsgd_ramp_steps: int = 0
    accel_warmup_steps: int = 0
    shadow_until_steps: int = 0

    qpos_margin: float = 0.05
    dq_limit_scale: float = 0.25
    ddq_limit_scale: float = 0.25
    tau_limit_scale: float = 0.25

    self_collision_min_dist: float = 0.08
    env_collision_min_dist: float = 0.08
    cable_min_dist: float = 0.12

    lock_torso: int = 1
    lock_chassis: int = 1
    stop_on_missing_safety_state: int = 1

    distill_safe_action_weight: float = 0.0
    log_every_steps: int = 100

    safety_yaml: str = ""

@dataclass
class AlgoConfig:
    algo: str = 'metra' # metra, dads, idk_csd
    dim_skill: int = 2
    discrete: int = 0 # 0 or 1
    inner: int = 1 # 0 or 1
    unit_length: int = 1 # 0 or 1 (only continuous)
    traj_latent_norm: str = 'off'
    traj_latent_norm_eps: float = 1e-5
    dual_dist: str = 'one' # l2, s2_from_s, one, ...
    dual_reg: int = 1
    dual_lam: float = 30
    dual_slack: float = 1e-3
    num_alt_samples: int = 100
    split_group: int = 65536
    use_target_traj_encoder: bool = False
    alpha: float = 0.01
    
    # Contrastive
    contrastive_n_epochs: int = 5
    contrastive_m_epochs: int = 5
    contrastive_warmup_epochs: int = 5
    contrastive_temperature: float = 0.1
    idk_update_interval: int = 200
    
    # KME / IDK
    use_kme: bool = False
    update_idk: int = 1000
    idk_subsample_size: int = 256
    idk_init: str = 'gaussian'
    idk_from: str = 'traj'
    idk_groups: int = 1
    kernel_map: bool = False
    use_novelty_reward: bool = False
    use_hierarchical_policy: bool = False
    use_hierarchical_skill: bool = False
    num_skill_levels: int = 1
    epochs_per_skill_stage: int = 50
    use_hierarchical_phi: bool = False
    hierarchical_phi_depth: int = 0
    beta_mode: str = 'uniform'
    beta_rho: float = 0.5
    log_beta_values: bool = True

@dataclass
class TrainConfig:
    n_parallel: int = 4
    n_thread: int = 1
    traj_batch_size: int = 8
    trans_minibatch_size: int = 256
    trans_optimization_epochs: int = 200
    lr_op: Optional[float] = None
    lr_te: Optional[float] = None
    dual_lr: Optional[float] = None
    sac_lr_q: Optional[float] = None
    sac_lr_a: Optional[float] = None
    common_lr: float = 1e-4
    grad_clip_norm: float = 50.0
    sac_tau: float = 5e-3
    sac_discount: float = 0.99
    sac_scale_reward: float = 1.
    sac_target_coef: float = 1.
    sac_min_buffer_size: int = 10000
    sac_max_buffer_size: int = 300000
    policy_delay: int = 1
    actor_start_steps: int = 0
    skill_policy_path: str = ''

@dataclass
class MetraConfig:
    seed: int = 0
    device: str = 'cuda'
    use_gpu: int = 1
    sample_cpu: int = 1
    
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    auto_branch: AutoBranchConfig = field(default_factory=AutoBranchConfig)
    motion_analysis: MotionAnalysisConfig = field(default_factory=MotionAnalysisConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # For backward compatibility with flattened access in agent/trainer
    def __getattr__(self, name):
        # ``copy.deepcopy`` reconstructs dataclasses before all fields are
        # restored and probes special methods such as ``__setstate__``. Reading
        # ``self.env`` in that window calls this method again, so consult the
        # raw instance dict and skip fields that are not present yet.
        try:
            instance_dict = object.__getattribute__(self, "__dict__")
        except AttributeError:
            instance_dict = {}
        for field_name in (
                "env",
                "log",
                "net",
                "algo",
                "train",
                "cascade",
                "auto_branch",
                "motion_analysis",
                "safety"):
            sub_cfg = instance_dict.get(field_name)
            if sub_cfg is None:
                continue
            if hasattr(sub_cfg, name):
                return getattr(sub_cfg, name)
        raise AttributeError(f"'MetraConfig' object has no attribute '{name}'")
